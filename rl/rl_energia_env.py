"""
Energia Phase 7 — Reinforcement Learning Environment
======================================================
Custom Gymnasium environment that wraps the Energia Digital Twin.

The agent observes:
  - Current grid electricity price
  - 12-step price forecast (next 60 minutes, 5-min intervals)
  - Pending job queue depth and urgency
  - Current GPU utilization and PUE
  - UPS charge state
  - Cyclical time-of-day and day-of-week encodings

The agent can take 9 discrete actions (3x3 MultiDiscrete):
  Dimension 0 — Job Dispatch:   0=Hold,   1=Dispatch now,  2=Defer 15min
  Dimension 1 — Cooling mode:   0=Reduce, 1=Maintain,      2=Boost
  Dimension 2 — UPS mode:       0=Charge, 1=Idle,          2=Discharge

Reward = cost_savings_vs_naive  - sla_penalty - efficiency_penalty

Install into your project:
    cp energia_env.py ~/energia/rl/energia_env.py

Usage:
    from rl.energia_env import EnergiaEnv
    env = EnergiaEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Optional
import math


# ── Simulated job types matching the existing scheduler ────────────────────────
JOB_CATALOG = [
    {'name': 'LLM_Training',        'duration_min': 45, 'power_kw': 8.0, 'priority': 2},
    {'name': 'Diffusion_Inference',  'duration_min': 15, 'power_kw': 4.5, 'priority': 3},
    {'name': 'RL_Training',          'duration_min': 60, 'power_kw': 7.5, 'priority': 1},
    {'name': 'Batch_Inference',      'duration_min': 20, 'power_kw': 5.0, 'priority': 3},
    {'name': 'Embedding_Generation', 'duration_min': 10, 'power_kw': 3.0, 'priority': 4},
    {'name': 'Fine_Tuning',          'duration_min': 90, 'power_kw': 9.0, 'priority': 1},
    {'name': 'Data_Preprocessing',   'duration_min': 30, 'power_kw': 2.5, 'priority': 4},
]

# ── Reward shaping constants ───────────────────────────────────────────────────
SLA_PENALTY_PER_MIN  = 0.05   # penalty per minute a job is overdue
MAX_DEFER_MINUTES    = 45     # after this, job is considered SLA breach
COOLING_MISMATCH_PEN = 0.02   # per-step penalty for overcooling/undercooling
UPS_DISCHARGE_BONUS  = 0.015  # small bonus for using stored energy during peaks
DISPATCH_BONUS       = 0.10   # bonus for dispatching during a cheap window


class PriceSimulator:
    """
    Synthetic electricity price simulator.
    Generates realistic TOU-style price curves with noise.
    On production, replace price_at()/forecast() to pull from the EIA connector.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.t   = 0
        self.day = 0

    def reset(self, t_start: Optional[int] = None):
        self.t   = t_start if t_start is not None else int(self.rng.integers(0, 1440))
        self.day = int(self.rng.integers(0, 7))

    def _base_price(self, t_min: int, day: int) -> float:
        hour    = (t_min % 1440) / 60
        weekend = 0.85 if day in (5, 6) else 1.0
        if 6 <= hour < 8:
            base = 0.04 + (hour - 6) / 2 * 0.04
        elif 8 <= hour < 22:
            morning_peak = math.exp(-0.5 * ((hour - 12) / 3) ** 2)
            evening_peak = math.exp(-0.5 * ((hour - 18) / 2) ** 2)
            base = 0.06 + 0.04 * max(morning_peak, evening_peak)
        else:
            base = 0.025 + 0.015 * math.exp(-0.5 * ((hour - 3) / 2) ** 2)
        return base * weekend

    def price_at(self, t_offset_min: int = 0) -> float:
        t     = self.t + t_offset_min
        base  = self._base_price(t, self.day)
        noise = self.rng.normal(0, 0.004)
        return float(max(0.005, base + noise))

    def forecast(self, n_steps: int = 12, step_min: int = 5) -> np.ndarray:
        return np.array([self.price_at(i * step_min) for i in range(n_steps)])

    def step(self, dt_min: int = 5):
        self.t += dt_min
        if self.t >= 1440:
            self.t -= 1440
            self.day = (self.day + 1) % 7


class JobQueue:
    """Manages a queue of pending GPU jobs with urgency tracking."""

    def __init__(self, rng: np.random.Generator):
        self.rng   = rng
        self.queue = deque()
        self.done  = []

    def reset(self):
        self.queue.clear()
        self.done.clear()
        for _ in range(int(self.rng.integers(2, 5))):
            self._spawn_job(age_min=int(self.rng.integers(0, 20)))

    def _spawn_job(self, age_min: int = 0):
        template = self.rng.choice(JOB_CATALOG)
        self.queue.append({
            **template,
            'age_min': age_min,
            'job_id':  int(self.rng.integers(1000, 9999)),
        })

    def step(self, dt_min: int = 5):
        for job in self.queue:
            job['age_min'] += dt_min
        if self.rng.random() < 0.2:
            self._spawn_job()

    def dispatch_next(self):
        return self.queue.popleft() if self.queue else None

    def peek(self):
        return self.queue[0] if self.queue else None

    @property
    def depth(self) -> int:
        return len(self.queue)

    @property
    def max_age(self) -> float:
        return max((j['age_min'] for j in self.queue), default=0.0)

    @property
    def sla_breach_count(self) -> int:
        return sum(1 for j in self.queue if j['age_min'] > MAX_DEFER_MINUTES)

    @property
    def urgency(self) -> float:
        return min(1.0, self.max_age / MAX_DEFER_MINUTES) if self.queue else 0.0


class EnergiaEnv(gym.Env):
    """
    Energia RL Environment — 26-dim observation, MultiDiscrete([3,3,3]) actions.

    Observation vector:
      [0]     current price (normalized)
      [1-12]  12-step price forecast
      [13]    queue depth (normalized, max=10)
      [14]    job urgency (0-1)
      [15]    GPU utilization (0-1)
      [16]    PUE normalized (1.0-2.0 → 0-1)
      [17]    UPS charge (0-1)
      [18]    cooling power (0-1)
      [19-20] sin/cos time-of-day
      [21-22] sin/cos day-of-week
      [23]    episode progress (0-1)
      [24]    last dispatch action (0-1)
      [25]    last reward (normalized)

    Actions:
      [0] Dispatch: 0=Hold  1=Dispatch  2=Defer+15min
      [1] Cooling:  0=Reduce  1=Maintain  2=Boost
      [2] UPS:      0=Charge  1=Idle  2=Discharge
    """

    metadata  = {'render_modes': ['human']}
    MAX_STEPS = 288       # 24 hours at 5-min steps
    PRICE_MIN = 0.005
    PRICE_MAX = 0.15

    def __init__(
        self,
        n_racks:     int            = 4,
        seed:        Optional[int]  = None,
        render_mode: Optional[str]  = None,
    ):
        super().__init__()
        self.n_racks     = n_racks
        self.render_mode = render_mode

        self._rng        = np.random.default_rng(seed)
        self._price_sim  = PriceSimulator(seed=seed)
        self._job_queue  = JobQueue(self._rng)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # State vars
        self._gpu_util      = 0.5
        self._pue           = 1.4
        self._ups_charge    = 0.5
        self._cooling_power = 0.5
        self._step_count    = 0
        self._last_action   = np.zeros(3, dtype=np.float32)
        self._last_reward   = 0.0
        self._episode_cost  = 0.0
        self._naive_episode_cost = 0.0
        self._dispatched_jobs    = []
        self._total_sla_penalty  = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng       = np.random.default_rng(seed)
            self._price_sim = PriceSimulator(seed=seed)
            self._job_queue = JobQueue(self._rng)

        self._price_sim.reset()
        self._job_queue.reset()

        self._gpu_util           = float(self._rng.uniform(0.2, 0.8))
        self._pue                = float(self._rng.uniform(1.2, 1.6))
        self._ups_charge         = float(self._rng.uniform(0.3, 0.8))
        self._cooling_power      = float(self._rng.uniform(0.3, 0.7))
        self._step_count         = 0
        self._last_action        = np.zeros(3, dtype=np.float32)
        self._last_reward        = 0.0
        self._episode_cost       = 0.0
        self._naive_episode_cost = 0.0
        self._dispatched_jobs    = []
        self._total_sla_penalty  = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action):
        dispatch_act, cooling_act, ups_act = int(action[0]), int(action[1]), int(action[2])
        dt_min = 5

        current_price      = self._price_sim.price_at()
        forecast           = self._price_sim.forecast()
        min_forecast_price = float(np.min(forecast))
        reward             = 0.0

        # ── 1. Job dispatch ───────────────────────────────────────────────────
        if self._job_queue.depth > 0:
            job      = self._job_queue.peek()
            power_kw = job['power_kw'] * self.n_racks
            naive_cost = (power_kw * job['duration_min'] / 60.0) * current_price

            if dispatch_act == 1:
                dispatched = self._job_queue.dispatch_next()
                job_cost   = (power_kw * dispatched['duration_min'] / 60.0) * current_price
                self._gpu_util = min(1.0, self._gpu_util + 0.3)
                self._dispatched_jobs.append({
                    'job': dispatched, 'price': current_price, 'step': self._step_count
                })
                if current_price <= min_forecast_price * 1.05:
                    reward += DISPATCH_BONUS
            elif dispatch_act == 2:
                for j in self._job_queue.queue:
                    j['age_min'] += 15
                job_cost = 0.0
            else:
                job_cost = 0.0

            self._episode_cost       += job_cost
            self._naive_episode_cost += naive_cost
            if naive_cost > 0:
                reward += ((naive_cost - job_cost) / naive_cost) * 0.5
            else:
                reward -= 0.01

        # ── 2. SLA penalty ────────────────────────────────────────────────────
        sla_penalty = self._job_queue.sla_breach_count * SLA_PENALTY_PER_MIN * dt_min
        reward -= sla_penalty
        self._total_sla_penalty += sla_penalty

        # ── 3. Cooling ────────────────────────────────────────────────────────
        target_cooling = min(1.0, self._gpu_util * 1.2)
        if cooling_act == 0:
            self._cooling_power = max(0.1, self._cooling_power - 0.1)
        elif cooling_act == 2:
            self._cooling_power = min(1.0, self._cooling_power + 0.1)
        cooling_mismatch = abs(self._cooling_power - target_cooling)
        reward -= cooling_mismatch * COOLING_MISMATCH_PEN
        cooling_cost = (self._cooling_power * 2.0 * self.n_racks * dt_min / 60.0) * current_price
        reward -= cooling_cost * 0.1

        # ── 4. UPS ────────────────────────────────────────────────────────────
        if ups_act == 0:
            self._ups_charge = min(1.0, self._ups_charge + 0.05)
            if current_price < 0.04:
                reward += UPS_DISCHARGE_BONUS
        elif ups_act == 2:
            if self._ups_charge > 0.1:
                self._ups_charge = max(0.0, self._ups_charge - 0.05)
                if current_price > 0.07:
                    reward += UPS_DISCHARGE_BONUS * 2
            else:
                reward -= 0.02

        # ── 5. PUE update ─────────────────────────────────────────────────────
        self._pue = float(np.clip(
            1.0 + self._cooling_power * 0.6 + (1 - self._gpu_util) * 0.2, 1.0, 2.0
        ))

        # ── 6. GPU decay ──────────────────────────────────────────────────────
        self._gpu_util = max(0.1, self._gpu_util - 0.05)

        # ── 7. Advance time ───────────────────────────────────────────────────
        self._price_sim.step(dt_min)
        self._job_queue.step(dt_min)
        self._step_count += 1

        self._last_action = np.array([dispatch_act, cooling_act, ups_act], dtype=np.float32)
        self._last_reward = float(reward)

        terminated = False
        truncated  = self._step_count >= self.MAX_STEPS

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _normalize_price(self, p: float) -> float:
        return float(np.clip((p - self.PRICE_MIN) / (self.PRICE_MAX - self.PRICE_MIN), 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        current_price = self._price_sim.price_at()
        forecast      = self._price_sim.forecast(n_steps=12)
        tod = (self._price_sim.t % 1440) / 1440 * 2 * math.pi
        dow = self._price_sim.day / 7 * 2 * math.pi
        return np.array([
            self._normalize_price(current_price),
            *[self._normalize_price(p) for p in forecast],
            min(1.0, self._job_queue.depth / 10.0),
            self._job_queue.urgency,
            self._gpu_util,
            (self._pue - 1.0),
            self._ups_charge,
            self._cooling_power,
            math.sin(tod), math.cos(tod),
            math.sin(dow), math.cos(dow),
            self._step_count / self.MAX_STEPS,
            float(self._last_action[0]) / 2.0,
            float(np.clip(self._last_reward, -1.0, 1.0)) * 0.5 + 0.5,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        current_price = self._price_sim.price_at()
        savings_pct = 0.0
        if self._naive_episode_cost > 0:
            savings_pct = (
                (self._naive_episode_cost - self._episode_cost)
                / self._naive_episode_cost * 100
            )
        return {
            'step':               self._step_count,
            'current_price':      round(current_price, 5),
            'gpu_utilization':    round(self._gpu_util, 3),
            'pue':                round(float(self._pue), 4),
            'ups_charge':         round(self._ups_charge, 3),
            'queue_depth':        self._job_queue.depth,
            'sla_breaches':       self._job_queue.sla_breach_count,
            'episode_cost':       round(self._episode_cost, 5),
            'naive_episode_cost': round(self._naive_episode_cost, 5),
            'savings_pct':        round(savings_pct, 2),
            'total_sla_penalty':  round(self._total_sla_penalty, 4),
            'dispatched_jobs':    len(self._dispatched_jobs),
        }

    def render(self):
        if self.render_mode == 'human':
            info = self._get_info()
            print(
                f"Step {info['step']:3d} | "
                f"Price ${info['current_price']:.4f} | "
                f"PUE {info['pue']:.3f} | "
                f"GPU {info['gpu_utilization']*100:.0f}% | "
                f"UPS {info['ups_charge']*100:.0f}% | "
                f"Queue {info['queue_depth']} | "
                f"Savings {info['savings_pct']:+.1f}%"
            )

    def close(self):
        pass
