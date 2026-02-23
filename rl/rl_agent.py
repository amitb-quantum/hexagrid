"""
Energia Phase 7 — RL Agent Inference Wrapper
=============================================
Loads a trained PPO model and exposes a clean recommend() interface
for the FastAPI layer to call in real-time.

Install into project:
    cp rl_agent.py ~/energia/rl/agent.py

Usage:
    from rl.agent import EnergiaAgent

    agent = EnergiaAgent()           # auto-loads best_model.zip
    rec   = agent.recommend({
        'current_price':  0.045,
        'price_forecast': [0.044, 0.041, 0.038, 0.036, ...],  # 12 values
        'queue_depth':    3,
        'job_urgency':    0.4,
        'gpu_utilization':0.65,
        'pue':            1.35,
        'ups_charge':     0.55,
        'cooling_power':  0.5,
        'hour_of_day':    14.5,
        'day_of_week':    1,
    })
    # Returns structured recommendation dict with dispatch/cooling/ups actions + reasoning
"""

import os
import sys
import json
import time
import math
import numpy as np
from typing import Optional

# ── Paths (relative to project root) ──────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT, 'models', 'rl')
LOG_DIR   = os.path.join(ROOT, 'logs',   'rl')

# ── Action label maps ──────────────────────────────────────────────────────────
DISPATCH_LABELS = {0: 'hold',    1: 'dispatch_now', 2: 'defer_15min'}
COOLING_LABELS  = {0: 'reduce',  1: 'maintain',     2: 'boost'}
UPS_LABELS      = {0: 'charge',  1: 'idle',          2: 'discharge'}

DISPATCH_DESC = {
    'hold':         'Hold — price is above forecast minimum, wait for cheaper window',
    'dispatch_now': 'Dispatch now — current price is at or near the forecast minimum',
    'defer_15min':  'Defer 15 min — a cheaper price window is expected shortly',
}
COOLING_DESC = {
    'reduce':   'Reduce cooling power — GPU load is low, save cooling energy',
    'maintain': 'Maintain cooling — load is balanced, no adjustment needed',
    'boost':    'Boost cooling — high GPU utilization, prevent thermal throttling',
}
UPS_DESC = {
    'charge':    'Charge UPS — grid price is low, store energy for future peaks',
    'idle':      'UPS idle — price/state neutral, no charge or discharge action',
    'discharge': 'Discharge UPS — grid price is high, use stored energy instead',
}


class EnergiaAgent:
    """
    Wraps a trained Stable-Baselines3 PPO model for live inference.

    Auto-loads in this order:
      1. models/rl/best_model.zip   (highest-reward checkpoint)
      2. models/rl/final_model.zip  (fallback)

    Thread-safe: predict() is stateless and fast (<5ms per call).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self._model      = None
        self._model_path = None
        self._loaded_at  = None
        self._device     = device   # cpu for inference — avoids CUDA memory contention
        self._ready      = False
        self._load_error = None

        if model_path:
            self._load(model_path)
        else:
            self._autoload()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _autoload(self):
        for name in ('best_model', 'final_model'):
            path = os.path.join(MODEL_DIR, name)
            if os.path.exists(path + '.zip'):
                self._load(path)
                return
        self._load_error = (
            f"No trained model found in {MODEL_DIR}. "
            f"Train first:  python rl/train.py --steps 200000 --run run_001"
        )
        print(f"  [Agent] {self._load_error}")

    def _load(self, path: str):
        try:
            from stable_baselines3 import PPO
            self._model      = PPO.load(path, device=self._device)
            self._model_path = path + '.zip' if not path.endswith('.zip') else path
            self._loaded_at  = time.strftime('%Y-%m-%dT%H:%M:%S')
            self._ready      = True
            self._load_error = None
            print(f"  [Agent] Loaded: {self._model_path}  (device={self._model.device})")
        except Exception as e:
            self._load_error = str(e)
            self._ready      = False
            print(f"  [Agent] Load failed: {e}")

    def reload(self):
        """Hot-reload after training completes — called by the API train endpoint."""
        print("  [Agent] Hot-reloading model...")
        self._autoload()

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Observation builder ────────────────────────────────────────────────────

    def build_observation(
        self,
        current_price:    float,
        price_forecast:   list,
        queue_depth:      int   = 1,
        job_urgency:      float = 0.3,
        gpu_utilization:  float = 0.5,
        pue:              float = 1.4,
        ups_charge:       float = 0.5,
        cooling_power:    float = 0.5,
        hour_of_day:      float = 12.0,
        day_of_week:      int   = 0,
        step_progress:    float = 0.5,
        last_dispatch:    int   = 0,
        last_reward:      float = 0.0,
        **kwargs,                         # absorb extra keys gracefully
    ) -> np.ndarray:
        """
        Build the 26-dim observation vector from live API/dashboard data.
        Matches EnergiaEnv._get_obs() exactly.
        """
        PRICE_MIN, PRICE_MAX = 0.005, 0.15

        def norm(p):
            return float(np.clip((p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN), 0.0, 1.0))

        # Pad/trim forecast to exactly 12 steps
        fc = list(price_forecast)
        while len(fc) < 12:
            fc.append(current_price)
        fc = fc[:12]

        tod = (hour_of_day / 24.0) * 2 * math.pi
        dow = (day_of_week  / 7.0)  * 2 * math.pi

        return np.array([
            norm(current_price),
            *[norm(p) for p in fc],
            float(min(1.0, queue_depth / 10.0)),
            float(np.clip(job_urgency, 0.0, 1.0)),
            float(np.clip(gpu_utilization, 0.0, 1.0)),
            float(np.clip(pue - 1.0, 0.0, 1.0)),
            float(np.clip(ups_charge, 0.0, 1.0)),
            float(np.clip(cooling_power, 0.0, 1.0)),
            math.sin(tod), math.cos(tod),
            math.sin(dow), math.cos(dow),
            float(np.clip(step_progress, 0.0, 1.0)),
            float(last_dispatch) / 2.0,
            float(np.clip(last_reward, -1.0, 1.0)) * 0.5 + 0.5,
        ], dtype=np.float32)

    # ── Inference ──────────────────────────────────────────────────────────────

    def recommend(self, obs_dict: dict) -> dict:
        """
        Main inference entry point.

        Args:
            obs_dict: dict with current_price + price_forecast required;
                      all other keys optional (defaults used if missing).

        Returns:
            {
              dispatch:               'dispatch_now' | 'hold' | 'defer_15min'
              cooling:                'boost' | 'maintain' | 'reduce'
              ups:                    'charge' | 'idle' | 'discharge'
              confidence:             0.0 – 1.0
              estimated_savings_pct:  float
              reasoning:              str
              action_descriptions:    {dispatch, cooling, ups}
              raw_action:             [int, int, int]
            }
        """
        if not self._ready:
            return {
                'status':   'not_ready',
                'error':    self._load_error or 'Model not loaded',
                'dispatch': None, 'cooling': None, 'ups': None,
            }

        obs = self.build_observation(**obs_dict)
        action, _ = self._model.predict(obs, deterministic=True)
        d_idx, c_idx, u_idx = int(action[0]), int(action[1]), int(action[2])

        dispatch = DISPATCH_LABELS[d_idx]
        cooling  = COOLING_LABELS[c_idx]
        ups      = UPS_LABELS[u_idx]

        confidence          = self._estimate_confidence(obs)
        estimated_savings   = self._estimate_savings(obs_dict, dispatch)
        reasoning           = self._build_reasoning(obs_dict, dispatch, cooling, ups)

        return {
            'status':                'ok',
            'dispatch':              dispatch,
            'cooling':               cooling,
            'ups':                   ups,
            'confidence':            round(confidence, 3),
            'estimated_savings_pct': round(estimated_savings, 2),
            'reasoning':             reasoning,
            'action_descriptions': {
                'dispatch': DISPATCH_DESC[dispatch],
                'cooling':  COOLING_DESC[cooling],
                'ups':      UPS_DESC[ups],
            },
            'raw_action':  [d_idx, c_idx, u_idx],
            'model_path':  self._model_path,
            'loaded_at':   self._loaded_at,
            'timestamp':   time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

    def _estimate_confidence(self, obs: np.ndarray) -> float:
        """Confidence from policy distribution entropy. Low entropy = high confidence."""
        try:
            import torch
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._model.device)
            with torch.no_grad():
                dist = self._model.policy.get_distribution(obs_t)
                if hasattr(dist, 'distribution'):
                    # MultiDiscrete: list of Categorical distributions
                    entropies   = [d.entropy().item() for d in dist.distribution]
                    max_entropy = math.log(3)   # Categorical(3) max entropy
                    mean_ent    = float(np.mean(entropies))
                    return float(np.clip(1.0 - mean_ent / max_entropy, 0.0, 1.0))
        except Exception:
            pass
        return 0.5

    def _estimate_savings(self, obs_dict: dict, dispatch: str) -> float:
        """Rough savings estimate vs naive always-dispatch baseline."""
        current = obs_dict.get('current_price', 0.05)
        fc      = obs_dict.get('price_forecast', [current] * 12)
        min_p   = min(fc) if fc else current
        if dispatch == 'dispatch_now':
            ratio = current / max(min_p, 0.001)
            return max(0.0, (1.0 - ratio) * 100)
        elif dispatch == 'defer_15min':
            ratio = min_p / max(current, 0.001)
            return max(0.0, (1.0 - ratio) * 100)
        return 0.0

    def _build_reasoning(self, obs_dict: dict, dispatch: str, cooling: str, ups: str) -> str:
        current    = obs_dict.get('current_price', 0.05)
        fc         = obs_dict.get('price_forecast', [current] * 12)
        min_p      = min(fc) if fc else current
        min_offset = int(np.argmin(fc)) * 5 if fc else 0
        urgency    = obs_dict.get('job_urgency', 0.0)
        ups_soc    = obs_dict.get('ups_charge', 0.5)
        queue      = obs_dict.get('queue_depth', 0)

        parts = []

        if dispatch == 'dispatch_now':
            spread = ((current - min_p) / max(min_p, 0.001)) * 100
            parts.append(
                f"Dispatching now at ${current:.4f}/kWh "
                f"({spread:+.1f}% vs forecast minimum ${min_p:.4f}/kWh)"
            )
        elif dispatch == 'defer_15min':
            parts.append(
                f"Deferring — cheaper window in ~{min_offset}min "
                f"(${min_p:.4f}/kWh vs current ${current:.4f}/kWh)"
            )
        else:
            parts.append(f"Holding at ${current:.4f}/kWh, waiting for better price")

        if urgency > 0.8:
            parts.append(f"SLA pressure high ({urgency*100:.0f}% of deadline)")
        elif queue == 0:
            parts.append("No jobs in queue")

        if cooling == 'boost':
            parts.append("Boosting cooling for high GPU load")
        elif cooling == 'reduce':
            parts.append("Reducing cooling — GPU utilization is low")

        if ups == 'charge':
            parts.append(f"Charging UPS (currently {ups_soc*100:.0f}%) during low-price period")
        elif ups == 'discharge':
            parts.append(f"Discharging UPS ({ups_soc*100:.0f}% SOC) to offset high grid price")

        return '. '.join(parts) + '.'

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return agent + training status for GET /api/v1/rl/status."""
        training_log = {}
        summary      = {}

        for path, target in [
            (os.path.join(LOG_DIR, 'training_log.json'),    training_log),
            (os.path.join(LOG_DIR, 'training_summary.json'), summary),
        ]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        target.update(json.load(f))
                except Exception:
                    pass

        return {
            'ready':      self._ready,
            'model_path': self._model_path,
            'loaded_at':  self._loaded_at,
            'error':      self._load_error,
            'training':   training_log,
            'summary':    summary,
        }
