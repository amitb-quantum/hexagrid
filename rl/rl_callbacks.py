"""
HexaGrid Phase 7 — Training Callbacks
=====================================
Custom Stable-Baselines3 callbacks for:
  - Live training metrics logged to JSON (feeds the dashboard)
  - Best model checkpointing
  - Early stopping on plateau

Install:
    cp rl_callbacks.py ~/hexagrid/rl/callbacks.py
"""

import os
import json
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HexaGridTrainingLogger(BaseCallback):
    """
    Logs per-episode metrics to training_log.json every `log_freq` episodes.
    The dashboard polls GET /api/v1/rl/training_log to display live curves.
    """

    def __init__(self, log_dir: str, log_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir  = log_dir
        self.log_freq = log_freq
        self.log_path = os.path.join(log_dir, 'training_log.json')
        os.makedirs(log_dir, exist_ok=True)

        self._episode_rewards: list = []
        self._episode_savings: list = []
        self._episode_sla:     list = []
        self._ep_count = 0
        self._t_start  = time.time()
        self._flush()   # write empty log on init

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._episode_rewards.append(float(info['episode']['r']))
                self._ep_count += 1
            if 'savings_pct' in info:
                self._episode_savings.append(float(info['savings_pct']))
            if 'total_sla_penalty' in info:
                self._episode_sla.append(float(info['total_sla_penalty']))

        if self._ep_count > 0 and self._ep_count % self.log_freq == 0:
            self._flush()
        return True

    def _flush(self):
        n = len(self._episode_rewards)
        w = min(100, n) if n > 0 else 1

        if n == 0:
            data = {
                'episodes': 0, 'timesteps': self.num_timesteps,
                'elapsed_sec': round(time.time() - self._t_start, 1),
                'mean_reward': None, 'mean_savings_pct': None,
                'reward_history': [], 'savings_history': [],
                'status': 'initializing',
            }
        else:
            data = {
                'episodes':         n,
                'timesteps':        self.num_timesteps,
                'elapsed_sec':      round(time.time() - self._t_start, 1),
                'mean_reward':      round(float(np.mean(self._episode_rewards[-w:])), 4),
                'std_reward':       round(float(np.std(self._episode_rewards[-w:])), 4),
                'best_reward':      round(float(np.max(self._episode_rewards)), 4),
                'mean_savings_pct': round(float(np.mean(self._episode_savings[-w:])), 2)
                                    if self._episode_savings else None,
                'mean_sla_penalty': round(float(np.mean(self._episode_sla[-w:])), 4)
                                    if self._episode_sla else None,
                'reward_history':   [round(r, 3) for r in self._episode_rewards[-200:]],
                'savings_history':  [round(s, 2) for s in self._episode_savings[-200:]]
                                    if self._episode_savings else [],
                'status': 'training',
            }
            if self.verbose >= 1:
                print(
                    f"  [Ep {n:4d}] reward={data['mean_reward']:+.3f} | "
                    f"savings={data['mean_savings_pct']}% | "
                    f"steps={data['timesteps']:,}"
                )

        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)


class BestModelCheckpoint(BaseCallback):
    """Saves the model whenever mean reward over last 50 episodes improves."""

    def __init__(self, save_path: str, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path   = save_path
        self.check_freq  = check_freq
        self.best_reward = -np.inf
        self._rewards:   list = []
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._rewards.append(float(info['episode']['r']))

        if self.n_calls % self.check_freq == 0 and len(self._rewards) >= 10:
            mean_r = float(np.mean(self._rewards[-50:]))
            if mean_r > self.best_reward:
                self.best_reward = mean_r
                path = os.path.join(self.save_path, 'best_model')
                self.model.save(path)
                if self.verbose >= 1:
                    print(f"  [Checkpoint] New best: {mean_r:+.4f} → {path}.zip")
        return True


class EarlyStoppingCallback(BaseCallback):
    """Stops training if mean reward hasn't improved by min_delta over patience checks."""

    def __init__(self, check_freq: int = 5000, patience: int = 5,
                 min_delta: float = 0.01, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq  = check_freq
        self.patience    = patience
        self.min_delta   = min_delta
        self._best       = -np.inf
        self._no_improve = 0
        self._rewards:   list = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._rewards.append(float(info['episode']['r']))

        if self.n_calls % self.check_freq == 0 and len(self._rewards) >= 20:
            mean_r = float(np.mean(self._rewards[-50:]))
            if mean_r > self._best + self.min_delta:
                self._best       = mean_r
                self._no_improve = 0
            else:
                self._no_improve += 1
                if self.verbose:
                    print(f"  [EarlyStop] No improvement {self._no_improve}/{self.patience}")
                if self._no_improve >= self.patience:
                    if self.verbose:
                        print(f"  [EarlyStop] Stopping. Best reward: {self._best:+.4f}")
                    return False
        return True
