"""
Energia Phase 7 — PPO Training Script
=======================================
Trains a PPO agent on the EnergiaEnv using Stable-Baselines3.

Install into project:
    cp rl_train.py ~/energia/rl/train.py

Usage:
    # Quick smoke test (~30s)
    python rl/train.py --steps 5000 --envs 1 --run smoke

    # Standard training (~15 min on RTX GPU)
    python rl/train.py --steps 500000 --envs 8 --run full_v1

    # Evaluate a saved model
    python rl/train.py --eval models/rl/best_model

    # Resume from checkpoint
    python rl/train.py --steps 500000 --resume models/rl/best_model --run full_v2
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from energia_env import EnergiaEnv
from callbacks import EnergiaTrainingLogger, BestModelCheckpoint, EarlyStoppingCallback

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT, 'models', 'rl')
LOG_DIR   = os.path.join(ROOT, 'logs',   'rl')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
PPO_KWARGS = dict(
    policy        = 'MlpPolicy',
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.995,    # high gamma: future energy savings matter
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,     # encourage exploration of dispatch strategies
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(
        net_arch      = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn = torch.nn.ReLU,
    ),
    verbose = 1,
)


def train(
    total_steps: int  = 200_000,
    n_envs:      int  = 8,
    n_racks:     int  = 4,
    run_name:    str  = 'run_001',
    resume_path: str  = None,
    eval_freq:   int  = 10_000,
    log_freq:    int  = 20,
    device:      str  = 'auto',
) -> PPO:

    print(f"\n{'='*58}")
    print(f"  Energia Phase 7 — PPO Training")
    print(f"{'='*58}")
    print(f"  Run:     {run_name}")
    print(f"  Steps:   {total_steps:,}  |  Envs: {n_envs}  |  Racks: {n_racks}")
    print(f"  Device:  {device}  |  Models: {MODEL_DIR}")
    print(f"{'='*58}\n")

    # ── Environments ──────────────────────────────────────────────────────────
    # Use DummyVecEnv for single env (avoids subprocess overhead in containers)
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_cls = DummyVecEnv  # safe in all environments

    vec_env  = make_vec_env(lambda: Monitor(EnergiaEnv(n_racks=n_racks)),
                            n_envs=n_envs, seed=42, vec_env_cls=vec_cls)
    eval_env = make_vec_env(lambda: Monitor(EnergiaEnv(n_racks=n_racks, seed=999)),
                            n_envs=1, vec_env_cls=vec_cls)

    # ── Model ─────────────────────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path + '.zip'):
        print(f"  Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device=device)
        model.set_env(vec_env)
    else:
        model = PPO(env=vec_env, device=device, **PPO_KWARGS)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}  |  Running on: {model.device}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    run_log_dir = os.path.join(LOG_DIR, run_name)
    run_mdl_dir = os.path.join(MODEL_DIR, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_mdl_dir, exist_ok=True)

    logger_cb    = EnergiaTrainingLogger(run_log_dir, log_freq=log_freq, verbose=1)
    checkpoint_cb = BestModelCheckpoint(MODEL_DIR, check_freq=2000, verbose=1)
    early_stop_cb = EarlyStoppingCallback(check_freq=20_000, patience=5,
                                          min_delta=0.005, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = os.path.join(run_log_dir, 'eval'),
        eval_freq            = max(eval_freq // n_envs, 1000),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    callbacks = CallbackList([logger_cb, checkpoint_cb, early_stop_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model.learn(
            total_timesteps     = total_steps,
            callback            = callbacks,
            progress_bar        = True,
            reset_num_timesteps = (resume_path is None),
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted.")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    final = os.path.join(MODEL_DIR, 'final_model')
    model.save(final)
    print(f"  Final model → {final}.zip")

    summary = {
        'run_name': run_name, 'total_steps': int(model.num_timesteps),
        'elapsed_sec': round(elapsed, 1), 'n_envs': n_envs, 'n_racks': n_racks,
        'device': str(model.device), 'final_model': final + '.zip',
        'best_model': os.path.join(MODEL_DIR, 'best_model.zip'),
        'status': 'complete', 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(LOG_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Sync run log to canonical path for API polling
    import shutil
    run_log = os.path.join(run_log_dir, 'training_log.json')
    if os.path.exists(run_log):
        shutil.copy(run_log, os.path.join(LOG_DIR, 'training_log.json'))

    vec_env.close()
    eval_env.close()
    return model


def evaluate(model_path: str, n_episodes: int = 20, n_racks: int = 4, render: bool = False):
    """Evaluate a saved model and print a performance report."""
    print(f"\n{'='*58}")
    print(f"  Energia RL — Evaluation  |  {model_path}")
    print(f"{'='*58}\n")

    env   = Monitor(EnergiaEnv(n_racks=n_racks, seed=42,
                               render_mode='human' if render else None))
    model = PPO.load(model_path)

    ep_rewards, ep_savings, ep_sla, ep_pue, ep_jobs = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done, ep_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            done  = term or trunc
        ep_rewards.append(ep_r)
        ep_savings.append(info.get('savings_pct', 0.0))
        ep_sla.append(info.get('total_sla_penalty', 0.0))
        ep_pue.append(info.get('pue', 1.5))
        ep_jobs.append(info.get('dispatched_jobs', 0))
        if render:
            print(f"  Ep {ep+1:2d}: reward={ep_r:+.3f}  savings={info['savings_pct']:+.1f}%")

    print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*55}")
    for lbl, arr in [('Episode Reward', ep_rewards), ('Cost Savings %', ep_savings),
                     ('SLA Penalty', ep_sla), ('Final PUE', ep_pue), ('Jobs Dispatched', ep_jobs)]:
        a = np.array(arr)
        print(f"  {lbl:<22}  {np.mean(a):>+8.3f}  {np.std(a):>7.3f}  "
              f"{np.min(a):>7.3f}  {np.max(a):>7.3f}")
    print(f"\n  Mean cost savings vs naive: {np.mean(ep_savings):+.1f}%")
    env.close()
    return {'mean_reward': float(np.mean(ep_rewards)), 'mean_savings': float(np.mean(ep_savings))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energia Phase 7 — RL Training')
    parser.add_argument('--steps',  type=int, default=200_000)
    parser.add_argument('--envs',   type=int, default=8)
    parser.add_argument('--racks',  type=int, default=4)
    parser.add_argument('--run',    type=str, default='run_001')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval',   type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, n_episodes=20, n_racks=args.racks, render=args.render)
    else:
        model = train(total_steps=args.steps, n_envs=args.envs, n_racks=args.racks,
                      run_name=args.run, resume_path=args.resume, device=args.device)
        best = os.path.join(MODEL_DIR, 'best_model')
        if os.path.exists(best + '.zip'):
            evaluate(best, n_episodes=20, n_racks=args.racks)
