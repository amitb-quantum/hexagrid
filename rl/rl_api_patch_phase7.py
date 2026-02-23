"""
Energia Phase 7 — API Patch
============================
Adds RL agent endpoints to api/api.py.

Run from project root (one time):
    python rl/api_patch_phase7.py

New endpoints:
    GET  /api/v1/rl/status           → agent readiness + training metrics
    POST /api/v1/rl/train            → launch background PPO training job
    GET  /api/v1/rl/training_log     → live training metrics (polled by dashboard)
    POST /api/v1/rl/recommend        → get action recommendation from trained agent
    GET  /api/v1/rl/recommend/live   → recommendation using live price feed automatically
"""

import os, sys

API_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api', 'api.py'))

if not os.path.exists(API_PATH):
    print(f"ERROR: Cannot find {API_PATH}")
    print("Run this script from the energia project root:  python rl/api_patch_phase7.py")
    sys.exit(1)

content = open(API_PATH).read()

if '/api/v1/rl/status' in content:
    print("  [SKIP] Phase 7 RL endpoints already present in api.py")
    sys.exit(0)

# ── RL block to inject ─────────────────────────────────────────────────────────
RL_BLOCK = '''

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7 — RL AGENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
import json   as _json_rl
import threading as _threading_rl

_rl_agent      = None
_rl_agent_lock = _threading_rl.Lock()

def _get_rl_agent():
    """Lazy-init the RL agent singleton (thread-safe)."""
    global _rl_agent
    if _rl_agent is None:
        with _rl_agent_lock:
            if _rl_agent is None:
                try:
                    _rl_sys_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                    sys.path.insert(0, _rl_sys_path)
                    from agent import EnergiaAgent
                    _rl_agent = EnergiaAgent()
                except Exception as _e:
                    print(f"  [RL] Agent init failed: {_e}")
                    _rl_agent = None
    return _rl_agent


# ── Request models ─────────────────────────────────────────────────────────────
class RLTrainRequest(BaseModel):
    total_steps: int   = Field(200_000, ge=2000, le=2_000_000)
    n_envs:      int   = Field(4,       ge=1,    le=16)
    n_racks:     int   = Field(4,       ge=1,    le=64)
    run_name:    str   = Field("run_001")
    resume_path: str   = Field(None)

class RLRecommendRequest(BaseModel):
    current_price:    float = Field(...,  description="Current grid price $/kWh")
    price_forecast:   list  = Field(...,  description="12 price values, 5-min intervals")
    queue_depth:      int   = Field(1)
    job_urgency:      float = Field(0.3)
    gpu_utilization:  float = Field(0.5)
    pue:              float = Field(1.4)
    ups_charge:       float = Field(0.5)
    cooling_power:    float = Field(0.5)
    hour_of_day:      float = Field(12.0)
    day_of_week:      int   = Field(0)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/rl/status", tags=["Phase 7 — RL Agent"])
async def rl_status():
    """Agent readiness, model path, and latest training metrics."""
    agent = _get_rl_agent()
    if agent is None:
        return JSONResponse({"ready": False, "error": "RL module unavailable"})
    return JSONResponse(agent.status())


@app.post("/api/v1/rl/train", tags=["Phase 7 — RL Agent"])
async def rl_train(req: RLTrainRequest, background_tasks: BackgroundTasks):
    """
    Launch a background PPO training job.
    Returns run_id immediately — poll GET /api/v1/jobs/{run_id} for status.
    """
    run_id = f"rl_train_{req.run_name}_{uuid.uuid4().hex[:6]}"
    _new_job(run_id)

    def _do_train():
        try:
            _update_job(run_id, 'running')
            _rl_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
            sys.path.insert(0, _rl_path)
            import importlib
            import train as rl_train_mod
            importlib.reload(rl_train_mod)
            model = rl_train_mod.train(
                total_steps = req.total_steps,
                n_envs      = req.n_envs,
                n_racks     = req.n_racks,
                run_name    = req.run_name,
                resume_path = req.resume_path,
            )
            # Hot-reload the agent after training
            agent = _get_rl_agent()
            if agent:
                agent.reload()
            _update_job(run_id, 'completed', {
                'run_name':   req.run_name,
                'steps':      int(model.num_timesteps),
                'model_path': 'models/rl/best_model.zip',
            })
        except Exception as _e:
            _update_job(run_id, 'failed', error=str(_e))

    background_tasks.add_task(_do_train)
    return JSONResponse({"run_id": run_id, "status": "queued", "run_name": req.run_name})


@app.get("/api/v1/rl/training_log", tags=["Phase 7 — RL Agent"])
async def rl_training_log():
    """Live training metrics — polled by the dashboard every 5 seconds during training."""
    import glob
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'logs', 'rl', 'training_log.json'),
    ] + sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), '..', 'logs', 'rl', '*', 'training_log.json')),
        key=os.path.getmtime, reverse=True
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return JSONResponse(_json_rl.load(f))
            except Exception:
                pass
    return JSONResponse({"status": "no_training_data", "episodes": 0, "reward_history": []})


@app.post("/api/v1/rl/recommend", tags=["Phase 7 — RL Agent"])
async def rl_recommend(req: RLRecommendRequest):
    """
    Get a real-time action recommendation from the trained RL agent.
    Provide current price, 12-step forecast, and system state.
    """
    agent = _get_rl_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="RL agent not available")
    if not agent.is_ready:
        raise HTTPException(status_code=503,
            detail="No trained model found. Run: POST /api/v1/rl/train first.")
    return JSONResponse(agent.recommend(req.dict()))


@app.get("/api/v1/rl/recommend/live", tags=["Phase 7 — RL Agent"])
async def rl_recommend_live():
    """
    Convenience: pulls current price feed automatically and returns agent recommendation.
    No request body needed — uses live grid data from the existing connector.
    """
    agent = _get_rl_agent()
    if agent is None or not agent.is_ready:
        return JSONResponse({"status": "not_ready", "dispatch": None})

    # Pull from existing price feed
    current_price, price_forecast = 0.05, [0.05] * 12
    try:
        _gc_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        sys.path.insert(0, _gc_path)
        from grid_connector import GridConnector
        gc = GridConnector()
        pf = gc.get_price_feed(horizon_min=60)
        current_price  = pf.get('current_price_usd_kwh', 0.05)
        price_forecast = [p['price_usd_kwh'] for p in pf.get('price_curve', [])][:12]
    except Exception:
        pass

    import datetime
    now = datetime.datetime.now()
    return JSONResponse(agent.recommend({
        'current_price':   current_price,
        'price_forecast':  price_forecast,
        'queue_depth':     1,
        'job_urgency':     0.3,
        'gpu_utilization': 0.6,
        'pue':             1.35,
        'ups_charge':      0.6,
        'cooling_power':   0.5,
        'hour_of_day':     float(now.hour) + float(now.minute) / 60.0,
        'day_of_week':     now.weekday(),
    }))

# ── end Phase 7 block ──────────────────────────────────────────────────────────
'''

# Inject before `if __name__ == '__main__':` or at end of file
ANCHOR = "if __name__ == '__main__':"
if ANCHOR in content:
    content = content.replace(ANCHOR, RL_BLOCK + '\n\n' + ANCHOR, 1)
else:
    content += RL_BLOCK

open(API_PATH, 'w').write(content)

print(f"  [OK] Phase 7 RL endpoints added to {API_PATH}")
print()
print("  New endpoints:")
print("    GET  /api/v1/rl/status")
print("    POST /api/v1/rl/train")
print("    GET  /api/v1/rl/training_log")
print("    POST /api/v1/rl/recommend")
print("    GET  /api/v1/rl/recommend/live")
print()
print("  Restart uvicorn to activate:")
print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
