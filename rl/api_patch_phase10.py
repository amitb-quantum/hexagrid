"""
HexaGrid Phase 10 — API Patch
=============================
Adds GPU health monitoring endpoints to api/api.py.

New endpoints:
  GET  /api/v1/hardware/snapshot          → live readings for all GPUs
  GET  /api/v1/hardware/history/{gpu_idx} → rolling telemetry history
  GET  /api/v1/hardware/alerts            → recent alert log

Run from ~/hexagrid:
    pip install pynvml --break-system-packages   # first time only
    python rl/api_patch_phase10.py
"""

import os, sys

API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'api', 'api.py')
)

if not os.path.exists(API_PATH):
    print(f"  ERROR: {API_PATH} not found. Run from ~/hexagrid.")
    sys.exit(1)

content = open(API_PATH).read()

if '/api/v1/hardware/snapshot' in content:
    print("  [WARN] Phase 10 endpoints already present in api.py — skipping.")
    sys.exit(0)

PHASE10_BLOCK = '''
# ════════════════════════════════════════════════════════════════════════════
#  PHASE 10 — GPU HEALTH MONITORING
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading10

_gpu_monitor      = None
_gpu_monitor_lock = _threading10.Lock()

def _get_gpu_monitor():
    global _gpu_monitor
    with _gpu_monitor_lock:
        if _gpu_monitor is None:
            try:
                _hw_path = os.path.join(os.path.dirname(__file__), \'..\', \'rl\')
                sys.path.insert(0, _hw_path)
                from gpu_monitor import GPUMonitor
                _gpu_monitor = GPUMonitor(poll_interval_s=10)
                _gpu_monitor.start_background_polling()
                print("  [Hardware] GPU monitor ready")
            except Exception as _e:
                print(f"  [Hardware] Init failed: {_e}")
    return _gpu_monitor


@app.get("/api/v1/hardware/snapshot", tags=["Phase 10 — Hardware"])
async def hardware_snapshot():
    """
    Live GPU telemetry snapshot for all detected GPUs.
    Returns real NVML data when pynvml is installed and GPUs are present;
    falls back to realistic synthetic readings for development.
    """
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    try:
        return mon.get_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/hardware/history/{gpu_idx}", tags=["Phase 10 — Hardware"])
async def hardware_history(gpu_idx: int, minutes: int = 60):
    """
    Rolling telemetry history for one GPU.
    gpu_idx: 0-based GPU index
    minutes: how far back to look (max 1440 = 24h)
    """
    if minutes < 1 or minutes > 1440:
        raise HTTPException(status_code=422, detail="minutes must be 1–1440")
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    history = mon.get_history(gpu_idx=gpu_idx, minutes=minutes)
    return {"gpu_idx": gpu_idx, "minutes": minutes, "history": history}


@app.get("/api/v1/hardware/alerts", tags=["Phase 10 — Hardware"])
async def hardware_alerts(limit: int = 50):
    """Most recent GPU health alerts across all devices."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be 1–500")
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    return {"alerts": mon.get_alerts(limit=limit)}

# ════ END PHASE 10 ════════════════════════════════════════════════════════════
'''

INSERT_BEFORE = "if __name__ == '__main__':"
if INSERT_BEFORE in content:
    content = content.replace(INSERT_BEFORE, PHASE10_BLOCK + "\n" + INSERT_BEFORE)
else:
    content = content.rstrip() + "\n" + PHASE10_BLOCK + "\n"

open(API_PATH, 'w').write(content)
print(f"  [OK] Phase 10 hardware endpoints added to {API_PATH}")
print()
print("  New endpoints:")
print("    GET  /api/v1/hardware/snapshot")
print("    GET  /api/v1/hardware/history/{gpu_idx}")
print("    GET  /api/v1/hardware/alerts")
print()
print("  Install pynvml if not already done:")
print("    pip install pynvml --break-system-packages")
print()
print("  Restart uvicorn to activate:")
print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
