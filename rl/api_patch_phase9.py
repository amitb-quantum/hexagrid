"""
HexaGrid Phase 9 — API Patch
============================
Adds multi-site fleet orchestration endpoints to api/api.py.

New endpoints:
  GET  /api/v1/fleet/summary              → all sites, live price + carbon
  GET  /api/v1/fleet/sites                → site configs
  POST /api/v1/fleet/sites/{id}           → update site config
  POST /api/v1/fleet/route                → route a workload across fleet
  GET  /api/v1/fleet/history              → last N routing decisions

Run from ~/hexagrid:
    python rl/api_patch_phase9.py
"""

import os, sys

API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'api', 'api.py')
)

if not os.path.exists(API_PATH):
    print(f"  ERROR: {API_PATH} not found. Run from ~/hexagrid.")
    sys.exit(1)

content = open(API_PATH).read()

if '/api/v1/fleet/summary' in content:
    print("  [WARN] Phase 9 endpoints already present in api.py — skipping.")
    sys.exit(0)

PHASE9_BLOCK = '''
# ════════════════════════════════════════════════════════════════════════════
#  PHASE 9 — MULTI-SITE FLEET ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading9
from pydantic import BaseModel as _BaseModel9
from typing import Optional as _Optional9

_site_orch       = None
_site_orch_lock  = _threading9.Lock()

def _get_orch():
    global _site_orch
    with _site_orch_lock:
        if _site_orch is None:
            try:
                _orch_path = os.path.join(os.path.dirname(__file__), \'..\', \'rl\')
                sys.path.insert(0, _orch_path)
                from site_orchestrator import SiteOrchestrator
                _site_orch = SiteOrchestrator()
                print("  [Fleet] Site orchestrator ready")
            except Exception as _e:
                print(f"  [Fleet] Init failed: {_e}")
    return _site_orch


class _RouteRequest(_BaseModel9):
    job_type:     str   = "custom"
    duration_min: int   = 60
    gpu_count:    int   = 4
    priority:     str   = "balanced"   # balanced | cost | carbon | capacity

class _SiteUpdate(_BaseModel9):
    capacity_pct: _Optional9[float] = None
    online:       _Optional9[bool]  = None
    pue:          _Optional9[float] = None
    num_racks:    _Optional9[int]   = None


@app.get("/api/v1/fleet/summary", tags=["Phase 9 — Fleet"])
async def fleet_summary():
    """Live fleet overview — all sites with price, carbon, capacity."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    try:
        return orch.fleet_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fleet/sites", tags=["Phase 9 — Fleet"])
async def fleet_sites():
    """Return all site configurations."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    return {"sites": orch.get_sites()}


@app.post("/api/v1/fleet/sites/{site_id}", tags=["Phase 9 — Fleet"])
async def update_site(site_id: str, body: _SiteUpdate):
    """Update a site\'s capacity, online status, PUE, or rack count."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")
    ok = orch.update_site(site_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Site not found: {site_id}")
    return {"status": "ok", "site_id": site_id, "updated": updates}


@app.post("/api/v1/fleet/route", tags=["Phase 9 — Fleet"])
async def fleet_route(body: _RouteRequest):
    """
    Route a workload across the fleet.
    Returns ranked site list + recommended site + cost/carbon savings vs worst site.

    priority options:
      balanced  — 40% cost, 30% carbon, 20% capacity, 10% PUE
      cost      — 70% cost, 10% carbon, 15% capacity,  5% PUE
      carbon    — 10% cost, 70% carbon, 15% capacity,  5% PUE
      capacity  — 20% cost, 20% carbon, 50% capacity, 10% PUE
    """
    if body.priority not in ("balanced", "cost", "carbon", "capacity"):
        raise HTTPException(status_code=422, detail="priority must be: balanced | cost | carbon | capacity")
    if body.gpu_count < 1 or body.gpu_count > 512:
        raise HTTPException(status_code=422, detail="gpu_count must be 1–512")
    if body.duration_min < 1 or body.duration_min > 10080:
        raise HTTPException(status_code=422, detail="duration_min must be 1–10080 (1 week)")

    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    try:
        return orch.route_workload(
            job_type=body.job_type,
            duration_min=body.duration_min,
            gpu_count=body.gpu_count,
            priority=body.priority,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fleet/history", tags=["Phase 9 — Fleet"])
async def fleet_history(limit: int = 50):
    """Last N workload routing decisions with full scoring breakdown."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be 1–200")
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    return {"history": orch.get_routing_history(limit)}

# ════ END PHASE 9 ════════════════════════════════════════════════════════════
'''

INSERT_BEFORE = "if __name__ == '__main__':"
if INSERT_BEFORE in content:
    content = content.replace(INSERT_BEFORE, PHASE9_BLOCK + "\n" + INSERT_BEFORE)
else:
    content = content.rstrip() + "\n" + PHASE9_BLOCK + "\n"

open(API_PATH, 'w').write(content)
print(f"  [OK] Phase 9 fleet endpoints added to {API_PATH}")
print()
print("  New endpoints:")
print("    GET  /api/v1/fleet/summary")
print("    GET  /api/v1/fleet/sites")
print("    POST /api/v1/fleet/sites/{site_id}")
print("    POST /api/v1/fleet/route")
print("    GET  /api/v1/fleet/history")
print()
print("  Restart uvicorn to activate:")
print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
