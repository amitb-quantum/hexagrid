"""
HexaGrid Phase 8 — API Patch
============================
Adds carbon intensity endpoints to api/api.py.

New endpoints:
  GET  /api/v1/carbon/snapshot        → live carbon intensity for all zones
  GET  /api/v1/carbon/history/{iso}   → 24h history for one zone
  GET  /api/v1/carbon/pareto          → cost vs carbon Pareto analysis
  GET  /api/v1/carbon/recommend       → carbon-aware dispatch recommendation

Run from ~/hexagrid:
    python rl/api_patch_phase8.py
"""

import os, sys, re

API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'api', 'api.py')
)

if not os.path.exists(API_PATH):
    print(f"  ERROR: {API_PATH} not found. Run from ~/hexagrid.")
    sys.exit(1)

content = open(API_PATH).read()

# ── Guard against double-patching ────────────────────────────────────────────
if '/api/v1/carbon/snapshot' in content:
    print("  [WARN] Phase 8 endpoints already present in api.py — skipping.")
    sys.exit(0)

# ── The block to inject ───────────────────────────────────────────────────────
PHASE8_BLOCK = '''
# ════════════════════════════════════════════════════════════════════════════
#  PHASE 8 — CARBON INTENSITY ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading8

_carbon_connector = None
_carbon_lock      = _threading8.Lock()

def _get_carbon() -> 'CarbonConnector':
    global _carbon_connector
    with _carbon_lock:
        if _carbon_connector is None:
            try:
                _carbon_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                sys.path.insert(0, _carbon_path)
                from carbon_connector import CarbonConnector
                _carbon_connector = CarbonConnector()
            except Exception as _ce:
                print(f"  [Carbon] Init failed: {_ce}")
    return _carbon_connector


@app.get("/api/v1/carbon/snapshot", tags=["Phase 8 — Carbon"])
async def carbon_snapshot():
    """Live carbon intensity (gCO2eq/kWh) for all 5 monitored ISO zones."""
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    try:
        return cc.get_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/carbon/history/{iso}", tags=["Phase 8 — Carbon"])
async def carbon_history(iso: str):
    """24-hour hourly carbon intensity history for one ISO zone.
    ISO codes: CAISO, ERCOT, NYISO, ISONE, PJM
    """
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    history = cc.get_history(iso.upper())
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for zone: {iso}")
    return {"iso": iso.upper(), "history": history}


@app.get("/api/v1/carbon/pareto", tags=["Phase 8 — Carbon"])
async def carbon_pareto():
    """
    Pareto analysis: cost vs carbon tradeoff across all zones.
    Returns scatter plot data and recommendations for three strategies:
      cost_only   — minimize $/kWh regardless of carbon
      balanced    — equal weight to cost and carbon
      carbon_only — minimize gCO2/kWh regardless of cost
    """
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    try:
        return cc.get_pareto_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/carbon/recommend", tags=["Phase 8 — Carbon"])
async def carbon_recommend(cost_weight: float = 0.5):
    """
    Carbon-aware workload dispatch recommendation.
    cost_weight: 0.0 = pure carbon minimization, 1.0 = pure cost minimization
    Default 0.5 = balanced.
    """
    if not 0.0 <= cost_weight <= 1.0:
        raise HTTPException(status_code=422, detail="cost_weight must be between 0.0 and 1.0")

    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")

    try:
        snap   = cc.get_snapshot()
        pareto = cc.get_pareto_data()
        zones  = snap.get("zones", {})
        points = pareto.get("points", [])

        if not points:
            return {"status": "no_data", "error": "No zone data available"}

        carbon_w = 1.0 - cost_weight
        best = min(
            points,
            key=lambda p: cost_weight * p["price_norm"] + carbon_w * p["carbon_norm"]
        )

        # Build reasoning text
        strategy = (
            "cost minimization" if cost_weight >= 0.9
            else "carbon minimization" if cost_weight <= 0.1
            else f"balanced ({int(cost_weight*100)}% cost / {int(carbon_w*100)}% carbon)"
        )
        reasoning = (
            f"Using {strategy} strategy. "
            f"{best['label']} offers ${best['price_usd_kwh']:.4f}/kWh at "
            f"{best['carbon_intensity']:.0f} gCO2eq/kWh ({best['carbon_label']})."
        )
        if best.get("on_pareto_frontier"):
            reasoning += " This zone is on the Pareto frontier — optimal for this weighting."

        return {
            "status":            "ok",
            "recommended_iso":   best["iso"],
            "recommended_label": best["label"],
            "price_usd_kwh":     best["price_usd_kwh"],
            "carbon_intensity":  best["carbon_intensity"],
            "carbon_label":      best["carbon_label"],
            "carbon_color":      best["carbon_color"],
            "fossil_free_pct":   best.get("fossil_free_pct"),
            "on_pareto_frontier":best.get("on_pareto_frontier", False),
            "cost_weight":       cost_weight,
            "carbon_weight":     carbon_w,
            "reasoning":         reasoning,
            "all_zones":         zones,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ════ END PHASE 8 ════════════════════════════════════════════════════════════
'''

# ── Insert before the final if __name__ block ─────────────────────────────────
INSERT_BEFORE = "if __name__ == '__main__':"
if INSERT_BEFORE in content:
    content = content.replace(INSERT_BEFORE, PHASE8_BLOCK + "\n" + INSERT_BEFORE)
else:
    # Fall back: append before last line
    content = content.rstrip() + "\n" + PHASE8_BLOCK + "\n"

open(API_PATH, 'w').write(content)
print(f"  [OK] Phase 8 carbon endpoints added to {API_PATH}")
print()
print("  New endpoints:")
print("    GET  /api/v1/carbon/snapshot")
print("    GET  /api/v1/carbon/history/{iso}")
print("    GET  /api/v1/carbon/pareto")
print("    GET  /api/v1/carbon/recommend?cost_weight=0.5")
print()
print("  Before restarting, set your API key:")
print("    echo 'ELECTRICITY_MAPS_API_KEY=your_key' >> ~/hexagrid/.env")
print()
print("  Then restart uvicorn:")
print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
