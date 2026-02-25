"""
patch_api_savings.py — HexaGrid API Patch: Savings Ledger + Spike Warning
==========================================================================
Adds the following endpoints to your existing FastAPI app:

    GET  /api/v1/savings/summary          — cumulative + windowed savings totals
    GET  /api/v1/savings/ledger           — paginated decision log
    GET  /api/v1/savings/by-region        — breakdown by ISO region
    POST /api/v1/savings/record           — record a new dispatch decision
    GET  /api/v1/savings/spike-warning    — price spike forecast for all regions

HOW TO APPLY:
    Copy this file into ~/hexagrid/api/
    Then add this line to the bottom of api.py (before if __name__ == "__main__":):

        from patch_api_savings import router as savings_router
        app.include_router(savings_router)

    Restart uvicorn. New endpoints appear immediately at /docs.
"""

import os
import sys
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Make sure savings_ledger is importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from savings_ledger import SavingsLedger

router = APIRouter(prefix="/api/v1/savings", tags=["Savings Ledger"])
_ledger = SavingsLedger()


# ── Pydantic models ───────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    region: str
    naive_price: float          # $/kWh — price if job ran immediately
    actual_price: float         # $/kWh — price after optimized deferral
    power_kw: float             # GPU rack power draw
    duration_minutes: float     # Job duration
    decision: str               # run_now | defer_15 | defer_30 | defer_60
    job_id: Optional[str] = None
    source: str = "rl_agent"    # rl_agent | qaoa_scheduler | manual


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/summary")
def get_savings_summary():
    """
    Returns cumulative savings summary with today / 7d / 30d / all-time windows.
    Powers the savings panel on the Overview tab.
    """
    try:
        return {"status": "ok", "summary": _ledger.summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ledger")
def get_ledger(
    limit: int = Query(default=50, le=500),
    region: Optional[str] = Query(default=None)
):
    """
    Returns recent dispatch decisions with cost comparisons, newest first.
    """
    try:
        entries = _ledger.ledger(limit=limit, region=region)
        return {"status": "ok", "count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-region")
def get_savings_by_region():
    """
    Returns savings totals broken down by ISO region.
    """
    try:
        return {"status": "ok", "regions": _ledger.by_region()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record")
def record_decision(req: RecordRequest):
    """
    Records a single dispatch decision and returns the computed savings.
    Called automatically by the RL agent and QAOA scheduler after each decision.
    """
    try:
        result = _ledger.record(
            region=req.region,
            naive_price=req.naive_price,
            actual_price=req.actual_price,
            power_kw=req.power_kw,
            duration_minutes=req.duration_minutes,
            decision=req.decision,
            job_id=req.job_id,
            source=req.source,
        )
        return {"status": "ok", "recorded": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spike-warning")
def get_spike_warning():
    """
    Analyses the LSTM forecast across all regions and returns spike risk levels.

    Risk levels:
        critical  — forecast price > 2x current price within 60 minutes
        warning   — forecast price > 1.5x current price within 120 minutes
        elevated  — forecast price > 1.25x current price within 120 minutes
        normal    — no significant spike predicted

    Powers the banner, tab badge, and modal on the Overview tab.
    """
    try:
        import sqlite3
        DB_PATH = os.environ.get(
            "HEXAGRID_DB", os.path.expanduser("~/hexagrid/hexagrid.db")
        )

        warnings = []
        overall_level = "normal"
        level_rank = {"normal": 0, "elevated": 1, "warning": 2, "critical": 3}

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # Pull latest cached prices and forecasts per region
        regions = ["CAISO", "ERCOT", "NYISO", "ISONE", "PJM"]
        for region in regions:
            # Current price — most recent actual reading
            cur_row = conn.execute("""
                SELECT price_per_kwh FROM grid_prices
                WHERE region = ?
                ORDER BY recorded_at DESC LIMIT 1
            """, (region,)).fetchone()

            # Forecast rows for this region
            forecast_rows = conn.execute("""
                SELECT forecast_price, minutes_ahead FROM price_forecasts
                WHERE region = ?
                ORDER BY forecast_time DESC LIMIT 8
            """, (region,)).fetchall()

            if not cur_row or not forecast_rows:
                continue

            current_price = cur_row["price_per_kwh"]
            if current_price <= 0:
                continue

            peak_forecast = max(r["forecast_price"] for r in forecast_rows)
            peak_minutes  = min(
                (r["minutes_ahead"] for r in forecast_rows
                 if r["forecast_price"] == peak_forecast),
                default=120
            )
            ratio = peak_forecast / current_price

            if ratio >= 2.0 and peak_minutes <= 60:
                level = "critical"
            elif ratio >= 1.5:
                level = "warning"
            elif ratio >= 1.25:
                level = "elevated"
            else:
                level = "normal"

            if level != "normal":
                warnings.append({
                    "region":         region,
                    "level":          level,
                    "current_price":  round(current_price, 4),
                    "peak_forecast":  round(peak_forecast, 4),
                    "ratio":          round(ratio, 2),
                    "peak_in_minutes":peak_minutes,
                    "message": (
                        f"{region}: price spike {ratio:.1f}× current "
                        f"predicted in ~{peak_minutes} min "
                        f"(${current_price:.3f} → ${peak_forecast:.3f}/kWh)"
                    )
                })

            if level_rank[level] > level_rank[overall_level]:
                overall_level = level

        conn.close()

        return {
            "status":        "ok",
            "overall_level": overall_level,
            "warning_count": len(warnings),
            "warnings":      warnings,
            "checked_at":    datetime.utcnow().isoformat(),
        }

    except Exception as e:
        # Return safe fallback — never crash the dashboard
        return {
            "status":        "error",
            "overall_level": "normal",
            "warning_count": 0,
            "warnings":      [],
            "checked_at":    datetime.utcnow().isoformat(),
            "detail":        str(e)
        }
