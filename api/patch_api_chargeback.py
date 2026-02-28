"""
patch_api_chargeback.py — HexaGrid API Patch: Cost Allocation & Chargeback
===========================================================================
Adds the following endpoints:

    GET  /api/v1/chargeback/report      — grouped cost report (JSON)
    GET  /api/v1/chargeback/report.csv  — same report as CSV download
    GET  /api/v1/chargeback/entries     — raw job entries (audit view)
    GET  /api/v1/chargeback/dimensions  — available filter values
    POST /api/v1/chargeback/record      — manually record a job cost entry

HOW TO APPLY:
    Copy chargeback_ledger.py and this file into ~/hexagrid/api/
    Add these two lines to the bottom of api.py (before the last include_router block):

        from patch_api_chargeback import router as chargeback_router
        app.include_router(chargeback_router)

    Also add cost_center / team / project fields to JobSpec in api.py —
    see the patch_api_chargeback_jobspec.py hotfix script.

    Restart uvicorn.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from chargeback_ledger import ChargebackLedger

router = APIRouter(prefix="/api/v1/chargeback", tags=["Chargeback"])
_ledger = ChargebackLedger()


# ── Request models ────────────────────────────────────────────────────────────

class ManualEntryRequest(BaseModel):
    job_id:         str
    job_name:       str   = ""
    cost_center:    str   = "untagged"
    team:           str   = "untagged"
    project:        str   = "untagged"
    region:         str   = "unknown"
    node_id:        str   = "unknown"
    scheduler:      str   = "manual"
    duration_min:   float = 0.0
    power_kw:       float = 0.0
    energy_kwh:     float = 0.0
    price_per_kwh:  float = 0.0
    cost_usd:       float = 0.0
    naive_cost_usd: float = 0.0

    model_config = {"json_schema_extra": {"example": {
        "job_id": "train_042", "job_name": "ResNet Training",
        "cost_center": "ML-Platform", "team": "infra", "project": "ImageNet-2026",
        "region": "CAISO", "node_id": "PF57VBJL", "scheduler": "qaoa",
        "duration_min": 45, "power_kw": 4.2, "energy_kwh": 3.15,
        "price_per_kwh": 0.031, "cost_usd": 0.097, "naive_cost_usd": 0.201,
    }}}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/report", summary="Cost allocation report (JSON)")
def chargeback_report(
    period_start: Optional[str] = Query(None, description="ISO datetime, e.g. 2026-02-01T00:00:00"),
    period_end:   Optional[str] = Query(None, description="ISO datetime, e.g. 2026-02-28T23:59:59"),
    cost_center:  Optional[str] = Query(None, description="Filter by cost center"),
    team:         Optional[str] = Query(None, description="Filter by team"),
    project:      Optional[str] = Query(None, description="Filter by project"),
    group_by:     str           = Query("cost_center",
                                        description="Group by: cost_center | team | project | region | scheduler"),
):
    """
    Return a cost allocation report grouped by the chosen dimension.
    All filters are optional — omit for all-time totals across everything.

    Example use cases:
      - Monthly bill for ML-Platform cost center:
        ?period_start=2026-02-01&period_end=2026-02-28&cost_center=ML-Platform
      - All teams this quarter:
        ?period_start=2026-01-01&group_by=team
      - Project breakdown for finance review:
        ?group_by=project&period_start=2026-01-01
    """
    try:
        return _ledger.report(
            period_start=period_start, period_end=period_end,
            cost_center=cost_center, team=team, project=project,
            group_by=group_by,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report.csv", summary="Cost allocation report (CSV download)")
def chargeback_report_csv(
    period_start: Optional[str] = Query(None),
    period_end:   Optional[str] = Query(None),
    cost_center:  Optional[str] = Query(None),
    team:         Optional[str] = Query(None),
    project:      Optional[str] = Query(None),
    group_by:     str           = Query("cost_center"),
):
    """
    Download the chargeback report as a CSV file.
    Same parameters as /report — suitable for Excel / finance systems.
    """
    try:
        csv_data = _ledger.report_csv(
            period_start=period_start, period_end=period_end,
            cost_center=cost_center, team=team, project=project,
            group_by=group_by,
        )
        # Build a descriptive filename
        suffix = period_start[:7] if period_start else "all-time"
        filename = f"hexagrid_chargeback_{group_by}_{suffix}.csv"
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entries", summary="Raw job cost entries (audit view)")
def chargeback_entries(
    period_start: Optional[str] = Query(None),
    period_end:   Optional[str] = Query(None),
    cost_center:  Optional[str] = Query(None),
    team:         Optional[str] = Query(None),
    project:      Optional[str] = Query(None),
    limit:        int           = Query(200, ge=1, le=1000),
):
    """
    Return individual job cost entries for auditing.
    Useful for finance teams who want to see every job that ran
    against a cost center, not just the roll-up.
    """
    try:
        entries = _ledger.entries(
            period_start=period_start, period_end=period_end,
            cost_center=cost_center, team=team, project=project,
            limit=limit,
        )
        return {"count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimensions", summary="Available filter values for report UI")
def chargeback_dimensions():
    """
    Return all known cost_center, team, project, and region values.
    Used to populate filter dropdowns in the dashboard chargeback tab.
    """
    try:
        return _ledger.dimensions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record", summary="Manually record a job cost entry", status_code=201)
def record_entry(req: ManualEntryRequest):
    """
    Manually record a job's energy cost for jobs run outside the
    HexaGrid scheduler (e.g. jobs submitted directly to SLURM or k8s).
    The scheduler auto-records via the schedule endpoint — this is for
    external workloads you still want to appear in chargeback reports.
    """
    try:
        row_id = _ledger.record(
            job_id=req.job_id, job_name=req.job_name,
            cost_center=req.cost_center, team=req.team, project=req.project,
            region=req.region, node_id=req.node_id, scheduler=req.scheduler,
            duration_min=req.duration_min, power_kw=req.power_kw,
            energy_kwh=req.energy_kwh, price_per_kwh=req.price_per_kwh,
            cost_usd=req.cost_usd, naive_cost_usd=req.naive_cost_usd,
        )
        return {"status": "ok", "id": row_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
