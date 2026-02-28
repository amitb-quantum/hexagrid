"""
patch_api_incidents.py — HexaGrid API Patch: Incident Response
==============================================================
Adds the following endpoints:

    GET  /api/v1/incidents/fleet         — current state of all nodes
    GET  /api/v1/incidents/history       — incident timeline (fleet or per-node)
    POST /api/v1/incidents/trigger       — manually trigger an incident event
    POST /api/v1/incidents/{node}/drain  — manually drain a node
    POST /api/v1/incidents/{node}/recover— mark node as recovering
    POST /api/v1/incidents/{node}/clear  — mark node as healthy
    GET  /api/v1/incidents/config        — current incident config thresholds
    POST /api/v1/incidents/config        — update thresholds

HOW TO APPLY:
    Copy incident_engine.py and this file into ~/hexagrid/api/
    Add these two lines to the bottom of api.py:

        from patch_api_incidents import router as incidents_router
        app.include_router(incidents_router)

    Optional env vars for outbound paging:
        HEXAGRID_PAGERDUTY_KEY   — PagerDuty Events API v2 routing key
        HEXAGRID_OPSGENIE_KEY    — OpsGenie API key
        HEXAGRID_DRAIN_THRESHOLD_TEMP_C  — temp to trigger drain (default 91)
        HEXAGRID_DEGRADE_THRESHOLD_TEMP_C — temp to trigger degrade (default 85)
        HEXAGRID_DEGRADE_HEALTH_SCORE     — health score to trigger degrade (default 60)

    To wire alert_manager → incident_engine automatically, add this to
    alert_manager.py's fire() method (see hotfix_wire_incidents.py).

    Restart uvicorn.
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from incident_engine import (
    IncidentEngine, HEALTHY, DEGRADED, DRAINING, DRAINED, RECOVERING,
    DRAIN_THRESHOLD, DEGRADE_THRESHOLD, HEALTH_THRESHOLD,
    PAGERDUTY_KEY, OPSGENIE_KEY,
)

router = APIRouter(prefix="/api/v1/incidents", tags=["Incident Response"])
_engine = IncidentEngine()


# ── Request models ────────────────────────────────────────────────────────────

class TriggerRequest(BaseModel):
    event_type: str
    node_id:    str
    gpu_uuid:   str  = ""
    detail:     dict = {}

    model_config = {"json_schema_extra": {"example": {
        "event_type": "gpu_temperature",
        "node_id":    "PF57VBJL",
        "gpu_uuid":   "GPU-abc123",
        "detail":     {"temp_c": 93},
    }}}


class ConfigUpdate(BaseModel):
    drain_threshold_temp_c:   Optional[int] = None
    degrade_threshold_temp_c: Optional[int] = None
    degrade_health_score:     Optional[int] = None
    pagerduty_key:            Optional[str] = None
    opsgenie_key:             Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/fleet", summary="Current incident state of all nodes")
def fleet_status():
    """
    Return the current incident state for every node that has had an event.
    Nodes that have never had an incident are omitted (they are implicitly healthy).
    """
    try:
        nodes = _engine.fleet_status()
        healthy_count  = sum(1 for n in nodes if n["state"] == HEALTHY)
        degraded_count = sum(1 for n in nodes if n["state"] == DEGRADED)
        draining_count = sum(1 for n in nodes if n["state"] == DRAINING)
        drained_count  = sum(1 for n in nodes if n["state"] == DRAINED)
        return {
            "summary": {
                "total_tracked": len(nodes),
                "healthy":       healthy_count,
                "degraded":      degraded_count,
                "draining":      draining_count,
                "drained":       drained_count,
            },
            "nodes": nodes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Incident timeline")
def incident_history(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    limit:   int           = Query(100, ge=1, le=1000),
):
    """Return the incident state transition log for a node or the whole fleet."""
    try:
        return {
            "history": _engine.incident_history(node_id=node_id, limit=limit)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger", summary="Manually trigger an incident event")
def trigger_incident(req: TriggerRequest):
    """
    Manually simulate an alert event to test the incident response pipeline
    or to handle alerts from external monitoring systems (Prometheus, Datadog, etc.)
    """
    try:
        result = _engine.handle_alert(
            event_type=req.event_type,
            node_id=req.node_id,
            gpu_uuid=req.gpu_uuid,
            detail=req.detail,
        )
        if result is None:
            return {
                "status":  "no_action",
                "message": f"Node {req.node_id} is already in state "
                           f"'{_engine.get_node_state(req.node_id)}' — no transition needed.",
            }
        return {"status": "ok", "transition": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{node_id}/drain", summary="Manually drain a node")
def drain_node(node_id: str):
    """
    Manually initiate a node drain — marks it as draining and fires notifications.
    Use when you need to take a node offline for maintenance.
    """
    try:
        result = _engine._transition_to_draining(
            node_id=node_id, gpu_uuid="",
            reason="manual_drain",
            detail={"source": "operator"},
            urgency="warning",
        )
        return {"status": "ok", "transition": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{node_id}/recover", summary="Mark a node as recovering")
def recover_node(node_id: str):
    """Mark a drained or degraded node as recovering after maintenance."""
    try:
        return {"status": "ok", "transition": _engine.recover_node(node_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{node_id}/clear", summary="Clear a node back to healthy")
def clear_node(node_id: str):
    """Mark a recovering node as fully healthy."""
    try:
        return {"status": "ok", "transition": _engine.clear_node(node_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", summary="Current incident response configuration")
def get_config():
    """Return current thresholds and notification config (keys masked)."""
    return {
        "thresholds": {
            "drain_threshold_temp_c":   DRAIN_THRESHOLD,
            "degrade_threshold_temp_c": DEGRADE_THRESHOLD,
            "degrade_health_score":     HEALTH_THRESHOLD,
        },
        "notifications": {
            "pagerduty": bool(PAGERDUTY_KEY),
            "opsgenie":  bool(OPSGENIE_KEY),
            "alert_manager_webhook": True,
        },
    }


@router.post("/config", summary="Update incident response thresholds")
def update_config(req: ConfigUpdate):
    """
    Update incident response thresholds at runtime.
    Changes are applied immediately but not persisted — add to .env.auth
    for persistence across restarts.
    """
    import incident_engine as _ie
    changed = {}
    if req.drain_threshold_temp_c is not None:
        _ie.DRAIN_THRESHOLD = req.drain_threshold_temp_c
        changed["drain_threshold_temp_c"] = req.drain_threshold_temp_c
    if req.degrade_threshold_temp_c is not None:
        _ie.DEGRADE_THRESHOLD = req.degrade_threshold_temp_c
        changed["degrade_threshold_temp_c"] = req.degrade_threshold_temp_c
    if req.degrade_health_score is not None:
        _ie.HEALTH_THRESHOLD = req.degrade_health_score
        changed["degrade_health_score"] = req.degrade_health_score
    if req.pagerduty_key is not None:
        _ie.PAGERDUTY_KEY = req.pagerduty_key
        changed["pagerduty"] = "configured"
    if req.opsgenie_key is not None:
        _ie.OPSGENIE_KEY = req.opsgenie_key
        changed["opsgenie"] = "configured"
    return {"status": "ok", "updated": changed}
