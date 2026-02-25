"""
patch_api_alerts.py — HexaGrid API Patch: Alerting & Notifications
===================================================================
Adds the following endpoints:

    GET  /api/v1/alerts/config          — get current alert config (password masked)
    POST /api/v1/alerts/config          — update alert config
    GET  /api/v1/alerts/history         — paginated alert history log
    GET  /api/v1/alerts/summary         — counts by event type
    POST /api/v1/alerts/test            — send a test alert
    POST /api/v1/alerts/check           — manually trigger a full platform check

HOW TO APPLY:
    Copy alert_manager.py and this file into ~/hexagrid/api/
    Add these two lines to the bottom of api.py:

        from patch_api_alerts import router as alerts_router
        app.include_router(alerts_router)

    Restart uvicorn.
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from alert_manager import AlertManager

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])
_am = AlertManager()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    enabled:               Optional[bool]   = None
    dedup_window_minutes:  Optional[int]    = None
    email:                 Optional[dict]   = None
    webhook:               Optional[dict]   = None
    thresholds:            Optional[dict]   = None
    events:                Optional[dict]   = None


class TestRequest(BaseModel):
    channel: str = "both"   # "email" | "webhook" | "both"


class FireRequest(BaseModel):
    event_type: str
    message:    str
    level:      str = "warning"
    title:      Optional[str] = None
    metadata:   Optional[dict] = None
    force:      bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/config")
def get_alert_config():
    """Returns current alert configuration. SMTP password is masked."""
    try:
        return {"status": "ok", "config": _am.get_config()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
def update_alert_config(body: ConfigUpdate):
    """
    Update alert configuration. Only provided fields are changed.
    Pass smtp_password as empty string to leave it unchanged.
    """
    try:
        updates = {k: v for k, v in body.dict().items() if v is not None}
        updated = _am.update_config(updates)
        return {"status": "ok", "config": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_alert_history(
    limit:      int            = Query(default=50, le=500),
    event_type: Optional[str]  = Query(default=None),
):
    """Returns recent alert history, newest first."""
    try:
        entries = _am.history(limit=limit, event_type=event_type)
        return {"status": "ok", "count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
def get_alert_summary():
    """Returns alert counts by type and last-24h summary."""
    try:
        return {"status": "ok", "summary": _am.history_summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
def send_test_alert(body: TestRequest, background_tasks: BackgroundTasks):
    """
    Send a test alert to verify email and/or webhook configuration.
    Runs in background so it doesn't block the response.
    """
    try:
        background_tasks.add_task(_am.send_test, channel=body.channel)
        return {
            "status": "ok",
            "message": f"Test alert queued for channel: {body.channel}. Check your inbox / endpoint."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fire")
def fire_alert(body: FireRequest, background_tasks: BackgroundTasks):
    """
    Manually fire an alert. Useful for testing or external integrations.
    """
    try:
        background_tasks.add_task(
            _am.fire,
            event_type=body.event_type,
            message=body.message,
            level=body.level,
            title=body.title,
            metadata=body.metadata,
            force=body.force,
        )
        return {"status": "ok", "message": "Alert queued for delivery."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
def run_platform_check(background_tasks: BackgroundTasks):
    """
    Manually trigger a full platform alert check across all monitors:
    - Price spike forecast
    - GPU health scores
    - GPU anomaly detection
    - GPU temperatures
    - Fleet site status

    Runs in background. Results appear in /alerts/history.
    """
    try:
        background_tasks.add_task(_run_full_check)
        return {"status": "ok", "message": "Platform alert check triggered. Results in /alerts/history."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_full_check():
    """
    Full platform check — queries existing endpoints and fires alerts
    for any conditions that breach configured thresholds.
    Designed to be called on a schedule (e.g. every 5 minutes via APScheduler).
    """
    import sqlite3 as sq
    import json

    DB_PATH = os.environ.get("HEXAGRID_DB", os.path.expanduser("~/hexagrid/hexagrid.db"))
    cfg = _am.config
    thresholds = cfg.get("thresholds", {})

    try:
        conn = sq.connect(DB_PATH)
        conn.row_factory = sq.Row

        # ── 1. Price spike check ──────────────────────────────────────────────
        if cfg["events"].get("price_spike", True):
            regions = ["CAISO", "ERCOT", "NYISO", "ISONE", "PJM"]
            allowed_levels = thresholds.get("price_spike_levels", ["critical", "warning"])

            for region in regions:
                cur = conn.execute("""
                    SELECT price_per_kwh FROM grid_prices
                    WHERE region = ? ORDER BY recorded_at DESC LIMIT 1
                """, (region,)).fetchone()

                forecasts = conn.execute("""
                    SELECT forecast_price, minutes_ahead FROM price_forecasts
                    WHERE region = ? ORDER BY forecast_time DESC LIMIT 8
                """, (region,)).fetchall()

                if not cur or not forecasts:
                    continue

                current_price = cur["price_per_kwh"]
                if current_price <= 0:
                    continue

                peak = max(r["forecast_price"] for r in forecasts)
                ratio = peak / current_price
                peak_mins = min(
                    (r["minutes_ahead"] for r in forecasts if r["forecast_price"] == peak),
                    default=120
                )

                if ratio >= 2.0 and peak_mins <= 60:
                    level = "critical"
                elif ratio >= 1.5:
                    level = "warning"
                elif ratio >= 1.25:
                    level = "elevated"
                else:
                    continue

                if level not in allowed_levels:
                    continue

                _am.fire(
                    event_type="price_spike",
                    level=level,
                    title=f"⚡ Price Spike Alert — {region}",
                    message=(
                        f"{region} grid price forecast shows {ratio:.1f}× spike "
                        f"predicted in approximately {peak_mins} minutes. "
                        f"Current: ${current_price:.4f}/kWh → Peak forecast: ${peak:.4f}/kWh. "
                        f"Consider deferring flexible workloads now."
                    ),
                    metadata={
                        "region": region,
                        "current_price_kwh": round(current_price, 4),
                        "peak_forecast_kwh": round(peak, 4),
                        "ratio": round(ratio, 2),
                        "eta_minutes": peak_mins,
                    }
                )

        # ── 2. GPU health check ───────────────────────────────────────────────
        if cfg["events"].get("gpu_health", True):
            min_score = thresholds.get("gpu_health_min_score", 60)
            gpu_rows = conn.execute("""
                SELECT gpu_idx, health_score, temp_c, power_w
                FROM gpu_telemetry
                WHERE recorded_at = (SELECT MAX(recorded_at) FROM gpu_telemetry)
            """).fetchall()

            for gpu in gpu_rows:
                if gpu["health_score"] is not None and gpu["health_score"] < min_score:
                    level = "critical" if gpu["health_score"] < 40 else "warning"
                    _am.fire(
                        event_type="gpu_health",
                        level=level,
                        title=f"🖥️ GPU {gpu['gpu_idx']} Health Degraded",
                        message=(
                            f"GPU {gpu['gpu_idx']} health score has dropped to "
                            f"{gpu['health_score']:.0f}/100, below the configured "
                            f"threshold of {min_score}. Review hardware telemetry."
                        ),
                        metadata={
                            "gpu_idx": gpu["gpu_idx"],
                            "health_score": gpu["health_score"],
                            "temp_c": gpu["temp_c"],
                            "power_w": gpu["power_w"],
                        }
                    )

        # ── 3. GPU temperature check ──────────────────────────────────────────
        if cfg["events"].get("gpu_temperature", True):
            temp_crit = thresholds.get("gpu_temp_critical_c", 88)
            gpu_rows = conn.execute("""
                SELECT gpu_idx, temp_c, power_w, gpu_util_pct
                FROM gpu_telemetry
                WHERE recorded_at = (SELECT MAX(recorded_at) FROM gpu_telemetry)
            """).fetchall()

            for gpu in gpu_rows:
                if gpu["temp_c"] is not None and gpu["temp_c"] >= temp_crit:
                    _am.fire(
                        event_type="gpu_temperature",
                        level="critical",
                        title=f"🌡️ GPU {gpu['gpu_idx']} Temperature Critical",
                        message=(
                            f"GPU {gpu['gpu_idx']} temperature is {gpu['temp_c']:.0f}°C, "
                            f"at or above the critical threshold of {temp_crit}°C. "
                            f"Immediate attention required — throttling or hardware damage risk."
                        ),
                        metadata={
                            "gpu_idx":      gpu["gpu_idx"],
                            "temp_c":       gpu["temp_c"],
                            "power_w":      gpu["power_w"],
                            "util_pct":     gpu["gpu_util_pct"],
                            "threshold_c":  temp_crit,
                        }
                    )

        # ── 4. GPU anomaly check ──────────────────────────────────────────────
        if cfg["events"].get("gpu_anomaly", True):
            anomaly_rows = conn.execute("""
                SELECT gpu_idx, anomaly_score, temp_c, power_w, mem_pct
                FROM gpu_telemetry
                WHERE anomaly_flag = 1
                  AND recorded_at >= datetime('now', '-2 minutes')
            """).fetchall()

            for gpu in anomaly_rows:
                _am.fire(
                    event_type="gpu_anomaly",
                    level="warning",
                    title=f"🔍 GPU {gpu['gpu_idx']} Anomaly Detected",
                    message=(
                        f"IsolationForest anomaly detected on GPU {gpu['gpu_idx']}. "
                        f"Combined telemetry signature (temp, power, VRAM, utilization) "
                        f"deviates significantly from baseline. "
                        f"This may indicate a developing hardware issue."
                    ),
                    metadata={
                        "gpu_idx":      gpu["gpu_idx"],
                        "anomaly_score": gpu["anomaly_score"],
                        "temp_c":       gpu["temp_c"],
                        "power_w":      gpu["power_w"],
                        "mem_pct":      gpu["mem_pct"],
                    }
                )

        # ── 5. Fleet offline/capacity check ──────────────────────────────────
        if cfg["events"].get("fleet_offline", True):
            cap_threshold = thresholds.get("fleet_capacity_critical_pct", 90)
            site_rows = conn.execute("""
                SELECT site_id, site_name, status, capacity_pct
                FROM fleet_sites
            """).fetchall()

            for site in site_rows:
                if site["status"] == "offline":
                    _am.fire(
                        event_type="fleet_offline",
                        level="critical",
                        title=f"🌐 Fleet Site Offline — {site['site_name']}",
                        message=(
                            f"Fleet site '{site['site_name']}' ({site['site_id']}) "
                            f"has gone offline. Workloads may be affected. "
                            f"Check site connectivity and API endpoints."
                        ),
                        metadata={"site_id": site["site_id"], "site_name": site["site_name"]}
                    )
                elif site["capacity_pct"] is not None and site["capacity_pct"] >= cap_threshold:
                    _am.fire(
                        event_type="fleet_offline",
                        level="warning",
                        title=f"🌐 Fleet Site Capacity Critical — {site['site_name']}",
                        message=(
                            f"Fleet site '{site['site_name']}' is at "
                            f"{site['capacity_pct']:.0f}% capacity, "
                            f"above the configured threshold of {cap_threshold}%. "
                            f"Consider re-routing workloads to lower-capacity regions."
                        ),
                        metadata={
                            "site_id":      site["site_id"],
                            "site_name":    site["site_name"],
                            "capacity_pct": site["capacity_pct"],
                        }
                    )

        conn.close()

    except Exception as e:
        # Log but never crash the background task
        print(f"[HexaGrid Alert Check Error] {e}")
