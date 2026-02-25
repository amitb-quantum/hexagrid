"""
patch_api_weather.py — HexaGrid API Patch: Weather & Resilience Monitor
========================================================================
Adds the following endpoints:

    GET  /api/v1/weather/summary          — all sites, overall status
    GET  /api/v1/weather/sites            — site configurations
    POST /api/v1/weather/sites/{site_id}  — upsert site config
    GET  /api/v1/weather/check/{site_id}  — live check single site
    GET  /api/v1/weather/check-all        — live check all sites
    GET  /api/v1/weather/runway/{site_id} — backup power runway calc
    GET  /api/v1/weather/history          — alert history log

HOW TO APPLY:
    Copy weather_monitor.py and this file into ~/hexagrid/api/
    Add to the bottom of api.py:

        from patch_api_weather import router as weather_router
        app.include_router(weather_router)

    Restart uvicorn.
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from weather_monitor import WeatherMonitor

router = APIRouter(prefix="/api/v1/weather", tags=["Weather & Resilience"])
_wm = WeatherMonitor()


# ── Pydantic models ───────────────────────────────────────────────────────────
class SiteConfig(BaseModel):
    name:                  str
    lat:                   float
    lon:                   float
    iso_region:            str
    ups_capacity_mwh:      float   = 10.0
    generator_kw:          float   = 1500.0
    normal_draw_kw:        float   = 500.0
    monitor_earthquake:    bool    = True
    enabled:               bool    = True


class RunwayRequest(BaseModel):
    event_duration_hrs:    float
    current_draw_kw:       Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/summary")
def get_weather_summary():
    """
    Returns cached weather alert summary for all sites.
    Uses last DB entries — does not trigger live API calls.
    For live data use /check-all.
    """
    try:
        _wm.sites = _wm._load_sites()
        from weather_monitor import _get_conn
        from datetime import datetime, timedelta
        conn = _get_conn()

        sites_summary = []
        overall = "normal"
        level_rank = {"normal": 0, "elevated": 1, "warning": 2, "critical": 3}
        total_alerts = 0

        for site_id, site in _wm.sites.items():
            if not site.get("enabled", True):
                continue

            # Get most recent alerts for this site (last 12 hours)
            since = (datetime.utcnow() - timedelta(hours=12)).isoformat()
            rows  = conn.execute("""
                SELECT * FROM weather_alerts
                WHERE site_id = ? AND fetched_at >= ?
                ORDER BY fetched_at DESC LIMIT 10
            """, (site_id, since)).fetchall()

            alerts = [dict(r) for r in rows]
            total_alerts += len(alerts)

            if any(a["level"] == "critical" for a in alerts):
                status = "critical"
            elif any(a["level"] == "warning" for a in alerts):
                status = "warning"
            elif alerts:
                status = "elevated"
            else:
                status = "normal"

            if level_rank.get(status, 0) > level_rank.get(overall, 0):
                overall = status

            sites_summary.append({
                "site_id":     site_id,
                "site_name":   site["name"],
                "iso_region":  site.get("iso_region", ""),
                "lat":         site["lat"],
                "lon":         site["lon"],
                "status":      status,
                "alert_count": len(alerts),
                "alerts":      alerts[:3],
                "ups_mwh":     site.get("ups_capacity_mwh", 0),
                "gen_kw":      site.get("generator_kw", 0),
                "normal_draw_kw": site.get("normal_draw_kw", 500),
            })

        conn.close()
        return {
            "status":         "ok",
            "overall_status": overall,
            "total_alerts":   total_alerts,
            "sites":          sites_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sites")
def get_sites():
    """Returns all configured site weather profiles."""
    try:
        return {"status": "ok", "sites": _wm.get_sites()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sites/{site_id}")
def upsert_site(site_id: str, body: SiteConfig):
    """Create or update a site weather profile."""
    try:
        updated = _wm.upsert_site(site_id, body.dict())
        return {"status": "ok", "site": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/{site_id}")
def check_site(site_id: str):
    """
    Triggers a live NOAA + USGS check for a single site.
    Makes real API calls — may take 2-5 seconds.
    """
    try:
        result = _wm.check_site(site_id)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-all")
def check_all_sites(background_tasks: BackgroundTasks):
    """
    Triggers live checks for all enabled sites.
    Runs in background — results appear in /summary within ~30s.
    """
    try:
        background_tasks.add_task(_wm.check_all_sites)
        return {
            "status": "ok",
            "message": "Weather check triggered for all sites. Results available in /weather/summary shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runway/{site_id}")
def calculate_runway(site_id: str, body: RunwayRequest):
    """
    Calculates backup power runway for a site under a hypothetical event duration.
    """
    try:
        sites = _wm.get_sites()
        if site_id not in sites:
            raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found")
        site = sites[site_id]
        site["id"] = site_id
        runway = _wm.calculate_runway(
            site,
            event_duration_hrs=body.event_duration_hrs,
            current_draw_kw=body.current_draw_kw,
        )
        return {"status": "ok", "runway": runway}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_weather_history(
    site_id: Optional[str] = Query(default=None),
    limit:   int            = Query(default=50, le=500),
):
    """Returns weather alert history, newest first."""
    try:
        entries = _wm.alert_history(site_id=site_id, limit=limit)
        return {"status": "ok", "count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
