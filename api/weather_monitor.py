"""
weather_monitor.py — HexaGrid Weather & Resilience Monitor
===========================================================
Monitors severe weather and seismic events for configured data center
sites, correlates with grid price risk, and calculates backup power
runway under each event scenario.

Data sources (all free, no API key required):
    NOAA NWS   — api.weather.gov    — weather alerts + hourly forecasts
    USGS       — earthquake.usgs.gov — real-time seismic events

Usage:
    from weather_monitor import WeatherMonitor
    wm = WeatherMonitor()
    wm.add_site("dc_east", "Ashburn VA", 39.0438, -77.4874,
                ups_capacity_mwh=12.0, generator_kw=2000.0,
                iso_region="PJM")
    summary = wm.get_summary()
"""

import json
import os
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.expanduser("~/hexagrid")
DB_PATH     = os.environ.get("HEXAGRID_DB", os.path.join(BASE_DIR, "hexagrid.db"))
CONFIG_PATH = os.environ.get("HEXAGRID_WEATHER_CONFIG",
                             os.path.join(BASE_DIR, "weather_sites.json"))

# ── Event type definitions ────────────────────────────────────────────────────
SEVERE_EVENT_KEYWORDS = {
    "tornado":    ["tornado", "twister"],
    "hurricane":  ["hurricane", "tropical storm", "typhoon", "cyclone"],
    "blizzard":   ["blizzard", "ice storm", "winter storm", "freezing", "snow squall", "arctic"],
    "heat":       ["excessive heat", "heat wave", "extreme heat"],
    "earthquake": ["earthquake"],  # handled via USGS separately
}

# NWS severity → internal level mapping
NWS_SEVERITY = {
    "Extreme":  "critical",
    "Severe":   "critical",
    "Moderate": "warning",
    "Minor":    "elevated",
    "Unknown":  "elevated",
}

# Grid correlation: event type + severity → estimated price spike probability
GRID_CORRELATION = {
    ("tornado",   "critical"): {"prob": 0.80, "est_spike_pct": 85,  "duration_hrs": 3},
    ("tornado",   "warning"):  {"prob": 0.55, "est_spike_pct": 45,  "duration_hrs": 2},
    ("hurricane", "critical"): {"prob": 0.90, "est_spike_pct": 120, "duration_hrs": 24},
    ("hurricane", "warning"):  {"prob": 0.70, "est_spike_pct": 75,  "duration_hrs": 12},
    ("blizzard",  "critical"): {"prob": 0.85, "est_spike_pct": 95,  "duration_hrs": 18},
    ("blizzard",  "warning"):  {"prob": 0.65, "est_spike_pct": 55,  "duration_hrs": 10},
    ("heat",      "critical"): {"prob": 0.75, "est_spike_pct": 70,  "duration_hrs": 8},
    ("heat",      "warning"):  {"prob": 0.50, "est_spike_pct": 40,  "duration_hrs": 6},
    ("earthquake","critical"): {"prob": 0.60, "est_spike_pct": 50,  "duration_hrs": 2},
    ("earthquake","warning"):  {"prob": 0.35, "est_spike_pct": 25,  "duration_hrs": 1},
}

DEFAULT_GRID_CORRELATION = {"prob": 0.20, "est_spike_pct": 15, "duration_hrs": 2}

# ── DB init ───────────────────────────────────────────────────────────────────
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at   TEXT NOT NULL,
                site_id      TEXT NOT NULL,
                event_type   TEXT NOT NULL,
                nws_event    TEXT,
                headline     TEXT,
                description  TEXT,
                severity     TEXT,
                level        TEXT,
                onset        TEXT,
                expires      TEXT,
                source       TEXT DEFAULT 'NWS',
                magnitude    REAL,
                active       INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_alerts_site
            ON weather_alerts (site_id, fetched_at)
        """)
        conn.commit()

_init_db()


# ── WeatherMonitor ────────────────────────────────────────────────────────────
class WeatherMonitor:

    def __init__(self):
        self.sites = self._load_sites()

    # ── Site management ───────────────────────────────────────────────────────
    def _load_sites(self) -> dict:
        if not os.path.exists(CONFIG_PATH):
            # Seed with the 5 default fleet sites
            defaults = {
                "dc_east": {
                    "name": "East Coast DC",
                    "lat": 39.0438, "lon": -77.4874,
                    "iso_region": "PJM",
                    "ups_capacity_mwh": 10.0,
                    "generator_kw": 1500.0,
                    "normal_draw_kw": 500.0,
                    "monitor_earthquake": True,
                    "enabled": True,
                },
                "dc_mid": {
                    "name": "Mid-Atlantic DC",
                    "lat": 38.9072, "lon": -77.0369,
                    "iso_region": "PJM",
                    "ups_capacity_mwh": 8.0,
                    "generator_kw": 1200.0,
                    "normal_draw_kw": 420.0,
                    "monitor_earthquake": True,
                    "enabled": True,
                },
                "dc_texas": {
                    "name": "Texas DC",
                    "lat": 30.2672, "lon": -97.7431,
                    "iso_region": "ERCOT",
                    "ups_capacity_mwh": 12.0,
                    "generator_kw": 2000.0,
                    "normal_draw_kw": 650.0,
                    "monitor_earthquake": False,
                    "enabled": True,
                },
                "dc_california": {
                    "name": "California DC",
                    "lat": 37.3382, "lon": -121.8863,
                    "iso_region": "CAISO",
                    "ups_capacity_mwh": 15.0,
                    "generator_kw": 2500.0,
                    "normal_draw_kw": 800.0,
                    "monitor_earthquake": True,
                    "enabled": True,
                },
                "dc_newengland": {
                    "name": "New England DC",
                    "lat": 42.3601, "lon": -71.0589,
                    "iso_region": "ISONE",
                    "ups_capacity_mwh": 6.0,
                    "generator_kw": 1000.0,
                    "normal_draw_kw": 350.0,
                    "monitor_earthquake": False,
                    "enabled": True,
                },
            }
            self._save_sites(defaults)
            return defaults
        with open(CONFIG_PATH) as f:
            return json.load(f)

    def _save_sites(self, sites: dict):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(sites, f, indent=2)

    def get_sites(self) -> dict:
        self.sites = self._load_sites()
        return self.sites

    def upsert_site(self, site_id: str, data: dict) -> dict:
        self.sites = self._load_sites()
        if site_id not in self.sites:
            self.sites[site_id] = {}
        self.sites[site_id].update(data)
        self._save_sites(self.sites)
        return self.sites[site_id]

    # ── NOAA NWS fetch ────────────────────────────────────────────────────────
    def _fetch_nws_alerts(self, lat: float, lon: float) -> list:
        """
        Fetches active NWS alerts for a lat/lon point.
        Returns list of parsed alert dicts.
        """
        url = f"https://api.weather.gov/alerts/active?point={lat},{lon}&status=actual"
        headers = {
            "User-Agent": "HexaGrid/1.0 (energy-management; contact@hexagrid.ai)",
            "Accept": "application/geo+json",
        }
        try:
            req  = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("features", [])
        except Exception as e:
            print(f"[NWS fetch error] {e}")
            return []

    def _fetch_nws_forecast(self, lat: float, lon: float) -> dict:
        """Fetches hourly forecast for a lat/lon — used for heat/cold severity."""
        try:
            # Step 1: get grid endpoint
            url = f"https://api.weather.gov/points/{lat},{lon}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "HexaGrid/1.0",
                "Accept": "application/geo+json",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                point_data = json.loads(resp.read().decode("utf-8"))
            hourly_url = point_data["properties"]["forecastHourly"]

            req2 = urllib.request.Request(hourly_url, headers={
                "User-Agent": "HexaGrid/1.0",
                "Accept": "application/geo+json",
            })
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                forecast = json.loads(resp2.read().decode("utf-8"))

            periods = forecast["properties"]["periods"][:24]
            return {
                "periods":   periods,
                "max_temp_f": max(p["temperature"] for p in periods if p["temperatureUnit"] == "F") if periods else None,
                "min_temp_f": min(p["temperature"] for p in periods if p["temperatureUnit"] == "F") if periods else None,
            }
        except Exception as e:
            print(f"[NWS forecast error] {e}")
            return {}

    # ── USGS Earthquake fetch ─────────────────────────────────────────────────
    def _fetch_usgs_earthquakes(self, lat: float, lon: float,
                                 radius_km: int = 200, min_mag: float = 4.0) -> list:
        """Fetches recent earthquakes within radius of a point."""
        now    = datetime.utcnow()
        start  = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        end    = now.strftime("%Y-%m-%dT%H:%M:%S")
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query"
            f"?format=geojson&starttime={start}&endtime={end}"
            f"&latitude={lat}&longitude={lon}&maxradiuskm={radius_km}"
            f"&minmagnitude={min_mag}&orderby=magnitude"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HexaGrid/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("features", [])
        except Exception as e:
            print(f"[USGS fetch error] {e}")
            return []

    # ── Event classification ──────────────────────────────────────────────────
    def _classify_nws_event(self, event_name: str) -> Optional[str]:
        event_lower = event_name.lower()
        for event_type, keywords in SEVERE_EVENT_KEYWORDS.items():
            if event_type == "earthquake":
                continue
            if any(kw in event_lower for kw in keywords):
                return event_type
        return None

    # ── Backup power runway ───────────────────────────────────────────────────
    def calculate_runway(self, site: dict, event_duration_hrs: float,
                          current_draw_kw: Optional[float] = None) -> dict:
        """
        Calculates backup power runway for a site under a weather event.

        Returns full operational hours, reduced capacity options,
        and recommended deferral actions.
        """
        ups_mwh     = site.get("ups_capacity_mwh", 0)
        gen_kw      = site.get("generator_kw", 0)
        normal_kw   = current_draw_kw or site.get("normal_draw_kw", 500)
        ups_kwh     = ups_mwh * 1000

        # Generator covers ongoing load, UPS covers initial gap
        gen_covers_load  = gen_kw >= normal_kw
        total_backup_kwh = ups_kwh + (gen_kw * event_duration_hrs if gen_covers_load else 0)

        # Hours at full capacity
        if gen_covers_load:
            full_capacity_hrs = event_duration_hrs  # Generator sustains indefinitely
            runway_note = "Generator covers full load — UPS is buffer only"
        else:
            full_capacity_hrs = round(ups_kwh / normal_kw, 1) if normal_kw > 0 else 0
            runway_note = "Generator undersized — UPS provides primary runway"

        event_covered = full_capacity_hrs >= event_duration_hrs

        # Scaled capacity options if runway is insufficient
        scale_options = []
        if not event_covered:
            for scale_pct in [90, 75, 60, 50]:
                scaled_kw  = normal_kw * (scale_pct / 100)
                scaled_hrs = round(ups_kwh / scaled_kw, 1) if scaled_kw > 0 else 0
                gen_gap    = max(0, scaled_kw - gen_kw)
                scale_options.append({
                    "scale_pct":   scale_pct,
                    "draw_kw":     round(scaled_kw, 1),
                    "runway_hrs":  scaled_hrs,
                    "covers_event": scaled_hrs >= event_duration_hrs,
                    "gen_gap_kw":  round(gen_gap, 1),
                })

        # Deferral recommendations
        deferral_savings_kw = normal_kw * 0.20  # Assume 20% of load is deferrable GPU jobs
        deferred_draw_kw    = normal_kw - deferral_savings_kw
        deferred_runway_hrs = round(ups_kwh / deferred_draw_kw, 1) if deferred_draw_kw > 0 else 0

        return {
            "site_id":              site.get("id", "unknown"),
            "normal_draw_kw":       round(normal_kw, 1),
            "ups_capacity_kwh":     round(ups_kwh, 1),
            "generator_kw":         round(gen_kw, 1),
            "gen_covers_load":      gen_covers_load,
            "full_capacity_hrs":    full_capacity_hrs,
            "event_duration_hrs":   event_duration_hrs,
            "event_covered":        event_covered,
            "runway_note":          runway_note,
            "scale_options":        scale_options,
            "deferred_draw_kw":     round(deferred_draw_kw, 1),
            "deferred_runway_hrs":  deferred_runway_hrs,
            "deferral_savings_kw":  round(deferral_savings_kw, 1),
            "recommendation": (
                "Full capacity sustainable for entire event duration."
                if event_covered else
                f"Recommend deferring non-critical workloads immediately — "
                f"reduces draw to {round(deferred_draw_kw, 0):.0f} kW, "
                f"extending runway to {deferred_runway_hrs} hrs."
            ),
        }

    # ── Full site check ───────────────────────────────────────────────────────
    def check_site(self, site_id: str) -> dict:
        """
        Runs a full weather + seismic check for a single site.
        Returns all active alerts, grid correlation, and runway analysis.
        """
        self.sites = self._load_sites()
        site = self.sites.get(site_id)
        if not site or not site.get("enabled", True):
            return {"site_id": site_id, "alerts": [], "status": "disabled"}

        lat, lon = site["lat"], site["lon"]
        active_alerts = []

        # ── NWS alerts ────────────────────────────────────────────────────────
        nws_features = self._fetch_nws_alerts(lat, lon)
        for feature in nws_features:
            props      = feature.get("properties", {})
            event_name = props.get("event", "")
            event_type = self._classify_nws_event(event_name)
            if not event_type:
                continue

            nws_sev = props.get("severity", "Unknown")
            level   = NWS_SEVERITY.get(nws_sev, "elevated")
            corr    = GRID_CORRELATION.get((event_type, level), DEFAULT_GRID_CORRELATION)

            onset   = props.get("onset", "")
            expires = props.get("expires", "")

            # Estimate duration
            try:
                onset_dt   = datetime.fromisoformat(onset.replace("Z", "+00:00"))
                expires_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                duration_hrs = max(1.0, (expires_dt - onset_dt).total_seconds() / 3600)
            except Exception:
                duration_hrs = corr["duration_hrs"]

            runway = self.calculate_runway(site, duration_hrs)

            alert = {
                "site_id":          site_id,
                "site_name":        site["name"],
                "event_type":       event_type,
                "nws_event":        event_name,
                "headline":         props.get("headline", event_name),
                "description":      (props.get("description", "") or "")[:400],
                "severity":         nws_sev,
                "level":            level,
                "onset":            onset,
                "expires":          expires,
                "duration_hrs":     round(duration_hrs, 1),
                "source":           "NWS",
                "grid_correlation": corr,
                "iso_region":       site.get("iso_region", ""),
                "runway":           runway,
            }
            active_alerts.append(alert)

            # Persist to DB
            with _get_conn() as conn:
                conn.execute("""
                    INSERT INTO weather_alerts
                      (fetched_at, site_id, event_type, nws_event, headline,
                       description, severity, level, onset, expires, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    site_id, event_type, event_name,
                    alert["headline"], alert["description"],
                    nws_sev, level, onset, expires, "NWS"
                ))
                conn.commit()

        # ── USGS earthquakes ──────────────────────────────────────────────────
        if site.get("monitor_earthquake", True):
            quakes = self._fetch_usgs_earthquakes(lat, lon)
            for q in quakes[:3]:  # max 3 most significant
                props = q.get("properties", {})
                mag   = props.get("mag", 0)
                level = "critical" if mag >= 6.0 else "warning" if mag >= 5.0 else "elevated"
                corr  = GRID_CORRELATION.get(("earthquake", level), DEFAULT_GRID_CORRELATION)
                runway = self.calculate_runway(site, corr["duration_hrs"])
                place  = props.get("place", "nearby region")
                time_ms = props.get("time", 0)
                quake_time = datetime.utcfromtimestamp(time_ms / 1000).isoformat() if time_ms else ""

                alert = {
                    "site_id":          site_id,
                    "site_name":        site["name"],
                    "event_type":       "earthquake",
                    "nws_event":        f"M{mag:.1f} Earthquake",
                    "headline":         f"M{mag:.1f} earthquake — {place}",
                    "description":      f"Magnitude {mag:.1f} earthquake detected near {place}.",
                    "severity":         "Severe" if mag >= 6.0 else "Moderate",
                    "level":            level,
                    "onset":            quake_time,
                    "expires":          "",
                    "duration_hrs":     corr["duration_hrs"],
                    "source":           "USGS",
                    "magnitude":        mag,
                    "grid_correlation": corr,
                    "iso_region":       site.get("iso_region", ""),
                    "runway":           runway,
                }
                active_alerts.append(alert)

                with _get_conn() as conn:
                    conn.execute("""
                        INSERT INTO weather_alerts
                          (fetched_at, site_id, event_type, nws_event, headline,
                           description, severity, level, onset, source, magnitude)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.utcnow().isoformat(),
                        site_id, "earthquake",
                        f"M{mag:.1f} Earthquake", alert["headline"],
                        alert["description"], alert["severity"],
                        level, quake_time, "USGS", mag
                    ))
                    conn.commit()

        # Overall site status
        if any(a["level"] == "critical" for a in active_alerts):
            status = "critical"
        elif any(a["level"] == "warning" for a in active_alerts):
            status = "warning"
        elif active_alerts:
            status = "elevated"
        else:
            status = "normal"

        return {
            "site_id":       site_id,
            "site_name":     site["name"],
            "iso_region":    site.get("iso_region", ""),
            "lat":           lat,
            "lon":           lon,
            "status":        status,
            "alert_count":   len(active_alerts),
            "alerts":        active_alerts,
            "checked_at":    datetime.utcnow().isoformat(),
        }

    # ── Full fleet check ──────────────────────────────────────────────────────
    def check_all_sites(self) -> dict:
        """Check all enabled sites and return combined summary."""
        self.sites = self._load_sites()
        results = {}
        overall = "normal"
        level_rank = {"normal": 0, "elevated": 1, "warning": 2, "critical": 3}

        for site_id in self.sites:
            result = self.check_site(site_id)
            results[site_id] = result
            if level_rank.get(result.get("status", "normal"), 0) > level_rank.get(overall, 0):
                overall = result["status"]

        total_alerts = sum(r.get("alert_count", 0) for r in results.values())

        return {
            "overall_status": overall,
            "total_alerts":   total_alerts,
            "sites":          results,
            "checked_at":     datetime.utcnow().isoformat(),
        }

    # ── History ───────────────────────────────────────────────────────────────
    def alert_history(self, site_id: Optional[str] = None, limit: int = 50) -> list:
        with _get_conn() as conn:
            if site_id:
                rows = conn.execute("""
                    SELECT * FROM weather_alerts WHERE site_id = ?
                    ORDER BY fetched_at DESC LIMIT ?
                """, (site_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM weather_alerts
                    ORDER BY fetched_at DESC LIMIT ?
                """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── Demo data ─────────────────────────────────────────────────────────────
    def inject_demo_alerts(self):
        """Inject realistic demo alerts for dashboard testing."""
        demo = [
            {
                "site_id": "dc_texas", "event_type": "tornado",
                "nws_event": "Tornado Watch",
                "headline": "Tornado Watch in effect until 10 PM CDT",
                "description": "Conditions favorable for tornado development across central Texas.",
                "severity": "Severe", "level": "critical",
                "onset": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(hours=4)).isoformat(),
                "source": "NWS", "magnitude": None,
            },
            {
                "site_id": "dc_newengland", "event_type": "blizzard",
                "nws_event": "Blizzard Warning",
                "headline": "Blizzard Warning — 18 to 24 inches expected",
                "description": "Heavy snow with winds 35-50 mph. Whiteout conditions possible.",
                "severity": "Extreme", "level": "critical",
                "onset": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(hours=18)).isoformat(),
                "source": "NWS", "magnitude": None,
            },
            {
                "site_id": "dc_california", "event_type": "earthquake",
                "nws_event": "M5.8 Earthquake",
                "headline": "M5.8 earthquake — 12 km E of San Jose CA",
                "description": "Magnitude 5.8 earthquake detected. Check infrastructure for damage.",
                "severity": "Severe", "level": "warning",
                "onset": datetime.utcnow().isoformat(),
                "expires": "",
                "source": "USGS", "magnitude": 5.8,
            },
            {
                "site_id": "dc_east", "event_type": "heat",
                "nws_event": "Excessive Heat Warning",
                "headline": "Excessive Heat Warning through Sunday",
                "description": "Dangerous heat expected. High temperatures 105-112°F.",
                "severity": "Extreme", "level": "critical",
                "onset": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(hours=48)).isoformat(),
                "source": "NWS", "magnitude": None,
            },
        ]
        with _get_conn() as conn:
            for d in demo:
                conn.execute("""
                    INSERT INTO weather_alerts
                      (fetched_at, site_id, event_type, nws_event, headline,
                       description, severity, level, onset, expires, source, magnitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    d["site_id"], d["event_type"], d["nws_event"],
                    d["headline"], d["description"], d["severity"],
                    d["level"], d["onset"], d["expires"],
                    d["source"], d["magnitude"]
                ))
            conn.commit()
        print(f"Injected {len(demo)} demo weather alerts.")


if __name__ == "__main__":
    wm = WeatherMonitor()
    wm.inject_demo_alerts()
    print("\nSite configs:")
    for sid, site in wm.get_sites().items():
        print(f"  {sid}: {site['name']} ({site['iso_region']}) — "
              f"UPS {site['ups_capacity_mwh']} MWh, Gen {site['generator_kw']} kW")
    print("\nRunway calculation demo (Texas DC, 6hr tornado event):")
    site = wm.get_sites()["dc_texas"]
    runway = wm.calculate_runway(site, event_duration_hrs=6.0)
    print(json.dumps(runway, indent=2))
