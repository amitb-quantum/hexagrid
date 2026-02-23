"""
Energia Phase 8 — Carbon Intensity Connector
=============================================
Pulls real-time carbon intensity (gCO2eq/kWh) and fuel mix from
Electricity Maps API v3 for the same ISO regions as grid_connector.py.

Features:
  - Live carbon intensity per ISO zone
  - 24-hour history per zone
  - Fuel mix breakdown (wind, solar, nuclear, gas, coal, hydro...)
  - Cleanest zone detection across all monitored regions
  - SQLite cache (15-min TTL) for fault tolerance
  - .env support — API key never hardcoded

Zones mapped to Energia ISOs:
  CAISO  → US-CAL-CISO
  ERCOT  → US-TEX-ERCO
  NYISO  → US-NY-NYIS
  ISONE  → US-NE-ISNE
  PJM    → US-MIDA-PJM

Usage:
  from carbon_connector import CarbonConnector
  cc = CarbonConnector()
  snapshot = cc.get_snapshot()        # all zones, current
  history  = cc.get_history('US-CAL-CISO')  # 24h history
"""

import os, sqlite3, json, logging, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY   = os.getenv("ELECTRICITY_MAPS_API_KEY", "")
BASE_URL  = "https://api.electricitymaps.com/v3"
CACHE_DB  = Path(__file__).parent / "cache" / "carbon_cache.db"
CACHE_TTL = 900   # seconds — 15 minutes (API updates hourly)
LOG_DIR   = Path(__file__).parent / "logs" / "carbon"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carbon_connector")

# ── ISO → Electricity Maps zone mapping ──────────────────────────────────────
ZONES = {
    "CAISO":  {"em_zone": "US-CAL-CISO",  "label": "California (CAISO)",        "lat": 36.7, "lon": -119.4},
    "ERCOT":  {"em_zone": "US-TEX-ERCO",  "label": "Texas (ERCOT)",             "lat": 31.0, "lon":  -99.0},
    "NYISO":  {"em_zone": "US-NY-NYIS",   "label": "New York (NYISO)",          "lat": 42.9, "lon":  -75.1},
    "ISONE":  {"em_zone": "US-NE-ISNE",   "label": "New England (ISONE)",       "lat": 42.4, "lon":  -71.4},
    "PJM":    {"em_zone": "US-MIDA-PJM",  "label": "Mid-Atlantic (PJM)",        "lat": 38.9, "lon":  -77.0},
}

# Carbon intensity thresholds (gCO2eq/kWh) — industry reference values
THRESHOLDS = {
    "very_clean":  50,    # mostly renewable
    "clean":       150,   # low carbon (nuclear + renewables)
    "moderate":    300,   # mixed grid
    "dirty":       450,   # fossil-heavy
    # above 450 = very dirty
}


def carbon_label(gco2: float) -> str:
    if gco2 < THRESHOLDS["very_clean"]:  return "Very Clean"
    if gco2 < THRESHOLDS["clean"]:       return "Clean"
    if gco2 < THRESHOLDS["moderate"]:    return "Moderate"
    if gco2 < THRESHOLDS["dirty"]:       return "Dirty"
    return "Very Dirty"

def carbon_color(gco2: float) -> str:
    if gco2 < THRESHOLDS["very_clean"]:  return "#00ff88"
    if gco2 < THRESHOLDS["clean"]:       return "#44dd88"
    if gco2 < THRESHOLDS["moderate"]:    return "#ffaa00"
    if gco2 < THRESHOLDS["dirty"]:       return "#ff6644"
    return "#ff4444"


class CarbonConnector:
    def __init__(self):
        self._init_cache()
        self._init_logs()
        if not API_KEY:
            logger.warning(
                "ELECTRICITY_MAPS_API_KEY not set. "
                "Add it to ~/energia/.env: ELECTRICITY_MAPS_API_KEY=your_key"
            )

    # ── Cache ─────────────────────────────────────────────────────────────────
    def _init_cache(self):
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(CACHE_DB), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS carbon_cache (
                key        TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )""")
        self._db.commit()

    def _init_logs(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_get(self, key: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT data, fetched_at FROM carbon_cache WHERE key=?", (key,)
        ).fetchone()
        if row and (time.time() - row[1]) < CACHE_TTL:
            return json.loads(row[0])
        return None

    def _cache_set(self, key: str, data: dict):
        self._db.execute(
            "INSERT OR REPLACE INTO carbon_cache (key, data, fetched_at) VALUES (?,?,?)",
            (key, json.dumps(data), time.time())
        )
        self._db.commit()

    # ── API calls ─────────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        if not API_KEY:
            return None
        try:
            r = requests.get(
                f"{BASE_URL}/{endpoint}",
                headers={"auth-token": API_KEY},
                params=params,
                timeout=10
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            logger.warning(f"Electricity Maps HTTP error {e.response.status_code}: {endpoint} {params}")
            return None
        except Exception as e:
            logger.warning(f"Electricity Maps error: {e}")
            return None

    # ── Public interface ──────────────────────────────────────────────────────
    def get_zone_latest(self, iso: str) -> Optional[dict]:
        """
        Get latest carbon intensity + fuel mix for one ISO zone.
        Returns dict with carbonIntensity, fossilFreePercentage, renewablePercentage,
        powerConsumptionBreakdown, label, color.
        """
        zone_info = ZONES.get(iso)
        if not zone_info:
            return None

        em_zone = zone_info["em_zone"]
        cache_key = f"latest:{em_zone}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        # Fetch carbon intensity
        ci_data = self._get("carbon-intensity/latest", {"zone": em_zone})

        # Fetch power breakdown for fuel mix
        pb_data = self._get("power-breakdown/latest", {"zone": em_zone})

        if not ci_data:
            return self._fallback(iso)

        gco2 = ci_data.get("carbonIntensity", 0)

        result = {
            "iso":                iso,
            "zone":               em_zone,
            "label":              zone_info["label"],
            "carbon_intensity":   gco2,
            "carbon_label":       carbon_label(gco2),
            "carbon_color":       carbon_color(gco2),
            "is_estimated":       ci_data.get("isEstimated", True),
            "datetime":           ci_data.get("datetime", ""),
            "fossil_free_pct":    None,
            "renewable_pct":      None,
            "fuel_mix":           {},
        }

        if pb_data:
            result["fossil_free_pct"] = pb_data.get("fossilFreePercentage")
            result["renewable_pct"]   = pb_data.get("renewablePercentage")
            breakdown = pb_data.get("powerConsumptionBreakdown", {})
            total = sum(v for v in breakdown.values() if isinstance(v, (int, float)) and v > 0)
            if total > 0:
                result["fuel_mix"] = {
                    k: round(v / total * 100, 1)
                    for k, v in breakdown.items()
                    if isinstance(v, (int, float)) and v > 0
                }

        self._cache_set(cache_key, result)
        return result

    def get_snapshot(self) -> dict:
        """
        Get current carbon intensity for all monitored zones.
        Returns dict with per-zone data + cleanest zone + summary stats.
        """
        zones = {}
        for iso in ZONES:
            data = self.get_zone_latest(iso)
            if data:
                zones[iso] = data

        if not zones:
            return {"zones": {}, "cleanest_iso": None, "dirtiest_iso": None,
                    "avg_carbon_intensity": None, "error": "No data available"}

        # Find cleanest / dirtiest
        by_carbon = sorted(zones.items(), key=lambda x: x[1]["carbon_intensity"])
        cleanest  = by_carbon[0][0]
        dirtiest  = by_carbon[-1][0]
        avg_ci    = round(sum(z["carbon_intensity"] for z in zones.values()) / len(zones), 1)

        return {
            "zones":               zones,
            "cleanest_iso":        cleanest,
            "cleanest_label":      zones[cleanest]["label"],
            "cleanest_ci":         zones[cleanest]["carbon_intensity"],
            "dirtiest_iso":        dirtiest,
            "dirtiest_label":      zones[dirtiest]["label"],
            "dirtiest_ci":         zones[dirtiest]["carbon_intensity"],
            "avg_carbon_intensity": avg_ci,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

    def get_history(self, iso: str) -> list:
        """
        Get last 24h of hourly carbon intensity for one ISO zone.
        Returns list of {datetime, carbon_intensity, carbon_label, carbon_color}.
        """
        zone_info = ZONES.get(iso)
        if not zone_info:
            return []

        em_zone   = zone_info["em_zone"]
        cache_key = f"history:{em_zone}"
        cached    = self._cache_get(cache_key)
        if cached:
            return cached

        data = self._get("carbon-intensity/history", {"zone": em_zone})
        if not data or "history" not in data:
            return []

        result = [
            {
                "datetime":         h.get("datetime", ""),
                "carbon_intensity": h.get("carbonIntensity", 0),
                "carbon_label":     carbon_label(h.get("carbonIntensity", 0)),
                "carbon_color":     carbon_color(h.get("carbonIntensity", 0)),
                "is_estimated":     h.get("isEstimated", True),
            }
            for h in data["history"]
        ]

        self._cache_set(cache_key, result)
        return result

    def get_pareto_data(self, local_api_base: str = "http://localhost:8000/api/v1") -> dict:
        """
        Compute cost vs carbon Pareto tradeoff across all zones.
        Fetches live CAISO price from local pricefeed API; uses EIA regional
        averages for other ISOs. No GridConnector import needed.
        """
        snapshot = self.get_snapshot()
        zones    = snapshot.get("zones", {})

        # EIA 2024 average wholesale prices by region (USD/kWh)
        REGIONAL_PRICES = {
            "CAISO": 0.0620,
            "ERCOT": 0.0410,
            "NYISO": 0.0580,
            "ISONE": 0.0550,
            "PJM":   0.0480,
        }
        price_map = dict(REGIONAL_PRICES)

        # Try to get live CAISO price from local pricefeed API
        try:
            r = requests.get(f"{local_api_base}/pricefeed?horizon_min=5", timeout=5)
            if r.ok:
                feed = r.json()
                caiso_price = feed.get("current_price_usd_kwh")
                if caiso_price:
                    price_map["CAISO"] = caiso_price
        except Exception as e:
            logger.warning(f"Could not fetch live pricefeed for Pareto: {e}")

        points = []
        for iso, z in zones.items():
            price = price_map.get(iso, 0.05)
            points.append({
                "iso":              iso,
                "label":            z["label"],
                "carbon_intensity": z["carbon_intensity"],
                "carbon_label":     z["carbon_label"],
                "carbon_color":     z["carbon_color"],
                "price_usd_kwh":    round(price, 5),
                "fossil_free_pct":  z.get("fossil_free_pct"),
            })

        if not points:
            return {"points": [], "recommendations": {}}

        # Normalize for scoring (0=best, 1=worst)
        min_price  = min(p["price_usd_kwh"]    for p in points)
        max_price  = max(p["price_usd_kwh"]    for p in points)
        min_carbon = min(p["carbon_intensity"]  for p in points)
        max_carbon = max(p["carbon_intensity"]  for p in points)

        price_range  = max_price  - min_price  or 1
        carbon_range = max_carbon - min_carbon or 1

        for p in points:
            p["price_norm"]  = (p["price_usd_kwh"]    - min_price)  / price_range
            p["carbon_norm"] = (p["carbon_intensity"]  - min_carbon) / carbon_range

        # Recommendations at three weightings
        recommendations = {}
        for weight_label, cost_w, carbon_w in [
            ("cost_only",    1.0, 0.0),
            ("balanced",     0.5, 0.5),
            ("carbon_only",  0.0, 1.0),
        ]:
            best = min(points, key=lambda p: cost_w * p["price_norm"] + carbon_w * p["carbon_norm"])
            recommendations[weight_label] = {
                "iso":              best["iso"],
                "label":            best["label"],
                "price_usd_kwh":    best["price_usd_kwh"],
                "carbon_intensity": best["carbon_intensity"],
                "carbon_label":     best["carbon_label"],
            }

        # Pareto frontier — points not dominated on both axes
        pareto = []
        for p in points:
            dominated = any(
                q["price_norm"] <= p["price_norm"] and q["carbon_norm"] <= p["carbon_norm"]
                and (q["price_norm"] < p["price_norm"] or q["carbon_norm"] < p["carbon_norm"])
                for q in points if q["iso"] != p["iso"]
            )
            p["on_pareto_frontier"] = not dominated
            pareto.append(p)

        return {
            "points":          pareto,
            "recommendations": recommendations,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    # ── Fallback (when API unavailable) ──────────────────────────────────────
    def _fallback(self, iso: str) -> dict:
        """Return synthetic carbon data based on known regional averages."""
        FALLBACK_CI = {
            "CAISO": 210,   # California — significant solar/wind
            "ERCOT": 380,   # Texas — heavy gas
            "NYISO": 170,   # New York — nuclear + hydro
            "ISONE": 220,   # New England — nuclear + gas
            "PJM":   310,   # Mid-Atlantic — mixed
        }
        gco2 = FALLBACK_CI.get(iso, 300)
        zone_info = ZONES.get(iso, {})
        return {
            "iso":              iso,
            "zone":             zone_info.get("em_zone", iso),
            "label":            zone_info.get("label", iso),
            "carbon_intensity": gco2,
            "carbon_label":     carbon_label(gco2),
            "carbon_color":     carbon_color(gco2),
            "is_estimated":     True,
            "is_fallback":      True,
            "datetime":         datetime.now(timezone.utc).isoformat(),
            "fossil_free_pct":  None,
            "renewable_pct":    None,
            "fuel_mix":         {},
        }


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    cc = CarbonConnector()
    print("\n=== Carbon Snapshot ===")
    snap = cc.get_snapshot()
    for iso, z in snap.get("zones", {}).items():
        est = " (est)" if z.get("is_estimated") else ""
        print(f"  {iso:6s} {z['carbon_intensity']:5.0f} gCO2/kWh  [{z['carbon_label']}]{est}")
    print(f"\n  Cleanest: {snap.get('cleanest_iso')} — {snap.get('cleanest_ci')} gCO2/kWh")
    print(f"  Dirtiest: {snap.get('dirtiest_iso')} — {snap.get('dirtiest_ci')} gCO2/kWh")
    print(f"  Average:  {snap.get('avg_carbon_intensity')} gCO2/kWh")
    print("\n=== Pareto Analysis ===")
    pareto = cc.get_pareto_data()
    for label, rec in pareto.get("recommendations", {}).items():
        print(f"  {label:15s} → {rec['iso']:6s} ${rec['price_usd_kwh']:.4f}/kWh  {rec['carbon_intensity']:.0f} gCO2/kWh")
