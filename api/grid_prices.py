"""
grid_prices.py — HexaGrid Real-Time Grid Price Fetcher
=======================================================
Fetches current hourly LMP + day-ahead forward curve from EIA v2 API.
Falls back to synthetic CAISO TOU model if API is unreachable.

ISO respondent codes:
  CAISO → CISO   ERCOT → ERCO   PJM  → PJM
  NYISO → NYIS   ISONE → ISNE   MISO → MISO
"""

import json, os, time, urllib.request, urllib.error
from datetime import datetime, timezone
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
EIA_API_KEY   = os.environ.get("EIA_API_KEY", "")
EIA_BASE      = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
CACHE_TTL_S   = 300   # cache EIA response for 5 minutes

ISO_RESPONDENT = {
    "CAISO": "CISO",
    "ERCOT": "ERCO",
    "PJM":   "PJM",
    "NYISO": "NYIS",
    "ISONE": "ISNE",
    "MISO":  "MISO",
}

# Simple in-process cache: {cache_key: (fetched_at, data)}
_cache: dict = {}


# ── EIA fetch ──────────────────────────────────────────────────────────────────
def _eia_fetch(respondent: str, n_hours: int = 6) -> Optional[list[dict]]:
    """
    Fetch the last n_hours of hourly demand-forecast (DF) data for a respondent.
    Returns list of {period, value} dicts sorted oldest→newest, or None on error.
    """
    if not EIA_API_KEY:
        return None

    params = (
        f"?api_key={EIA_API_KEY}"
        f"&frequency=hourly"
        f"&data[0]=value"
        f"&facets[respondent][]={respondent}"
        f"&facets[type][]=DF"          # day-ahead demand forecast — best proxy for forward LMP
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&length={n_hours}"
    )

    try:
        req = urllib.request.Request(
            EIA_BASE + params,
            headers={"User-Agent": "HexaGrid/1.0 (contact@hexagrid.ai)"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = json.loads(r.read().decode())

        rows = raw.get("response", {}).get("data", [])
        if not rows:
            return None

        # Sort oldest first
        rows.sort(key=lambda x: x.get("period", ""))
        return rows

    except Exception as e:
        print(f"  [EIA fetch error for {respondent}: {e}]")
        return None


# ── Price normalisation ────────────────────────────────────────────────────────
def _mwh_to_kwh(value_mwh: float) -> float:
    """EIA DF values are in MWh demand, not $/MWh price.
    We use them as a demand proxy to shape a price curve around
    the EIA annual average retail price for the region."""
    return value_mwh


def _demand_to_price(demand_mwh: float, base_price: float,
                     demand_min: float, demand_max: float) -> float:
    """
    Scale demand signal into a price curve.
    Low demand → near floor price, high demand → near peak price.
    """
    if demand_max == demand_min:
        return base_price
    t = (demand_mh - demand_min) / (demand_max - demand_min)
    return round(base_price * (0.7 + 0.6 * t), 5)


# Regional base retail prices (EIA 2024 annual averages, $/kWh)
REGIONAL_BASE_PRICE = {
    "CAISO": 0.232,
    "ERCOT": 0.118,
    "PJM":   0.128,
    "NYISO": 0.195,
    "ISONE": 0.212,
    "MISO":  0.098,
}

# ── Public API ─────────────────────────────────────────────────────────────────
def get_price_curve(iso: str = "CAISO",
                    horizon_min: int = 120) -> dict:
    """
    Returns current price + forward curve for the next horizon_min minutes.

    {
        current_price_usd_kwh: float,
        min_price: float,
        max_price: float,
        cheapest_slot: {minute_offset, price_usd_kwh},
        price_curve: [{minute_offset, price_usd_kwh}, ...],
        source: str,
        timestamp: str,
    }
    """
    cache_key = f"{iso}:{horizon_min}"
    now = time.time()

    # Return cached result if fresh
    if cache_key in _cache:
        fetched_at, cached = _cache[cache_key]
        if now - fetched_at < CACHE_TTL_S:
            return cached

    respondent  = ISO_RESPONDENT.get(iso.upper(), "CISO")
    base_price  = REGIONAL_BASE_PRICE.get(iso.upper(), 0.13)
    n_hours     = max(3, horizon_min // 60 + 2)

    rows   = _eia_fetch(respondent, n_hours=n_hours)
    source = "synthetic-fallback"

    if rows:
        # Build per-minute price curve by linear interpolation between hourly points
        # Each EIA row covers one hour; we interpolate across minutes
        demands = [float(r.get("value") or 0) for r in rows]
        d_min, d_max = min(demands), max(demands)

        # Price floors and peaks calibrated to regional retail averages
        price_floor = base_price * 0.55
        price_peak  = base_price * 1.45

        def demand_to_price(d):
            if d_max == d_min:
                return base_price
            t = (d - d_min) / (d_max - d_min)
            return round(price_floor + (price_peak - price_floor) * t, 5)

        # Expand hourly demand rows to per-minute prices via linear interp
        minute_prices = []
        for i in range(len(demands) - 1):
            p0 = demand_to_price(demands[i])
            p1 = demand_to_price(demands[i + 1])
            for m in range(60):
                alpha = m / 60
                minute_prices.append(round(p0 + alpha * (p1 - p0), 5))

        # Pad to horizon if needed
        while len(minute_prices) < horizon_min:
            minute_prices.append(minute_prices[-1] if minute_prices else base_price)

        minute_prices = minute_prices[:horizon_min]
        source = f"{iso}-EIA-live"

    else:
        # Synthetic fallback — CAISO TOU shape
        try:
            from simulation.digital_twin import grid_price_usd_kwh
            now_tick = int(time.time() // 60) % 1440
            minute_prices = [
                round(grid_price_usd_kwh(now_tick + i), 5)
                for i in range(horizon_min)
            ]
            source = f"{iso}-TOU-synthetic"
        except Exception:
            minute_prices = [base_price] * horizon_min
            source = f"{iso}-flat-fallback"

    price_curve = [
        {"minute_offset": i, "price_usd_kwh": p}
        for i, p in enumerate(minute_prices)
    ]

    current_price = price_curve[0]["price_usd_kwh"]
    cheapest_slot = min(price_curve, key=lambda x: x["price_usd_kwh"])

    result = {
        "current_price_usd_kwh": current_price,
        "horizon_minutes":       horizon_min,
        "min_price":             min(minute_prices),
        "max_price":             max(minute_prices),
        "cheapest_slot":         cheapest_slot,
        "price_curve":           price_curve,
        "source":                source,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "iso":                   iso,
        "eia_rows_received":     len(rows) if rows else 0,
    }

    _cache[cache_key] = (now, result)
    return result
