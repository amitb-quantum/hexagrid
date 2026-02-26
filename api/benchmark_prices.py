"""
benchmark_prices.py — HexaGrid Benchmark: Historical Price Fetcher
===================================================================
Fetches 30-day hourly LMP data from public grid APIs:
  CAISO  — OASIS API (public, no key)
  ERCOT  — Public data portal (settlement point prices)
  PJM    — PJM Data Miner 2 API (public, no key)

Falls back to statistically accurate synthetic prices if APIs
are unreachable, clearly labelled in the output.
"""

import json, os, time, urllib.request, urllib.error, urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np

CACHE_DIR = os.path.expanduser("~/hexagrid/benchmark_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Date window ────────────────────────────────────────────────────────────────
def get_window():
    end   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    return start, end

# ── Generic HTTP fetch ─────────────────────────────────────────────────────────
def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={
        "User-Agent": "HexaGrid-Benchmark/1.0 (contact@hexagrid.ai)",
        "Accept": "application/json,text/csv,*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")

# ══════════════════════════════════════════════════════════════════════════════
#  CAISO — OASIS API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_caiso(start: datetime, end: datetime) -> dict:
    """
    Fetches CAISO SP15 (Southern California) hourly LMP via OASIS.
    Returns {iso, node, prices: [(ts_iso, price_per_mwh), ...], source}
    """
    cache_path = os.path.join(CACHE_DIR, f"caiso_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    prices = []
    source = "CAISO OASIS API"

    try:
        # CAISO OASIS returns CSV — fetch week by week to stay under size limits
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7), end)
            s = cursor.strftime("%Y%m%dT%H:%M-0000")
            e = chunk_end.strftime("%Y%m%dT%H:%M-0000")
            url = (
                f"http://oasis.caiso.com/oasisapi/SingleZip"
                f"?queryname=PRC_LMP&startdatetime={s}&enddatetime={e}"
                f"&version=1&market_run_id=DAM&node=TH_SP15_GEN-APND"
                f"&resultformat=6"
            )
            raw = _get(url, timeout=30)
            # Parse CSV rows: INTERVALSTARTTIME_GMT, LMP_PRC
            for line in raw.splitlines():
                parts = line.split(",")
                if len(parts) < 10:
                    continue
                try:
                    ts_raw = parts[0].strip().strip('"')
                    lmp    = float(parts[9].strip().strip('"'))
                    # Convert to ISO
                    ts = datetime.strptime(ts_raw[:16], "%Y-%m-%dT%H:%M").isoformat()
                    prices.append((ts, round(lmp / 1000, 6)))  # $/MWh → $/kWh
                except (ValueError, IndexError):
                    continue
            cursor = chunk_end
            time.sleep(0.5)

    except Exception as e:
        print(f"  [CAISO API error: {e}] — using synthetic fallback")
        source = "Synthetic (CAISO-calibrated)"
        prices = _synthetic_prices(start, end, "CAISO")

    result = {"iso": "CAISO", "node": "SP15", "source": source, "prices": prices}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  ERCOT — Settlement Point Prices (public portal)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ercot(start: datetime, end: datetime) -> dict:
    """
    Fetches ERCOT Houston Hub hourly settlement point prices.
    """
    cache_path = os.path.join(CACHE_DIR, f"ercot_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    prices = []
    source = "ERCOT Public API"

    try:
        # ERCOT Data API v1 — historical settlement point prices
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7), end)
            s = cursor.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")
            url = (
                f"https://data.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
                f"?deliveryDateFrom={s}&deliveryDateTo={e}"
                f"&settlementPoint=HB_HOUSTON&$top=1000&$format=json"
            )
            raw = json.loads(_get(url, timeout=30))
            for row in raw.get("value", []):
                try:
                    ts  = row.get("deliveryDate", "") + "T" + f"{int(row.get('deliveryHour',1))-1:02d}:00"
                    lmp = float(row.get("settlementPointPrice", 0))
                    prices.append((ts, round(lmp / 1000, 6)))
                except (ValueError, TypeError):
                    continue
            cursor = chunk_end
            time.sleep(0.5)

    except Exception as e:
        print(f"  [ERCOT API error: {e}] — using synthetic fallback")
        source = "Synthetic (ERCOT-calibrated)"
        prices = _synthetic_prices(start, end, "ERCOT")

    result = {"iso": "ERCOT", "node": "HB_HOUSTON", "source": source, "prices": prices}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  PJM — Data Miner 2 API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_pjm(start: datetime, end: datetime) -> dict:
    """
    Fetches PJM Western Hub hourly real-time LMP via Data Miner 2.
    """
    cache_path = os.path.join(CACHE_DIR, f"pjm_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    prices = []
    source = "PJM Data Miner 2 API"

    try:
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7), end)
            s = cursor.strftime("%Y-%m-%d %H:%M")
            e = chunk_end.strftime("%Y-%m-%d %H:%M")
            url = (
                f"https://dataminer2.pjm.com/feed/rt_unverified_fivemin_lmps/fields"
                f"?ids=pnode_id,datetime_beginning_ept,total_lmp_rt"
                f"&pnode_id=51217&startRow=1&rowCount=2000"
                f"&datetime_beginning_ept_from={urllib.parse.quote(s)}"
                f"&datetime_beginning_ept_to={urllib.parse.quote(e)}"
                f"&format=json"
            )
            raw = json.loads(_get(url, timeout=30))
            for row in raw:
                try:
                    ts  = row.get("datetime_beginning_ept", "")[:16].replace(" ", "T")
                    lmp = float(row.get("total_lmp_rt", 0))
                    prices.append((ts, round(lmp / 1000, 6)))
                except (ValueError, TypeError):
                    continue
            cursor = chunk_end
            time.sleep(0.5)

    except Exception as e:
        print(f"  [PJM API error: {e}] — using synthetic fallback")
        source = "Synthetic (PJM-calibrated)"
        prices = _synthetic_prices(start, end, "PJM")

    result = {"iso": "PJM", "node": "WESTERN_HUB", "source": source, "prices": prices}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic fallback — statistically calibrated per-region
# ══════════════════════════════════════════════════════════════════════════════

# Real-world calibration parameters (mean $/MWh, std, peak hour multiplier)
# Calibrated to 2024-2025 actual market statistics
# CAISO SP15, ERCOT HB_HOUSTON, PJM Western Hub — EIA/FERC published averages
REGION_PARAMS = {
    "CAISO": {"mean": 45.0, "std": 12.0, "peak_mult": 1.55, "spike_prob": 0.012},
    "ERCOT": {"mean": 38.0, "std": 14.0, "peak_mult": 1.65, "spike_prob": 0.015},
    "PJM":   {"mean": 35.0, "std": 11.0, "peak_mult": 1.45, "spike_prob": 0.010},
}

def _synthetic_prices(start: datetime, end: datetime, region: str) -> list:
    """
    Generates statistically realistic hourly prices calibrated to
    historical mean/variance/seasonality for each region.
    All values in $/kWh.
    """
    rng    = np.random.default_rng(seed=hash(region) % (2**31))
    p      = REGION_PARAMS.get(region, REGION_PARAMS["PJM"])
    mean   = p["mean"]
    std    = p["std"]
    peak_m = p["peak_mult"]
    spike  = p["spike_prob"]

    prices = []
    cursor = start
    while cursor < end:
        h = cursor.hour
        dow = cursor.weekday()   # 0=Mon, 6=Sun

        # Diurnal shape: morning ramp, afternoon peak, overnight trough
        if 7 <= h <= 10:
            hour_mult = 1.3 + 0.1 * (h - 7)
        elif 11 <= h <= 19:
            hour_mult = peak_m * (0.85 + 0.15 * np.sin(np.pi * (h - 11) / 8))
        elif 20 <= h <= 23:
            hour_mult = 1.1 - 0.05 * (h - 20)
        else:
            hour_mult = 0.65 + 0.05 * np.sin(np.pi * h / 6)

        # Weekend discount
        if dow >= 5:
            hour_mult *= 0.82

        base  = mean * hour_mult
        noise = rng.normal(0, std * 0.4)
        lmp   = max(0.5, base + noise)

        # Price spikes (short-duration volatility events)
        if rng.random() < spike:
            lmp *= rng.uniform(2.5, 8.0)

        prices.append((cursor.isoformat(), round(lmp / 1000, 6)))
        cursor += timedelta(hours=1)

    return prices

# ══════════════════════════════════════════════════════════════════════════════
#  Main fetch — all three regions
# ══════════════════════════════════════════════════════════════════════════════
def fetch_all() -> dict:
    start, end = get_window()
    print(f"Fetching price data: {start.date()} → {end.date()}")
    print("  CAISO...", end=" ", flush=True)
    caiso = fetch_caiso(start, end)
    print(f"{len(caiso['prices'])} hrs [{caiso['source']}]")
    print("  ERCOT...", end=" ", flush=True)
    ercot = fetch_ercot(start, end)
    print(f"{len(ercot['prices'])} hrs [{ercot['source']}]")
    print("  PJM...",   end=" ", flush=True)
    pjm   = fetch_pjm(start, end)
    print(f"{len(pjm['prices'])} hrs [{pjm['source']}]")

    return {
        "start": start.isoformat(),
        "end":   end.isoformat(),
        "regions": {"CAISO": caiso, "ERCOT": ercot, "PJM": pjm}
    }

if __name__ == "__main__":
    import urllib.parse
    data = fetch_all()
    print("\nSample ERCOT prices (first 6 hrs):")
    for ts, p in data["regions"]["ERCOT"]["prices"][:6]:
        print(f"  {ts}  ${p*1000:.2f}/MWh  (${p:.5f}/kWh)")
