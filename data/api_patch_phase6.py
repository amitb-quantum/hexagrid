"""
Energia Phase 6 — API patch for grid_connector integration.

Changes to apply to api.py:

1. Import get_connector at top
2. Initialize connector in lifespan
3. Replace /pricefeed endpoint with live data version
4. Add /grid endpoint for full connector status + fuel mix

These are the EXACT replacements to make in api.py.
"""

# ═══════════════════════════════════════════════════
# PATCH 1: Add to imports (after existing imports)
# ═══════════════════════════════════════════════════

PATCH_IMPORT = """
# Phase 6: Live grid connector
try:
    from data.grid_connector import get_connector as _get_grid_connector
    _GRID_CONNECTOR_AVAILABLE = True
except ImportError:
    _GRID_CONNECTOR_AVAILABLE = False
    _get_grid_connector = None
"""

# ═══════════════════════════════════════════════════
# PATCH 2: Add to lifespan (after _load_engines())
# ═══════════════════════════════════════════════════

PATCH_LIFESPAN = """
    # Phase 6: Start live grid connector
    if _GRID_CONNECTOR_AVAILABLE:
        iso  = os.environ.get('ENERGIA_ISO',  'caiso')
        node = os.environ.get('ENERGIA_NODE', None)
        gc   = _get_grid_connector(iso=iso, node=node, verbose=True)
        print(f"  ✓  Grid connector: {iso.upper()} / {gc.node}")
    else:
        print("  ⚠  Grid connector unavailable — using synthetic model")
"""

# ═══════════════════════════════════════════════════
# PATCH 3: Replace /pricefeed endpoint
# ═══════════════════════════════════════════════════

PATCH_PRICEFEED = '''
@app.get("/api/v1/pricefeed", tags=["Grid"])
async def price_feed(
    horizon_min: int = Query(120, ge=1, le=1440,
                             description="How many minutes ahead to forecast"),
    start_tick:  int = Query(0,   ge=0, le=1440),
    iso:         Optional[str] = Query(None,
                             description="Override ISO (caiso/pjm/ercot/isone/nyiso)"),
):
    """
    Live grid price + forward curve for the next N minutes.
    Phase 6: Uses real ISO market data via gridstatus when available.
    Falls back gracefully to synthetic CAISO TOU model.
    """

    if _GRID_CONNECTOR_AVAILABLE:
        try:
            gc = _get_grid_connector()
            curve_data = gc.price_curve(minutes=horizon_min)

            current = curve_data[0]["price_usd_kwh"]
            prices  = [c["price_usd_kwh"] for c in curve_data]
            min_p   = min(prices)
            max_p   = max(prices)
            cheapest_idx = prices.index(min_p)
            sources  = list(set(c["source"] for c in curve_data))
            live_pct = sum(1 for c in curve_data if c["source"] != "synthetic") / len(curve_data) * 100

            return {
                "current_price_usd_kwh": current,
                "current_price_usd_mwh": round(current * 1000, 2),
                "horizon_minutes":       horizon_min,
                "min_price":             min_p,
                "max_price":             max_p,
                "cheapest_slot": {
                    "minute_offset":  cheapest_idx,
                    "price_usd_kwh":  min_p,
                    "saving_pct":     round((current - min_p) / current * 100, 2)
                                      if current > min_p else 0.0,
                },
                "price_curve":     curve_data,
                "data_sources":    sources,
                "live_data_pct":   round(live_pct, 1),
                "iso":             gc.iso,
                "node":            gc.node,
                "timestamp":       datetime.utcnow().isoformat(),
            }
        except Exception as e:
            pass  # Fall through to synthetic

    # Synthetic fallback
    from simulation.digital_twin import grid_price_usd_kwh
    now_tick = int(time.time() // 60) % 1440
    prices = [
        {
            "minute_offset": i,
            "price_usd_kwh": round(grid_price_usd_kwh(now_tick + start_tick + i), 5),
            "source":        "synthetic",
        }
        for i in range(horizon_min)
    ]
    current = prices[0]["price_usd_kwh"]
    min_p   = min(p["price_usd_kwh"] for p in prices)
    cheapest = min(prices, key=lambda x: x["price_usd_kwh"])
    return {
        "current_price_usd_kwh": current,
        "current_price_usd_mwh": round(current * 1000, 2),
        "horizon_minutes":       horizon_min,
        "min_price":             min_p,
        "max_price":             max(p["price_usd_kwh"] for p in prices),
        "cheapest_slot":         cheapest,
        "price_curve":           prices,
        "data_sources":          ["synthetic"],
        "live_data_pct":         0.0,
        "iso":                   "caiso-synthetic",
        "node":                  "TOU-model",
        "timestamp":             datetime.utcnow().isoformat(),
    }
'''

# ═══════════════════════════════════════════════════
# PATCH 4: Add new /grid endpoint
# ═══════════════════════════════════════════════════

PATCH_GRID_ENDPOINT = '''
@app.get("/api/v1/grid", tags=["Grid"])
async def grid_status():
    """
    Full live grid connector status: current price, fuel mix,
    carbon intensity, connector health, and data source breakdown.
    """
    if not _GRID_CONNECTOR_AVAILABLE:
        raise HTTPException(503, "Grid connector not available")

    gc    = _get_grid_connector()
    stats = gc.stats()
    fm    = gc.fuel_mix()

    return {
        "connector":  stats,
        "fuel_mix":   fm,
        "timestamp":  datetime.utcnow().isoformat(),
    }
'''
