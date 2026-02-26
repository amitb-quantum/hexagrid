"""
patch_api_benchmark.py — HexaGrid API Patch: Benchmark Endpoints
=================================================================
    GET  /api/v1/benchmark/summary    — latest benchmark results
    POST /api/v1/benchmark/run        — trigger a fresh benchmark run
    GET  /api/v1/benchmark/status     — run status (idle/running/done)

HOW TO APPLY:
    cp benchmark_prices.py benchmark_engine.py patch_api_benchmark.py ~/hexagrid/api/
    echo "
from patch_api_benchmark import router as benchmark_router
app.include_router(benchmark_router)" >> ~/hexagrid/api/api.py
    Restart uvicorn.
"""

import json, os, threading, sys
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException

sys.path.insert(0, os.path.dirname(__file__))

router = APIRouter(prefix="/api/v1/benchmark", tags=["Benchmark"])

RESULTS_PATH = os.path.expanduser("~/hexagrid/benchmark_results.json")
_status = {"state": "idle", "started_at": None, "finished_at": None, "error": None}

def _load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)

def _run_benchmark_bg():
    global _status
    _status = {"state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None, "error": None}
    try:
        from benchmark_prices import fetch_all
        from benchmark_engine import run_benchmark
        price_data = fetch_all()
        results    = run_benchmark(price_data)
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        _status["state"]       = "done"
        _status["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        _status["state"] = "error"
        _status["error"] = str(e)

@router.get("/summary")
def get_benchmark_summary():
    results = _load_results()
    if not results:
        return {"status": "no_data",
                "message": "No benchmark results yet. POST /api/v1/benchmark/run to generate."}
    s = results["summary"]
    regions_out = {}
    for region, rd in results["regions"].items():
        regions_out[region] = {
            "source":              rd["source"],
            "price_mean":          rd["price_mean"],
            "price_min":           rd["price_min"],
            "price_max":           rd["price_max"],
            "hourly_prices":       rd["hourly_prices"],
            "naive_cost":          rd["naive"]["total_cost"],
            "threshold_cost":      rd["threshold"]["total_cost"],
            "hexagrid_cost":       rd["hexagrid"]["total_cost"],
            "threshold_savings_pct": rd["threshold"]["savings_pct"],
            "hexagrid_savings_pct":  rd["hexagrid"]["savings_pct"],
            "hexagrid_savings":      rd["hexagrid"]["savings_vs_naive"],
            "annual_savings":        rd["hexagrid"]["annual_savings"],
            "co2_avoided_t":         rd["hexagrid"]["carbon"]["avoided_co2_t"],
            "jobs_deferred":         rd["hexagrid"].get("jobs_deferred", 0),
        }
    return {
        "status": "ok",
        "generated_at": results.get("end", ""),
        "period_start":  results["start"],
        "period_end":    results["end"],
        "summary": s,
        "regions": regions_out,
    }

@router.post("/run")
def run_benchmark(background_tasks: BackgroundTasks):
    if _status["state"] == "running":
        return {"status": "already_running", "message": "Benchmark is already running."}
    background_tasks.add_task(_run_benchmark_bg)
    return {"status": "started",
            "message": "Benchmark started in background. Check /benchmark/status for progress."}

@router.get("/status")
def get_status():
    return {"status": "ok", "benchmark": _status}
