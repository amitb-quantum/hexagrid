"""
Energia - Phase 4: FastAPI Platform Layer
==========================================
Wraps all three Energia engines into a REST API:

  POST /api/v1/simulate        → Digital Twin simulation
  POST /api/v1/forecast        → LSTM power demand forecast
  POST /api/v1/schedule        → QAOA / Greedy job scheduler
  GET  /api/v1/health          → System health + GPU status
  GET  /api/v1/pricefeed       → Live grid price (current + 120min ahead)
  GET  /api/v1/report/{run_id} → Fetch a saved simulation report

All heavy computation runs in background threads (non-blocking).
Results are cached by run_id for retrieval.

Usage:
    python api.py                          # start on 0.0.0.0:8000
    python api.py --port 8080 --reload     # dev mode with auto-reload
    curl http://localhost:8000/docs        # Swagger UI (auto-generated)
    curl http://localhost:8000/api/v1/health
"""

import os, sys, uuid, time, asyncio, threading, warnings, subprocess
from datetime import datetime
from typing import Optional, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── Lazy imports for heavy engines (loaded once at startup) ────────────────────
_twin_module        = None
_forecaster_module  = None
_scheduler_module   = None

def _load_engines():
    global _twin_module, _forecaster_module, _scheduler_module
    from simulation.digital_twin import DataCenterDigitalTwin, grid_price_usd_kwh
    from intelligence.forecaster  import run_forecast_pipeline, engineer_features
    from optimization.scheduler   import (
        run_scheduler_pipeline, greedy_schedule,
        QAOAScheduler, default_jobs, GPUJob, get_price_forecast
    )
    _twin_module       = sys.modules['simulation.digital_twin']
    _forecaster_module = sys.modules['intelligence.forecaster']
    _scheduler_module  = sys.modules['optimization.scheduler']
    print("  ✓  All Energia engines loaded")


# ══════════════════════════════════════════════════════════════════════════════
#  APP LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load engines on startup, cleanup on shutdown."""
    print("\n  ⚡ Energia API starting up...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_engines)
    yield
    print("  Energia API shutting down.")

app = FastAPI(
    title       = "Energia API",
    description = (
        "AI Data Center Energy Intelligence Platform\n\n"
        "Provides three core services:\n"
        "- **Digital Twin** simulation of full power chain (Grid→GPU)\n"
        "- **LSTM Forecasting** of power demand at 30/60/120min horizons\n"
        "- **QAOA Scheduler** for energy-cost-optimal GPU job scheduling\n\n"
        "Built with SimPy, TensorFlow, Cirq/TFQ, and FastAPI."
    ),
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Dashboard: serve index.html at GET / ──────────────────────────────────────
_DASHBOARD = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'index.html')

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the Energia dashboard."""
    if not os.path.exists(_DASHBOARD):
        return HTMLResponse(
            content="<h2>Dashboard not found</h2>"
                    "<p>Place <code>index.html</code> in <code>energia/dashboard/</code></p>"
                    "<p><a href='/docs'>API docs →</a></p>",
            status_code=404,
        )
    return HTMLResponse(content=open(_DASHBOARD).read())

# ── In-memory job store (replace with Redis in production) ────────────────────
_jobs: dict[str, dict] = {}       # run_id -> {status, result, error, created_at}
_jobs_lock = threading.Lock()


def _new_job(run_id: str):
    with _jobs_lock:
        _jobs[run_id] = {
            "run_id":     run_id,
            "status":     "queued",
            "result":     None,
            "error":      None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

def _set_running(run_id: str):
    with _jobs_lock:
        _jobs[run_id]["status"] = "running"
        _jobs[run_id]["started_at"] = datetime.utcnow().isoformat()

def _set_done(run_id: str, result: Any):
    with _jobs_lock:
        _jobs[run_id]["status"]       = "completed"
        _jobs[run_id]["result"]       = result
        _jobs[run_id]["completed_at"] = datetime.utcnow().isoformat()

def _set_error(run_id: str, error: str):
    with _jobs_lock:
        _jobs[run_id]["status"]       = "failed"
        _jobs[run_id]["error"]        = error
        _jobs[run_id]["completed_at"] = datetime.utcnow().isoformat()

def _get_job(run_id: str) -> dict:
    with _jobs_lock:
        return _jobs.get(run_id)


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SimulateRequest(BaseModel):
    num_racks:          int   = Field(4,          ge=1, le=128,
                                      description="Number of GPU racks (1-128)")
    duration_minutes:   int   = Field(480,        ge=60, le=10080,
                                      description="Sim duration in minutes (60-10080)")
    efficiency_profile: str   = Field("standard", pattern="^(standard|heron)$",
                                      description="'standard' or 'heron'")
    facility_name:      str   = Field("Energia-DC-01",
                                      description="Facility identifier")
    compare_profiles:   bool  = Field(False,
                                      description="Run both profiles and return delta")

    model_config = {"json_schema_extra": {"example": {
        "num_racks": 4, "duration_minutes": 480,
        "efficiency_profile": "standard", "compare_profiles": True
    }}}


class ForecastRequest(BaseModel):
    num_racks:  int = Field(4,  ge=1, le=32)
    sim_hours:  int = Field(72, ge=12, le=336,
                            description="Hours of training data to generate (12-336)")
    profile:    str = Field("standard", pattern="^(standard|heron)$")

    model_config = {"json_schema_extra": {"example": {
        "num_racks": 4, "sim_hours": 72, "profile": "standard"
    }}}


class JobSpec(BaseModel):
    job_id:       int   = Field(..., description="Unique job ID")
    name:         str   = Field(..., description="Job name")
    duration_min: int   = Field(..., ge=1, le=480)
    power_kw:     float = Field(..., ge=0.1, le=100.0)
    priority:     int   = Field(1, ge=1, le=2)
    deadline_min: Optional[int] = Field(None, description="Latest start slot (None=flexible)")


class ScheduleRequest(BaseModel):
    jobs:           Optional[list[JobSpec]] = Field(
        None, description="Jobs to schedule. If null, uses default AI workload mix.")
    n_slots:        int   = Field(120, ge=30,  le=1440,
                                  description="Scheduling window in minutes")
    p_layers:       int   = Field(2,   ge=1,   le=5,
                                  description="QAOA circuit depth")
    n_candidates:   int   = Field(3,   ge=2,   le=4,
                                  description="Candidate slots per job (keep <=4 for sim)")
    n_restarts:     int   = Field(3,   ge=1,   le=10)
    start_tick:     int   = Field(480, ge=0,   le=1440)
    classical_only: bool  = Field(False, description="Greedy only, skip QAOA")

    model_config = {"json_schema_extra": {"example": {
        "n_slots": 120, "p_layers": 2, "n_candidates": 3,
        "n_restarts": 3, "classical_only": False
    }}}


class JobStatusResponse(BaseModel):
    run_id:       str
    status:       str
    created_at:   str
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None
    result:       Optional[Any] = None
    error:        Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASK RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_simulation(run_id: str, req: SimulateRequest):
    """Background: run Digital Twin simulation."""
    try:
        _set_running(run_id)
        from simulation.digital_twin import (
            DataCenterDigitalTwin, DEFAULT_EFFICIENCY, HERON_EFFICIENCY
        )
        import numpy as np

        def _run_profile(profile):
            twin = DataCenterDigitalTwin(
                num_racks         = req.num_racks,
                duration_minutes  = req.duration_minutes,
                efficiency_profile= profile,
                facility_name     = req.facility_name,
                seed              = 42,
            )
            df = twin.run()
            total_h    = req.duration_minutes / 60.0
            scale      = (365 * 24) / total_h
            total_kwh  = float((df['grid_demand_kw'].sum() * 1) / 60.0)
            it_kwh     = float((df['gpu_total_kw'].sum()   * 1) / 60.0)
            loss_kwh   = float((df['total_loss_kw'].sum()  * 1) / 60.0)
            total_cost = float(df['cost_per_min_usd'].sum())
            avg_pue    = float(df['pue'].mean())
            peak_kw    = float(df['grid_demand_kw'].max())
            avg_price  = float(df['grid_price_usd_kwh'].mean())

            chain_eff = 1.0
            eff = HERON_EFFICIENCY if profile == "heron" else DEFAULT_EFFICIENCY
            for v in eff.values():
                chain_eff *= v

            return {
                "profile":           profile,
                "grid_kwh":          round(total_kwh,  3),
                "it_kwh":            round(it_kwh,     3),
                "loss_kwh":          round(loss_kwh,   3),
                "loss_pct":          round((loss_kwh / total_kwh) * 100, 2),
                "avg_pue":           round(avg_pue,    4),
                "peak_demand_kw":    round(peak_kw,    2),
                "avg_price_usd_kwh": round(avg_price,  5),
                "period_cost_usd":   round(total_cost, 4),
                "annual_cost_usd":   round(total_cost * scale, 0),
                "annual_kwh":        round(total_kwh   * scale, 0),
                "chain_efficiency":  round(chain_eff,  5),
                "nodes": {
                    name: {
                        "efficiency": node.efficiency,
                        "loss_pct":   round(node.loss_pct, 3)
                    }
                    for name, node in twin.nodes.items()
                },
                "telemetry_sample": df.tail(5).to_dict(orient='records'),
            }

        if req.compare_profiles:
            std   = _run_profile("standard")
            heron = _run_profile("heron")
            kwh_saved  = std["grid_kwh"] - heron["grid_kwh"]
            cost_saved = std["period_cost_usd"] - heron["period_cost_usd"]
            scale      = (365 * 24) / (req.duration_minutes / 60.0)
            result = {
                "standard": std,
                "heron":    heron,
                "delta": {
                    "kwh_saved_period":  round(kwh_saved,  3),
                    "kwh_saved_annual":  round(kwh_saved * scale, 0),
                    "cost_saved_period": round(cost_saved, 4),
                    "cost_saved_annual": round(cost_saved * scale, 0),
                    "pue_improvement":   round(std["avg_pue"] - heron["avg_pue"], 4),
                    "loss_reduction_pct":round(
                        (std["loss_kwh"] - heron["loss_kwh"]) / std["loss_kwh"] * 100, 1
                    ),
                }
            }
        else:
            result = _run_profile(req.efficiency_profile)

        _set_done(run_id, result)

    except Exception as e:
        import traceback
        _set_error(run_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _run_forecast(run_id: str, req: ForecastRequest):
    """Background: train LSTM forecaster."""
    try:
        _set_running(run_id)
        from intelligence.forecaster import run_forecast_pipeline

        model, eval_results, feat_scaler, tgt_scaler = run_forecast_pipeline(
            num_racks  = req.num_racks,
            sim_hours  = req.sim_hours,
            profile    = req.profile,
            load_model = False,
        )

        result = {
            "model_path": os.path.join(
                os.path.dirname(__file__), '..', 'models', 'energia_lstm_best.h5'
            ),
            "horizons": {
                f"+{h}min": {
                    "mae":  round(eval_results[h]['mae'],  4),
                    "rmse": round(eval_results[h]['rmse'], 4),
                    "mape": round(eval_results[h]['mape'], 4),
                }
                for h in eval_results
            },
            "avg_mape": round(
                sum(v['mape'] for v in eval_results.values()) / len(eval_results), 4
            ),
            "training_data": {
                "sim_hours":  req.sim_hours,
                "num_racks":  req.num_racks,
                "profile":    req.profile,
            }
        }
        _set_done(run_id, result)

    except Exception as e:
        import traceback
        _set_error(run_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _run_schedule(run_id: str, req: ScheduleRequest):
    """Background: run QAOA/Greedy scheduler."""
    try:
        _set_running(run_id)
        from optimization.scheduler import (
            greedy_schedule, QAOAScheduler,
            default_jobs, GPUJob, get_price_forecast, _build_result, _job_cost
        )
        import numpy as np

        # Build job list
        if req.jobs:
            jobs = [
                GPUJob(
                    job_id       = j.job_id,
                    name         = j.name,
                    duration_min = j.duration_min,
                    power_kw     = j.power_kw,
                    priority     = j.priority,
                    deadline_min = j.deadline_min,
                )
                for j in req.jobs
            ]
        else:
            jobs = default_jobs()

        prices = get_price_forecast(req.n_slots, start_tick=req.start_tick)

        results = {}

        # Greedy
        greedy = greedy_schedule(jobs, prices, req.n_slots)
        results["greedy"] = {
            "assignments": {
                job.job_id: {
                    "job_name":  job.name,
                    "start_min": greedy.job_assignments[job.job_id],
                    "price":     float(prices[greedy.job_assignments[job.job_id]]),
                    "cost_usd":  round(_job_cost(job, greedy.job_assignments[job.job_id], prices), 5),
                }
                for job in jobs
            },
            "total_cost_usd":   round(greedy.total_cost_usd,   5),
            "total_energy_kwh": round(greedy.total_energy_kwh, 4),
            "solve_time_s":     round(greedy.solve_time_s,     4),
        }

        # QAOA
        if not req.classical_only:
            scheduler = QAOAScheduler(
                jobs         = jobs,
                prices       = prices,
                n_slots      = req.n_slots,
                p_layers     = req.p_layers,
                n_candidates = req.n_candidates,
                n_restarts   = req.n_restarts,
            )
            qaoa = scheduler.solve()
            results["qaoa"] = {
                "assignments": {
                    job.job_id: {
                        "job_name":  job.name,
                        "start_min": qaoa.job_assignments[job.job_id],
                        "price":     float(prices[qaoa.job_assignments[job.job_id]]),
                        "cost_usd":  round(_job_cost(job, qaoa.job_assignments[job.job_id], prices), 5),
                    }
                    for job in jobs
                },
                "total_cost_usd":   round(qaoa.total_cost_usd,   5),
                "total_energy_kwh": round(qaoa.total_energy_kwh, 4),
                "solve_time_s":     round(qaoa.solve_time_s,     3),
                "quantum_state":    qaoa.quantum_state,
            }

            # Delta
            saving = greedy.total_cost_usd - qaoa.total_cost_usd
            scale  = (365 * 24 * 60) / req.n_slots
            results["delta"] = {
                "cost_saved_period":   round(saving,         5),
                "cost_saved_pct":      round(saving / greedy.total_cost_usd * 100, 3),
                "cost_saved_annual":   round(saving * scale, 0),
                "n_qubits":            qaoa.quantum_state['n_qubits'],
                "qaoa_p_layers":       req.p_layers,
            }

        _set_done(run_id, results)

    except Exception as e:
        import traceback
        _set_error(run_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"])
async def health():
    """System health check — GPU status, engine readiness, job queue depth."""
    gpu_info = _get_gpu_info()
    with _jobs_lock:
        queue_depth = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
        total_jobs  = len(_jobs)

    return {
        "status":   "healthy",
        "version":  "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu":       gpu_info,
        "engines": {
            "digital_twin": _twin_module is not None,
            "forecaster":   _forecaster_module is not None,
            "scheduler":    _scheduler_module is not None,
        },
        "job_queue": {
            "active":  queue_depth,
            "total":   total_jobs,
        }
    }


def _get_gpu_info() -> dict:
    """Probe GPU via subprocess to avoid CUDA context collision."""
    try:
        out = subprocess.run(
            [sys.executable, '-c',
             'import warnings; warnings.filterwarnings("ignore"); '
             'import torch; '
             'cuda=torch.cuda.is_available(); '
             'name=torch.cuda.get_device_name(0) if cuda else "N/A"; '
             'mem=round(torch.cuda.get_device_properties(0).total_memory/1e9,2) if cuda else 0; '
             'print(f"{cuda}|{name}|{mem}")'],
            capture_output=True, text=True, timeout=10
        )
        parts = out.stdout.strip().split('|')
        if len(parts) == 3:
            return {
                "available":  parts[0] == 'True',
                "device":     parts[1],
                "memory_gb":  float(parts[2]),
            }
    except Exception:
        pass
    return {"available": False, "device": "unknown", "memory_gb": 0}


# ── Price Feed ────────────────────────────────────────────────────────────────
@app.get("/api/v1/pricefeed", tags=["Grid"])
async def price_feed(
    horizon_min: int = Query(120, ge=1, le=1440,
                             description="How many minutes ahead to forecast"),
    start_tick:  int = Query(0,   ge=0, le=1440),
):
    """
    Current grid price + forward price curve for the next N minutes.
    Based on CAISO TOU model (production: swap for live grid API).
    """
    from simulation.digital_twin import grid_price_usd_kwh
    import math

    now_tick = int(time.time() // 60) % 1440   # current minute of day
    prices   = [
        {
            "minute_offset": i,
            "price_usd_kwh": round(grid_price_usd_kwh(now_tick + start_tick + i), 5),
        }
        for i in range(horizon_min)
    ]

    current_price = prices[0]["price_usd_kwh"]
    min_price     = min(p["price_usd_kwh"] for p in prices)
    max_price     = max(p["price_usd_kwh"] for p in prices)
    cheapest_slot = min(prices, key=lambda x: x["price_usd_kwh"])

    return {
        "current_price_usd_kwh": current_price,
        "horizon_minutes":       horizon_min,
        "min_price":             min_price,
        "max_price":             max_price,
        "cheapest_slot":         cheapest_slot,
        "price_curve":           prices,
        "source":                "CAISO-TOU-synthetic",
        "timestamp":             datetime.utcnow().isoformat(),
    }


# ── Simulate ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/simulate", tags=["Digital Twin"], status_code=202)
async def simulate(req: SimulateRequest, background_tasks: BackgroundTasks):
    """
    Launch an async Digital Twin simulation.
    Returns run_id immediately; poll /api/v1/jobs/{run_id} for results.
    """
    run_id = str(uuid.uuid4())[:8]
    _new_job(run_id)
    background_tasks.add_task(_run_simulation, run_id, req)
    return {
        "run_id":   run_id,
        "status":   "queued",
        "poll_url": f"/api/v1/jobs/{run_id}",
        "message":  "Simulation queued. Poll poll_url for status and results.",
    }


# ── Forecast ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/forecast", tags=["Intelligence"], status_code=202)
async def forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Launch LSTM training + evaluation pipeline.
    WARNING: Takes 30-120s depending on sim_hours. Poll for completion.
    """
    run_id = str(uuid.uuid4())[:8]
    _new_job(run_id)
    background_tasks.add_task(_run_forecast, run_id, req)
    return {
        "run_id":   run_id,
        "status":   "queued",
        "poll_url": f"/api/v1/jobs/{run_id}",
        "message":  f"Forecast training queued ({req.sim_hours}h data). "
                    "LSTM training takes ~30-120s. Poll for results.",
    }


# ── Schedule ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/schedule", tags=["Scheduler"], status_code=202)
async def schedule(req: ScheduleRequest, background_tasks: BackgroundTasks):
    """
    Launch QAOA or Greedy job scheduler.
    QAOA (classical_only=false) takes ~30-120s. Greedy is instant.
    """
    # Validate qubit count won't OOM
    n_jobs = len(req.jobs) if req.jobs else 5
    max_qubits = n_jobs * req.n_candidates
    if max_qubits > 20 and not req.classical_only:
        raise HTTPException(
            status_code=422,
            detail=f"n_jobs({n_jobs}) * n_candidates({req.n_candidates}) = "
                   f"{max_qubits} qubits exceeds safe simulation limit of 20. "
                   "Reduce n_candidates to 3 or enable classical_only=true."
        )

    run_id = str(uuid.uuid4())[:8]
    _new_job(run_id)
    background_tasks.add_task(_run_schedule, run_id, req)
    return {
        "run_id":   run_id,
        "status":   "queued",
        "poll_url": f"/api/v1/jobs/{run_id}",
        "n_qubits": max_qubits if not req.classical_only else 0,
        "message":  f"Scheduler queued. {'QAOA will take ~30-120s.' if not req.classical_only else 'Greedy is instant.'}",
    }


# ── Job Status / Results ──────────────────────────────────────────────────────
@app.get("/api/v1/jobs/{run_id}", tags=["Jobs"], response_model=JobStatusResponse)
async def job_status(run_id: str):
    """Poll job status and retrieve results when complete."""
    job = _get_job(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
    return JobStatusResponse(**job)


@app.get("/api/v1/jobs", tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit:  int           = Query(20,   ge=1, le=100)
):
    """List recent jobs."""
    with _jobs_lock:
        jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"total": len(jobs), "jobs": jobs[:limit]}


# ── Reports ───────────────────────────────────────────────────────────────────
@app.get("/api/v1/reports", tags=["Reports"])
async def list_reports():
    """List all generated dashboard PNG files."""
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    if not os.path.exists(reports_dir):
        return {"reports": []}

    files = sorted(
        [f for f in os.listdir(reports_dir) if f.endswith('.png')],
        reverse=True
    )
    return {
        "count":   len(files),
        "reports": [
            {
                "filename": f,
                "url":      f"/api/v1/reports/{f}",
                "size_kb":  round(os.path.getsize(
                    os.path.join(reports_dir, f)) / 1024, 1),
            }
            for f in files
        ]
    }


@app.get("/api/v1/reports/{filename}", tags=["Reports"])
async def get_report(filename: str):
    """Download a specific dashboard PNG."""
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    path        = os.path.join(reports_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Report {filename} not found")
    return FileResponse(path, media_type='image/png')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7 — RL AGENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
import json   as _json_rl
import threading as _threading_rl

_rl_agent      = None
_rl_agent_lock = _threading_rl.Lock()

def _get_rl_agent():
    """Lazy-init the RL agent singleton (thread-safe)."""
    global _rl_agent
    if _rl_agent is None:
        with _rl_agent_lock:
            if _rl_agent is None:
                try:
                    _rl_sys_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                    sys.path.insert(0, _rl_sys_path)
                    from rl_agent import EnergiaAgent
                    _rl_agent = EnergiaAgent()
                except Exception as _e:
                    print(f"  [RL] Agent init failed: {_e}")
                    _rl_agent = None
    return _rl_agent


# ── Request models ─────────────────────────────────────────────────────────────
class RLTrainRequest(BaseModel):
    total_steps: int   = Field(200_000, ge=2000, le=2_000_000)
    n_envs:      int   = Field(4,       ge=1,    le=16)
    n_racks:     int   = Field(4,       ge=1,    le=64)
    run_name:    str   = Field("run_001")
    resume_path: str   = Field(None)

class RLRecommendRequest(BaseModel):
    current_price:    float = Field(...,  description="Current grid price $/kWh")
    price_forecast:   list  = Field(...,  description="12 price values, 5-min intervals")
    queue_depth:      int   = Field(1)
    job_urgency:      float = Field(0.3)
    gpu_utilization:  float = Field(0.5)
    pue:              float = Field(1.4)
    ups_charge:       float = Field(0.5)
    cooling_power:    float = Field(0.5)
    hour_of_day:      float = Field(12.0)
    day_of_week:      int   = Field(0)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/rl/status", tags=["Phase 7 — RL Agent"])
async def rl_status():
    """Agent readiness, model path, and latest training metrics."""
    agent = _get_rl_agent()
    if agent is None:
        return JSONResponse({"ready": False, "error": "RL module unavailable"})
    return JSONResponse(agent.status())


@app.post("/api/v1/rl/train", tags=["Phase 7 — RL Agent"])
async def rl_train(req: RLTrainRequest, background_tasks: BackgroundTasks):
    """
    Launch a background PPO training job.
    Returns run_id immediately — poll GET /api/v1/jobs/{run_id} for status.
    """
    run_id = f"rl_train_{req.run_name}_{uuid.uuid4().hex[:6]}"
    _new_job(run_id)

    def _do_train():
        try:
            _set_running(run_id)
            _rl_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
            sys.path.insert(0, _rl_path)
            import importlib
            import rl_train as rl_train_mod
            importlib.reload(rl_train_mod)
            model = rl_train_mod.train(
                total_steps = req.total_steps,
                n_envs      = req.n_envs,
                n_racks     = req.n_racks,
                run_name    = req.run_name,
                resume_path = req.resume_path,
            )
            # Hot-reload the agent after training
            agent = _get_rl_agent()
            if agent:
                agent.reload()
            _set_done(run_id, {
                'run_name':   req.run_name,
                'steps':      int(model.num_timesteps),
                'model_path': 'models/rl/best_model.zip',
            })
        except Exception as _e:
            _set_error(run_id, str(_e))

    background_tasks.add_task(_do_train)
    return JSONResponse({"run_id": run_id, "status": "queued", "run_name": req.run_name})


@app.get("/api/v1/rl/training_log", tags=["Phase 7 — RL Agent"])
async def rl_training_log():
    """Live training metrics — polled by the dashboard every 5 seconds during training."""
    import glob
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'logs', 'rl', 'training_log.json'),
    ] + sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), '..', 'logs', 'rl', '*', 'training_log.json')),
        key=os.path.getmtime, reverse=True
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return JSONResponse(_json_rl.load(f))
            except Exception:
                pass
    return JSONResponse({"status": "no_training_data", "episodes": 0, "reward_history": []})


@app.post("/api/v1/rl/recommend", tags=["Phase 7 — RL Agent"])
async def rl_recommend(req: RLRecommendRequest):
    """
    Get a real-time action recommendation from the trained RL agent.
    Provide current price, 12-step forecast, and system state.
    """
    agent = _get_rl_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="RL agent not available")
    if not agent.is_ready:
        raise HTTPException(status_code=503,
            detail="No trained model found. Run: POST /api/v1/rl/train first.")
    return JSONResponse(agent.recommend(req.dict()))


@app.get("/api/v1/rl/recommend/live", tags=["Phase 7 — RL Agent"])
async def rl_recommend_live():
    """
    Convenience: pulls current price feed automatically and returns agent recommendation.
    No request body needed — uses live grid data from the existing connector.
    """
    agent = _get_rl_agent()
    if agent is None or not agent.is_ready:
        return JSONResponse({"status": "not_ready", "dispatch": None})

    # Pull from existing price feed
    current_price, price_forecast = 0.05, [0.05] * 12
    try:
        _gc_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        sys.path.insert(0, _gc_path)
        from grid_connector import GridConnector
        gc = GridConnector()
        pf = gc.get_price_feed(horizon_min=60)
        current_price  = pf.get('current_price_usd_kwh', 0.05)
        price_forecast = [p['price_usd_kwh'] for p in pf.get('price_curve', [])][:12]
    except Exception:
        pass

    import datetime
    now = datetime.datetime.now()
    return JSONResponse(agent.recommend({
        'current_price':   current_price,
        'price_forecast':  price_forecast,
        'queue_depth':     1,
        'job_urgency':     0.3,
        'gpu_utilization': 0.6,
        'pue':             1.35,
        'ups_charge':      0.6,
        'cooling_power':   0.5,
        'hour_of_day':     float(now.hour) + float(now.minute) / 60.0,
        'day_of_week':     now.weekday(),
    }))

# ── end Phase 7 block ──────────────────────────────────────────────────────────



# ════════════════════════════════════════════════════════════════════════════
#  PHASE 8 — CARBON INTENSITY ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading8

_carbon_connector = None
_carbon_lock      = _threading8.Lock()

def _get_carbon() -> 'CarbonConnector':
    global _carbon_connector
    with _carbon_lock:
        if _carbon_connector is None:
            try:
                _carbon_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                sys.path.insert(0, _carbon_path)
                from carbon_connector import CarbonConnector
                _carbon_connector = CarbonConnector()
            except Exception as _ce:
                print(f"  [Carbon] Init failed: {_ce}")
    return _carbon_connector


@app.get("/api/v1/carbon/snapshot", tags=["Phase 8 — Carbon"])
async def carbon_snapshot():
    """Live carbon intensity (gCO2eq/kWh) for all 5 monitored ISO zones."""
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    try:
        return cc.get_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/carbon/history/{iso}", tags=["Phase 8 — Carbon"])
async def carbon_history(iso: str):
    """24-hour hourly carbon intensity history for one ISO zone.
    ISO codes: CAISO, ERCOT, NYISO, ISONE, PJM
    """
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    history = cc.get_history(iso.upper())
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for zone: {iso}")
    return {"iso": iso.upper(), "history": history}


@app.get("/api/v1/carbon/pareto", tags=["Phase 8 — Carbon"])
async def carbon_pareto():
    """
    Pareto analysis: cost vs carbon tradeoff across all zones.
    Returns scatter plot data and recommendations for three strategies:
      cost_only   — minimize $/kWh regardless of carbon
      balanced    — equal weight to cost and carbon
      carbon_only — minimize gCO2/kWh regardless of cost
    """
    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")
    try:
        return cc.get_pareto_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/carbon/recommend", tags=["Phase 8 — Carbon"])
async def carbon_recommend(cost_weight: float = 0.5):
    """
    Carbon-aware workload dispatch recommendation.
    cost_weight: 0.0 = pure carbon minimization, 1.0 = pure cost minimization
    Default 0.5 = balanced.
    """
    if not 0.0 <= cost_weight <= 1.0:
        raise HTTPException(status_code=422, detail="cost_weight must be between 0.0 and 1.0")

    cc = _get_carbon()
    if not cc:
        raise HTTPException(status_code=503, detail="Carbon connector unavailable")

    try:
        snap   = cc.get_snapshot()
        pareto = cc.get_pareto_data()
        zones  = snap.get("zones", {})
        points = pareto.get("points", [])

        if not points:
            return {"status": "no_data", "error": "No zone data available"}

        carbon_w = 1.0 - cost_weight
        best = min(
            points,
            key=lambda p: cost_weight * p["price_norm"] + carbon_w * p["carbon_norm"]
        )

        # Build reasoning text
        strategy = (
            "cost minimization" if cost_weight >= 0.9
            else "carbon minimization" if cost_weight <= 0.1
            else f"balanced ({int(cost_weight*100)}% cost / {int(carbon_w*100)}% carbon)"
        )
        reasoning = (
            f"Using {strategy} strategy. "
            f"{best['label']} offers ${best['price_usd_kwh']:.4f}/kWh at "
            f"{best['carbon_intensity']:.0f} gCO2eq/kWh ({best['carbon_label']})."
        )
        if best.get("on_pareto_frontier"):
            reasoning += " This zone is on the Pareto frontier — optimal for this weighting."

        return {
            "status":            "ok",
            "recommended_iso":   best["iso"],
            "recommended_label": best["label"],
            "price_usd_kwh":     best["price_usd_kwh"],
            "carbon_intensity":  best["carbon_intensity"],
            "carbon_label":      best["carbon_label"],
            "carbon_color":      best["carbon_color"],
            "fossil_free_pct":   best.get("fossil_free_pct"),
            "on_pareto_frontier":best.get("on_pareto_frontier", False),
            "cost_weight":       cost_weight,
            "carbon_weight":     carbon_w,
            "reasoning":         reasoning,
            "all_zones":         zones,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ════ END PHASE 8 ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 9 — MULTI-SITE FLEET ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading9
from pydantic import BaseModel as _BaseModel9
from typing import Optional as _Optional9

_site_orch       = None
_site_orch_lock  = _threading9.Lock()

def _get_orch():
    global _site_orch
    with _site_orch_lock:
        if _site_orch is None:
            try:
                _orch_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                sys.path.insert(0, _orch_path)
                from site_orchestrator import SiteOrchestrator
                _site_orch = SiteOrchestrator()
                print("  [Fleet] Site orchestrator ready")
            except Exception as _e:
                print(f"  [Fleet] Init failed: {_e}")
    return _site_orch


class _RouteRequest(_BaseModel9):
    job_type:     str   = "custom"
    duration_min: int   = 60
    gpu_count:    int   = 4
    priority:     str   = "balanced"   # balanced | cost | carbon | capacity

class _SiteUpdate(_BaseModel9):
    capacity_pct: _Optional9[float] = None
    online:       _Optional9[bool]  = None
    pue:          _Optional9[float] = None
    num_racks:    _Optional9[int]   = None


@app.get("/api/v1/fleet/summary", tags=["Phase 9 — Fleet"])
async def fleet_summary():
    """Live fleet overview — all sites with price, carbon, capacity."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    try:
        return orch.fleet_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fleet/sites", tags=["Phase 9 — Fleet"])
async def fleet_sites():
    """Return all site configurations."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    return {"sites": orch.get_sites()}


@app.post("/api/v1/fleet/sites/{site_id}", tags=["Phase 9 — Fleet"])
async def update_site(site_id: str, body: _SiteUpdate):
    """Update a site's capacity, online status, PUE, or rack count."""
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")
    ok = orch.update_site(site_id, updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Site not found: {site_id}")
    return {"status": "ok", "site_id": site_id, "updated": updates}


@app.post("/api/v1/fleet/route", tags=["Phase 9 — Fleet"])
async def fleet_route(body: _RouteRequest):
    """
    Route a workload across the fleet.
    Returns ranked site list + recommended site + cost/carbon savings vs worst site.

    priority options:
      balanced  — 40% cost, 30% carbon, 20% capacity, 10% PUE
      cost      — 70% cost, 10% carbon, 15% capacity,  5% PUE
      carbon    — 10% cost, 70% carbon, 15% capacity,  5% PUE
      capacity  — 20% cost, 20% carbon, 50% capacity, 10% PUE
    """
    if body.priority not in ("balanced", "cost", "carbon", "capacity"):
        raise HTTPException(status_code=422, detail="priority must be: balanced | cost | carbon | capacity")
    if body.gpu_count < 1 or body.gpu_count > 512:
        raise HTTPException(status_code=422, detail="gpu_count must be 1–512")
    if body.duration_min < 1 or body.duration_min > 10080:
        raise HTTPException(status_code=422, detail="duration_min must be 1–10080 (1 week)")

    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    try:
        return orch.route_workload(
            job_type=body.job_type,
            duration_min=body.duration_min,
            gpu_count=body.gpu_count,
            priority=body.priority,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fleet/history", tags=["Phase 9 — Fleet"])
async def fleet_history(limit: int = 50):
    """Last N workload routing decisions with full scoring breakdown."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be 1–200")
    orch = _get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Fleet orchestrator unavailable")
    return {"history": orch.get_routing_history(limit)}

# ════ END PHASE 9 ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 10 — GPU HEALTH MONITORING
# ════════════════════════════════════════════════════════════════════════════
import threading as _threading10

_gpu_monitor      = None
_gpu_monitor_lock = _threading10.Lock()

def _get_gpu_monitor():
    global _gpu_monitor
    with _gpu_monitor_lock:
        if _gpu_monitor is None:
            try:
                _hw_path = os.path.join(os.path.dirname(__file__), '..', 'rl')
                sys.path.insert(0, _hw_path)
                from gpu_monitor import GPUMonitor
                _gpu_monitor = GPUMonitor(poll_interval_s=10)
                _gpu_monitor.start_background_polling()
                print("  [Hardware] GPU monitor ready")
            except Exception as _e:
                print(f"  [Hardware] Init failed: {_e}")
    return _gpu_monitor


@app.get("/api/v1/hardware/snapshot", tags=["Phase 10 — Hardware"])
async def hardware_snapshot():
    """
    Live GPU telemetry snapshot for all detected GPUs.
    Returns real NVML data when pynvml is installed and GPUs are present;
    falls back to realistic synthetic readings for development.
    """
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    try:
        return mon.get_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/hardware/history/{gpu_idx}", tags=["Phase 10 — Hardware"])
async def hardware_history(gpu_idx: int, minutes: int = 60):
    """
    Rolling telemetry history for one GPU.
    gpu_idx: 0-based GPU index
    minutes: how far back to look (max 1440 = 24h)
    """
    if minutes < 1 or minutes > 1440:
        raise HTTPException(status_code=422, detail="minutes must be 1–1440")
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    history = mon.get_history(gpu_idx=gpu_idx, minutes=minutes)
    return {"gpu_idx": gpu_idx, "minutes": minutes, "history": history}


@app.get("/api/v1/hardware/alerts", tags=["Phase 10 — Hardware"])
async def hardware_alerts(limit: int = 50):
    """Most recent GPU health alerts across all devices."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be 1–500")
    mon = _get_gpu_monitor()
    if not mon:
        raise HTTPException(status_code=503, detail="GPU monitor unavailable")
    return {"alerts": mon.get_alerts(limit=limit)}

# ════ END PHASE 10 ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Energia FastAPI Server')
    parser.add_argument('--host',   default='0.0.0.0')
    parser.add_argument('--port',   type=int, default=8000)
    parser.add_argument('--reload', action='store_true',
                        help='Auto-reload on code changes (dev mode)')
    args = parser.parse_args()

    print(f"""
  ╔══════════════════════════════════════════════╗
  ║   ⚡  ENERGIA API  v0.1.0                    ║
  ║      http://{args.host}:{args.port}           ║
  ║      Swagger UI: /docs                       ║
  ║      ReDoc:      /redoc                      ║
  ╚══════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host      = args.host,
        port      = args.port,
        workers   = 1,
        log_level = "info",
    )
