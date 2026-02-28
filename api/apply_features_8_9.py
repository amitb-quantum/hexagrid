#!/usr/bin/env python3
"""
HexaGrid — Deploy Feature 8 (Chargeback) + Feature 9 (Incident Response)
=========================================================================
Run from ~/hexagrid/api/:  python apply_features_8_9.py

What it does:
  1. Copies chargeback_ledger.py + patch_api_chargeback.py
  2. Copies incident_engine.py + patch_api_incidents.py
  3. Patches JobSpec in api.py to add cost_center / team / project fields
  4. Patches _run_schedule() to auto-record chargeback entries
  5. Wires alert_manager.fire() → incident_engine.handle_alert()
  6. Mounts both new routers in api.py
  7. Adds PagerDuty/OpsGenie env vars to .env.auth template
"""

import os, sys, shutil, textwrap
from pathlib import Path
from datetime import datetime

HEXAGRID_ROOT = Path.home() / "hexagrid"
API_DIR       = HEXAGRID_ROOT / "api"
API_PY        = API_DIR / "api.py"
ALERT_MGR     = API_DIR / "alert_manager.py"
ENV_AUTH      = HEXAGRID_ROOT / ".env.auth"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def step(msg): print(f"\n  ➤  {msg}")
def banner(msg):
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}")

def backup(path):
    bak = Path(str(path) + BACKUP_SUFFIX)
    shutil.copy2(path, bak)
    ok(f"Backed up → {bak.name}")

def patch(path, old, new, label):
    content = path.read_text()
    if new.strip() in content:
        ok(f"Already patched: {label}")
        return True
    if old not in content:
        warn(f"Anchor not found — skipping: {label}")
        return False
    path.write_text(content.replace(old, new, 1))
    ok(f"Patched: {label}")
    return True

def write(path, content):
    path.write_text(content)
    ok(f"Written: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  FILE CONTENTS (embedded)
# ══════════════════════════════════════════════════════════════════════════════

# These are imported from the files in the same directory as this script
def _read_sibling(name):
    p = Path(__file__).parent / name
    if not p.exists():
        print(f"  ✗  Required file not found: {p}")
        print(f"     Place {name} in the same directory as this script.")
        sys.exit(1)
    return p.read_text()


# ══════════════════════════════════════════════════════════════════════════════
#  PATCHES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. JobSpec: add cost_center, team, project ────────────────────────────────
JOBSPEC_OLD = """\
class JobSpec(BaseModel):
    job_id:       int   = Field(..., description="Unique job ID")
    name:         str   = Field(..., description="Job name")
    duration_min: int   = Field(..., ge=1, le=480)
    power_kw:     float = Field(..., ge=0.1, le=100.0)
    priority:     int   = Field(1, ge=1, le=2)
    deadline_min: Optional[int] = Field(None, description="Latest start slot (None=flexible)")"""

JOBSPEC_NEW = """\
class JobSpec(BaseModel):
    job_id:       int   = Field(..., description="Unique job ID")
    name:         str   = Field(..., description="Job name")
    duration_min: int   = Field(..., ge=1, le=480)
    power_kw:     float = Field(..., ge=0.1, le=100.0)
    priority:     int   = Field(1, ge=1, le=2)
    deadline_min: Optional[int] = Field(None, description="Latest start slot (None=flexible)")
    # ── Chargeback tags ───────────────────────────────────────────────────────
    cost_center:  str   = Field("untagged", description="Cost center for chargeback reporting")
    team:         str   = Field("untagged", description="Team owning this job")
    project:      str   = Field("untagged", description="Project this job belongs to")"""

# ── 2. Auto-record chargeback after schedule runs ─────────────────────────────
SCHEDULE_RECORD_OLD = "        _set_done(run_id, results)"

SCHEDULE_RECORD_NEW = """\
        _set_done(run_id, results)

        # ── Auto-record chargeback entries ────────────────────────────────────
        try:
            from chargeback_ledger import ChargebackLedger
            _cb = ChargebackLedger()
            # Use QAOA assignments if available, otherwise greedy
            assignments = results.get("qaoa", results.get("greedy", {})).get("assignments", {})
            sched_name  = "qaoa" if "qaoa" in results else "greedy"
            for job in jobs:
                assignment = assignments.get(job.job_id, {})
                cost_usd   = assignment.get("cost_usd", 0)
                price_kwh  = assignment.get("price", 0)
                energy_kwh = (job.power_kw * job.duration_min / 60.0)
                # Naive cost = peak price × energy
                naive_price = float(max(prices)) if prices else price_kwh
                naive_cost  = naive_price * energy_kwh
                _cb.record(
                    job_id        = str(job.job_id),
                    job_name      = job.name,
                    cost_center   = getattr(job, "cost_center", "untagged"),
                    team          = getattr(job, "team",         "untagged"),
                    project       = getattr(job, "project",      "untagged"),
                    region        = "CAISO",
                    node_id       = os.environ.get("NODE_ID", "unknown"),
                    scheduler     = sched_name,
                    duration_min  = job.duration_min,
                    power_kw      = job.power_kw,
                    energy_kwh    = round(energy_kwh, 4),
                    price_per_kwh = round(price_kwh, 5),
                    cost_usd      = round(cost_usd, 5),
                    naive_cost_usd= round(naive_cost, 5),
                )
        except Exception as _cb_err:
            import logging
            logging.getLogger("hexagrid.chargeback").warning(
                "Chargeback record failed: %s", _cb_err)"""

# ── 3. Wire alert_manager → incident_engine ────────────────────────────────────
ALERT_WIRE_OLD = "    def fire(self, event_type, message, level=\"warning\","

ALERT_WIRE_NEW = "    def fire(self, event_type, message, level=\"warning\","

# We'll append to alert_manager.fire() using a different anchor
ALERT_FIRE_ANCHOR = "        self._persist_alert(event_type, title, level, message)"

ALERT_FIRE_PATCH = """\
        self._persist_alert(event_type, title, level, message)

        # ── Incident engine hook ──────────────────────────────────────────────
        if event_type in ("gpu_temperature", "gpu_health", "gpu_anomaly", "ecc_error"):
            try:
                import sys as _sys
                import os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
                from incident_engine import IncidentEngine as _IE
                _node  = (metadata or {}).get("node_id", "unknown")
                _gpu   = (metadata or {}).get("gpu_uuid", "")
                _IE().handle_alert(event_type, node_id=_node, gpu_uuid=_gpu,
                                   detail=metadata or {})
            except Exception as _ie_err:
                pass  # Never let incident engine crash alert delivery"""

# ── 4. Mount routers ──────────────────────────────────────────────────────────
ROUTER_OLD = """\
from auth_routes import router as auth_router
app.include_router(auth_router)"""

ROUTER_NEW = """\
from auth_routes import router as auth_router
app.include_router(auth_router)
from patch_api_chargeback import router as chargeback_router
app.include_router(chargeback_router)
from patch_api_incidents import router as incidents_router
app.include_router(incidents_router)"""

# ── 5. .env.auth additions ────────────────────────────────────────────────────
ENV_ADDITIONS = """
# ── Incident Response: PagerDuty ─────────────────────────────────────────────
# export HEXAGRID_PAGERDUTY_KEY=your-pagerduty-events-api-v2-routing-key

# ── Incident Response: OpsGenie ──────────────────────────────────────────────
# export HEXAGRID_OPSGENIE_KEY=your-opsgenie-api-key

# ── Incident Response: Thresholds ────────────────────────────────────────────
# export HEXAGRID_DRAIN_THRESHOLD_TEMP_C=91    # temp °C to trigger node drain
# export HEXAGRID_DEGRADE_THRESHOLD_TEMP_C=85  # temp °C to trigger degraded state
# export HEXAGRID_DEGRADE_HEALTH_SCORE=60      # health score to trigger degraded
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("VALIDATING PATHS")
    for p, label in [(API_PY, "api.py"), (API_DIR, "api/"), (ALERT_MGR, "alert_manager.py")]:
        if Path(p).exists():
            ok(f"Found: {p}")
        else:
            print(f"  ✗  Not found: {p}")
            sys.exit(1)

    banner("COPYING MODULE FILES")
    for fname in ["chargeback_ledger.py", "patch_api_chargeback.py",
                  "incident_engine.py", "patch_api_incidents.py"]:
        src = Path(__file__).parent / fname
        dst = API_DIR / fname
        if not src.exists():
            print(f"  ✗  Missing: {src}")
            print(f"     Ensure all 5 files are in the same directory as this script.")
            sys.exit(1)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            ok(f"Copied: {fname}")
        else:
            ok(f"Already in place: {fname}")

    banner("PATCHING api.py")
    backup(API_PY)
    patch(API_PY, JOBSPEC_OLD, JOBSPEC_NEW, "JobSpec: add cost_center/team/project")
    patch(API_PY, SCHEDULE_RECORD_OLD, SCHEDULE_RECORD_NEW, "Auto-record chargeback on schedule")
    patch(API_PY, ROUTER_OLD, ROUTER_NEW, "Mount chargeback + incidents routers")

    banner("WIRING alert_manager → incident_engine")
    # Check what the actual fire() method looks like in alert_manager.py
    alert_content = ALERT_MGR.read_text()
    if ALERT_FIRE_ANCHOR in alert_content:
        backup(ALERT_MGR)
        patch(ALERT_MGR, ALERT_FIRE_ANCHOR, ALERT_FIRE_PATCH,
              "Wire alert_manager.fire() → incident_engine")
    else:
        warn("Could not find _persist_alert anchor in alert_manager.py — wire manually")
        warn("See ALERT_FIRE_PATCH in this script for the code to add")

    banner("UPDATING .env.auth")
    if ENV_AUTH.exists():
        env_content = ENV_AUTH.read_text()
        if "PAGERDUTY" not in env_content:
            with open(ENV_AUTH, "a") as f:
                f.write(ENV_ADDITIONS)
            ok("Added PagerDuty/OpsGenie/threshold vars to .env.auth")
        else:
            ok("PagerDuty config already in .env.auth")
    else:
        warn(".env.auth not found — skipping")

    banner("DEPLOYMENT COMPLETE")
    print("""
  Next steps:

  1. Restart the API:
       source ~/hexagrid/.env.auth
       ./hexagrid.sh restart

  2. Verify new endpoints in Swagger:
       http://localhost:8000/docs

  3. Test chargeback — run a schedule job then check:
       curl http://localhost:8000/api/v1/chargeback/report | python -m json.tool

  4. Test incident response:
       curl -s -X POST http://localhost:8000/api/v1/incidents/trigger \\
         -H 'Content-Type: application/json' \\
         -H 'Authorization: Bearer <token>' \\
         -d '{"event_type":"gpu_temperature","node_id":"test-node","detail":{"temp_c":93}}'

  5. (Optional) Add PagerDuty/OpsGenie keys to .env.auth for live paging.
""")

if __name__ == "__main__":
    main()
