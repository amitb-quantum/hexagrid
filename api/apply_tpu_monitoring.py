#!/usr/bin/env python3
"""
HexaGrid — Deploy TPU Monitoring (Feature 10)
=============================================
Run from ~/hexagrid/api/:  python apply_tpu_monitoring.py

What it does:
  1. Copies tpu_receiver.py + tpu_collector_agent.py into api/
  2. Mounts the TPU router in api.py
  3. Calls init_tpu_db() on startup
  4. Patches dashboard (runs patch_dashboard_tpu.py)
"""

import os, sys, shutil
from pathlib import Path
from datetime import datetime

API_DIR   = Path.home() / "hexagrid" / "api"
API_PY    = API_DIR / "api.py"
DASH_DIR  = Path.home() / "hexagrid" / "dashboard"
HERE      = Path(__file__).parent
BACKUP    = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def ok(m):   print(f"  ✓  {m}")
def warn(m): print(f"  ⚠  {m}")
def banner(m): print(f"\n{'═'*55}\n  {m}\n{'═'*55}")

def patch(path, old, new, label):
    content = path.read_text()
    if new.strip() in content:
        ok(f"Already patched: {label}"); return
    if old not in content:
        warn(f"Anchor not found — skipping: {label}"); return
    bak = Path(str(path) + BACKUP)
    shutil.copy2(path, bak)
    path.write_text(content.replace(old, new, 1))
    ok(f"Patched: {label}")

banner("COPYING MODULE FILES")
for fname in ["tpu_receiver.py", "tpu_collector_agent.py"]:
    src, dst = HERE / fname, API_DIR / fname
    if src.resolve() == dst.resolve():
        ok(f"Already in place: {fname}")
    else:
        shutil.copy2(src, dst); ok(f"Copied: {fname}")

banner("PATCHING api.py")

# Mount router
patch(API_PY,
    "from telemetry_receiver import router as telemetry_router, init_telemetry_db",
    "from telemetry_receiver import router as telemetry_router, init_telemetry_db\nfrom tpu_receiver import router as tpu_router, init_tpu_db",
    "Import tpu_receiver"
)

patch(API_PY,
    "app.include_router(telemetry_router)",
    "app.include_router(telemetry_router)\napp.include_router(tpu_router)\ninit_tpu_db()",
    "Mount TPU router and init DB"
)

banner("PATCHING DASHBOARD")
dash_patch = HERE / "patch_dashboard_tpu.py"
if dash_patch.exists():
    import subprocess
    r = subprocess.run([sys.executable, str(dash_patch)], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        warn(f"Dashboard patch error: {r.stderr}")
else:
    warn("patch_dashboard_tpu.py not found — run it manually")

banner("DONE")
print("""
  Next steps:

  1. Restart API:
       cd ~/hexagrid && ./hexagrid.sh restart

  2. Hard refresh dashboard (Ctrl+Shift+R)
     → TPU section appears in Fleet tab

  3. On a GCP TPU node:
       export HEXAGRID_ENDPOINT=http://<your-host>:8000
       export HEXAGRID_TOKEN=hg_...
       export TPU_PROVIDER=gcp
       export GCP_PROJECT_ID=my-project
       export GCP_ZONE=us-central2-b
       export GCP_TPU_NODE=my-tpu-node
       python tpu_collector_agent.py

  4. On an AWS Trainium/Inferentia node:
       neuron-monitor &
       export HEXAGRID_ENDPOINT=http://<your-host>:8000
       export HEXAGRID_TOKEN=hg_...
       export TPU_PROVIDER=aws
       python tpu_collector_agent.py

  5. Verify API:
       curl http://localhost:8000/api/v1/telemetry/tpu/fleet
""")
