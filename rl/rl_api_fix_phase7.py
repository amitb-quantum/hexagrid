"""
HexaGrid Phase 7 — API Bug Fix
==============================
Fixes two bugs introduced by rl_api_patch_phase7.py:

  Bug 1: _update_job() doesn't exist — real functions are
          _set_running(), _set_done(), _set_error()

  Bug 2: 'from agent import HexaGridAgent' fails because the
          file is named rl_agent.py, not agent.py

Run from ~/hexagrid:
    python rl/rl_api_fix_phase7.py
"""

import os, sys, re

API_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'api', 'api.py')
)

if not os.path.exists(API_PATH):
    print(f"ERROR: {API_PATH} not found. Run from ~/hexagrid.")
    sys.exit(1)

content = open(API_PATH).read()
original = content

# ── Fix 1: wrong function names ───────────────────────────────────────────────
# Replace _update_job(run_id, 'running') → _set_running(run_id)
content = content.replace(
    "_update_job(run_id, 'running')",
    "_set_running(run_id)"
)

# Replace _update_job(run_id, 'completed', {...}) → _set_done(run_id, {...})
# This pattern appears as: _update_job(run_id, 'completed', { ... })
content = re.sub(
    r"_update_job\(run_id,\s*'completed',\s*(\{[^}]+\})\)",
    r"_set_done(run_id, \1)",
    content
)

# Replace _update_job(run_id, 'failed', error=str(e)) → _set_error(run_id, str(e))
content = re.sub(
    r"_update_job\(run_id,\s*'failed',\s*error=str\((\w+)\)\)",
    r"_set_error(run_id, str(\1))",
    content
)

# ── Fix 2: wrong module name ──────────────────────────────────────────────────
content = content.replace(
    "from agent import HexaGridAgent",
    "from rl_agent import HexaGridAgent"
)

if content == original:
    print("  [WARN] No changes made — bugs may already be fixed or patch not found.")
    print("  Check that rl_api_patch_phase7.py was applied first.")
    sys.exit(0)

open(API_PATH, 'w').write(content)
print(f"  [OK] Fixed api.py: {API_PATH}")
print()

# Verify fixes landed
fixes = [
    ("_set_running(run_id)",        "Fix 1a: _set_running"),
    ("_set_done(run_id,",           "Fix 1b: _set_done"),
    ("_set_error(run_id,",          "Fix 1c: _set_error"),
    ("from rl_agent import",        "Fix 2:  rl_agent import"),
]
all_ok = True
for needle, label in fixes:
    found = needle in open(API_PATH).read()
    status = "✓" if found else "✗"
    print(f"  {status} {label}")
    if not found:
        all_ok = False

print()
if all_ok:
    print("  All fixes applied. Restart uvicorn:")
    print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
else:
    print("  Some fixes may have missed. Check api.py manually around the Phase 7 block.")
