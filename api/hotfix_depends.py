#!/usr/bin/env python3
"""
HexaGrid Auth — Hotfix: add Depends to FastAPI imports in api.py
Run from ~/hexagrid/api/:  python hotfix_depends.py
"""
from pathlib import Path

API_PY = Path.home() / "hexagrid" / "api" / "api.py"

content = API_PY.read_text()

OLD = "from fastapi import FastAPI, HTTPException, BackgroundTasks, Query"
NEW = "from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends"

if NEW in content:
    print("✓ Already fixed — Depends is already imported")
elif OLD in content:
    fixed = content.replace(OLD, NEW, 1)
    API_PY.write_text(fixed)
    print("✓ Fixed — added Depends to FastAPI imports on line 34")
    print(f"  {OLD}")
    print(f"  → {NEW}")
else:
    print("✗ Could not find import line — check api.py manually")
    print(f"  Looking for: {OLD}")
