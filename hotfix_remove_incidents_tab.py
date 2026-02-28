#!/usr/bin/env python3
"""Remove orphaned page-incidents div from index.html"""
import shutil
from datetime import datetime
from pathlib import Path

CANDIDATES = [
    Path.home() / "hexagrid" / "dashboard" / "index.html",
    Path.home() / "hexagrid" / "api"       / "index.html",
    Path.home() / "hexagrid"               / "index.html",
]
INDEX = next((p for p in CANDIDATES if p.exists()), None)
if not INDEX:
    raise SystemExit("✗  index.html not found")

html = INDEX.read_text()

START = '\n<!-- ═══════════════════════════ INCIDENTS TAB ══════════════════════════ -->\n<div class="page" id="page-incidents">'
END   = '<!-- ═══════════════════════════ END INCIDENTS TAB ══════════════════════════ -->'

i = html.find(START)
j = html.find(END)

if i == -1 or j == -1:
    print("✗  Could not find orphaned incidents block — may already be removed")
    raise SystemExit(0)

bak = Path(str(INDEX) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(INDEX, bak)
print(f"✓  Backed up → {bak.name}")

html = html[:i] + html[j + len(END):]
INDEX.write_text(html)
print("✓  Removed orphaned page-incidents div")
print("   Hard refresh the dashboard (Ctrl+Shift+R)")
