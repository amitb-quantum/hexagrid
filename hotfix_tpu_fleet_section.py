#!/usr/bin/env python3
"""
Hotfix: insert TPU Fleet section into Fleet tab using correct anchor.
Run from anywhere: python hotfix_tpu_fleet_section.py
"""
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

# Check if already patched
if 'id="tpu-kpi-nodes"' in html:
    print("✓  TPU section already present in Fleet tab")
    raise SystemExit(0)

bak = Path(str(INDEX) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(INDEX, bak)
print(f"✓  Backed up → {bak.name}")

# Exact anchor from the live file
ANCHOR = '  <!-- ══ END LIVE GPU TELEMETRY ════════════════════════════════════════ -->'

TPU_SECTION = '''  <!-- ══ END LIVE GPU TELEMETRY ════════════════════════════════════════ -->

  <!-- ══ TPU TELEMETRY ════════════════════════════════════════════════ -->

  <!-- TPU header row -->
  <div class="col-12" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px;margin-top:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:15px;font-weight:700;color:var(--text);">
        <i class="fa-solid fa-microchip" style="color:var(--purple);margin-right:6px;"></i>TPU Fleet
      </span>
      <span style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Provider</span>
      <select id="tpu-provider-filter" onchange="loadTPUTelemetry()"
        style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;
               padding:4px 10px;font-size:13px;color:var(--text);width:120px;">
        <option value="">All</option>
        <option value="gcp">GCP</option>
        <option value="aws">AWS</option>
      </select>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
      <span id="tpu-age-badge"
        style="font-size:11px;padding:3px 10px;border-radius:12px;
               background:rgba(191,127,255,0.1);color:var(--purple);
               border:1px solid rgba(191,127,255,0.25);">
        &#9679; no data
      </span>
      <button class="btn-secondary" onclick="loadTPUTelemetry()" style="padding:4px 12px;font-size:12px;">
        <i class="fa-solid fa-rotate"></i>
      </button>
    </div>
  </div>

  <!-- TPU KPI strip -->
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-server" style="color:var(--purple)"></i>TPU Nodes</div>
    <div class="kpi-value" style="color:var(--purple)" id="tpu-kpi-nodes">0</div>
    <div class="kpi-delta neu" id="tpu-kpi-nodes-sub">no collectors yet</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-microchip" style="color:var(--cyan)"></i>Total Chips</div>
    <div class="kpi-value cyan" id="tpu-kpi-chips">0</div>
    <div class="kpi-delta neu" id="tpu-kpi-chips-sub">deploy collector to populate</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-gauge-high" style="color:var(--green)"></i>Avg Utilisation</div>
    <div class="kpi-value green" id="tpu-kpi-util">—</div>
    <div class="kpi-delta neu" id="tpu-kpi-util-sub">matrix unit</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-triangle-exclamation" style="color:var(--amber)"></i>Runtime Errors</div>
    <div class="kpi-value green" id="tpu-kpi-errors">0</div>
    <div class="kpi-delta neu" id="tpu-kpi-errors-sub">since last poll</div>
  </div>

  <!-- TPU node grid -->
  <div class="card col-12">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div class="card-title" style="margin-bottom:0;">
        <i class="fa-solid fa-chart-network" style="color:var(--purple)"></i>TPU Node Grid
        <span style="font-size:11px;font-weight:400;color:var(--muted);margin-left:8px;">
          GCP Cloud TPU &middot; AWS Trainium / Inferentia
        </span>
      </div>
      <div style="display:flex;gap:12px;font-size:11px;color:var(--muted);">
        <span><span style="color:var(--purple);">&#9679;</span> GCP TPU</span>
        <span><span style="color:var(--cyan);">&#9679;</span> AWS Trainium</span>
        <span><span style="color:var(--amber);">&#9679;</span> AWS Inferentia</span>
        <span><span style="color:var(--muted);">&#9679;</span> stale</span>
      </div>
    </div>
    <div id="tpu-node-grid"
         style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;">
      <div style="color:var(--muted);font-size:13px;padding:20px 0;grid-column:span 6;">
        <i class="fa-solid fa-satellite-dish" style="margin-right:8px;color:var(--purple);"></i>
        No TPU nodes reporting yet — deploy <strong>tpu_collector_agent.py</strong>
        on your GCP or AWS nodes and point it at this HexaGrid API.
        See the Help drawer (TPU section) for setup instructions.
      </div>
    </div>
  </div>

  <div class="col-12" style="border-bottom:1px solid var(--border);margin:8px 0 20px"></div>
  <!-- ══ END TPU TELEMETRY ═════════════════════════════════════════════ -->'''

if ANCHOR not in html:
    print(f"✗  Anchor not found in {INDEX}")
    print("   Looking for:")
    print(f"   {repr(ANCHOR)}")
    raise SystemExit(1)

html = html.replace(ANCHOR, TPU_SECTION, 1)
INDEX.write_text(html)
print(f"✓  TPU Fleet section inserted into {INDEX.name}")
print("   Hard refresh the dashboard (Ctrl+Shift+R)")
