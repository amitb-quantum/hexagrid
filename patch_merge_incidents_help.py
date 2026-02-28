#!/usr/bin/env python3
"""
HexaGrid Dashboard — Patch: Merge Incidents into Alerts tab + add Help content
===============================================================================
What this does:
  1. Renames "Incidents" tab to "Alerts & Incidents" (replaces Alerts tab button)
  2. Merges the incidents page into the alerts page as a second section
  3. Removes the now-redundant standalone Incidents tab button
  4. Adds help content for Alerts & Incidents (combined) and Chargeback
  5. Adds help drawer buttons for both new sections

Run from anywhere:  python patch_merge_incidents_help.py
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
    print("✗  index.html not found")
    raise SystemExit(1)

print(f"  ✓  Found: {INDEX}")
bak = Path(str(INDEX) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(INDEX, bak)
print(f"  ✓  Backed up → {bak.name}")

html = INDEX.read_text()

def patch(old, new, label):
    global html
    if new.strip() in html:
        print(f"  ✓  Already patched: {label}")
        return True
    if old not in html:
        print(f"  ⚠  Anchor not found — skipping: {label}")
        return False
    html = html.replace(old, new, 1)
    print(f"  ✓  Patched: {label}")
    return True

# ══════════════════════════════════════════════════════════════════════════════
#  1. TAB BAR — replace Alerts button + remove standalone Incidents button
# ══════════════════════════════════════════════════════════════════════════════

patch(
    '    <button class="tab" onclick="switchTab(\'alerts\', this)"><i class="fa-solid fa-bell"></i>Alerts</button>',
    '    <button class="tab" onclick="switchTab(\'alerts\', this)" id="tab-alerts-incidents"><i class="fa-solid fa-bell"></i>Alerts &amp; Incidents</button>',
    "Rename Alerts tab button"
)

patch(
    '\n    <button class="tab" onclick="switchTab(\'incidents\', this)" id="tab-incidents"><i class="fa-solid fa-triangle-exclamation"></i>Incidents</button>',
    '',
    "Remove standalone Incidents tab button"
)

# ══════════════════════════════════════════════════════════════════════════════
#  2. MERGE INCIDENTS CONTENT INTO ALERTS PAGE
#     Append incidents section just before the closing </div></div> of alerts page
# ══════════════════════════════════════════════════════════════════════════════

INCIDENTS_SECTION = '''
<!-- ═══════════════════════ INCIDENTS SECTION (merged into Alerts) ═══════════ -->
<div style="margin-top:28px;">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:17px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:10px;">
        <i class="fa-solid fa-triangle-exclamation" style="color:var(--amber)"></i>
        Incident Response
      </div>
      <div style="font-size:12px;color:var(--muted);margin-top:3px;">Automated node state machine — thermal, ECC, and health-driven remediation</div>
    </div>
    <div style="display:flex;gap:10px;">
      <button onclick="loadIncidents()" style="background:var(--surface);border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 14px;font-size:12px;cursor:pointer;">↻ Refresh</button>
    </div>
  </div>

  <!-- KPI strip -->
  <div class="grid" style="margin-bottom:18px;">
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-server"></i>Tracked Nodes</div>
      <div class="kpi-value" id="incKpiTotal">—</div>
      <div class="kpi-unit">ever had an event</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-circle-check" style="color:var(--green)"></i>Healthy</div>
      <div class="kpi-value green" id="incKpiHealthy">—</div>
      <div class="kpi-unit">nodes nominal</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-triangle-exclamation" style="color:var(--amber)"></i>Degraded</div>
      <div class="kpi-value" style="color:var(--amber)" id="incKpiDegraded">—</div>
      <div class="kpi-unit">under monitoring</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-circle-xmark" style="color:var(--red)"></i>Draining / Drained</div>
      <div class="kpi-value" style="color:var(--red)" id="incKpiDrained">—</div>
      <div class="kpi-unit">offline / draining</div>
    </div>
  </div>

  <!-- Fleet grid + config -->
  <div class="grid" style="margin-bottom:18px;">
    <div class="card col-8" style="padding:20px;">
      <div class="card-title" style="margin-bottom:16px;"><i class="fa-solid fa-network-wired" style="color:var(--cyan)"></i>Fleet Node States</div>
      <div id="incFleetGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;">
        <div style="color:var(--muted);font-size:13px;grid-column:span 4;padding:24px;text-align:center;">
          No incidents recorded yet — all nodes implicitly healthy.<br>
          <span style="font-size:11px;">Nodes appear here when their first alert fires.</span>
        </div>
      </div>
    </div>
    <div class="card col-4" style="padding:20px;">
      <div class="card-title" style="margin-bottom:16px;"><i class="fa-solid fa-sliders" style="color:var(--amber)"></i>Response Thresholds</div>
      <div class="alerts-field"><label>Drain Temp (°C)</label>
        <input type="number" id="incDrainTemp" placeholder="91" min="70" max="110" style="width:100%;">
      </div>
      <div class="alerts-field"><label>Degrade Temp (°C)</label>
        <input type="number" id="incDegradeTemp" placeholder="85" min="60" max="105" style="width:100%;">
      </div>
      <div class="alerts-field"><label>Health Score Min</label>
        <input type="number" id="incHealthMin" placeholder="60" min="0" max="100" style="width:100%;">
      </div>
      <button onclick="saveIncidentConfig()" style="margin-top:14px;width:100%;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:var(--green);border-radius:8px;padding:9px;font-size:12px;font-weight:600;cursor:pointer;">Save Thresholds</button>
      <div id="incConfigStatus" style="font-size:11px;color:var(--muted);margin-top:8px;text-align:center;"></div>
      <div class="card-title" style="margin:18px 0 12px;"><i class="fa-solid fa-bell" style="color:var(--purple)"></i>Paging Channels</div>
      <div id="incNotifStatus" style="font-size:12px;color:var(--text2);line-height:2;">Loading...</div>
    </div>
  </div>

  <!-- Incident timeline -->
  <div class="card col-12" style="padding:20px;margin-bottom:8px;">
    <div class="card-title" style="margin-bottom:14px;">
      <i class="fa-solid fa-clock-rotate-left" style="color:var(--cyan)"></i>Incident Timeline
      <span style="margin-left:auto;">
        <input type="text" id="incNodeFilter" placeholder="Filter by node ID..." oninput="filterIncidentHistory()" style="background:var(--surface2);border:1px solid var(--border);color:var(--text2);border-radius:6px;padding:5px 10px;font-size:11px;width:180px;">
      </span>
    </div>
    <div style="overflow-x:auto;">
      <table class="data-table" id="incHistoryTable">
        <thead>
          <tr>
            <th>Time</th><th>Node</th><th>Transition</th><th>Trigger</th><th>Detail</th>
            <th style="text-align:right">Actions</th>
          </tr>
        </thead>
        <tbody id="incHistoryBody">
          <tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px;">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>
<!-- ═══════════════════════ END INCIDENTS SECTION ═══════════════════════════ -->
'''

# Anchor: the closing tags of the alerts page
ALERTS_PAGE_END = '''</div>
</div>
<!-- ═══════════════════════════ END ALERTS TAB ═══════════════════════ -->'''

patch(
    ALERTS_PAGE_END,
    INCIDENTS_SECTION + ALERTS_PAGE_END,
    "Merge incidents section into alerts page"
)

# ══════════════════════════════════════════════════════════════════════════════
#  3. UPDATE switchTab to load incidents when alerts tab opens
# ══════════════════════════════════════════════════════════════════════════════

patch(
    "  if (name === 'alerts')  { setTimeout(initAlertsTab, 100); }",
    "  if (name === 'alerts')  { setTimeout(initAlertsTab, 100); setTimeout(loadIncidents, 200); }",
    "Wire loadIncidents() to alerts tab switch"
)

# Also update the _origSwitchTab_89 hook (from previous patch) to not double-load incidents
patch(
    "  if (name === 'incidents')  setTimeout(loadIncidents,  100);",
    "  // incidents merged into alerts tab — loaded via initAlertsTab hook",
    "Remove orphaned incidents switchTab hook"
)

# ══════════════════════════════════════════════════════════════════════════════
#  4. FIX tab badge — point to alerts tab button instead of incidents
# ══════════════════════════════════════════════════════════════════════════════

patch(
    "    const tabBtn  = document.getElementById('tab-incidents');",
    "    const tabBtn  = document.getElementById('tab-alerts-incidents');",
    "Point incident badge to merged Alerts & Incidents tab button"
)

# ══════════════════════════════════════════════════════════════════════════════
#  5. HELP DRAWER — add buttons for Chargeback and Alerts & Incidents
# ══════════════════════════════════════════════════════════════════════════════

patch(
    '    <button class="help-tab-btn" onclick="showHelpTab(\'alerts\',this)"><i class="fa-solid fa-bell"></i> Alerts</button>',
    '    <button class="help-tab-btn" onclick="showHelpTab(\'alerts\',this)"><i class="fa-solid fa-bell"></i> Alerts &amp; Incidents</button>\n    <button class="help-tab-btn" onclick="showHelpTab(\'chargeback\',this)"><i class="fa-solid fa-receipt"></i> Chargeback</button>',
    "Add Chargeback help button, rename Alerts help button"
)

# ══════════════════════════════════════════════════════════════════════════════
#  6. HELP CONTENT — replace alerts entry + add chargeback entry
# ══════════════════════════════════════════════════════════════════════════════

ALERTS_HELP_OLD = "  ,alerts: {\n    label: 'Alerts Tab',"
ALERTS_HELP_NEW = "  ,alerts: {\n    label: 'Alerts \\u0026 Incidents',"
patch(ALERTS_HELP_OLD, ALERTS_HELP_NEW, "Rename alerts help label")

# Add chargeback and incidents help content after benchmark entry
HELP_ANCHOR = "  ,benchmark: {"
CHARGEBACK_AND_INCIDENTS_HELP = '''  ,chargeback: {
    label: 'Chargeback Tab',
    html: `
      <div class="help-section">
        <h3><i class="fa-solid fa-receipt"></i>What is the Chargeback tab?</h3>
        <p>The Chargeback tab attributes GPU energy costs to specific cost centers, teams, and projects. Every job run through the HexaGrid scheduler is automatically recorded with its actual energy consumption and cost, enabling accurate internal billing and finance reporting.</p>
        <p>Finance teams can pull monthly CSV exports directly from this tab — no engineering involvement required once jobs are tagged.</p>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-tags"></i>How to Tag Jobs</h3>
        <p>Add three optional fields to any job submitted via <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">POST /api/v1/schedule</code>:</p>
        <div class="help-metric"><div class="hm-name">cost_center</div><div class="hm-desc">The billing cost center — e.g. "ML-Platform", "Research", "Infra-Ops". Defaults to "untagged" if omitted.</div></div>
        <div class="help-metric"><div class="hm-name">team</div><div class="hm-desc">The team owning the job — e.g. "infra", "nlp-team", "cv-research". Used for team-level reporting.</div></div>
        <div class="help-metric"><div class="hm-name">project</div><div class="hm-desc">The project this job belongs to — e.g. "ImageNet-2026", "LLM-Finetune-Q1". Used for project budget tracking.</div></div>
        <div class="help-tip"><strong>Tip:</strong> Jobs submitted without tags still appear in reports under "untagged" — you can see at a glance how much unattributed spend exists and work to tag it down over time.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-chart-pie"></i>Reports &amp; Filters</h3>
        <div class="help-metric"><div class="hm-name">Group By</div><div class="hm-desc">Switch between cost center, team, project, region, or scheduler grouping to slice the data differently for different audiences.</div></div>
        <div class="help-metric"><div class="hm-name">Month Picker</div><div class="hm-desc">Filter to a specific calendar month for monthly billing cycles. Leave blank for all-time totals.</div></div>
        <div class="help-metric"><div class="hm-name">Export CSV</div><div class="hm-desc">Downloads a finance-ready CSV with all breakdown rows plus period totals. Safe to open in Excel or upload to expense systems.</div></div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-piggy-bank"></i>Scheduler Savings Column</h3>
        <p>The <strong style="color:var(--green)">Saved ($)</strong> column shows how much each cost center saved compared to naive dispatch — running jobs immediately at whatever the grid price was at queue time. This is the CFO-friendly number: it quantifies the ROI of running HexaGrid vs not.</p>
        <div class="help-tip"><strong>Tip:</strong> For external jobs not submitted through the HexaGrid scheduler, use <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">POST /api/v1/chargeback/record</code> to manually add entries to the ledger.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-plug-circle-exclamation"></i>API Endpoints</h3>
        <div class="help-metric"><div class="hm-name">GET /api/v1/chargeback/report</div><div class="hm-desc">JSON report grouped by any dimension with period filters. Used by this tab.</div></div>
        <div class="help-metric"><div class="hm-name">GET /api/v1/chargeback/report.csv</div><div class="hm-desc">Direct CSV download — same filters as the JSON report.</div></div>
        <div class="help-metric"><div class="hm-name">GET /api/v1/chargeback/entries</div><div class="hm-desc">Raw per-job audit trail — every individual job entry with full cost breakdown.</div></div>
        <div class="help-metric"><div class="hm-name">POST /api/v1/chargeback/record</div><div class="hm-desc">Manually record a job cost entry for workloads run outside the scheduler.</div></div>
      </div>`
  }

  ,incidents_help: {
    label: 'Alerts \\u0026 Incidents',
    html: `
      <div class="help-section">
        <h3><i class="fa-solid fa-triangle-exclamation"></i>What is Incident Response?</h3>
        <p>The Incident Response section (bottom of the Alerts tab) automates GPU node remediation. When a thermal event, ECC error, or health score drop is detected, HexaGrid transitions the affected node through a state machine — from healthy to degraded to draining — without requiring manual intervention.</p>
        <p>At 10,000+ GPU scale, automated triage is the difference between a contained event and a cascading failure.</p>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-diagram-project"></i>Node State Machine</h3>
        <div class="help-metric"><div class="hm-name" style="color:var(--green)">Healthy</div><div class="hm-desc">Normal operation. No active incident.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--amber)">Degraded</div><div class="hm-desc">Temperature or health score is elevated but below drain threshold. Node is monitored closely. New jobs may still be scheduled here.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--red)">Draining</div><div class="hm-desc">A critical event triggered job migration. The node is flagged — no new jobs will be assigned here. Existing jobs should complete or be migrated by the operator.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--red)">Drained</div><div class="hm-desc">The node is empty and awaiting maintenance. Safe to take offline.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--cyan)">Recovering</div><div class="hm-desc">Node is back online and health metrics are returning to normal. Mark as Clear once you're satisfied it's stable.</div></div>
        <div class="help-tip"><strong>Trigger thresholds:</strong> ECC double-bit errors → immediate drain. Temp ≥ 91°C (configurable) → drain. Temp ≥ 85°C → degrade. Health score &lt; 60 → degrade.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-bell"></i>Outbound Paging</h3>
        <div class="help-metric"><div class="hm-name">PagerDuty</div><div class="hm-desc">Add HEXAGRID_PAGERDUTY_KEY to .env.auth. Uses Events API v2 — the dedup key is hexagrid-node-{node_id} so repeated events for the same node don't create duplicate incidents.</div></div>
        <div class="help-metric"><div class="hm-name">OpsGenie</div><div class="hm-desc">Add HEXAGRID_OPSGENIE_KEY to .env.auth. Uses OpsGenie Alerts API with alias-based deduplication. Recovery events automatically close the alert.</div></div>
        <div class="help-metric"><div class="hm-name">Alert Manager</div><div class="hm-desc">Always fires through HexaGrid's own alert manager — so existing email/webhook delivery works without any additional configuration.</div></div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-hand"></i>Manual Operator Controls</h3>
        <p>Every node card has action buttons appropriate to its current state:</p>
        <div class="help-metric"><div class="hm-name">Drain</div><div class="hm-desc">Manually initiate a drain on a healthy or degraded node — useful when scheduling maintenance before an alert fires.</div></div>
        <div class="help-metric"><div class="hm-name">Recover</div><div class="hm-desc">Mark a drained node as recovering after you've addressed the underlying issue.</div></div>
        <div class="help-metric"><div class="hm-name">Clear</div><div class="hm-desc">Return a recovering node to fully healthy once you're confident it's stable.</div></div>
        <div class="help-tip"><strong>Tip:</strong> Use the <strong style="color:var(--cyan)">Node filter</strong> in the timeline to quickly find the full incident history for a specific node during post-incident review.</div>
      </div>`
  }

  ,benchmark: {'''

patch(HELP_ANCHOR, CHARGEBACK_AND_INCIDENTS_HELP, "Add Chargeback + Incidents help content")

# ══════════════════════════════════════════════════════════════════════════════
#  WRITE
# ══════════════════════════════════════════════════════════════════════════════

INDEX.write_text(html)
print(f"\n  ✓  Written: {INDEX}")
print("""
  ══════════════════════════════════════════════
  Done — hard refresh the dashboard (Ctrl+Shift+R)

  Changes:
    • "Alerts" tab renamed → "Alerts & Incidents"
    • Incident Response section now lives at the
      bottom of the Alerts & Incidents tab
    • Help drawer: Alerts & Incidents + Chargeback
      entries added
    • Standalone Incidents tab removed
  ══════════════════════════════════════════════
""")
