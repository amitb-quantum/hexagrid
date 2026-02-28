#!/usr/bin/env python3
"""
HexaGrid Dashboard — Patch: Chargeback + Incidents tabs
========================================================
Adds two new tabs to index.html:
  - Chargeback: cost breakdown by cost_center/team/project, CSV export
  - Incidents:  fleet health grid, node state badges, incident timeline

Run from ~/hexagrid/api/:  python patch_dashboard_features89.py
Or from ~/hexagrid/:       python api/patch_dashboard_features89.py
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

# ── Locate index.html ─────────────────────────────────────────────────────────
CANDIDATES = [
    Path.home() / "hexagrid" / "dashboard" / "index.html",
    Path.home() / "hexagrid" / "api"       / "index.html",
    Path.home() / "hexagrid"               / "index.html",
]
INDEX = next((p for p in CANDIDATES if p.exists()), None)
if not INDEX:
    print("✗  index.html not found — tried:")
    for c in CANDIDATES: print(f"     {c}")
    raise SystemExit(1)

print(f"  ✓  Found: {INDEX}")
bak = Path(str(INDEX) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(INDEX, bak)
print(f"  ✓  Backed up → {bak.name}")

html = INDEX.read_text()

# ══════════════════════════════════════════════════════════════════════════════
#  1. TAB BUTTONS
# ══════════════════════════════════════════════════════════════════════════════

TAB_OLD = '    <button class="tab" onclick="switchTab(\'benchmark\', this)"><i class="fa-solid fa-chart-bar"></i>Benchmark</button>'
TAB_NEW = '''    <button class="tab" onclick="switchTab('benchmark', this)"><i class="fa-solid fa-chart-bar"></i>Benchmark</button>
    <button class="tab" onclick="switchTab('chargeback', this)"><i class="fa-solid fa-receipt"></i>Chargeback</button>
    <button class="tab" onclick="switchTab('incidents', this)" id="tab-incidents"><i class="fa-solid fa-triangle-exclamation"></i>Incidents</button>'''

# ══════════════════════════════════════════════════════════════════════════════
#  2. CHARGEBACK PAGE HTML
# ══════════════════════════════════════════════════════════════════════════════

CHARGEBACK_HTML = '''
<!-- ═══════════════════════════ CHARGEBACK TAB ══════════════════════════ -->
<div class="page" id="page-chargeback">
<div class="page-inner">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:20px;font-weight:800;color:var(--text)">Cost Allocation &amp; Chargeback</div>
      <div style="font-size:12px;color:var(--muted);margin-top:3px;">Energy costs attributed by cost center, team, and project</div>
    </div>
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <select id="cbGroupBy" onchange="loadChargeback()" style="background:var(--surface);border:1px solid var(--border);color:var(--text2);border-radius:8px;padding:7px 12px;font-size:12px;cursor:pointer;">
        <option value="cost_center">Group by Cost Center</option>
        <option value="team">Group by Team</option>
        <option value="project">Group by Project</option>
        <option value="region">Group by Region</option>
        <option value="scheduler">Group by Scheduler</option>
      </select>
      <input type="month" id="cbMonth" onchange="loadChargeback()" style="background:var(--surface);border:1px solid var(--border);color:var(--text2);border-radius:8px;padding:7px 12px;font-size:12px;">
      <button onclick="downloadChargebackCSV()" style="background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:var(--green);border-radius:8px;padding:8px 16px;font-size:12px;font-weight:600;cursor:pointer;">⬇ Export CSV</button>
      <button onclick="loadChargeback()" style="background:var(--surface);border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 14px;font-size:12px;cursor:pointer;">↻ Refresh</button>
    </div>
  </div>

  <!-- KPI strip -->
  <div class="grid" style="margin-bottom:18px;">
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-sack-dollar"></i>Total Cost</div>
      <div class="kpi-value green" id="cbKpiCost">—</div>
      <div class="kpi-unit" id="cbKpiPeriod">selected period</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-bolt"></i>Total Energy</div>
      <div class="kpi-value cyan" id="cbKpiEnergy">—</div>
      <div class="kpi-unit">kWh consumed</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-briefcase"></i>Jobs Scheduled</div>
      <div class="kpi-value" id="cbKpiJobs">—</div>
      <div class="kpi-unit">job runs recorded</div>
    </div>
    <div class="card col-3">
      <div class="card-title"><i class="fa-solid fa-piggy-bank"></i>Scheduler Savings</div>
      <div class="kpi-value" style="color:var(--amber)" id="cbKpiSaved">—</div>
      <div class="kpi-unit">vs naive dispatch</div>
    </div>
  </div>

  <!-- Chart + Table grid -->
  <div class="grid" style="margin-bottom:18px;">

    <!-- Donut chart -->
    <div class="card col-4" style="padding:20px;">
      <div class="card-title" style="margin-bottom:14px;"><i class="fa-solid fa-chart-pie" style="color:var(--cyan)"></i>Cost Distribution</div>
      <div style="position:relative;height:220px;display:flex;align-items:center;justify-content:center;">
        <canvas id="cbDonutChart"></canvas>
        <div id="cbDonutEmpty" style="display:none;color:var(--muted);font-size:13px;position:absolute;">No data yet — run jobs with cost_center tags</div>
      </div>
      <div id="cbLegend" style="margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;"></div>
    </div>

    <!-- Bar chart -->
    <div class="card col-8" style="padding:20px;">
      <div class="card-title" style="margin-bottom:14px;"><i class="fa-solid fa-chart-bar" style="color:var(--green)"></i>Cost &amp; Savings by Group</div>
      <div style="position:relative;height:220px;">
        <canvas id="cbBarChart"></canvas>
        <div id="cbBarEmpty" style="display:none;color:var(--muted);font-size:13px;padding-top:80px;text-align:center;">No chargeback data recorded yet.<br><span style="font-size:11px;">Submit jobs with cost_center/team/project tags via the Scheduler tab or API.</span></div>
      </div>
    </div>

  </div>

  <!-- Breakdown table -->
  <div class="card col-12" style="margin-bottom:18px;padding:20px;">
    <div class="card-title" style="margin-bottom:14px;">
      <i class="fa-solid fa-table" style="color:var(--amber)"></i>Chargeback Breakdown
      <span style="margin-left:auto;font-size:11px;color:var(--muted);font-weight:400;" id="cbTableMeta"></span>
    </div>
    <div style="overflow-x:auto;">
      <table class="data-table" id="cbBreakdownTable">
        <thead>
          <tr>
            <th>Group</th>
            <th style="text-align:right">Jobs</th>
            <th style="text-align:right">Energy (kWh)</th>
            <th style="text-align:right">Actual Cost</th>
            <th style="text-align:right">Naive Cost</th>
            <th style="text-align:right">Saved</th>
            <th style="text-align:right">Avg $/kWh</th>
          </tr>
        </thead>
        <tbody id="cbBreakdownBody">
          <tr><td colspan="7" style="color:var(--muted);text-align:center;padding:24px;">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Info note -->
  <div style="background:rgba(0,255,136,0.04);border:1px solid rgba(0,255,136,0.15);border-radius:10px;padding:14px 18px;font-size:12px;color:var(--text2);line-height:1.7;">
    <span style="color:var(--green);font-weight:700;">How to tag jobs: </span>
    Add <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">cost_center</code>,
    <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">team</code>, and
    <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">project</code> fields to any job submitted via
    <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">POST /api/v1/schedule</code>.
    The scheduler automatically records costs to the chargeback ledger on every run.
    Use <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">POST /api/v1/chargeback/record</code> for external jobs.
  </div>

</div>
</div>
<!-- ═══════════════════════════ END CHARGEBACK TAB ══════════════════════════ -->
'''

# ══════════════════════════════════════════════════════════════════════════════
#  3. INCIDENTS PAGE HTML
# ══════════════════════════════════════════════════════════════════════════════

INCIDENTS_HTML = '''
<!-- ═══════════════════════════ INCIDENTS TAB ══════════════════════════ -->
<div class="page" id="page-incidents">
<div class="page-inner">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:20px;font-weight:800;color:var(--text)">Incident Response</div>
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

    <!-- Node state grid -->
    <div class="card col-8" style="padding:20px;">
      <div class="card-title" style="margin-bottom:16px;"><i class="fa-solid fa-network-wired" style="color:var(--cyan)"></i>Fleet Node States</div>
      <div id="incFleetGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;">
        <div style="color:var(--muted);font-size:13px;grid-column:span 4;padding:24px;text-align:center;">
          No incidents recorded yet — all nodes implicitly healthy.<br>
          <span style="font-size:11px;">Nodes appear here when their first alert fires.</span>
        </div>
      </div>
    </div>

    <!-- Config panel -->
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
  <div class="card col-12" style="padding:20px;margin-bottom:18px;">
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
            <th>Time</th>
            <th>Node</th>
            <th>Transition</th>
            <th>Trigger</th>
            <th>Detail</th>
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
</div>
<!-- ═══════════════════════════ END INCIDENTS TAB ══════════════════════════ -->
'''

# ══════════════════════════════════════════════════════════════════════════════
#  4. JAVASCRIPT
# ══════════════════════════════════════════════════════════════════════════════

JS = '''
// ═══════════════════════════════════════════════════════════════════
//  CHARGEBACK
// ═══════════════════════════════════════════════════════════════════
let cbDonutChart = null;
let cbBarChart   = null;
let _cbAllRows   = [];

function getChargebackParams() {
  const month = document.getElementById('cbMonth').value; // "2026-02"
  let start = '', end = '';
  if (month) {
    const [y, m] = month.split('-').map(Number);
    start = `${month}-01T00:00:00`;
    const lastDay = new Date(y, m, 0).getDate();
    end   = `${month}-${String(lastDay).padStart(2,'0')}T23:59:59`;
  }
  const group = document.getElementById('cbGroupBy').value;
  return { start, end, group };
}

async function loadChargeback() {
  const { start, end, group } = getChargebackParams();
  let url = `/chargeback/report?group_by=${group}`;
  if (start) url += `&period_start=${encodeURIComponent(start)}&period_end=${encodeURIComponent(end)}`;

  try {
    const d = await apiFetch(url);

    // KPIs
    const t = d.totals || {};
    document.getElementById('cbKpiCost').textContent   = t.total_cost_usd   != null ? `$${(+t.total_cost_usd).toFixed(4)}`   : '—';
    document.getElementById('cbKpiEnergy').textContent = t.total_energy_kwh != null ? `${(+t.total_energy_kwh).toFixed(2)} kWh` : '—';
    document.getElementById('cbKpiJobs').textContent   = t.job_count        != null ? t.job_count : '—';
    document.getElementById('cbKpiSaved').textContent  = t.total_saved_usd  != null ? `$${(+t.total_saved_usd).toFixed(4)}`  : '—';
    document.getElementById('cbKpiPeriod').textContent = d.period_start !== 'all-time'
      ? `${d.period_start.slice(0,10)} → ${d.period_end.slice(0,10)}`
      : 'all time';

    const rows = d.breakdown || [];
    _cbAllRows = rows;

    // Table
    document.getElementById('cbTableMeta').textContent = `${rows.length} group${rows.length!==1?'s':''}`;
    const tbody = document.getElementById('cbBreakdownBody');
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:24px;">No data for the selected period — submit jobs with cost center tags to populate this report.</td></tr>';
    } else {
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td><span style="color:var(--text);font-weight:600;">${r.group||'untagged'}</span></td>
          <td style="text-align:right;color:var(--text2)">${r.job_count}</td>
          <td style="text-align:right;font-family:var(--mono)">${(+r.total_energy_kwh).toFixed(3)}</td>
          <td style="text-align:right;font-family:var(--mono);color:var(--amber)">$${(+r.total_cost_usd).toFixed(4)}</td>
          <td style="text-align:right;font-family:var(--mono);color:var(--muted)">$${(+r.total_naive_cost_usd).toFixed(4)}</td>
          <td style="text-align:right;font-family:var(--mono);color:var(--green)">$${(+r.total_saved_usd).toFixed(4)}</td>
          <td style="text-align:right;font-family:var(--mono);font-size:11px;color:var(--text2)">${(+r.avg_price_per_kwh).toFixed(5)}</td>
        </tr>`).join('');
    }

    // Charts
    renderCbCharts(rows);

  } catch(e) {
    console.error('Chargeback load failed:', e);
    document.getElementById('cbBreakdownBody').innerHTML =
      '<tr><td colspan="7" style="color:var(--red);text-align:center;padding:18px;">Error loading data — check API connection.</td></tr>';
  }
}

function renderCbCharts(rows) {
  const COLORS = ['#00ff88','#00ccff','#ffaa00','#ff4444','#bf7fff','#00e5ff','#76ff03','#ff6d00','#e040fb','#40c4ff'];
  const labels = rows.map(r => r.group || 'untagged');
  const costs  = rows.map(r => +(+r.total_cost_usd).toFixed(4));
  const saved  = rows.map(r => +(+r.total_saved_usd).toFixed(4));
  const colors = rows.map((_,i) => COLORS[i % COLORS.length]);

  // Donut
  const donutCtx = document.getElementById('cbDonutChart').getContext('2d');
  if (cbDonutChart) cbDonutChart.destroy();
  if (!rows.length) {
    document.getElementById('cbDonutEmpty').style.display = 'flex';
    document.getElementById('cbDonutChart').style.display = 'none';
  } else {
    document.getElementById('cbDonutEmpty').style.display = 'none';
    document.getElementById('cbDonutChart').style.display = 'block';
    cbDonutChart = new Chart(donutCtx, {
      type: 'doughnut',
      data: { labels, datasets: [{ data: costs, backgroundColor: colors, borderWidth: 2, borderColor: '#0a0e1a' }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: {
          callbacks: { label: ctx => ` ${ctx.label}: $${ctx.parsed.toFixed(4)}` }
        }},
        cutout: '65%',
      }
    });
    document.getElementById('cbLegend').innerHTML = rows.map((r,i) =>
      `<div style="display:flex;align-items:center;gap:5px;font-size:11px;color:var(--text2);">
        <div style="width:10px;height:10px;border-radius:2px;background:${colors[i]};flex-shrink:0"></div>
        <span>${r.group||'untagged'}</span>
       </div>`).join('');
  }

  // Bar
  const barCtx = document.getElementById('cbBarChart').getContext('2d');
  if (cbBarChart) cbBarChart.destroy();
  if (!rows.length) {
    document.getElementById('cbBarEmpty').style.display = 'block';
    document.getElementById('cbBarChart').style.display = 'none';
  } else {
    document.getElementById('cbBarEmpty').style.display = 'none';
    document.getElementById('cbBarChart').style.display = 'block';
    cbBarChart = new Chart(barCtx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Actual Cost ($)', data: costs, backgroundColor: 'rgba(0,255,136,0.7)', borderRadius: 4 },
          { label: 'Saved ($)',       data: saved, backgroundColor: 'rgba(0,204,255,0.5)', borderRadius: 4 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#a0aec0', font: { size: 11 } } },
                   tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: $${ctx.parsed.y.toFixed(4)}` } } },
        scales: {
          x: { ticks: { color: '#6b7a99', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#6b7a99', font: { size: 11 }, callback: v => '$'+v }, grid: { color: 'rgba(255,255,255,0.06)' } },
        }
      }
    });
  }
}

async function downloadChargebackCSV() {
  const { start, end, group } = getChargebackParams();
  const token = sessionStorage.getItem('hg_access_token') || '';
  let url = `${API}/chargeback/report.csv?group_by=${group}`;
  if (start) url += `&period_start=${encodeURIComponent(start)}&period_end=${encodeURIComponent(end)}`;
  const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` }, credentials: 'include' });
  if (!r.ok) { alert('CSV export failed'); return; }
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `hexagrid_chargeback_${group}_${start.slice(0,7)||'all'}.csv`;
  a.click();
}

// ═══════════════════════════════════════════════════════════════════
//  INCIDENTS
// ═══════════════════════════════════════════════════════════════════
let _incAllHistory = [];

const INC_STATE_STYLES = {
  healthy:   { color: 'var(--green)',  icon: 'fa-circle-check',        bg: 'rgba(0,255,136,0.08)',  border: 'rgba(0,255,136,0.25)' },
  degraded:  { color: 'var(--amber)',  icon: 'fa-triangle-exclamation', bg: 'rgba(255,170,0,0.08)',  border: 'rgba(255,170,0,0.3)' },
  draining:  { color: 'var(--red)',    icon: 'fa-spinner fa-spin',      bg: 'rgba(255,68,68,0.08)',  border: 'rgba(255,68,68,0.3)' },
  drained:   { color: 'var(--red)',    icon: 'fa-circle-xmark',         bg: 'rgba(255,68,68,0.12)',  border: 'rgba(255,68,68,0.4)' },
  recovering:{ color: 'var(--cyan)',   icon: 'fa-rotate',               bg: 'rgba(0,204,255,0.08)',  border: 'rgba(0,204,255,0.25)' },
};

async function loadIncidents() {
  try {
    const [fleet, hist, cfg] = await Promise.all([
      apiFetch('/incidents/fleet'),
      apiFetch('/incidents/history?limit=100'),
      apiFetch('/incidents/config'),
    ]);

    // KPIs
    const s = fleet.summary || {};
    document.getElementById('incKpiTotal').textContent    = s.total_tracked ?? '0';
    document.getElementById('incKpiHealthy').textContent  = s.healthy        ?? '0';
    document.getElementById('incKpiDegraded').textContent = s.degraded       ?? '0';
    document.getElementById('incKpiDrained').textContent  = (s.draining||0) + (s.drained||0);

    // Badge on tab if any non-healthy nodes
    const badNode = (s.degraded||0) + (s.draining||0) + (s.drained||0);
    const tabBtn  = document.getElementById('tab-incidents');
    if (tabBtn) {
      const existing = tabBtn.querySelector('.tab-badge');
      if (existing) existing.remove();
      if (badNode > 0) {
        const badge = document.createElement('span');
        badge.className = `tab-badge ${s.drained||s.draining ? 'critical' : 'warning'}`;
        badge.textContent = badNode;
        tabBtn.appendChild(badge);
      }
    }

    // Fleet grid
    const grid = document.getElementById('incFleetGrid');
    const nodes = fleet.nodes || [];
    if (!nodes.length) {
      grid.innerHTML = '<div style="color:var(--muted);font-size:13px;grid-column:span 4;padding:24px;text-align:center;">No incidents recorded yet — all nodes implicitly healthy.<br><span style="font-size:11px;">Nodes appear here when their first alert fires.</span></div>';
    } else {
      grid.innerHTML = nodes.map(n => {
        const st = INC_STATE_STYLES[n.state] || INC_STATE_STYLES.healthy;
        const ts = n.updated_at ? new Date(n.updated_at).toLocaleString() : '—';
        return `
          <div style="background:${st.bg};border:1px solid ${st.border};border-radius:10px;padding:14px 16px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
              <i class="fa-solid ${st.icon}" style="color:${st.color};font-size:14px;"></i>
              <span style="font-weight:700;color:var(--text);font-size:13px;">${n.node_id}</span>
              <span style="margin-left:auto;font-size:10px;font-weight:700;color:${st.color};text-transform:uppercase;">${n.state}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-bottom:10px;line-height:1.5;">${n.reason||''}<br>${ts}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
              ${n.state !== 'healthy' && n.state !== 'draining' ?
                `<button onclick="incAction('${n.node_id}','drain')" style="font-size:10px;padding:4px 8px;background:rgba(255,68,68,0.15);border:1px solid rgba(255,68,68,0.3);color:var(--red);border-radius:5px;cursor:pointer;">Drain</button>` : ''}
              ${n.state === 'drained' || n.state === 'degraded' ?
                `<button onclick="incAction('${n.node_id}','recover')" style="font-size:10px;padding:4px 8px;background:rgba(0,204,255,0.1);border:1px solid rgba(0,204,255,0.3);color:var(--cyan);border-radius:5px;cursor:pointer;">Recover</button>` : ''}
              ${n.state === 'recovering' ?
                `<button onclick="incAction('${n.node_id}','clear')" style="font-size:10px;padding:4px 8px;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:var(--green);border-radius:5px;cursor:pointer;">Clear ✓</button>` : ''}
            </div>
          </div>`;
      }).join('');
    }

    // Config
    const th = cfg.thresholds || {};
    document.getElementById('incDrainTemp').value   = th.drain_threshold_temp_c   || 91;
    document.getElementById('incDegradeTemp').value = th.degrade_threshold_temp_c || 85;
    document.getElementById('incHealthMin').value   = th.degrade_health_score      || 60;
    const notif = cfg.notifications || {};
    document.getElementById('incNotifStatus').innerHTML = [
      `<div><i class="fa-solid fa-pager" style="color:${notif.pagerduty?'var(--green)':'var(--muted)'}"></i> PagerDuty: <span style="color:${notif.pagerduty?'var(--green)':'var(--muted)'}">${notif.pagerduty?'Configured':'Not configured'}</span></div>`,
      `<div><i class="fa-solid fa-bolt" style="color:${notif.opsgenie?'var(--green)':'var(--muted)'}"></i> OpsGenie: <span style="color:${notif.opsgenie?'var(--green)':'var(--muted)'}">${notif.opsgenie?'Configured':'Not configured'}</span></div>`,
      `<div><i class="fa-solid fa-bell" style="color:var(--cyan)"></i> Alert Manager: <span style="color:var(--cyan)">Always active</span></div>`,
    ].join('');

    // History
    _incAllHistory = hist.history || [];
    renderIncidentHistory(_incAllHistory);

  } catch(e) {
    console.error('Incidents load failed:', e);
  }
}

function renderIncidentHistory(rows) {
  const tbody = document.getElementById('incHistoryBody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px;">No incidents recorded.</td></tr>';
    return;
  }
  const STATE_COLORS = { healthy:'var(--green)', degraded:'var(--amber)', draining:'var(--red)', drained:'var(--red)', recovering:'var(--cyan)' };
  tbody.innerHTML = rows.map(r => {
    const fc = STATE_COLORS[r.from_state] || 'var(--muted)';
    const tc = STATE_COLORS[r.to_state]   || 'var(--text)';
    const ts = r.ts ? new Date(r.ts).toLocaleString() : '—';
    return `<tr>
      <td style="font-size:11px;color:var(--muted);font-family:var(--mono)">${ts}</td>
      <td style="font-weight:600;color:var(--text)">${r.node_id}</td>
      <td>
        <span style="color:${fc};font-size:11px;">${r.from_state}</span>
        <span style="color:var(--muted);margin:0 5px;">→</span>
        <span style="color:${tc};font-weight:600;font-size:11px;">${r.to_state}</span>
      </td>
      <td style="font-size:11px;color:var(--text2)">${r.trigger||'—'}</td>
      <td style="font-size:11px;color:var(--muted);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.detail||''}</td>
      <td style="text-align:right;">
        ${r.to_state === 'degraded' ? `<button onclick="incAction('${r.node_id}','drain')" style="font-size:10px;padding:3px 7px;background:rgba(255,68,68,0.12);border:1px solid rgba(255,68,68,0.25);color:var(--red);border-radius:4px;cursor:pointer;">Drain</button>` : ''}
        ${r.to_state === 'drained'  ? `<button onclick="incAction('${r.node_id}','recover')" style="font-size:10px;padding:3px 7px;background:rgba(0,204,255,0.1);border:1px solid rgba(0,204,255,0.25);color:var(--cyan);border-radius:4px;cursor:pointer;">Recover</button>` : ''}
      </td>
    </tr>`;
  }).join('');
}

function filterIncidentHistory() {
  const q = document.getElementById('incNodeFilter').value.toLowerCase();
  renderIncidentHistory(q ? _incAllHistory.filter(r => r.node_id.toLowerCase().includes(q)) : _incAllHistory);
}

async function incAction(nodeId, action) {
  try {
    await apiFetch(`/incidents/${nodeId}/${action}`, { method: 'POST' });
    await loadIncidents();
  } catch(e) {
    alert(`Action failed: ${e.message}`);
  }
}

async function saveIncidentConfig() {
  const body = {
    drain_threshold_temp_c:   parseInt(document.getElementById('incDrainTemp').value)   || null,
    degrade_threshold_temp_c: parseInt(document.getElementById('incDegradeTemp').value) || null,
    degrade_health_score:     parseInt(document.getElementById('incHealthMin').value)   || null,
  };
  const el = document.getElementById('incConfigStatus');
  try {
    await apiFetch('/incidents/config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    el.textContent = '✓ Saved'; el.style.color = 'var(--green)';
    setTimeout(() => el.textContent = '', 3000);
  } catch(e) {
    el.textContent = '✗ Failed'; el.style.color = 'var(--red)';
  }
}

// ── Hook into switchTab ────────────────────────────────────────────────────
const _origSwitchTab_89 = switchTab;
switchTab = function(name, btn) {
  _origSwitchTab_89(name, btn);
  if (name === 'chargeback') setTimeout(loadChargeback, 100);
  if (name === 'incidents')  setTimeout(loadIncidents,  100);
};

// Set default month to current month on load
(function() {
  const now = new Date();
  const m = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}`;
  const el = document.getElementById('cbMonth');
  if (el) el.value = m;
})();
'''

# ══════════════════════════════════════════════════════════════════════════════
#  APPLY PATCHES
# ══════════════════════════════════════════════════════════════════════════════

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

print("\n  ➤  Patching tabs bar")
patch(TAB_OLD, TAB_NEW, "Add Chargeback + Incidents tab buttons")

# Insert page divs before closing </body> or before <script>
print("\n  ➤  Inserting page HTML")
SCRIPT_ANCHOR = '\n<script>'
if SCRIPT_ANCHOR in html:
    # Insert before the first <script> tag
    html = html.replace(SCRIPT_ANCHOR, CHARGEBACK_HTML + INCIDENTS_HTML + SCRIPT_ANCHOR, 1)
    print("  ✓  Inserted Chargeback + Incidents page HTML")
else:
    print("  ⚠  Could not find <script> anchor — inserting before </body>")
    html = html.replace('</body>', CHARGEBACK_HTML + INCIDENTS_HTML + '</body>', 1)

# Insert JS before </script> (last one)
print("\n  ➤  Inserting JavaScript")
last_script_close = html.rfind('</script>')
if last_script_close != -1:
    html = html[:last_script_close] + JS + '\n' + html[last_script_close:]
    print("  ✓  Inserted Chargeback + Incidents JavaScript")
else:
    print("  ⚠  Could not find </script> — appending JS at end")
    html += f'<script>{JS}</script>'

# Write
INDEX.write_text(html)
print(f"\n  ✓  Written: {INDEX}")
print("""
  ═══════════════════════════════════════════════
  Dashboard patched — reload http://localhost:8000
  Two new tabs: Chargeback | Incidents
  ═══════════════════════════════════════════════
""")
