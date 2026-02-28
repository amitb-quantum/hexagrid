#!/usr/bin/env python3
"""
HexaGrid Dashboard — Patch: TPU Monitoring section in Fleet tab
===============================================================
Adds a TPU fleet monitoring section to the Fleet tab, between the
GPU telemetry block and the existing Fleet KPI bar.
Also adds a TPU help entry to the help drawer.

Run from anywhere:  python patch_dashboard_tpu.py
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
#  1. TPU SECTION HTML — inserted between GPU block and Fleet KPI bar
# ══════════════════════════════════════════════════════════════════════════════

TPU_HTML = '''
  <div class="col-12" style="border-bottom:1px solid var(--border);margin:8px 0 20px"></div>
  <!-- ══ TPU TELEMETRY ══════════════════════════════════════════════════════ -->

  <!-- TPU header row -->
  <div class="col-12" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px;">
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
        &#9679; loading
      </span>
      <button class="btn-secondary" onclick="loadTPUTelemetry()" style="padding:4px 12px;font-size:12px;">
        <i class="fa-solid fa-rotate"></i>
      </button>
    </div>
  </div>

  <!-- TPU KPI strip -->
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-server" style="color:var(--purple)"></i>TPU Nodes</div>
    <div class="kpi-value" style="color:var(--purple)" id="tpu-kpi-nodes">--</div>
    <div class="kpi-delta neu" id="tpu-kpi-nodes-sub">loading...</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-microchip" style="color:var(--cyan)"></i>Total Chips</div>
    <div class="kpi-value cyan" id="tpu-kpi-chips">--</div>
    <div class="kpi-delta neu" id="tpu-kpi-chips-sub">loading...</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-gauge-high" style="color:var(--green)"></i>Avg Utilisation</div>
    <div class="kpi-value green" id="tpu-kpi-util">--</div>
    <div class="kpi-delta neu" id="tpu-kpi-util-sub">loading...</div>
  </div>
  <div class="card col-3">
    <div class="card-title"><i class="fa-solid fa-triangle-exclamation" style="color:var(--amber)"></i>Runtime Errors</div>
    <div class="kpi-value" style="color:var(--amber)" id="tpu-kpi-errors">--</div>
    <div class="kpi-delta neu" id="tpu-kpi-errors-sub">since last poll</div>
  </div>

  <!-- TPU node grid -->
  <div class="card col-12">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div class="card-title" style="margin-bottom:0;">
        <i class="fa-solid fa-chart-network" style="color:var(--purple)"></i>TPU Node Grid
        <span style="font-size:11px;font-weight:400;color:var(--muted);margin-left:8px;">
          one card per node &middot; GCP &amp; AWS Trainium/Inferentia
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
        Waiting for TPU telemetry &mdash; deploy tpu_collector_agent.py on your GCP or AWS nodes.
      </div>
    </div>
  </div>

  <div class="col-12" style="border-bottom:1px solid var(--border);margin:8px 0 20px"></div>
  <!-- ══ END TPU TELEMETRY ═════════════════════════════════════════════════ -->
'''

# Anchor: the existing divider that separates GPU telemetry from Fleet KPIs
GPU_END_DIVIDER = '  <div class="col-12" style="border-bottom:1px solid var(--border);margin:8px 0 20px"></div>\n  <!-- ══ END LIVE GPU TELEMETRY'

patch(
    GPU_END_DIVIDER,
    TPU_HTML + '\n  <!-- ══ END LIVE GPU TELEMETRY',
    "Insert TPU section into Fleet tab"
)

# ══════════════════════════════════════════════════════════════════════════════
#  2. JAVASCRIPT
# ══════════════════════════════════════════════════════════════════════════════

TPU_JS = '''
// ═══════════════════════════════════════════════════════════════════
//  TPU TELEMETRY
// ═══════════════════════════════════════════════════════════════════

const TPU_PROVIDER_COLORS = {
  gcp:          { accent: 'var(--purple)', label: 'GCP Cloud TPU' },
  aws_trainium: { accent: 'var(--cyan)',   label: 'AWS Trainium'  },
  aws_inferentia:{ accent: 'var(--amber)', label: 'AWS Inferentia' },
  aws:          { accent: 'var(--cyan)',   label: 'AWS Neuron'    },
};

function _tpuAccent(node) {
  if (node.provider === 'gcp') return 'var(--purple)';
  const t = (node.accelerator_type || '').toLowerCase();
  if (t.includes('inf')) return 'var(--amber)';
  return 'var(--cyan)';
}

function _tpuLabel(node) {
  if (node.provider === 'gcp') return 'GCP Cloud TPU';
  const t = (node.accelerator_type || '').toLowerCase();
  if (t.includes('inf')) return 'AWS Inferentia';
  return 'AWS Trainium';
}

async function loadTPUTelemetry() {
  const provider = document.getElementById('tpu-provider-filter')?.value || '';
  let url = '/telemetry/tpu/fleet?stale_secs=120';
  if (provider) url += `&provider=${provider}`;

  const badge = document.getElementById('tpu-age-badge');

  try {
    const d = await apiFetch(url);

    // KPIs
    document.getElementById('tpu-kpi-nodes').textContent  = d.node_count  ?? '--';
    document.getElementById('tpu-kpi-chips').textContent  = d.total_chips  ?? '--';
    document.getElementById('tpu-kpi-util').textContent   =
      d.fleet_util_pct != null ? `${d.fleet_util_pct.toFixed(1)}%` : '--';

    const totalErrors = (d.nodes || []).reduce((s, n) => s + (n.total_errors || 0), 0);
    document.getElementById('tpu-kpi-errors').textContent = totalErrors;
    document.getElementById('tpu-kpi-errors-sub').textContent =
      totalErrors > 0 ? 'runtime errors detected' : 'no errors';
    document.getElementById('tpu-kpi-errors').style.color =
      totalErrors > 0 ? 'var(--red)' : 'var(--green)';

    const providers = (d.providers || []).join(', ') || 'none';
    document.getElementById('tpu-kpi-nodes-sub').textContent  = `providers: ${providers}`;
    document.getElementById('tpu-kpi-chips-sub').textContent  =
      d.node_count > 0 ? `${Math.round(d.total_chips / d.node_count)} chips/node avg` : 'no nodes';
    document.getElementById('tpu-kpi-util-sub').textContent  = 'matrix unit utilisation';

    // Age badge
    if (badge) {
      badge.textContent = `● ${new Date().toLocaleTimeString()}`;
      badge.style.color = d.node_count > 0 ? 'var(--purple)' : 'var(--muted)';
    }

    // Node grid
    const grid  = document.getElementById('tpu-node-grid');
    const nodes = d.nodes || [];

    if (!nodes.length) {
      grid.innerHTML = `
        <div style="color:var(--muted);font-size:13px;padding:20px 0;grid-column:span 6;">
          <i class="fa-solid fa-satellite-dish" style="margin-right:8px;color:var(--purple);"></i>
          Waiting for TPU telemetry &mdash; deploy tpu_collector_agent.py on your GCP or AWS nodes.
        </div>`;
      return;
    }

    grid.innerHTML = nodes.map(n => {
      const accent  = _tpuAccent(n);
      const label   = _tpuLabel(n);
      const util    = n.avg_util_pct != null ? `${n.avg_util_pct.toFixed(1)}%` : '—';
      const hbm     = n.avg_hbm_used_pct != null ? `${n.avg_hbm_used_pct.toFixed(1)}%` : '—';
      const stale   = n.stale;
      const cardBg  = stale
        ? 'rgba(255,255,255,0.03)'
        : `rgba(${accent === 'var(--purple)' ? '191,127,255' : accent === 'var(--cyan)' ? '0,204,255' : '255,170,0'},0.06)`;
      const cardBorder = stale
        ? 'rgba(255,255,255,0.08)'
        : `rgba(${accent === 'var(--purple)' ? '191,127,255' : accent === 'var(--cyan)' ? '0,204,255' : '255,170,0'},0.25)`;
      const dotColor = stale ? 'var(--muted)' : accent;
      const lastSeen = n.last_seen ? new Date(n.last_seen).toLocaleTimeString() : '—';

      // Utilisation bar width
      const barW = n.avg_util_pct != null ? Math.min(100, n.avg_util_pct) : 0;
      const barColor = barW > 90 ? 'var(--red)' : barW > 70 ? 'var(--amber)' : accent;

      return `
        <div style="background:${cardBg};border:1px solid ${cardBorder};border-radius:12px;padding:14px 16px;">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px;">
            <span style="width:8px;height:8px;border-radius:50%;background:${dotColor};display:inline-block;flex-shrink:0;"></span>
            <span style="font-weight:700;color:var(--text);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">${n.node_id}</span>
            <span style="font-size:10px;font-weight:600;color:${accent};white-space:nowrap;">${label}</span>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:8px;">
            ${n.accelerator_type || '—'} &middot; ${n.chip_count || '?'} chips &middot; ${n.zone || '—'}
          </div>

          <!-- Util bar -->
          <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-bottom:3px;">
              <span>Matrix Util</span><span style="color:${barColor};font-weight:600;">${util}</span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:3px;height:4px;overflow:hidden;">
              <div style="width:${barW}%;height:100%;background:${barColor};border-radius:3px;transition:width 0.5s;"></div>
            </div>
          </div>

          <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:8px;">
            <span style="color:var(--muted);">HBM</span>
            <span style="color:var(--text2);font-family:var(--mono);">${hbm}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;">
            <span style="color:var(--muted);">Errors</span>
            <span style="color:${(n.total_errors||0) > 0 ? 'var(--red)' : 'var(--green)'};font-family:var(--mono);">${n.total_errors || 0}</span>
          </div>
          <div style="font-size:10px;color:var(--muted);margin-top:8px;text-align:right;">
            ${stale ? '<span style="color:var(--amber);">⚠ stale</span>' : `● ${lastSeen}`}
          </div>
        </div>`;
    }).join('');

  } catch(e) {
    if (badge) { badge.textContent = '● error'; badge.style.color = 'var(--red)'; }
    console.warn('TPU telemetry load failed:', e);
  }
}

// Hook into fleet tab load
const _origSwitchTab_tpu = switchTab;
switchTab = function(name, btn) {
  _origSwitchTab_tpu(name, btn);
  if (name === 'fleet') setTimeout(loadTPUTelemetry, 300);
};
'''

# Insert JS before closing </script>
last_script_close = html.rfind('</script>')
if last_script_close != -1:
    html = html[:last_script_close] + TPU_JS + '\n' + html[last_script_close:]
    print("  ✓  Inserted TPU JavaScript")
else:
    html += f'<script>{TPU_JS}</script>'
    print("  ✓  Appended TPU JavaScript")

# ══════════════════════════════════════════════════════════════════════════════
#  3. HELP DRAWER BUTTON
# ══════════════════════════════════════════════════════════════════════════════

patch(
    '    <button class="help-tab-btn" onclick="showHelpTab(\'fleet\',this)"><i class="fa-solid fa-network-wired"></i> Fleet</button>',
    '    <button class="help-tab-btn" onclick="showHelpTab(\'fleet\',this)"><i class="fa-solid fa-network-wired"></i> Fleet</button>\n    <button class="help-tab-btn" onclick="showHelpTab(\'tpu\',this)"><i class="fa-solid fa-microchip"></i> TPU</button>',
    "Add TPU help drawer button"
)

# ══════════════════════════════════════════════════════════════════════════════
#  4. HELP CONTENT
# ══════════════════════════════════════════════════════════════════════════════

TPU_HELP = '''  ,tpu: {
    label: 'TPU Monitoring',
    html: `
      <div class="help-section">
        <h3><i class="fa-solid fa-microchip"></i>What is TPU Monitoring?</h3>
        <p>HexaGrid monitors Google Cloud TPU (v4, v5e, v5p) and AWS Trainium/Inferentia nodes alongside your GPU fleet. The same chargeback tagging, alerting, and incident response pipeline applies to TPU workloads — no separate tooling required.</p>
        <p>TPU metrics appear in the Fleet tab, grouped by provider. GPU and TPU nodes are visible side by side so mixed fleets have a single pane of glass.</p>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-chart-bar"></i>Metrics Collected</h3>
        <div class="help-metric"><div class="hm-name">Matrix Utilisation %</div><div class="hm-desc">Percentage of time the matrix multiply unit (MXU on TPU / NeuronCore on Trainium) is active. The primary throughput indicator — equivalent to SM utilisation on GPUs.</div></div>
        <div class="help-metric"><div class="hm-name">HBM Memory %</div><div class="hm-desc">High-bandwidth memory used as a percentage of total. High HBM usage with low matrix utilisation usually indicates memory-bound workloads — consider reducing batch size or using model parallelism.</div></div>
        <div class="help-metric"><div class="hm-name">ICI Bandwidth (GCP)</div><div class="hm-desc">Inter-Chip Interconnect bandwidth in GB/s. High ICI usage means chips are exchanging tensors frequently — expected in large model parallel runs, but saturation causes slowdowns.</div></div>
        <div class="help-metric"><div class="hm-name">Runtime Errors</div><div class="hm-desc">Cumulative NeuronCore runtime errors (AWS) or Cloud TPU reported errors (GCP). Any non-zero value warrants investigation — check neuron-monitor logs or GCP Cloud Logging.</div></div>
        <div class="help-tip"><strong>Note:</strong> Thermal data and per-chip power draw are not exposed by either GCP or AWS — these are managed internally by the cloud providers. ECC equivalents are also not surfaced at the API level.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-brands fa-google"></i>GCP Setup</h3>
        <p>Deploy <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">tpu_collector_agent.py</code> on each TPU node (or as a GKE sidecar). Metrics are pulled from the Cloud Monitoring API — the node must have the <strong style="color:var(--cyan)">monitoring.viewer</strong> IAM role on its service account.</p>
        <div class="help-metric"><div class="hm-name">GCP_PROJECT_ID</div><div class="hm-desc">Your GCP project ID — e.g. my-ml-project-123.</div></div>
        <div class="help-metric"><div class="hm-name">GCP_ZONE</div><div class="hm-desc">Zone of the TPU node — e.g. us-central2-b.</div></div>
        <div class="help-metric"><div class="hm-name">GCP_TPU_NODE</div><div class="hm-desc">The TPU node name as shown in Cloud Console.</div></div>
        <div class="help-metric"><div class="hm-name">GCP_ACCELERATOR_TYPE</div><div class="hm-desc">e.g. v5e-8, v4-32, v5p-128. Used for display and chargeback tagging.</div></div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-brands fa-aws"></i>AWS Setup</h3>
        <p>Run <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">neuron-monitor</code> as a background service on each Trainium or Inferentia instance. The collector reads its JSON output on <code style="color:var(--cyan);background:rgba(0,204,255,0.08);padding:1px 5px;border-radius:4px;">localhost:8888</code>.</p>
        <div class="help-metric"><div class="hm-name">Install</div><div class="hm-desc">pip install neuronx-cc aws-neuronx-runtime-lib (Trainium) or pip install neuron-cc (Inferentia). Part of the AWS Neuron SDK.</div></div>
        <div class="help-metric"><div class="hm-name">Start daemon</div><div class="hm-desc">neuron-monitor &amp; — runs in the background, serves JSON on port 8888.</div></div>
        <div class="help-metric"><div class="hm-name">Instance types</div><div class="hm-desc">trn1.2xlarge, trn1.32xlarge (Trainium); inf2.xlarge → inf2.48xlarge (Inferentia2).</div></div>
        <div class="help-tip"><strong>Tip:</strong> Set TPU_PROVIDER=aws explicitly on AWS nodes to skip the auto-detect step and reduce startup time.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-plug-circle-bolt"></i>Quick Start</h3>
        <p>Same three env vars as the GPU collector, plus provider-specific ones:</p>
        <div class="help-metric"><div class="hm-name">HEXAGRID_ENDPOINT</div><div class="hm-desc">URL of your HexaGrid API — e.g. http://hexagrid-host:8000</div></div>
        <div class="help-metric"><div class="hm-name">HEXAGRID_TOKEN</div><div class="hm-desc">Your collector API key from the bootstrap output (hg_...).</div></div>
        <div class="help-metric"><div class="hm-name">CLUSTER_ID</div><div class="hm-desc">Logical cluster name — groups nodes together in the Fleet view.</div></div>
        <div class="help-metric"><div class="hm-name">TPU_PROVIDER</div><div class="hm-desc">gcp or aws. Leave unset for auto-detection.</div></div>
      </div>`
  }

  ,benchmark: {'''

patch(
    '  ,benchmark: {',
    TPU_HELP,
    "Add TPU help content"
)

# ══════════════════════════════════════════════════════════════════════════════
#  WRITE
# ══════════════════════════════════════════════════════════════════════════════

INDEX.write_text(html)
print(f"\n  ✓  Written: {INDEX}")
print("""
  ══════════════════════════════════════════════
  Done — hard refresh the dashboard (Ctrl+Shift+R)

  Changes:
    • TPU Fleet section added to Fleet tab
      (between GPU telemetry and Fleet KPI bar)
    • GCP TPU (purple) and AWS Trainium/Inferentia
      (cyan/amber) node cards with util bars
    • Help drawer: TPU entry added
  ══════════════════════════════════════════════
""")
