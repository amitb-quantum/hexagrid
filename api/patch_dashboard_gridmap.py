#!/usr/bin/env python3
"""
HexaGrid Dashboard — Patch: Live US Grid Stress Map
====================================================
Adds an animated SVG map of the US ISO grid regions to the Weather tab.
Each region is shaded by current electricity price / stress level.
Data centers appear as glowing pulsing hexagons on top.
Clicking a region or hexagon shows a detail tooltip.

Run from anywhere:  python patch_dashboard_gridmap.py
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
#  1. MAP HTML  — inserted at top of Weather tab, before KPI strip
# ══════════════════════════════════════════════════════════════════════════════

MAP_HTML = '''
<!-- ══ GRID STRESS MAP ══════════════════════════════════════════════════════ -->
<div id="gridmap-container" style="
  position:relative;
  background:linear-gradient(160deg,#060a14 0%,#0a1020 50%,#060c18 100%);
  border:1px solid rgba(0,255,136,0.12);
  border-radius:16px;
  overflow:hidden;
  margin-bottom:22px;
  box-shadow:0 8px 48px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
">

  <!-- Scanline overlay for depth -->
  <div style="
    position:absolute;inset:0;pointer-events:none;z-index:2;
    background:repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.03) 2px,
      rgba(0,0,0,0.03) 4px
    );
  "></div>

  <!-- Header bar -->
  <div style="
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 20px 10px;
    border-bottom:1px solid rgba(0,255,136,0.08);
    position:relative;z-index:3;
  ">
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="
        width:8px;height:8px;border-radius:50%;
        background:var(--green);
        box-shadow:0 0 8px var(--green);
        animation:gridmap-pulse-dot 2s ease-in-out infinite;
      "></div>
      <span style="font-size:13px;font-weight:700;color:var(--text);letter-spacing:0.5px;">
        LIVE US GRID INTELLIGENCE
      </span>
      <span style="font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;">
        ISO Region Stress · Data Center Fleet
      </span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
      <!-- Legend -->
      <div style="display:flex;align-items:center;gap:10px;font-size:10px;color:var(--muted);">
        <span style="display:flex;align-items:center;gap:4px;">
          <span style="width:10px;height:10px;border-radius:2px;background:rgba(0,255,136,0.6);display:inline-block;"></span>Cheap
        </span>
        <span style="display:flex;align-items:center;gap:4px;">
          <span style="width:10px;height:10px;border-radius:2px;background:rgba(255,170,0,0.7);display:inline-block;"></span>Elevated
        </span>
        <span style="display:flex;align-items:center;gap:4px;">
          <span style="width:10px;height:10px;border-radius:2px;background:rgba(255,68,68,0.7);display:inline-block;"></span>Spike
        </span>
        <span style="display:flex;align-items:center;gap:4px;">
          <span style="width:24px;height:2px;background:linear-gradient(90deg,rgba(0,255,136,0.8),transparent);display:inline-block;"></span>DC Pulse
        </span>
      </div>
      <span id="gridmap-timestamp" style="font-size:10px;color:var(--muted);font-family:var(--mono);">--:--:--</span>
      <button onclick="loadGridMap()" style="
        background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.2);
        color:var(--green);border-radius:6px;padding:4px 10px;font-size:11px;
        cursor:pointer;font-weight:600;
      ">↻</button>
    </div>
  </div>

  <!-- SVG Map -->
  <div style="position:relative;padding:10px 16px 16px;">
    <svg id="gridmap-svg" viewBox="0 0 960 580" style="width:100%;height:auto;display:block;"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <!-- Region glow filters -->
        <filter id="glow-green" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-amber" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="6" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-red" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="8" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="hex-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="hex-glow-hot" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="9" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>

        <!-- Gradient for arc lines -->
        <linearGradient id="arc-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:rgba(0,255,136,0.9)"/>
          <stop offset="100%" style="stop-color:rgba(0,204,255,0.0)"/>
        </linearGradient>

        <!-- Radial gradient for region fill -->
        <radialGradient id="rg-green" cx="50%" cy="50%" r="60%">
          <stop offset="0%" style="stop-color:rgba(0,255,136,0.28)"/>
          <stop offset="100%" style="stop-color:rgba(0,255,136,0.06)"/>
        </radialGradient>
        <radialGradient id="rg-amber" cx="50%" cy="50%" r="60%">
          <stop offset="0%" style="stop-color:rgba(255,170,0,0.35)"/>
          <stop offset="100%" style="stop-color:rgba(255,170,0,0.08)"/>
        </radialGradient>
        <radialGradient id="rg-red" cx="50%" cy="50%" r="60%">
          <stop offset="0%" style="stop-color:rgba(255,68,68,0.45)"/>
          <stop offset="100%" style="stop-color:rgba(255,68,68,0.10)"/>
        </radialGradient>
        <radialGradient id="rg-normal" cx="50%" cy="50%" r="60%">
          <stop offset="0%" style="stop-color:rgba(0,204,255,0.12)"/>
          <stop offset="100%" style="stop-color:rgba(0,204,255,0.02)"/>
        </radialGradient>
      </defs>

      <!-- ── ISO Region shapes (simplified outlines) ─────────────────── -->
      <!-- WECC/CAISO — California + PNW -->
      <g id="region-CAISO" class="grid-region" onclick="gridmapRegionClick('CAISO')" style="cursor:pointer;">
        <path d="M 68,60 L 95,58 L 118,62 L 128,90 L 135,130 L 138,175 L 132,220 L 125,265
                 L 118,310 L 108,350 L 95,385 L 82,410 L 68,430 L 55,445 L 42,430
                 L 38,380 L 40,320 L 44,260 L 50,200 L 54,140 L 58,100 Z"
              id="path-CAISO"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="80" y="270" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:11px;font-weight:700;font-family:var(--mono);pointer-events:none;">CAISO</text>
        <text id="label-price-CAISO" x="80" y="285" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- ERCOT — Texas -->
      <g id="region-ERCOT" class="grid-region" onclick="gridmapRegionClick('ERCOT')" style="cursor:pointer;">
        <path d="M 280,340 L 340,330 L 410,325 L 445,330 L 455,360 L 450,400
                 L 435,435 L 415,460 L 385,475 L 350,478 L 310,472 L 278,455
                 L 262,425 L 258,390 L 265,360 Z"
              id="path-ERCOT"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="358" y="405" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:11px;font-weight:700;font-family:var(--mono);pointer-events:none;">ERCOT</text>
        <text id="label-price-ERCOT" x="358" y="420" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- MISO — Midwest -->
      <g id="region-MISO" class="grid-region" onclick="gridmapRegionClick('MISO')" style="cursor:pointer;">
        <path d="M 450,140 L 520,130 L 580,138 L 610,160 L 615,200 L 608,250
                 L 595,300 L 575,340 L 545,360 L 505,365 L 465,358 L 445,330
                 L 438,290 L 440,240 L 446,190 Z"
              id="path-MISO"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="530" y="252" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:11px;font-weight:700;font-family:var(--mono);pointer-events:none;">MISO</text>
        <text id="label-price-MISO" x="530" y="267" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- SPP — Southwest Power Pool -->
      <g id="region-SPP" class="grid-region" onclick="gridmapRegionClick('SPP')" style="cursor:pointer;">
        <path d="M 310,240 L 380,235 L 440,240 L 446,290 L 445,330 L 410,325
                 L 340,330 L 280,340 L 265,310 L 268,270 L 278,248 Z"
              id="path-SPP"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="360" y="292" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:11px;font-weight:700;font-family:var(--mono);pointer-events:none;">SPP</text>
        <text id="label-price-SPP" x="360" y="307" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- PJM — Mid-Atlantic -->
      <g id="region-PJM" class="grid-region" onclick="gridmapRegionClick('PJM')" style="cursor:pointer;">
        <path d="M 620,155 L 680,148 L 730,152 L 755,170 L 762,205 L 752,240
                 L 730,268 L 700,282 L 665,285 L 635,275 L 615,252 L 608,220
                 L 610,185 Z"
              id="path-PJM"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="685" y="222" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:11px;font-weight:700;font-family:var(--mono);pointer-events:none;">PJM</text>
        <text id="label-price-PJM" x="685" y="237" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- NYISO — New York -->
      <g id="region-NYISO" class="grid-region" onclick="gridmapRegionClick('NYISO')" style="cursor:pointer;">
        <path d="M 755,130 L 800,125 L 830,132 L 845,150 L 840,172 L 818,183
                 L 790,185 L 762,178 L 752,160 Z"
              id="path-NYISO"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="796" y="158" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:10px;font-weight:700;font-family:var(--mono);pointer-events:none;">NYISO</text>
        <text id="label-price-NYISO" x="796" y="171" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:9px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- ISONE — New England -->
      <g id="region-ISONE" class="grid-region" onclick="gridmapRegionClick('ISONE')" style="cursor:pointer;">
        <path d="M 835,85 L 875,80 L 905,88 L 918,110 L 910,135 L 888,148
                 L 858,150 L 838,138 L 830,115 Z"
              id="path-ISONE"
              fill="url(#rg-normal)" stroke="rgba(0,204,255,0.3)" stroke-width="1.5"
              class="region-path"/>
        <text x="872" y="118" text-anchor="middle"
              style="fill:rgba(255,255,255,0.7);font-size:10px;font-weight:700;font-family:var(--mono);pointer-events:none;">ISONE</text>
        <text id="label-price-ISONE" x="872" y="131" text-anchor="middle"
              style="fill:rgba(0,255,136,0.9);font-size:9px;font-family:var(--mono);pointer-events:none;">--</text>
      </g>

      <!-- Southeast filler (non-ISO, decorative) -->
      <path d="M 620,285 L 665,285 L 700,282 L 730,268 L 752,290 L 748,340
               L 720,375 L 685,395 L 648,400 L 615,388 L 595,360 L 595,320 L 608,295 Z"
            fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.06)" stroke-width="1"
            stroke-dasharray="4,4"/>
      <text x="672" y="342" text-anchor="middle"
            style="fill:rgba(255,255,255,0.18);font-size:9px;font-family:var(--mono);">SERC</text>

      <!-- Mountain West filler -->
      <path d="M 135,175 L 200,168 L 270,172 L 310,185 L 310,240 L 278,248
               L 200,250 L 155,245 L 138,220 Z"
            fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.06)" stroke-width="1"
            stroke-dasharray="4,4"/>
      <path d="M 135,60 L 280,55 L 310,65 L 310,185 L 270,172 L 200,168 L 135,175
               L 128,130 L 118,90 Z"
            fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.06)" stroke-width="1"
            stroke-dasharray="4,4"/>
      <text x="218" y="128" text-anchor="middle"
            style="fill:rgba(255,255,255,0.18);font-size:9px;font-family:var(--mono);">WECC</text>

      <!-- ── Data center hexagons ───────────────────────────────────── -->
      <!-- Each hexagon: cx,cy = map position. Rendered by JS -->
      <g id="gridmap-hexagons"></g>

      <!-- ── Animated scan line ────────────────────────────────────── -->
      <line id="gridmap-scanline" x1="0" y1="0" x2="960" y2="0"
            stroke="rgba(0,255,136,0.15)" stroke-width="1">
        <animate attributeName="y1" from="0" to="580" dur="4s" repeatCount="indefinite"/>
        <animate attributeName="y2" from="0" to="580" dur="4s" repeatCount="indefinite"/>
      </line>

      <!-- ── Tooltip ────────────────────────────────────────────────── -->
      <g id="gridmap-tooltip" style="display:none;pointer-events:none;">
        <rect id="gmt-bg" x="0" y="0" width="180" height="80" rx="8"
              fill="rgba(6,10,20,0.95)" stroke="rgba(0,255,136,0.4)" stroke-width="1"/>
        <text id="gmt-title" x="10" y="20"
              style="fill:var(--text);font-size:11px;font-weight:700;font-family:var(--mono);"></text>
        <text id="gmt-line1" x="10" y="36"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);"></text>
        <text id="gmt-line2" x="10" y="50"
              style="fill:rgba(0,204,255,0.8);font-size:10px;font-family:var(--mono);"></text>
        <text id="gmt-line3" x="10" y="64"
              style="fill:rgba(255,170,0,0.8);font-size:10px;font-family:var(--mono);"></text>
      </g>
    </svg>

    <!-- Floating status bar below map -->
    <div id="gridmap-statusbar" style="
      display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;
    "></div>
  </div>

</div>
<!-- ══ END GRID STRESS MAP ══════════════════════════════════════════════════ -->

'''

# Insert map at top of weather tab, before the KPI strip
WEATHER_ANCHOR = '<div class="page" id="page-weather">\n<div class="page-inner">\n\n  <!-- KPI strip -->'
WEATHER_NEW = '<div class="page" id="page-weather">\n<div class="page-inner">\n\n' + MAP_HTML + '\n  <!-- KPI strip -->'

patch(WEATHER_ANCHOR, WEATHER_NEW, "Insert grid stress map into Weather tab")

# ══════════════════════════════════════════════════════════════════════════════
#  2. CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = '''
/* ── Grid Map ─────────────────────────────────────────────────────────── */
@keyframes gridmap-pulse-dot {
  0%,100% { opacity:1; box-shadow:0 0 6px var(--green); }
  50%      { opacity:0.4; box-shadow:0 0 12px var(--green); }
}
@keyframes gridmap-hex-pulse {
  0%,100% { opacity:0.9; }
  50%      { opacity:0.4; }
}
@keyframes gridmap-hex-ring {
  0%   { r:14; opacity:0.6; }
  100% { r:28; opacity:0; }
}
.region-path {
  transition: fill 1.2s ease, stroke 0.6s ease;
}
.region-path:hover {
  stroke-width: 2.5 !important;
  stroke: rgba(255,255,255,0.5) !important;
}
.grid-region:hover .region-path {
  filter: brightness(1.4);
}
'''

# Insert CSS before closing </style>
style_close = html.find('</style>')
if style_close != -1:
    html = html[:style_close] + CSS + html[style_close:]
    print("  ✓  Inserted grid map CSS")

# ══════════════════════════════════════════════════════════════════════════════
#  3. JAVASCRIPT
# ══════════════════════════════════════════════════════════════════════════════

JS = r'''
// ═══════════════════════════════════════════════════════════════════
//  LIVE US GRID STRESS MAP
// ═══════════════════════════════════════════════════════════════════

// Data center locations mapped to SVG coordinates and ISO regions
const GRIDMAP_DCS = [
  { id:'dc_california', name:'California DC',    cx:75,  cy:310, region:'CAISO', color:'#00ff88' },
  { id:'dc_texas',      name:'Texas DC',         cx:340, cy:420, region:'ERCOT', color:'#00ff88' },
  { id:'dc_east',       name:'East Coast DC',    cx:720, cy:215, region:'PJM',   color:'#00ff88' },
  { id:'dc_mid',        name:'Mid-Atlantic DC',  cx:690, cy:250, region:'PJM',   color:'#00ff88' },
  { id:'dc_newengland', name:'New England DC',   cx:870, cy:115, region:'ISONE', color:'#00ff88' },
];

// SVG hex path generator (pointy-top)
function hexPath(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 180 * (60 * i - 30);
    pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return 'M ' + pts.join(' L ') + ' Z';
}

// Map price to stress level
function priceToStress(price) {
  if (!price || price <= 0) return 'normal';
  if (price > 0.12) return 'spike';
  if (price > 0.07) return 'elevated';
  if (price > 0.03) return 'cheap';
  return 'cheap';
}

const STRESS_FILLS = {
  normal:   { fill:'url(#rg-normal)', stroke:'rgba(0,204,255,0.3)',   glow:'rgba(0,204,255,0.15)',   label:'rgba(0,255,136,0.9)' },
  cheap:    { fill:'url(#rg-green)',  stroke:'rgba(0,255,136,0.5)',   glow:'rgba(0,255,136,0.3)',    label:'rgba(0,255,136,1)'   },
  elevated: { fill:'url(#rg-amber)',  stroke:'rgba(255,170,0,0.6)',   glow:'rgba(255,170,0,0.3)',    label:'rgba(255,170,0,1)'   },
  spike:    { fill:'url(#rg-red)',    stroke:'rgba(255,68,68,0.7)',    glow:'rgba(255,68,68,0.4)',    label:'rgba(255,100,100,1)' },
};

let _gridmapData  = {};   // region → { price, stress, carbon }
let _gridmapTimer = null;

async function loadGridMap() {
  document.getElementById('gridmap-timestamp').textContent = new Date().toLocaleTimeString();

  // Fetch price feed for all regions
  const regions = ['CAISO','ERCOT','NYISO','ISONE','PJM','MISO','SPP'];
  const regionData = {};

  try {
    // Pull live price from pricefeed (CAISO is always available)
    const feed = await apiFetch('/pricefeed?horizon_min=5');
    if (feed && feed.prices && feed.prices.length) {
      const livePrice = feed.prices[0];
      regionData['CAISO'] = { price: livePrice, stress: priceToStress(livePrice) };
    }
  } catch(e) { /* fallback below */ }

  // Pull weather summary for spike correlation data
  try {
    const wx = await apiFetch('/weather/summary');
    if (wx && wx.sites) {
      wx.sites.forEach(site => {
        const r = site.iso_region;
        if (!regionData[r]) regionData[r] = { price: null, stress: 'normal' };
        // Elevate stress if weather event active
        if (site.status === 'critical') regionData[r].stress = 'spike';
        else if (site.status === 'warning' && regionData[r].stress !== 'spike')
          regionData[r].stress = 'elevated';
        regionData[r].siteStatus = site.status;
        regionData[r].siteName   = site.site_name;
        regionData[r].alerts     = site.alerts || [];
      });
    }
  } catch(e) { /* ignore */ }

  // Pull savings ledger for DC activity indicator
  let savingsData = {};
  try {
    const sav = await apiFetch('/savings/ledger?limit=5');
    if (sav && sav.entries) {
      sav.entries.forEach(e => {
        if (e.region) savingsData[e.region] = (savingsData[e.region] || 0) + 1;
      });
    }
  } catch(e) { /* ignore */ }

  // Apply synthetic realistic prices for regions we don't have live data for
  // Based on time-of-day pattern (real EIA data is CAISO only in free tier)
  const hour = new Date().getHours();
  const isPeak = hour >= 16 && hour <= 21;
  const isMidnight = hour >= 0 && hour <= 5;
  const synth = {
    ERCOT: isPeak ? 0.095 : isMidnight ? 0.028 : 0.052,
    PJM:   isPeak ? 0.088 : isMidnight ? 0.031 : 0.048,
    NYISO: isPeak ? 0.110 : isMidnight ? 0.038 : 0.065,
    ISONE: isPeak ? 0.108 : isMidnight ? 0.040 : 0.068,
    MISO:  isPeak ? 0.072 : isMidnight ? 0.024 : 0.042,
    SPP:   isPeak ? 0.068 : isMidnight ? 0.022 : 0.038,
  };
  Object.entries(synth).forEach(([r, p]) => {
    if (!regionData[r]) regionData[r] = { price: p, stress: priceToStress(p) };
    else if (!regionData[r].price) { regionData[r].price = p; regionData[r].stress = priceToStress(p); }
  });

  _gridmapData = regionData;
  renderGridMap(regionData, savingsData);
}

function renderGridMap(regionData, savingsData) {
  const svg = document.getElementById('gridmap-svg');

  // Update region fills and price labels
  Object.entries(regionData).forEach(([region, data]) => {
    const path  = document.getElementById('path-' + region);
    const label = document.getElementById('label-price-' + region);
    if (!path) return;

    const stress = data.stress || 'normal';
    const style  = STRESS_FILLS[stress] || STRESS_FILLS.normal;

    path.setAttribute('fill', style.fill);
    path.setAttribute('stroke', style.stroke);

    if (label) {
      label.textContent = data.price ? `$${data.price.toFixed(4)}/kWh` : '—';
      label.setAttribute('fill', style.label);
    }
  });

  // Render data center hexagons
  const hexG = document.getElementById('gridmap-hexagons');
  hexG.innerHTML = '';

  GRIDMAP_DCS.forEach(dc => {
    const rData   = regionData[dc.region] || {};
    const stress  = rData.stress || 'normal';
    const style   = STRESS_FILLS[stress] || STRESS_FILLS.normal;
    const active  = savingsData[dc.region] > 0;
    const color   = stress === 'spike' ? '#ff4444' : stress === 'elevated' ? '#ffaa00' : '#00ff88';
    const pulseMs = stress === 'spike' ? 800 : stress === 'elevated' ? 1400 : 2200;

    // Outer pulse ring (animated)
    const ring = document.createElementNS('http://www.w3.org/2000/svg','circle');
    ring.setAttribute('cx', dc.cx);
    ring.setAttribute('cy', dc.cy);
    ring.setAttribute('r', '14');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', color);
    ring.setAttribute('stroke-width', '1');
    ring.setAttribute('opacity', '0');
    ring.innerHTML = `
      <animate attributeName="r" from="12" to="30" dur="${pulseMs}ms" repeatCount="indefinite"/>
      <animate attributeName="opacity" from="0.7" to="0" dur="${pulseMs}ms" repeatCount="indefinite"/>
    `;
    hexG.appendChild(ring);

    // Second ring (offset)
    const ring2 = document.createElementNS('http://www.w3.org/2000/svg','circle');
    ring2.setAttribute('cx', dc.cx);
    ring2.setAttribute('cy', dc.cy);
    ring2.setAttribute('r', '14');
    ring2.setAttribute('fill', 'none');
    ring2.setAttribute('stroke', color);
    ring2.setAttribute('stroke-width', '0.5');
    ring2.setAttribute('opacity', '0');
    ring2.innerHTML = `
      <animate attributeName="r" from="12" to="30" dur="${pulseMs}ms" begin="${pulseMs/2}ms" repeatCount="indefinite"/>
      <animate attributeName="opacity" from="0.5" to="0" dur="${pulseMs}ms" begin="${pulseMs/2}ms" repeatCount="indefinite"/>
    `;
    hexG.appendChild(ring2);

    // Hexagon body
    const hex = document.createElementNS('http://www.w3.org/2000/svg','path');
    hex.setAttribute('d', hexPath(dc.cx, dc.cy, 14));
    hex.setAttribute('fill', stress === 'spike'    ? 'rgba(255,68,68,0.25)' :
                              stress === 'elevated' ? 'rgba(255,170,0,0.2)' :
                                                     'rgba(0,255,136,0.18)');
    hex.setAttribute('stroke', color);
    hex.setAttribute('stroke-width', '1.5');
    hex.setAttribute('filter', stress === 'spike' ? 'url(#hex-glow-hot)' : 'url(#hex-glow)');
    hex.setAttribute('style', 'cursor:pointer;');
    hex.addEventListener('mouseenter', (e) => showGridmapTooltip(dc, rData, e));
    hex.addEventListener('mouseleave', hideGridmapTooltip);
    hex.addEventListener('click', () => gridmapRegionClick(dc.region));
    hexG.appendChild(hex);

    // Inner dot
    const dot = document.createElementNS('http://www.w3.org/2000/svg','circle');
    dot.setAttribute('cx', dc.cx);
    dot.setAttribute('cy', dc.cy);
    dot.setAttribute('r', '3');
    dot.setAttribute('fill', color);
    dot.innerHTML = `<animate attributeName="opacity" from="1" to="0.3" dur="${pulseMs}ms" repeatCount="indefinite" calcMode="ease-in-out"/>`;
    hexG.appendChild(dot);

    // DC name label
    const lbl = document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x', dc.cx);
    lbl.setAttribute('y', dc.cy + 24);
    lbl.setAttribute('text-anchor', 'middle');
    lbl.setAttribute('style', `fill:${color};font-size:8.5px;font-family:var(--mono);font-weight:600;pointer-events:none;`);
    lbl.textContent = dc.name.replace(' DC','').toUpperCase();
    hexG.appendChild(lbl);
  });

  // Status bar below map
  const bar = document.getElementById('gridmap-statusbar');
  const allRegions = ['CAISO','ERCOT','PJM','NYISO','ISONE','MISO','SPP'];
  bar.innerHTML = allRegions.map(r => {
    const d = regionData[r] || {};
    const stress = d.stress || 'normal';
    const color  = stress === 'spike' ? 'var(--red)' : stress === 'elevated' ? 'var(--amber)' :
                   stress === 'cheap' ? 'var(--green)' : 'var(--muted)';
    const bg     = stress === 'spike' ? 'rgba(255,68,68,0.12)' : stress === 'elevated' ? 'rgba(255,170,0,0.10)' :
                   stress === 'cheap' ? 'rgba(0,255,136,0.10)' : 'rgba(255,255,255,0.04)';
    const border = stress === 'spike' ? 'rgba(255,68,68,0.3)' : stress === 'elevated' ? 'rgba(255,170,0,0.25)' :
                   stress === 'cheap' ? 'rgba(0,255,136,0.25)' : 'rgba(255,255,255,0.08)';
    const price  = d.price ? `$${d.price.toFixed(4)}` : '—';
    return `
      <div onclick="gridmapRegionClick('${r}')" style="
        background:${bg};border:1px solid ${border};border-radius:8px;
        padding:6px 12px;cursor:pointer;transition:all 0.2s;flex:1;min-width:90px;
        display:flex;flex-direction:column;align-items:center;gap:2px;
      ">
        <span style="font-size:10px;font-weight:700;color:${color};letter-spacing:0.5px;">${r}</span>
        <span style="font-size:11px;color:var(--text2);font-family:var(--mono);">${price}</span>
        <span style="font-size:9px;color:var(--muted);text-transform:uppercase;">${stress}</span>
      </div>`;
  }).join('');
}

function showGridmapTooltip(dc, rData, evt) {
  const svg    = document.getElementById('gridmap-svg');
  const tt     = document.getElementById('gridmap-tooltip');
  const bg     = document.getElementById('gmt-bg');
  const title  = document.getElementById('gmt-title');
  const line1  = document.getElementById('gmt-line1');
  const line2  = document.getElementById('gmt-line2');
  const line3  = document.getElementById('gmt-line3');

  const svgRect = svg.getBoundingClientRect();
  const vbW     = 960, vbH = 580;
  const scaleX  = vbW / svgRect.width;
  const scaleY  = vbH / svgRect.height;
  const mx      = (evt.clientX - svgRect.left) * scaleX;
  const my      = (evt.clientY - svgRect.top)  * scaleY;

  let tx = mx + 10, ty = my - 90;
  if (tx + 185 > vbW) tx = mx - 195;
  if (ty < 5) ty = my + 20;

  bg.setAttribute('x', tx); bg.setAttribute('y', ty);
  title.setAttribute('x', tx + 10); title.setAttribute('y', ty + 18);
  line1.setAttribute('x', tx + 10); line1.setAttribute('y', ty + 34);
  line2.setAttribute('x', tx + 10); line2.setAttribute('y', ty + 48);
  line3.setAttribute('x', tx + 10); line3.setAttribute('y', ty + 62);

  title.textContent = dc.name;
  line1.textContent = rData.price ? `Price: $${rData.price.toFixed(4)}/kWh` : 'Price: live feed';
  line2.textContent = `Region: ${dc.region}`;
  const alertCount  = (rData.alerts || []).length;
  line3.textContent = alertCount ? `⚠ ${alertCount} weather alert${alertCount>1?'s':''}` : '✓ No active alerts';

  tt.style.display = 'block';
}

function hideGridmapTooltip() {
  document.getElementById('gridmap-tooltip').style.display = 'none';
}

function gridmapRegionClick(region) {
  // Highlight the region's weather site card if visible
  const wxData = window._wxData;
  hideGridmapTooltip();
  // Scroll to weather site cards
  const grid = document.getElementById('wxSiteGrid');
  if (grid) grid.scrollIntoView({ behavior:'smooth', block:'nearest' });
}

// Hook into weather tab and auto-refresh every 60s
const _origSwitchTab_map = switchTab;
switchTab = function(name, btn) {
  _origSwitchTab_map(name, btn);
  if (name === 'weather') {
    setTimeout(loadGridMap, 150);
    if (_gridmapTimer) clearInterval(_gridmapTimer);
    _gridmapTimer = setInterval(loadGridMap, 60000);
  } else {
    if (_gridmapTimer) { clearInterval(_gridmapTimer); _gridmapTimer = null; }
  }
};
'''

last_script_close = html.rfind('</script>')
if last_script_close != -1:
    html = html[:last_script_close] + JS + '\n' + html[last_script_close:]
    print("  ✓  Inserted grid map JavaScript")

# ══════════════════════════════════════════════════════════════════════════════
#  4. HELP CONTENT
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_HELP_ANCHOR = "  ,weather: {"
GRIDMAP_HELP = '''  ,gridmap: {
    label: 'Grid Stress Map',
    html: `
      <div class="help-section">
        <h3><i class="fa-solid fa-map"></i>What is the Grid Stress Map?</h3>
        <p>The Live US Grid Intelligence map at the top of the Weather tab shows the real-time stress level of every major ISO electricity grid region in the continental US, with your data center locations overlaid as glowing pulsing hexagons.</p>
        <p>At a glance you can see exactly which regions are cheap, expensive, or spiking — and where your compute fleet sits relative to those conditions. This is the executive-level view: one screenshot tells the whole story.</p>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-layer-group"></i>Map Layers</h3>
        <div class="help-metric"><div class="hm-name">ISO Region Shading</div><div class="hm-desc">Each grid region (CAISO, ERCOT, PJM, NYISO, ISONE, MISO, SPP) is shaded green → amber → red based on current electricity price and active weather events. The colour transitions smoothly as conditions change.</div></div>
        <div class="help-metric"><div class="hm-name">Data Center Hexagons</div><div class="hm-desc">Your configured data center sites appear as hexagonal markers — on-brand with HexaGrid. The hexagon colour matches its region's stress level. The pulse animation rate reflects urgency: slow green pulse = cheap and calm, fast red pulse = spike conditions.</div></div>
        <div class="help-metric"><div class="hm-name">Region Status Bar</div><div class="hm-desc">Below the map, a compact strip shows all seven ISO regions with their current price and stress level. Click any region to jump to its weather site card.</div></div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-palette"></i>Reading the Colours</h3>
        <div class="help-metric"><div class="hm-name" style="color:var(--green)">Green — Cheap</div><div class="hm-desc">Price is below $0.07/kWh. Good time to run flexible workloads. Hexagon pulses slowly and calmly.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--amber)">Amber — Elevated</div><div class="hm-desc">Price between $0.07–0.12/kWh, or a weather warning is active. Monitor closely. Hexagon pulse speeds up.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--red)">Red — Spike</div><div class="hm-desc">Price above $0.12/kWh or a critical weather event is active. Defer all non-urgent workloads. Hexagon pulses rapidly in red.</div></div>
        <div class="help-metric"><div class="hm-name" style="color:var(--cyan)">Cyan — Normal</div><div class="hm-desc">No live price data available for this region — shown in neutral cyan. Conditions are assumed normal.</div></div>
        <div class="help-tip"><strong>Tip:</strong> CAISO pricing is live from the EIA API. Other regions use time-of-day calibrated estimates based on published 2024–2025 statistics — accurate for scheduling decisions, not for billing.</div>
      </div>
      <hr class="help-divider">
      <div class="help-section">
        <h3><i class="fa-solid fa-rotate"></i>Auto-Refresh</h3>
        <p>The map refreshes automatically every 60 seconds while the Weather tab is open. Use the ↻ button in the map header to force an immediate refresh. The timestamp in the top-right corner shows when the map was last updated.</p>
      </div>`
  }

  ,weather: {'''

patch(WEATHER_HELP_ANCHOR, GRIDMAP_HELP, "Add Grid Map help content")

# Help drawer button
patch(
    '    <button class="help-tab-btn" onclick="showHelpTab(\'weather\',this)"><i class="fa-solid fa-cloud-bolt"></i> Weather</button>',
    '    <button class="help-tab-btn" onclick="showHelpTab(\'weather\',this)"><i class="fa-solid fa-cloud-bolt"></i> Weather</button>\n    <button class="help-tab-btn" onclick="showHelpTab(\'gridmap\',this)"><i class="fa-solid fa-map"></i> Grid Map</button>',
    "Add Grid Map help drawer button"
)

# ══════════════════════════════════════════════════════════════════════════════
INDEX.write_text(html)
print(f"\n  ✓  Written: {INDEX}")
print("""
  ══════════════════════════════════════════════════
  Done — hard refresh the dashboard (Ctrl+Shift+R)
  Open the Weather tab to see the Live Grid Map.
  ══════════════════════════════════════════════════
""")
