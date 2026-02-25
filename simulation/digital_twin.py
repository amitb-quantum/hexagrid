"""
HexaGrid - Phase 1: Digital Twin Simulation Engine
==================================================
Models the full data center power chain as a discrete-event simulation:

    Grid → Transformer → UPS → PDU → GPU Rack(s)

Each node has:
  - Configurable efficiency (%) representing real-world loss
  - Power draw telemetry logged every tick
  - Fault injection capability
  - Event bus for downstream modules (Phase 2/3) to subscribe to

Usage:
    python digital_twin.py                  # run default simulation
    python digital_twin.py --duration 480   # run for 480 minutes
    python digital_twin.py --racks 8        # simulate 8 GPU racks
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import simpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

# Real-world efficiency baselines (industry standard values)
DEFAULT_EFFICIENCY = {
    "grid_feed":     0.99,   # Utility feed / metering loss
    "transformer":   0.97,   # Medium-voltage step-down transformer
    "ups":           0.94,   # Uninterruptible Power Supply
    "pdu":           0.98,   # Power Distribution Unit
    "vrm":           0.92,   # Voltage Regulator Module (board-level)
}

# With Heron-class silicon (4x loss reduction scenario)
HERON_EFFICIENCY = {
    "grid_feed":     0.999,
    "transformer":   0.995,
    "ups":           0.985,
    "pdu":           0.995,
    "vrm":           0.98,
}

# GPU rack specs (NVIDIA H100 SXM5 80GB rack = 8 GPUs)
GPU_RACK_TDP_KW       = 10.2    # kW per rack at full load (H100 x8 = ~10.2kW)
GPU_IDLE_FRACTION     = 0.25    # idle = 25% of TDP
SIM_TICK_MINUTES      = 1       # simulation resolution
WORKLOAD_CYCLE_HOURS  = 4       # hours per workload pattern cycle

COLORS = {
    "grid":        "#00d4ff",
    "transformer": "#7b61ff",
    "ups":         "#ff6b6b",
    "pdu":         "#ffa94d",
    "gpu_rack":    "#69db7c",
    "loss":        "#ff4444",
    "pue":         "#ffffff",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerNode:
    """A single node in the power delivery chain."""
    name:        str
    efficiency:  float            # 0.0 - 1.0
    capacity_kw: float            # max kW this node can handle
    power_in_kw: float = 0.0
    power_out_kw: float = 0.0
    fault_active: bool = False

    @property
    def loss_kw(self) -> float:
        return self.power_in_kw - self.power_out_kw

    @property
    def loss_pct(self) -> float:
        if self.power_in_kw == 0:
            return 0.0
        return (self.loss_kw / self.power_in_kw) * 100.0

    def process(self, power_in_kw: float) -> float:
        """Apply efficiency loss and return output power."""
        self.power_in_kw  = min(power_in_kw, self.capacity_kw)
        eff = 0.0 if self.fault_active else self.efficiency
        self.power_out_kw = self.power_in_kw * eff
        return self.power_out_kw


@dataclass
class GPURack:
    """A GPU rack with a synthetic AI workload pattern."""
    rack_id:      int
    tdp_kw:       float = GPU_RACK_TDP_KW
    utilization:  float = 0.0    # 0.0 - 1.0
    num_gpus:     int   = 8

    @property
    def power_draw_kw(self) -> float:
        idle = self.tdp_kw * GPU_IDLE_FRACTION
        return idle + (self.tdp_kw - idle) * self.utilization

    def update_workload(self, tick: int, rng: np.random.Generator):
        """
        Synthetic AI workload pattern:
          - Base sinusoidal cycle (training batch patterns)
          - Random burst events (inference spikes)
          - Gaussian noise
        """
        cycle = WORKLOAD_CYCLE_HOURS * 60 / SIM_TICK_MINUTES
        base  = 0.55 + 0.35 * np.sin(2 * np.pi * tick / cycle + self.rack_id)
        burst = rng.choice([0, 0.3], p=[0.97, 0.03])       # 3% chance of burst
        noise = rng.normal(0, 0.02)
        self.utilization = float(np.clip(base + burst + noise, 0.1, 1.0))


@dataclass
class TelemetryRecord:
    """One tick of simulation telemetry."""
    tick:              int
    timestamp_min:     float
    grid_demand_kw:    float
    transformer_in_kw: float
    transformer_out_kw: float
    ups_in_kw:         float
    ups_out_kw:        float
    pdu_in_kw:         float
    pdu_out_kw:        float
    gpu_total_kw:      float
    total_loss_kw:     float
    pue:               float
    grid_price_usd_kwh: float
    cost_per_min_usd:  float


# ══════════════════════════════════════════════════════════════════════════════
#  GRID PRICING MODEL
# ══════════════════════════════════════════════════════════════════════════════

def grid_price_usd_kwh(tick: int) -> float:
    """
    Synthetic real-time electricity price ($/kWh).
    Based on CAISO TOU (Time of Use) pattern:
      - Off-peak night:   $0.06/kWh
      - Shoulder morning: $0.09/kWh
      - On-peak day:      $0.14/kWh
      - Super-peak:       $0.22/kWh (rare events)
    """
    minute_of_day = (tick * SIM_TICK_MINUTES) % (24 * 60)
    hour = minute_of_day / 60.0

    if   0   <= hour < 7:   base = 0.062
    elif 7   <= hour < 11:  base = 0.091
    elif 11  <= hour < 19:  base = 0.140
    elif 19  <= hour < 22:  base = 0.118
    else:                   base = 0.071

    # Occasional grid stress events (+40%)
    stress = 0.40 if (tick % 720 < 15) else 0.0
    return round(base * (1 + stress), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  DIGITAL TWIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class DataCenterDigitalTwin:
    """
    Full discrete-event simulation of a data center power chain.

    Architecture:
        Grid Feed → Transformer → UPS → PDU → [GPU Rack x N]

    Exposes:
        .run()          → run simulation, returns telemetry DataFrame
        .report()       → print summary statistics
        .plot()         → generate dashboard PNG
        .telemetry_df   → raw tick-level data
    """

    def __init__(
        self,
        num_racks:        int   = 4,
        duration_minutes: int   = 480,   # 8 hours default
        efficiency_profile: str = "standard",   # "standard" | "heron"
        facility_name:    str   = "HexaGrid-DC-01",
        seed:             int   = 42,
    ):
        self.facility_name      = facility_name
        self.num_racks          = num_racks
        self.duration_minutes   = duration_minutes
        self.efficiency_profile = efficiency_profile
        self.rng                = np.random.default_rng(seed)
        self.telemetry: list[TelemetryRecord] = []

        eff = HERON_EFFICIENCY if efficiency_profile == "heron" else DEFAULT_EFFICIENCY

        # Total capacity = racks * rack_TDP * 2x headroom
        total_capacity_kw = num_racks * GPU_RACK_TDP_KW * 2.0

        # Power chain nodes
        self.nodes = {
            "grid_feed":   PowerNode("Grid Feed",   eff["grid_feed"],   total_capacity_kw * 1.2),
            "transformer": PowerNode("Transformer", eff["transformer"], total_capacity_kw * 1.1),
            "ups":         PowerNode("UPS",         eff["ups"],         total_capacity_kw),
            "pdu":         PowerNode("PDU",         eff["pdu"],         total_capacity_kw),
        }

        # GPU racks
        self.racks = [GPURack(rack_id=i) for i in range(num_racks)]

        print(f"\n  ⚡ {facility_name} Digital Twin Initialized")
        print(f"     Profile   : {efficiency_profile.upper()}")
        print(f"     GPU Racks : {num_racks}  ({num_racks * 8} GPUs @ H100-class)")
        print(f"     Peak Load : {num_racks * GPU_RACK_TDP_KW:.1f} kW")
        print(f"     Duration  : {duration_minutes} min ({duration_minutes/60:.1f} hrs)\n")

    # ── SimPy process ──────────────────────────────────────────────────────────
    def _power_chain_process(self, env: simpy.Environment):
        tick = 0
        while True:
            # 1. Update GPU workloads
            for rack in self.racks:
                rack.update_workload(tick, self.rng)

            # 2. Total GPU demand
            gpu_total_kw = sum(r.power_draw_kw for r in self.racks)

            # 3. Propagate backwards through chain to get grid demand
            #    (what the grid must supply given all losses)
            #    We work backwards: gpu_demand → PDU_in → UPS_in → TX_in → Grid
            pdu_in  = gpu_total_kw / self.nodes["pdu"].efficiency
            ups_in  = pdu_in       / self.nodes["ups"].efficiency
            tx_in   = ups_in       / self.nodes["transformer"].efficiency
            grid_in = tx_in        / self.nodes["grid_feed"].efficiency

            # 4. Process nodes forward (records actual in/out)
            tx_out  = self.nodes["grid_feed"].process(grid_in)
            ups_in_ = self.nodes["transformer"].process(tx_out)
            pdu_in_ = self.nodes["ups"].process(ups_in_)
            gpu_in  = self.nodes["pdu"].process(pdu_in_)

            # 5. Total loss across chain
            total_loss_kw = grid_in - gpu_total_kw

            # 6. PUE (Power Usage Effectiveness) = total facility / IT load
            pue = grid_in / gpu_total_kw if gpu_total_kw > 0 else 1.0

            # 7. Grid price and cost
            price  = grid_price_usd_kwh(tick)
            cost_per_min = (grid_in / 60.0) * price   # kWh * $/kWh

            # 8. Record telemetry
            self.telemetry.append(TelemetryRecord(
                tick              = tick,
                timestamp_min     = tick * SIM_TICK_MINUTES,
                grid_demand_kw    = grid_in,
                transformer_in_kw = tx_out,
                transformer_out_kw= ups_in_,
                ups_in_kw         = ups_in_,
                ups_out_kw        = pdu_in_,
                pdu_in_kw         = pdu_in_,
                pdu_out_kw        = gpu_in,
                gpu_total_kw      = gpu_total_kw,
                total_loss_kw     = total_loss_kw,
                pue               = pue,
                grid_price_usd_kwh= price,
                cost_per_min_usd  = cost_per_min,
            ))

            tick += 1
            yield env.timeout(SIM_TICK_MINUTES)

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        print("  ▶  Running simulation...")
        t0  = time.time()
        env = simpy.Environment()
        env.process(self._power_chain_process(env))
        env.run(until=self.duration_minutes)
        elapsed = time.time() - t0

        self.telemetry_df = pd.DataFrame([vars(r) for r in self.telemetry])
        print(f"  ✓  Simulation complete in {elapsed:.2f}s  "
              f"({len(self.telemetry_df)} ticks recorded)\n")
        return self.telemetry_df

    # ── Report ────────────────────────────────────────────────────────────────
    def report(self):
        if not hasattr(self, 'telemetry_df'):
            print("Run simulation first.")
            return

        df = self.telemetry_df
        total_hours     = self.duration_minutes / 60.0
        total_kwh_grid  = (df['grid_demand_kw'].sum()  * SIM_TICK_MINUTES) / 60.0
        total_kwh_it    = (df['gpu_total_kw'].sum()    * SIM_TICK_MINUTES) / 60.0
        total_kwh_loss  = (df['total_loss_kw'].sum()   * SIM_TICK_MINUTES) / 60.0
        total_cost      = df['cost_per_min_usd'].sum()
        avg_pue         = df['pue'].mean()
        avg_price       = df['grid_price_usd_kwh'].mean()
        peak_demand     = df['grid_demand_kw'].max()
        peak_tick       = df.loc[df['grid_demand_kw'].idxmax(), 'timestamp_min']

        # Annualized projections
        scale = (365 * 24) / total_hours
        annual_kwh  = total_kwh_grid * scale
        annual_cost = total_cost     * scale

        # Efficiency chain product
        chain_eff = 1.0
        for node in self.nodes.values():
            chain_eff *= node.efficiency
        loss_reduction_pct = (1 - chain_eff) * 100

        print(f"{'═'*56}")
        print(f"  ⚡  HEXAGRID DIGITAL TWIN — SIMULATION REPORT")
        print(f"{'═'*56}")
        print(f"  Facility          : {self.facility_name}")
        print(f"  Profile           : {self.efficiency_profile.upper()}")
        print(f"  Racks / GPUs      : {self.num_racks} racks / {self.num_racks*8} GPUs")
        print(f"{'─'*56}")
        print(f"  Sim Duration      : {self.duration_minutes} min ({total_hours:.1f} hrs)")
        print(f"  Grid Consumed     : {total_kwh_grid:.2f} kWh")
        print(f"  IT Load (GPUs)    : {total_kwh_it:.2f} kWh")
        print(f"  Chain Loss        : {total_kwh_loss:.2f} kWh  ({loss_reduction_pct:.1f}% lost)")
        print(f"{'─'*56}")
        print(f"  Avg PUE           : {avg_pue:.4f}  (ideal = 1.0)")
        print(f"  Peak Grid Demand  : {peak_demand:.2f} kW  (at t={peak_tick:.0f} min)")
        print(f"  Avg Grid Price    : ${avg_price:.4f}/kWh")
        print(f"{'─'*56}")
        print(f"  Sim Period Cost   : ${total_cost:.2f}")
        print(f"  Annualized Cost   : ${annual_cost:,.0f}")
        print(f"  Annualized kWh    : {annual_kwh:,.0f} kWh")
        print(f"{'─'*56}")

        # Node-level breakdown
        print(f"  POWER CHAIN LOSSES:")
        for name, node in self.nodes.items():
            bar_len = int((1 - node.efficiency) * 200)
            bar = "█" * max(bar_len, 1)
            print(f"    {node.name:<18} eff={node.efficiency*100:.1f}%  "
                  f"loss={node.loss_pct:.2f}%  {bar}")
        print(f"{'═'*56}\n")

        # Heron comparison (if running standard profile)
        if self.efficiency_profile == "standard":
            heron_chain = 1.0
            for v in HERON_EFFICIENCY.values():
                heron_chain *= v
            std_chain = chain_eff
            improvement = (heron_chain - std_chain) / (1 - std_chain)
            saved_kwh   = total_kwh_loss * improvement
            saved_cost  = saved_kwh * avg_price * scale

            print(f"  💡 HERON-CLASS UPGRADE PROJECTION:")
            print(f"     Standard chain eff : {std_chain*100:.2f}%")
            print(f"     Heron chain eff    : {heron_chain*100:.2f}%")
            print(f"     Loss reduction     : {improvement*100:.1f}%")
            print(f"     Annual kWh saved   : {saved_kwh*scale:,.0f} kWh")
            print(f"     Annual $ saved     : ${saved_cost:,.0f}")
            print(f"{'═'*56}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    def plot(self, save_path: Optional[str] = None) -> str:
        if not hasattr(self, 'telemetry_df'):
            print("Run simulation first.")
            return ""

        df  = self.telemetry_df
        t   = df['timestamp_min'] / 60.0   # convert to hours

        fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
        fig.suptitle(
            f"⚡  HEXAGRID — {self.facility_name}  |  "
            f"{self.num_racks} Racks / {self.num_racks*8} GPUs  |  "
            f"Profile: {self.efficiency_profile.upper()}",
            color='white', fontsize=14, fontweight='bold', y=0.98
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        ax_bg = {'facecolor': '#161b22', 'labelcolor': 'white'}

        def style_ax(ax, title, xlabel='Time (hrs)', ylabel='kW'):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.set_title(title, color='white', fontsize=10, pad=8)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.8)

        # 1. Power Chain Flow
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.fill_between(t, df['grid_demand_kw'],  alpha=0.25, color=COLORS['grid'])
        ax1.fill_between(t, df['ups_out_kw'],       alpha=0.30, color=COLORS['ups'])
        ax1.fill_between(t, df['gpu_total_kw'],     alpha=0.45, color=COLORS['gpu_rack'])
        ax1.plot(t, df['grid_demand_kw'],  color=COLORS['grid'],     lw=1.5, label='Grid Demand')
        ax1.plot(t, df['ups_out_kw'],       color=COLORS['ups'],      lw=1.2, label='Post-UPS')
        ax1.plot(t, df['gpu_total_kw'],     color=COLORS['gpu_rack'], lw=1.5, label='GPU Load')
        ax1.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
        style_ax(ax1, '📊 Power Chain Flow (kW)')

        # 2. Chain Losses
        ax2 = fig.add_subplot(gs[0, 2])
        loss_labels = ['Grid\nFeed', 'Trans-\nformer', 'UPS', 'PDU', 'VRM\n(est.)']
        eff_vals    = list(DEFAULT_EFFICIENCY.values()) if self.efficiency_profile == "standard" \
                      else list(HERON_EFFICIENCY.values())
        loss_vals   = [(1-e)*100 for e in eff_vals]
        bars = ax2.bar(loss_labels, loss_vals,
                       color=[COLORS['grid'], COLORS['transformer'],
                               COLORS['ups'], COLORS['pdu'], COLORS['gpu_rack']],
                       edgecolor='#30363d', linewidth=0.5)
        for bar, val in zip(bars, loss_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=7)
        style_ax(ax2, '🔥 Loss per Node', xlabel='Node', ylabel='Loss %')

        # 3. PUE over time
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(t, df['pue'], color='#ffd700', lw=1.5, label='PUE')
        ax3.axhline(y=1.0,  color='#69db7c', lw=1.0, ls='--', label='Ideal PUE = 1.0')
        ax3.axhline(y=1.2,  color='#ffa94d', lw=1.0, ls='--', label='Industry Best = 1.2')
        ax3.fill_between(t, df['pue'], 1.0, alpha=0.15, color='#ffd700')
        ax3.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
        style_ax(ax3, '📈 Power Usage Effectiveness (PUE)', ylabel='PUE')

        # 4. Electricity Price
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(t, df['grid_price_usd_kwh'], color='#ff6b9d', lw=1.2)
        ax4.fill_between(t, df['grid_price_usd_kwh'], alpha=0.2, color='#ff6b9d')
        style_ax(ax4, '💰 Grid Price ($/kWh)', ylabel='$/kWh')

        # 5. Cost accumulation
        ax5 = fig.add_subplot(gs[2, :2])
        cumulative_cost = df['cost_per_min_usd'].cumsum()
        ax5.plot(t, cumulative_cost, color='#ff9f43', lw=2)
        ax5.fill_between(t, cumulative_cost, alpha=0.2, color='#ff9f43')
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}'))
        style_ax(ax5, '💵 Cumulative Energy Cost', ylabel='USD')

        # 6. Summary stats box
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.set_facecolor('#161b22')
        ax6.axis('off')
        total_kwh   = (df['grid_demand_kw'].sum() * SIM_TICK_MINUTES) / 60.0
        total_cost  = df['cost_per_min_usd'].sum()
        avg_pue     = df['pue'].mean()
        peak_kw     = df['grid_demand_kw'].max()
        loss_kwh    = (df['total_loss_kw'].sum() * SIM_TICK_MINUTES) / 60.0

        stats = [
            ("Facility",        self.facility_name),
            ("Profile",         self.efficiency_profile.upper()),
            ("GPUs",            f"{self.num_racks * 8}x H100-class"),
            ("Total kWh",       f"{total_kwh:.1f} kWh"),
            ("Chain Loss",      f"{loss_kwh:.2f} kWh"),
            ("Avg PUE",         f"{avg_pue:.4f}"),
            ("Peak Demand",     f"{peak_kw:.2f} kW"),
            ("Period Cost",     f"${total_cost:.2f}"),
        ]
        y_pos = 0.95
        ax6.text(0.05, y_pos, "📋 Summary", color='#00d4ff',
                 fontsize=10, fontweight='bold', transform=ax6.transAxes)
        for label, value in stats:
            y_pos -= 0.11
            ax6.text(0.05, y_pos, f"{label:<14}", color='#8b949e',
                     fontsize=8, transform=ax6.transAxes)
            ax6.text(0.55, y_pos, value, color='white',
                     fontsize=8, transform=ax6.transAxes)

        for spine in ax6.spines.values():
            spine.set_edgecolor('#30363d')

        # Save
        if save_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                os.path.dirname(__file__), '..', 'reports',
                f"hexagrid_twin_{self.efficiency_profile}_{ts}.png"
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d1117', edgecolor='none')
        plt.close()
        print(f"  📊 Dashboard saved → {os.path.abspath(save_path)}")
        return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON RUN: Standard vs Heron
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(num_racks: int = 4, duration_minutes: int = 480):
    """
    Run both standard and Heron-class profiles and produce a side-by-side
    comparison report and plot.
    """
    print("\n" + "═"*56)
    print("  ⚡  HEXAGRID — STANDARD vs HERON COMPARISON RUN")
    print("═"*56)

    results = {}
    for profile in ["standard", "heron"]:
        twin = DataCenterDigitalTwin(
            num_racks=num_racks,
            duration_minutes=duration_minutes,
            efficiency_profile=profile,
            facility_name=f"HexaGrid-DC-01-{profile.upper()}",
            seed=42,
        )
        df = twin.run()
        twin.report()
        path = twin.plot()
        results[profile] = {
            "df":   df,
            "twin": twin,
            "path": path,
        }

    # Delta analysis
    std_df   = results["standard"]["df"]
    heron_df = results["heron"]["df"]

    std_kwh   = (std_df['grid_demand_kw'].sum()   * SIM_TICK_MINUTES) / 60.0
    heron_kwh = (heron_df['grid_demand_kw'].sum() * SIM_TICK_MINUTES) / 60.0
    saved_kwh = std_kwh - heron_kwh

    avg_price  = std_df['grid_price_usd_kwh'].mean()
    scale      = (365 * 24) / (duration_minutes / 60.0)
    annual_saving = saved_kwh * avg_price * scale

    std_pue   = std_df['pue'].mean()
    heron_pue = heron_df['pue'].mean()

    print("═"*56)
    print("  📊 COMPARISON DELTA")
    print("═"*56)
    print(f"  Grid kWh (standard) : {std_kwh:.2f} kWh")
    print(f"  Grid kWh (heron)    : {heron_kwh:.2f} kWh")
    print(f"  Saved (sim period)  : {saved_kwh:.2f} kWh  "
          f"({saved_kwh/std_kwh*100:.1f}% reduction)")
    print(f"  Avg PUE standard    : {std_pue:.4f}")
    print(f"  Avg PUE heron       : {heron_pue:.4f}")
    print(f"  PUE improvement     : {(std_pue - heron_pue):.4f}")
    print(f"  Annual kWh saved    : {saved_kwh*scale:,.0f} kWh")
    print(f"  Annual $ saved      : ${annual_saving:,.0f}")
    print("═"*56 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HexaGrid Digital Twin Simulation")
    parser.add_argument('--racks',    type=int,  default=4,    help='Number of GPU racks')
    parser.add_argument('--duration', type=int,  default=480,  help='Simulation duration (minutes)')
    parser.add_argument('--profile',  type=str,  default='both',
                        choices=['standard', 'heron', 'both'],
                        help='Efficiency profile to run')
    args = parser.parse_args()

    if args.profile == 'both':
        run_comparison(num_racks=args.racks, duration_minutes=args.duration)
    else:
        twin = DataCenterDigitalTwin(
            num_racks=args.racks,
            duration_minutes=args.duration,
            efficiency_profile=args.profile,
            facility_name="HexaGrid-DC-01",
        )
        twin.run()
        twin.report()
        twin.plot()
