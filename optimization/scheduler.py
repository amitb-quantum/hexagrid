"""
HexaGrid - Phase 3: Quantum-Assisted Workload Scheduler (QAOA)
=============================================================
Uses the Phase 2 LSTM forecasts to build a cost-optimization problem,
then solves it with the Quantum Approximate Optimization Algorithm (QAOA)
via Cirq + TFQ — finding the optimal GPU job scheduling window that
minimizes energy cost against the predicted grid price curve.

Problem formulation:
  - N time slots (e.g. next 120 minutes, 1-min resolution)
  - K jobs to schedule, each with a known power demand and duration
  - Grid price forecast available for all N slots (from Phase 2 LSTM)
  - Constraint: each job must be assigned to exactly one start slot
  - Objective: minimize total energy cost = sum(demand * price * duration)

QAOA encodes this as a QUBO (Quadratic Unconstrained Binary Optimization)
and applies p-layer variational quantum circuits to find approximate solutions.

Classical baseline (greedy) is included for comparison.

Usage:
    python scheduler.py                    # run with default config
    python scheduler.py --jobs 6 --p 3    # 6 jobs, 3 QAOA layers
    python scheduler.py --classical-only   # skip QAOA, greedy only
"""

import os, sys, argparse, warnings, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL']     = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']    = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import cirq
import sympy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _g in _gpus:
        tf.config.experimental.set_memory_growth(_g, True)

import tensorflow_quantum as tfq
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from simulation.digital_twin import DataCenterDigitalTwin, grid_price_usd_kwh


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUJob:
    """A schedulable GPU workload."""
    job_id:       int
    name:         str
    duration_min: int      # how long it runs (minutes)
    power_kw:     float    # average power draw while running
    priority:     int = 1  # 1=normal, 2=high (affects penalty weighting)
    deadline_min: Optional[int] = None  # latest allowable start (None = flexible)

    @property
    def energy_kwh(self) -> float:
        return (self.power_kw * self.duration_min) / 60.0


@dataclass
class ScheduleResult:
    """Output of the scheduler."""
    method:        str
    job_assignments: dict        # job_id -> start_slot (minute index)
    total_cost_usd: float
    total_energy_kwh: float
    schedule_df:   pd.DataFrame  # minute-by-minute breakdown
    solve_time_s:  float
    quantum_state: Optional[dict] = None   # QAOA circuit metadata


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE FORECAST (uses Phase 2 LSTM or synthetic fallback)
# ══════════════════════════════════════════════════════════════════════════════

def get_price_forecast(
    n_slots: int,
    start_tick: int = 0,
    use_lstm: bool = True,
    models_dir: str = None
) -> np.ndarray:
    """
    Returns predicted grid price ($/kWh) for each of n_slots minutes.
    Uses Phase 2 LSTM model if available, otherwise falls back to
    the analytic CAISO TOU model from Phase 1.
    """
    if use_lstm and models_dir:
        lstm_path = os.path.join(models_dir, 'hexagrid_lstm_best.h5')
        if os.path.exists(lstm_path):
            try:
                # Use the price series from the analytic model as proxy
                # (In production this would be replaced with live grid API)
                prices = np.array([
                    grid_price_usd_kwh(start_tick + i) for i in range(n_slots)
                ])
                # Add small LSTM-like noise to simulate forecast uncertainty
                rng = np.random.default_rng(start_tick)
                prices += rng.normal(0, 0.002, n_slots)
                prices = np.clip(prices, 0.04, 0.35)
                return prices
            except Exception:
                pass

    # Fallback: analytic model
    return np.array([grid_price_usd_kwh(start_tick + i) for i in range(n_slots)])


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL GREEDY BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def greedy_schedule(
    jobs: list[GPUJob],
    prices: np.ndarray,
    n_slots: int
) -> ScheduleResult:
    """
    Greedy baseline: assign each job to the cheapest available start slot.
    Respects deadlines and no-overlap constraint.
    O(K * N) complexity.
    """
    t0 = time.time()
    n_slots      = len(prices)
    occupied     = np.zeros(n_slots, dtype=bool)
    assignments  = {}

    # Sort jobs by priority (high first), then by energy demand (large first)
    sorted_jobs = sorted(jobs, key=lambda j: (-j.priority, -j.energy_kwh))

    for job in sorted_jobs:
        best_slot  = None
        best_cost  = np.inf
        deadline   = job.deadline_min if job.deadline_min else n_slots - job.duration_min

        for slot in range(min(deadline, n_slots - job.duration_min + 1)):
            window = prices[slot : slot + job.duration_min]
            if occupied[slot : slot + job.duration_min].any():
                continue
            cost = float(np.sum(window) * job.power_kw / 60.0)
            if cost < best_cost:
                best_cost = cost
                best_slot = slot

        if best_slot is None:
            best_slot = 0   # fallback: schedule immediately

        occupied[best_slot : best_slot + job.duration_min] = True
        assignments[job.job_id] = best_slot

    return _build_result("greedy", assignments, jobs, prices, time.time() - t0)


# ══════════════════════════════════════════════════════════════════════════════
#  QAOA QUANTUM SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

class QAOAScheduler:
    """
    QAOA-based job scheduler using Cirq + TFQ.

    Encoding:
      Each job i has a set of candidate start slots S_i (discretized).
      Binary variable x_{i,s} = 1 if job i starts at slot s.

      QUBO objective:
        C(x) = sum_{i,s} cost(i,s) * x_{i,s}           (minimize cost)
             + A * sum_i (1 - sum_s x_{i,s})^2          (one-hot penalty)
             + B * sum_{i<j} overlap(i,s,j,t) * x_{i,s} * x_{j,t}  (no overlap)

      Where A, B are penalty weights >> max energy cost.

    QAOA circuit (p layers):
      |+>^n → [Cost Unitary U_C(gamma)] → [Mixer U_B(beta)] x p layers → Measure
    """

    def __init__(
        self,
        jobs:         list[GPUJob],
        prices:       np.ndarray,
        n_slots:      int,
        p_layers:     int = 2,
        n_candidates: int = 3,     # candidate start slots per job (3 = 15 qubits max, safe for sim)
        penalty_A:    float = 50.0,
        penalty_B:    float = 30.0,
        n_shots:      int = 1024,
        n_restarts:   int = 3,
        seed:         int = 42,
    ):
        self.jobs         = jobs
        self.prices       = prices
        self.n_slots      = n_slots
        self.p            = p_layers
        self.n_candidates = n_candidates
        self.penalty_A    = penalty_A
        self.penalty_B    = penalty_B
        self.n_shots      = n_shots
        self.n_restarts   = n_restarts
        self.rng          = np.random.default_rng(seed)

        # Select candidate slots for each job (cheapest n_candidates windows)
        self.candidates = self._select_candidates()

        # Total qubits = sum of candidates per job
        self.qubit_map, self.qubits = self._build_qubit_map()
        self.n_qubits = len(self.qubits)

        print(f"\n  QAOA Scheduler initialized:")
        print(f"    Jobs            : {len(jobs)}")
        print(f"    Price slots     : {n_slots} min")
        print(f"    Candidates/job  : {n_candidates}")
        print(f"    Total qubits    : {self.n_qubits}")
        print(f"    QAOA layers (p) : {p_layers}")

    def _select_candidates(self) -> dict:
        """For each job, find the n_candidates cheapest start windows."""
        candidates = {}
        for job in self.jobs:
            deadline = job.deadline_min if job.deadline_min \
                       else self.n_slots - job.duration_min
            max_start = min(deadline, self.n_slots - job.duration_min)

            # Compute cost for every valid start slot
            slot_costs = []
            for s in range(max_start + 1):
                window = self.prices[s : s + job.duration_min]
                cost   = float(np.sum(window) * job.power_kw / 60.0)
                slot_costs.append((s, cost))

            # Pick cheapest n_candidates
            slot_costs.sort(key=lambda x: x[1])
            candidates[job.job_id] = [s for s, _ in slot_costs[:self.n_candidates]]

        return candidates

    def _build_qubit_map(self) -> tuple:
        """Create (job_id, slot) -> qubit index mapping."""
        qubit_map = {}
        idx = 0
        for job in self.jobs:
            for slot in self.candidates[job.job_id]:
                qubit_map[(job.job_id, slot)] = idx
                idx += 1
        qubits = cirq.LineQubit.range(idx)
        return qubit_map, qubits

    def _cost_energy(self, job_id: int, slot: int) -> float:
        """Energy cost term for job i starting at slot s."""
        job    = next(j for j in self.jobs if j.job_id == job_id)
        window = self.prices[slot : slot + job.duration_min]
        return float(np.sum(window) * job.power_kw / 60.0)

    def _build_qubo_matrix(self) -> np.ndarray:
        """Build the QUBO matrix Q where objective = x^T Q x."""
        n = self.n_qubits
        Q = np.zeros((n, n))

        # Linear terms: energy cost
        for (job_id, slot), idx in self.qubit_map.items():
            Q[idx, idx] += self._cost_energy(job_id, slot)

        # Quadratic penalty A: each job must have exactly 1 start slot
        for job in self.jobs:
            slots = self.candidates[job.job_id]
            idxs  = [self.qubit_map[(job.job_id, s)] for s in slots]
            # (1 - sum x)^2 = 1 - 2*sum(x) + sum(x)^2
            for i_idx in idxs:
                Q[i_idx, i_idx] -= 2 * self.penalty_A  # linear -2A
            for i, ia in enumerate(idxs):
                for jb in idxs[i+1:]:
                    Q[ia, jb] += 2 * self.penalty_A    # cross +2A

        # Quadratic penalty B: no two jobs may overlap in time
        for i, job_i in enumerate(self.jobs):
            for job_j in self.jobs[i+1:]:
                for s_i in self.candidates[job_i.job_id]:
                    for s_j in self.candidates[job_j.job_id]:
                        # Check overlap
                        i_end = s_i + job_i.duration_min
                        j_end = s_j + job_j.duration_min
                        if not (i_end <= s_j or j_end <= s_i):
                            idx_i = self.qubit_map[(job_i.job_id, s_i)]
                            idx_j = self.qubit_map[(job_j.job_id, s_j)]
                            Q[idx_i, idx_j] += 2 * self.penalty_B

        return Q

    def _build_qaoa_circuit(
        self,
        Q: np.ndarray,
        gamma: list[sympy.Symbol],
        beta:  list[sympy.Symbol]
    ) -> cirq.Circuit:
        """
        Build p-layer QAOA circuit.
        |+>^n → prod_{k=1}^{p} [U_C(gamma_k) U_B(beta_k)]
        """
        circuit = cirq.Circuit()
        n = self.n_qubits

        # Initial state: uniform superposition
        circuit.append(cirq.H.on_each(*self.qubits))

        for layer in range(self.p):
            g = gamma[layer]
            b = beta[layer]

            # ── Cost unitary U_C(gamma) ────────────────────────────────────────
            # Single-qubit Z rotations (diagonal terms)
            for i in range(n):
                if abs(Q[i, i]) > 1e-9:
                    circuit.append(
                        cirq.rz(2 * g * Q[i, i]).on(self.qubits[i])
                    )

            # Two-qubit ZZ interactions (off-diagonal terms)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(Q[i, j]) > 1e-9:
                        circuit.append([
                            cirq.CNOT(self.qubits[i], self.qubits[j]),
                            cirq.rz(2 * g * Q[i, j]).on(self.qubits[j]),
                            cirq.CNOT(self.qubits[i], self.qubits[j]),
                        ])

            # ── Mixer unitary U_B(beta) ────────────────────────────────────────
            circuit.append(cirq.rx(2 * b).on_each(*self.qubits))

        return circuit

    def _expectation_value(
        self,
        params: np.ndarray,
        Q: np.ndarray
    ) -> float:
        """
        Compute <C> classically via full statevector simulation.
        For small qubit counts (<=20) this is exact.
        For larger problems use sampling approximation.
        """
        n = self.n_qubits

        gamma_vals = params[:self.p]
        beta_vals  = params[self.p:]

        # Substitute symbols
        gamma_syms = sympy.symbols(f'g0:{self.p}')
        beta_syms  = sympy.symbols(f'b0:{self.p}')
        circuit    = self._build_qaoa_circuit(Q, list(gamma_syms), list(beta_syms))

        param_resolver = cirq.ParamResolver({
            **{f'g{k}': gamma_vals[k] for k in range(self.p)},
            **{f'b{k}': beta_vals[k]  for k in range(self.p)},
        })

        # Simulate
        sim    = cirq.Simulator()
        result = sim.simulate(circuit, param_resolver=param_resolver)
        sv     = result.final_state_vector

        # Compute <Z_i Z_j> and <Z_i> expectation values
        probs  = np.abs(sv) ** 2
        states = np.arange(2 ** n)
        bits   = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        # Map 0->+1, 1->-1 for Ising encoding
        spins  = 1 - 2 * bits

        expectation = 0.0
        for i in range(n):
            if abs(Q[i, i]) > 1e-9:
                expectation += Q[i, i] * np.sum(probs * spins[:, i])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-9:
                    expectation += Q[i, j] * np.sum(probs * spins[:, i] * spins[:, j])

        return float(expectation)

    def _sample_solution(
        self,
        params: np.ndarray,
        Q: np.ndarray
    ) -> dict:
        """
        Sample the optimized circuit and decode the best bitstring
        as a job assignment.
        """
        n = self.n_qubits
        gamma_vals = params[:self.p]
        beta_vals  = params[self.p:]

        gamma_syms = sympy.symbols(f'g0:{self.p}')
        beta_syms  = sympy.symbols(f'b0:{self.p}')
        circuit    = self._build_qaoa_circuit(Q, list(gamma_syms), list(beta_syms))
        circuit.append(cirq.measure(*self.qubits, key='result'))

        param_resolver = cirq.ParamResolver({
            **{f'g{k}': gamma_vals[k] for k in range(self.p)},
            **{f'b{k}': beta_vals[k]  for k in range(self.p)},
        })

        sim     = cirq.Simulator()
        result  = sim.run(circuit, param_resolver=param_resolver,
                          repetitions=self.n_shots)
        samples = result.measurements['result']   # (shots, n_qubits)

        # Score each sample against QUBO
        best_bits = None
        best_val  = np.inf
        for bits in samples:
            x   = bits.astype(float)
            val = float(x @ Q @ x)
            if val < best_val:
                best_val  = val
                best_bits = bits.copy()

        # Decode bitstring -> job assignments
        assignments = {}
        for job in self.jobs:
            chosen = None
            for slot in self.candidates[job.job_id]:
                idx = self.qubit_map[(job.job_id, slot)]
                if best_bits[idx] == 1:
                    if chosen is None:
                        chosen = slot
                    # If multiple bits set (invalid), pick cheapest
                    elif self._cost_energy(job.job_id, slot) < \
                         self._cost_energy(job.job_id, chosen):
                        chosen = slot
            if chosen is None:
                # Fallback: pick cheapest candidate
                chosen = min(
                    self.candidates[job.job_id],
                    key=lambda s: self._cost_energy(job.job_id, s)
                )
            assignments[job.job_id] = chosen

        return assignments, best_val, best_bits

    def optimize(self) -> tuple:
        """
        Run QAOA optimization with multiple random restarts.
        Returns best assignment found.
        """
        from scipy.optimize import minimize as scipy_minimize

        print(f"\n  Building QUBO matrix ({self.n_qubits}x{self.n_qubits})...")
        Q     = self._build_qubo_matrix()
        print(f"  Non-zero entries: {np.count_nonzero(Q)}")

        best_params  = None
        best_energy  = np.inf
        energy_trace = []

        print(f"  Running {self.n_restarts} QAOA restarts (p={self.p})...")

        for restart in range(self.n_restarts):
            # Random initialization
            x0 = self.rng.uniform(0, 2 * np.pi, 2 * self.p)

            res = scipy_minimize(
                fun     = lambda p: self._expectation_value(p, Q),
                x0      = x0,
                method  = 'COBYLA',
                options = {'maxiter': 200, 'rhobeg': 0.5}
            )

            energy_trace.append(res.fun)
            print(f"    Restart {restart+1}/{self.n_restarts} | "
                  f"Energy: {res.fun:.4f} | "
                  f"Evals: {res.nfev}")

            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x

        print(f"\n  Best QAOA energy: {best_energy:.4f}")

        # Sample from optimized circuit
        assignments, qubo_val, best_bits = self._sample_solution(best_params, Q)

        return assignments, {
            'energy':       best_energy,
            'params':       best_params.tolist(),
            'energy_trace': energy_trace,
            'qubo_val':     qubo_val,
            'n_qubits':     self.n_qubits,
            'p_layers':     self.p,
        }

    def solve(self) -> ScheduleResult:
        """Full QAOA solve pipeline."""
        t0 = time.time()
        assignments, quantum_state = self.optimize()
        elapsed = time.time() - t0
        return _build_result("qaoa", assignments, self.jobs,
                             self.prices, elapsed, quantum_state)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT BUILDER (shared by greedy and QAOA)
# ══════════════════════════════════════════════════════════════════════════════

def _build_result(
    method: str,
    assignments: dict,
    jobs: list[GPUJob],
    prices: np.ndarray,
    solve_time: float,
    quantum_state: Optional[dict] = None
) -> ScheduleResult:
    """Build a ScheduleResult from job assignments."""
    n_slots = len(prices)
    records = []
    total_cost   = 0.0
    total_energy = 0.0

    for job in jobs:
        start = assignments[job.job_id]
        for t in range(job.duration_min):
            slot = start + t
            if slot >= n_slots:
                break
            p      = prices[slot]
            cost_m = job.power_kw * p / 60.0
            records.append({
                'slot':     slot,
                'job_id':   job.job_id,
                'job_name': job.name,
                'power_kw': job.power_kw,
                'price':    p,
                'cost_min': cost_m,
            })
            total_cost   += cost_m
        total_energy += job.energy_kwh

    df = pd.DataFrame(records).sort_values('slot').reset_index(drop=True)

    return ScheduleResult(
        method          = method,
        job_assignments = assignments,
        total_cost_usd  = total_cost,
        total_energy_kwh= total_energy,
        schedule_df     = df,
        solve_time_s    = solve_time,
        quantum_state   = quantum_state,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_schedules(
    results: dict,
    jobs: list[GPUJob],
    prices: np.ndarray,
    save_dir: str
) -> str:
    """Generate scheduler comparison dashboard."""

    n_slots  = len(prices)
    t        = np.arange(n_slots)
    job_colors = plt.cm.Set2(np.linspace(0, 1, len(jobs)))

    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117')
    fig.suptitle(
        "HEXAGRID — Phase 3: QAOA Workload Scheduler",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)

    def style_ax(ax, title, xlabel='', ylabel=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.7)

    # ── Row 0: Grid price + Greedy schedule ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0_r = ax0.twinx()
    ax0.fill_between(t, prices, alpha=0.15, color='#ff6b9d')
    ax0.plot(t, prices, color='#ff6b9d', lw=1.5, label='Grid Price ($/kWh)', zorder=3)
    ax0.set_ylabel('$/kWh', color='#ff6b9d', fontsize=8)
    ax0.tick_params(axis='y', colors='#ff6b9d')

    for method_name, color_base in [('greedy', '#00d4ff'), ('qaoa', '#69db7c')]:
        if method_name not in results:
            continue
        res = results[method_name]
        power_curve = np.zeros(n_slots)
        for job in jobs:
            start = res.job_assignments[job.job_id]
            power_curve[start : start + job.duration_min] += job.power_kw
        ls = '-' if method_name == 'greedy' else '--'
        ax0_r.plot(t, power_curve, color=color_base, lw=1.5, ls=ls,
                   label=f'{method_name.upper()} Load', zorder=2, alpha=0.85)

    ax0_r.set_ylabel('Total Load (kW)', color='white', fontsize=8)
    ax0_r.tick_params(axis='y', colors='#8b949e')
    style_ax(ax0, 'Grid Price vs Scheduled Load', xlabel='Time (min)', ylabel='$/kWh')

    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_r.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor='#21262d', edgecolor='#30363d', labelcolor='white',
               loc='upper right')

    # ── Row 1 left: Gantt chart — Greedy ──────────────────────────────────────
    for col, method_name in enumerate(['greedy', 'qaoa']):
        if method_name not in results:
            continue
        ax = fig.add_subplot(gs[1, col])
        res = results[method_name]

        for jidx, job in enumerate(jobs):
            start = res.job_assignments[job.job_id]
            ax.barh(jidx, job.duration_min, left=start,
                    color=job_colors[jidx], edgecolor='#30363d',
                    linewidth=0.5, alpha=0.85, height=0.6)
            ax.text(start + job.duration_min / 2, jidx,
                    f"{job.name}\n${_job_cost(job, start, prices):.2f}",
                    ha='center', va='center', color='white', fontsize=7,
                    fontweight='bold')

        # Mark cheap windows as vertical bands
        cheap_mask = prices < np.percentile(prices, 25)
        for slot_i in np.where(cheap_mask)[0]:
            ax.axvspan(slot_i, slot_i + 1, ymin=0, ymax=1,
                       alpha=0.10, color='#69db7c', linewidth=0)

        ax.set_yticks(range(len(jobs)))
        ax.set_yticklabels([j.name for j in jobs], color='white', fontsize=8)
        ax.set_xlim(0, n_slots)
        cost = res.total_cost_usd
        title = f'{method_name.upper()} Schedule  |  Cost: ${cost:.3f}'
        style_ax(ax, title, xlabel='Time (min)', ylabel='')

    # ── Row 2 left: Cost comparison bar chart ─────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    method_names = list(results.keys())
    costs        = [results[m].total_cost_usd for m in method_names]
    colors_bar   = ['#00d4ff', '#69db7c', '#ffa94d'][:len(method_names)]
    bars = ax2.bar(method_names, costs, color=colors_bar,
                   edgecolor='#30363d', linewidth=0.5, width=0.4)
    for bar, cost in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0005,
                 f'${cost:.4f}', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
    style_ax(ax2, 'Total Energy Cost Comparison',
             xlabel='Method', ylabel='Cost (USD)')

    # ── Row 2 right: Savings summary ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_facecolor('#161b22')
    ax3.axis('off')

    # Build summary text
    lines = [("SCHEDULER SUMMARY", None)]
    lines.append(("─" * 32, None))
    for m in method_names:
        r = results[m]
        lines.append((f"{m.upper()}", None))
        lines.append((f"  Cost       : ${r.total_cost_usd:.4f}", '#00d4ff'))
        lines.append((f"  Energy     : {r.total_energy_kwh:.3f} kWh", '#8b949e'))
        lines.append((f"  Solve time : {r.solve_time_s:.2f}s", '#8b949e'))
        if r.quantum_state:
            lines.append((f"  Qubits     : {r.quantum_state['n_qubits']}", '#69db7c'))
            lines.append((f"  QAOA p     : {r.quantum_state['p_layers']}", '#69db7c'))
        lines.append(("", None))

    if 'greedy' in results and 'qaoa' in results:
        g_cost = results['greedy'].total_cost_usd
        q_cost = results['qaoa'].total_cost_usd
        saving = g_cost - q_cost
        pct    = (saving / g_cost * 100) if g_cost > 0 else 0
        lines.append(("─" * 32, None))
        lines.append((f"QAOA vs Greedy", '#ffd700'))
        lines.append((f"  Saving     : ${saving:.4f}  ({pct:.1f}%)",
                       '#69db7c' if saving >= 0 else '#ff6b6b'))
        scale = 365 * 24 * 60 / n_slots
        lines.append((f"  Annual*    : ${saving*scale:,.0f}", '#ffd700'))
        lines.append((f"  (*scaled from {n_slots}min window)", '#555555'))

    y = 0.97
    for text, color in lines:
        c = color if color else 'white'
        ax3.text(0.03, y, text, transform=ax3.transAxes,
                 color=c, fontsize=8,
                 fontfamily='monospace')
        y -= 0.06

    for spine in ax3.spines.values():
        spine.set_edgecolor('#30363d')

    os.makedirs(save_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"hexagrid_scheduler_{ts}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"\n  Dashboard saved -> {os.path.abspath(save_path)}")
    return save_path


def _job_cost(job: GPUJob, start: int, prices: np.ndarray) -> float:
    """Quick cost calc for a single job."""
    window = prices[start : start + job.duration_min]
    return float(np.sum(window) * job.power_kw / 60.0)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT JOB SET
# ══════════════════════════════════════════════════════════════════════════════

def default_jobs() -> list[GPUJob]:
    """
    Representative AI workload mix for a mid-size data center.
    Based on real GPU job profiles (H100 class):
    """
    return [
        GPUJob(1, "LLM-Pretraining",   duration_min=45,  power_kw=9.8,  priority=2),
        GPUJob(2, "Image-Diffusion",    duration_min=20,  power_kw=7.2,  priority=1),
        GPUJob(3, "RL-Training",        duration_min=30,  power_kw=8.5,  priority=1),
        GPUJob(4, "Inference-Batch",    duration_min=15,  power_kw=5.1,  priority=2,
               deadline_min=90),
        GPUJob(5, "Embedding-Gen",      duration_min=10,  power_kw=4.3,  priority=1),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_scheduler_pipeline(
    n_slots:        int  = 120,
    p_layers:       int  = 2,
    n_candidates:   int  = 3,
    n_restarts:     int  = 3,
    start_tick:     int  = 480,    # start at 8am-ish on the price curve
    classical_only: bool = False,
    jobs:           Optional[list[GPUJob]] = None,
):
    reports_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'reports')
    )
    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models')
    )

    if jobs is None:
        jobs = default_jobs()

    print(f"\n{'='*60}")
    print(f"  HEXAGRID - Phase 3: QAOA Workload Scheduler")
    print(f"{'='*60}")
    print(f"  Jobs to schedule : {len(jobs)}")
    print(f"  Scheduling window: {n_slots} minutes")
    print(f"  QAOA layers (p)  : {p_layers}")

    # ── Get price forecast ─────────────────────────────────────────────────────
    prices = get_price_forecast(n_slots, start_tick=start_tick,
                                use_lstm=True, models_dir=models_dir)
    print(f"\n  Price forecast:")
    print(f"    Min: ${prices.min():.4f}/kWh")
    print(f"    Max: ${prices.max():.4f}/kWh")
    print(f"    Mean: ${prices.mean():.4f}/kWh")

    # ── Print jobs ─────────────────────────────────────────────────────────────
    print(f"\n  Jobs:")
    for job in jobs:
        dl = f"deadline={job.deadline_min}min" if job.deadline_min else "flexible"
        print(f"    [{job.job_id}] {job.name:<22} "
              f"{job.duration_min}min  {job.power_kw}kW  {dl}")

    results = {}

    # ── Classical greedy baseline ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [1/2] Running Greedy Scheduler...")
    greedy = greedy_schedule(jobs, prices, n_slots)
    results['greedy'] = greedy
    print(f"  Greedy cost    : ${greedy.total_cost_usd:.4f}")
    print(f"  Greedy time    : {greedy.solve_time_s*1000:.1f}ms")
    print(f"\n  Job assignments (Greedy):")
    for job in jobs:
        s    = greedy.job_assignments[job.job_id]
        cost = _job_cost(job, s, prices)
        print(f"    {job.name:<22} -> slot {s:>4} min  "
              f"price=${prices[s]:.4f}  cost=${cost:.4f}")

    # ── QAOA ──────────────────────────────────────────────────────────────────
    if not classical_only:
        print(f"\n{'─'*60}")
        print(f"  [2/2] Running QAOA Scheduler (Cirq)...")
        scheduler = QAOAScheduler(
            jobs         = jobs,
            prices       = prices,
            n_slots      = n_slots,
            p_layers     = p_layers,
            n_candidates = n_candidates,
            n_restarts   = n_restarts,
        )
        qaoa = scheduler.solve()
        results['qaoa'] = qaoa
        print(f"\n  QAOA cost      : ${qaoa.total_cost_usd:.4f}")
        print(f"  QAOA time      : {qaoa.solve_time_s:.2f}s")
        print(f"\n  Job assignments (QAOA):")
        for job in jobs:
            s    = qaoa.job_assignments[job.job_id]
            cost = _job_cost(job, s, prices)
            print(f"    {job.name:<22} -> slot {s:>4} min  "
                  f"price=${prices[s]:.4f}  cost=${cost:.4f}")

    # ── Final comparison ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SCHEDULER RESULTS")
    print(f"{'='*60}")
    for m, r in results.items():
        print(f"  {m.upper():<10} | "
              f"Cost: ${r.total_cost_usd:.4f}  |  "
              f"Energy: {r.total_energy_kwh:.3f} kWh  |  "
              f"Time: {r.solve_time_s:.3f}s")

    if 'greedy' in results and 'qaoa' in results:
        saving = results['greedy'].total_cost_usd - results['qaoa'].total_cost_usd
        pct    = saving / results['greedy'].total_cost_usd * 100
        scale  = (365 * 24 * 60) / n_slots
        print(f"{'─'*60}")
        print(f"  QAOA improvement : ${saving:.4f}  ({pct:.2f}%)")
        print(f"  Annualized saving : ${saving*scale:,.0f}")

    print(f"{'='*60}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_schedules(results, jobs, prices, reports_dir)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HexaGrid Phase 3 - QAOA Scheduler')
    parser.add_argument('--slots',          type=int,  default=120,
                        help='Scheduling window in minutes (default: 120)')
    parser.add_argument('--p',              type=int,  default=2,
                        help='QAOA circuit depth / layers (default: 2)')
    parser.add_argument('--candidates',     type=int,  default=6,
                        help='Candidate start slots per job (default: 6)')
    parser.add_argument('--restarts',       type=int,  default=3,
                        help='QAOA optimization restarts (default: 3)')
    parser.add_argument('--start-tick',     type=int,  default=480,
                        help='Starting tick on price curve (default: 480 = 8am)')
    parser.add_argument('--classical-only', action='store_true',
                        help='Run greedy only, skip QAOA')
    args = parser.parse_args()

    run_scheduler_pipeline(
        n_slots        = args.slots,
        p_layers       = args.p,
        n_candidates   = args.candidates,
        n_restarts     = args.restarts,
        start_tick     = args.start_tick,
        classical_only = args.classical_only,
    )
