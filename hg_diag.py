#!/usr/bin/env python3
"""
hg_diag.py — HexaGrid Fleet Diagnostic Tool
Quantum Clarity LLC · hexagrid.ai

Usage:
    python3 hg_diag.py [--api http://localhost:8000] [--cluster local-dev]
                       [--db /var/lib/hexagrid/telemetry.db]
                       [--json] [--watch] [--interval 30]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── ANSI colour palette ────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    # Foreground
    WHITE   = "\033[97m"
    GREY    = "\033[37m"
    MUTED   = "\033[90m"
    RED     = "\033[91m"
    ORANGE  = "\033[38;5;208m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    PURPLE  = "\033[38;5;141m"

    # Background
    BG_RED    = "\033[41m"
    BG_ORANGE = "\033[48;5;208m"
    BG_GREEN  = "\033[42m"
    BG_DARK   = "\033[48;5;235m"
    BG_DARKER = "\033[48;5;233m"

    @staticmethod
    def strip(text: str) -> str:
        """Remove all ANSI codes — used for plain width calculation."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)

    @staticmethod
    def width(text: str) -> int:
        return len(C.strip(text))


# ── Terminal width ─────────────────────────────────────────────────────────────
def term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 100


# ── Status levels ──────────────────────────────────────────────────────────────
class Status:
    OK       = "OK"
    WARN     = "WARN"
    ALERT    = "ALERT"
    CRITICAL = "CRITICAL"
    UNKNOWN  = "UNKNOWN"
    INFO     = "INFO"

STATUS_ICON = {
    Status.OK:       f"{C.GREEN}✓{C.RESET}",
    Status.WARN:     f"{C.YELLOW}⚠{C.RESET}",
    Status.ALERT:    f"{C.ORANGE}●{C.RESET}",
    Status.CRITICAL: f"{C.RED}✗{C.RESET}",
    Status.UNKNOWN:  f"{C.MUTED}?{C.RESET}",
    Status.INFO:     f"{C.CYAN}i{C.RESET}",
}

STATUS_COLOUR = {
    Status.OK:       C.GREEN,
    Status.WARN:     C.YELLOW,
    Status.ALERT:    C.ORANGE,
    Status.CRITICAL: C.RED,
    Status.UNKNOWN:  C.MUTED,
    Status.INFO:     C.CYAN,
}


@dataclass
class Check:
    name:    str
    status:  str
    value:   str
    unit:    str = ""
    detail:  str = ""
    source:  str = ""   # "api" | "db" | "local"


@dataclass
class Section:
    title:   str
    icon:    str
    checks:  list = field(default_factory=list)

    @property
    def worst_status(self) -> str:
        order = [Status.CRITICAL, Status.ALERT, Status.WARN,
                 Status.UNKNOWN,  Status.INFO,  Status.OK]
        for s in order:
            if any(c.status == s for c in self.checks):
                return s
        return Status.OK


# ── Rendering ──────────────────────────────────────────────────────────────────
def render_header(cluster_id: str, api_url: str, ts: datetime):
    w = term_width()
    print()
    # Top border
    print(f"{C.PURPLE}{'━' * w}{C.RESET}")

    # Brand line
    brand    = f"  {C.BOLD}{C.WHITE}HexaGrid™{C.RESET}  {C.MUTED}Fleet Diagnostic{C.RESET}"
    ts_str   = f"{C.MUTED}{ts.strftime('%Y-%m-%d  %H:%M:%S UTC')}  {C.RESET}"
    gap = w - C.width(brand) - C.width(ts_str) - 2
    print(f"{brand}{' ' * max(gap, 2)}{ts_str}")

    # Cluster / API line
    meta = (f"  {C.MUTED}cluster{C.RESET} {C.CYAN}{cluster_id}{C.RESET}"
            f"  {C.MUTED}api{C.RESET} {C.DIM}{api_url}{C.RESET}")
    print(meta)
    print(f"{C.PURPLE}{'━' * w}{C.RESET}")
    print()


def render_section(section: Section):
    w        = term_width()
    ws       = section.worst_status
    sc       = STATUS_COLOUR[ws]
    ok_count = sum(1 for c in section.checks if c.status == Status.OK)
    total    = len(section.checks)

    # Section header
    header = f"  {section.icon}  {C.BOLD}{C.WHITE}{section.title}{C.RESET}"
    summary_parts = []
    counts = {}
    for c in section.checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    for st in [Status.CRITICAL, Status.ALERT, Status.WARN]:
        if st in counts:
            summary_parts.append(
                f"{STATUS_COLOUR[st]}{counts[st]} {st}{C.RESET}")
    if not summary_parts:
        summary_parts.append(f"{C.GREEN}all clear{C.RESET}")
    summary = "  ".join(summary_parts)

    gap = w - C.width(header) - C.width(summary) - 4
    print(f"{header}{'  ' + summary:>{C.width(summary) + max(gap,2)}}")
    print(f"  {sc}{'─' * (w - 4)}{C.RESET}")

    # Check rows
    col1 = 32   # name column width
    col2 = 22   # value column width
    col3 = 10   # status badge

    for chk in section.checks:
        icon  = STATUS_ICON[chk.status]
        sc2   = STATUS_COLOUR[chk.status]

        name_raw  = chk.name
        name_pad  = name_raw + (C.MUTED + "·" * max(0, col1 - len(name_raw) - 1) + C.RESET)

        val_str   = f"{C.BOLD}{sc2}{chk.value}{C.RESET}"
        if chk.unit:
            val_str += f" {C.MUTED}{chk.unit}{C.RESET}"

        badge_raw = chk.status
        badge_col = sc2
        badge     = f"{badge_col}{badge_raw:<8}{C.RESET}"

        detail_str = f"  {C.MUTED}{chk.detail}{C.RESET}" if chk.detail else ""
        src_str    = f" {C.MUTED}[{chk.source}]{C.RESET}" if chk.source else ""

        line = f"    {icon}  {name_pad}  {val_str:<{col2 + 20}}  {badge}{detail_str}{src_str}"
        print(line)

    print()


def render_summary(sections: list[Section], elapsed: float):
    w = term_width()
    all_checks = [c for s in sections for c in s.checks]

    counts = {s: 0 for s in [Status.OK, Status.WARN, Status.ALERT,
                               Status.CRITICAL, Status.UNKNOWN, Status.INFO]}
    for c in all_checks:
        counts[c.status] = counts.get(c.status, 0) + 1

    critical = counts[Status.CRITICAL]
    alerts   = counts[Status.ALERT]
    warns    = counts[Status.WARN]

    if critical > 0:
        overall     = Status.CRITICAL
        overall_msg = "CRITICAL ISSUES REQUIRE IMMEDIATE ATTENTION"
    elif alerts > 0:
        overall     = Status.ALERT
        overall_msg = "ALERTS REQUIRE ATTENTION"
    elif warns > 0:
        overall     = Status.WARN
        overall_msg = "WARNINGS — REVIEW RECOMMENDED"
    else:
        overall     = Status.OK
        overall_msg = "ALL SYSTEMS NOMINAL"

    oc = STATUS_COLOUR[overall]
    print(f"{oc}{'━' * w}{C.RESET}")

    left  = (f"  {STATUS_ICON[overall]}  {C.BOLD}{oc}{overall_msg}{C.RESET}")
    right = (f"{C.GREEN}{counts[Status.OK]} OK{C.RESET}"
             f"  {C.YELLOW}{counts[Status.WARN]} WARN{C.RESET}"
             f"  {C.ORANGE}{counts[Status.ALERT]} ALERT{C.RESET}"
             f"  {C.RED}{counts[Status.CRITICAL]} CRITICAL{C.RESET}"
             f"  {C.MUTED}({elapsed:.2f}s){C.RESET}  ")

    gap = w - C.width(left) - C.width(right)
    print(f"{left}{' ' * max(gap, 2)}{right}")
    print(f"{oc}{'━' * w}{C.RESET}")
    print()

    # Action items — only things needing attention
    action_items = [c for c in all_checks
                    if c.status in (Status.CRITICAL, Status.ALERT, Status.WARN)]
    if action_items:
        print(f"  {C.BOLD}{C.WHITE}Action Items{C.RESET}")
        print(f"  {C.MUTED}{'─' * 60}{C.RESET}")
        for i, c in enumerate(action_items, 1):
            sc = STATUS_COLOUR[c.status]
            print(f"  {C.BOLD}{sc}{i:>2}.{C.RESET}  "
                  f"{sc}{c.status:<8}{C.RESET}  "
                  f"{C.WHITE}{c.name}{C.RESET}")
            if c.detail:
                print(f"        {C.MUTED}{c.detail}{C.RESET}")
        print()


# ── API helpers ────────────────────────────────────────────────────────────────
def api_get(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ── Diagnostic checks ──────────────────────────────────────────────────────────

def check_api_reachability(api_url: str) -> Section:
    s = Section("API & Control Plane", "⬡")

    # Probe multiple candidate endpoints — FastAPI apps may not have /health.
    # We accept any HTTP response (including 4xx) as "reachable".
    # Only connection errors, timeouts, and DNS failures mean truly down.
    PROBE_CANDIDATES = [
        "/health",
        "/api/v1/telemetry/fleet/now?cluster_id=default",
        "/openapi.json",
        "/docs",
        "/",
    ]

    reached    = False
    hit_url    = None
    hit_status = None
    t0         = time.time()

    for path in PROBE_CANDIDATES:
        url = api_url + path
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(url), timeout=3
            )
            reached    = True
            hit_url    = path
            hit_status = r.status
            break
        except urllib.error.HTTPError as e:
            # Any HTTP response = server is up
            reached    = True
            hit_url    = path
            hit_status = e.code
            break
        except (urllib.error.URLError, OSError):
            continue   # try next candidate

    latency = (time.time() - t0) * 1000

    if not reached:
        s.checks.append(Check(
            "API reachability", Status.CRITICAL,
            "UNREACHABLE",
            detail=f"No response on any probe path from {api_url} — is uvicorn running?",
            source="api",
        ))
        return s

    s.checks.append(Check(
        "API reachability", Status.OK,
        "ONLINE",
        detail=f"Responded on {hit_url} (HTTP {hit_status}) in {latency:.0f}ms",
        source="api",
    ))

    # Latency quality
    if latency < 50:
        lat_status = Status.OK
        lat_detail = "excellent"
    elif latency < 200:
        lat_status = Status.WARN
        lat_detail = "acceptable but elevated"
    else:
        lat_status = Status.ALERT
        lat_detail = "high — check API host load"

    s.checks.append(Check(
        "API response latency", lat_status,
        f"{latency:.0f}", "ms", lat_detail, source="api",
    ))

    # /docs returns HTML not JSON so use raw probe, not api_get
    docs_ok, docs_detail = check_host_reachable(f"{api_url}/docs", timeout=3)
    s.checks.append(Check(
        "OpenAPI / Swagger docs",
        Status.OK if docs_ok else Status.INFO,
        "AVAILABLE" if docs_ok else "NOT FOUND",
        detail="" if docs_ok else "No /docs route — normal for production deployments",
        source="api",
    ))

    return s


def check_fleet_telemetry(api_url: str, cluster_id: str) -> Section:
    s = Section("Fleet Telemetry", "◈")

    data = api_get(f"{api_url}/api/v1/telemetry/fleet/now?cluster_id={cluster_id}")

    if data is None:
        s.checks.append(Check(
            "Fleet endpoint", Status.CRITICAL, "FAILED",
            "Cannot reach /api/v1/telemetry/fleet/now",
            source="api",
        ))
        return s

    s.checks.append(Check(
        "Fleet endpoint", Status.OK, "OK", source="api",
    ))

    # Node count
    nodes = data.get("total_nodes", 0) or 0
    if nodes == 0:
        node_status = Status.CRITICAL
        node_detail = "No agents pushing telemetry — deploy collector_agent.py on GPU nodes"
    elif nodes < 2:
        node_status = Status.INFO
        node_detail = "Single node — expected for dev; check agent deployment for production"
    else:
        node_status = Status.OK
        node_detail = ""
    s.checks.append(Check(
        "Active nodes", node_status, str(nodes), "nodes", node_detail, source="api",
    ))

    # GPU count
    gpus = data.get("total_gpus", 0) or 0
    s.checks.append(Check(
        "Active GPUs", Status.OK if gpus > 0 else Status.CRITICAL,
        str(gpus), "GPUs",
        "" if gpus > 0 else "No GPU telemetry received",
        source="api",
    ))

    # Data freshness
    age = data.get("data_age_s")
    stale = data.get("stale", True)
    if age is None:
        age_status = Status.CRITICAL
        age_detail = "No telemetry received yet"
        age_val    = "N/A"
    elif age <= 15:
        age_status = Status.OK
        age_detail = "live"
        age_val    = f"{age:.1f}"
    elif age <= 30:
        age_status = Status.WARN
        age_detail = "slightly delayed — check agent interval"
        age_val    = f"{age:.1f}"
    elif age <= 60:
        age_status = Status.ALERT
        age_detail = "stale — agent may be struggling to push"
        age_val    = f"{age:.1f}"
    else:
        age_status = Status.CRITICAL
        age_detail = f"agent offline or unreachable ({age:.0f}s since last push)"
        age_val    = f"{age:.1f}"
    s.checks.append(Check(
        "Telemetry freshness", age_status, age_val, "s ago", age_detail, source="api",
    ))

    # Power draw
    power = data.get("total_power_w")
    if power is not None and power > 0:
        s.checks.append(Check(
            "Fleet power draw", Status.OK,
            f"{power:.1f}", "W",
            f"{power/1000:.3f} kW  ·  est. ${power/1000 * 0.08:.4f}/hr at $0.08/kWh",
            source="api",
        ))
    else:
        s.checks.append(Check(
            "Fleet power draw", Status.WARN, "0.0", "W",
            "No power data — check NVML permissions on agent host",
            source="api",
        ))

    # Avg utilisation
    util = data.get("avg_util_pct")
    if util is not None:
        if util < 10:
            u_status = Status.WARN
            u_detail = "very low — GPUs may be idle, consider workload scheduling"
        elif util > 95:
            u_status = Status.WARN
            u_detail = "saturated — memory pressure risk, monitor throttle flags"
        else:
            u_status = Status.OK
            u_detail = ""
        s.checks.append(Check(
            "Avg GPU utilisation", u_status, f"{util:.1f}", "%", u_detail, source="api",
        ))

    # Power cap throttling
    pcap = data.get("power_capped_count", 0) or 0
    s.checks.append(Check(
        "Power-cap throttled GPUs",
        Status.WARN if pcap > 0 else Status.OK,
        str(pcap),
        "GPUs",
        f"{pcap} GPU(s) throttled by power limit — RL agent efficiency reduced" if pcap > 0 else "",
        source="api",
    ))

    # ECC errors
    ecc = data.get("ecc_alert_count", 0) or 0
    s.checks.append(Check(
        "ECC double-bit errors",
        Status.CRITICAL if ecc > 0 else Status.OK,
        str(ecc),
        "GPUs",
        f"{ecc} GPU(s) with uncorrected memory errors — schedule for replacement" if ecc > 0 else "",
        source="api",
    ))

    # MIG
    mig = data.get("total_mig_instances", 0) or 0
    mig_gpus = data.get("mig_gpu_count", 0) or 0
    if mig_gpus > 0:
        s.checks.append(Check(
            "MIG instances", Status.INFO,
            str(mig),
            "instances",
            f"{mig_gpus} GPU(s) in MIG mode — fleet view shows MIG instances",
            source="api",
        ))

    return s


def check_database(db_path: str, cluster_id: str) -> Section:
    s = Section("Database & Storage", "▣")

    # File existence and size
    if not os.path.exists(db_path):
        s.checks.append(Check(
            "Database file", Status.CRITICAL, "MISSING",
            f"Expected at {db_path}",
            source="local",
        ))
        return s

    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    if size_mb < 0.001:
        db_status = Status.WARN
        db_detail = "empty — no data written yet"
    elif size_mb > 4000:
        db_status = Status.WARN
        db_detail = "large — consider enabling retention policy"
    else:
        db_status = Status.OK
        db_detail = ""

    s.checks.append(Check(
        "Database file", db_status,
        f"{size_mb:.2f}", "MB", db_detail, source="local",
    ))

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Row counts and time span
        row_count = conn.execute(
            "SELECT COUNT(*) FROM gpu_telemetry WHERE cluster_id=?",
            (cluster_id,)
        ).fetchone()[0]

        span_row = conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM gpu_telemetry WHERE cluster_id=?",
            (cluster_id,)
        ).fetchone()
        oldest, newest = span_row[0], span_row[1]

        if row_count == 0:
            s.checks.append(Check(
                "Telemetry rows", Status.CRITICAL, "0",
                "rows", f"No data for cluster '{cluster_id}'", source="db",
            ))
        else:
            span_hrs = (newest - oldest) / 3600 if oldest and newest else 0
            s.checks.append(Check(
                "Telemetry rows", Status.OK,
                f"{row_count:,}", "rows",
                f"{span_hrs:.1f}h of history  ·  {row_count/(span_hrs*360 or 1):.1f} rows/min avg",
                source="db",
            ))

        # Write rate (last 2 minutes)
        recent = conn.execute(
            "SELECT COUNT(*) FROM gpu_telemetry WHERE cluster_id=? AND ts > ?",
            (cluster_id, time.time() - 120)
        ).fetchone()[0]
        expected_per_2min = 12   # 1 GPU × 10s interval × 12 = 12 rows in 2min
        if recent == 0:
            wr_status = Status.CRITICAL
            wr_detail = "no writes in last 2 minutes — agent may be offline"
        elif recent < expected_per_2min * 0.5:
            wr_status = Status.WARN
            wr_detail = f"low write rate — expected ~{expected_per_2min}, got {recent} in 2min"
        else:
            wr_status = Status.OK
            wr_detail = f"{recent} rows in last 2min"
        s.checks.append(Check(
            "Write rate", wr_status, str(recent),
            "rows/2min", wr_detail, source="db",
        ))

        # Index coverage
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        required = {
            "idx_gpu_cluster_ts",
            "idx_gpu_telemetry_lookup",
            "idx_gpu_uuid_ts",
            "idx_node_cluster_ts",
        }
        missing = required - indexes
        if missing:
            s.checks.append(Check(
                "Index coverage", Status.WARN,
                f"{len(required)-len(missing)}/{len(required)}",
                "",
                f"Missing: {', '.join(sorted(missing))}",
                source="db",
            ))
        else:
            s.checks.append(Check(
                "Index coverage", Status.OK,
                f"{len(indexes)}", "indexes",
                "all required indexes present",
                source="db",
            ))

        # Node summary alignment
        ns_count = conn.execute(
            "SELECT COUNT(*) FROM node_summary WHERE cluster_id=?",
            (cluster_id,)
        ).fetchone()[0]
        if row_count > 0:
            drift_row = conn.execute("""
                SELECT AVG(ABS(g.ts - n.ts)) AS avg_drift
                FROM gpu_telemetry g
                JOIN node_summary n
                  ON  n.cluster_id = g.cluster_id
                  AND n.node_id    = g.node_id
                  AND ABS(n.ts - g.ts) < 1
                WHERE g.cluster_id = ?
                ORDER BY g.ts DESC
                LIMIT 100
            """, (cluster_id,)).fetchone()
            drift = drift_row[0] if drift_row[0] is not None else None
            if drift is None:
                s.checks.append(Check(
                    "node_summary alignment", Status.WARN,
                    "UNKNOWN", "",
                    "Cannot verify — check join between gpu_telemetry and node_summary",
                    source="db",
                ))
            elif drift < 0.01:
                s.checks.append(Check(
                    "node_summary alignment", Status.OK,
                    f"{drift*1000:.1f}", "ms drift",
                    "atomic writes confirmed", source="db",
                ))
            else:
                s.checks.append(Check(
                    "node_summary alignment", Status.WARN,
                    f"{drift*1000:.1f}", "ms drift",
                    "non-atomic writes — join queries may return mismatched rows",
                    source="db",
                ))

        # Stale nodes — nodes that haven't pushed in > 60s
        stale_nodes = conn.execute("""
            SELECT node_id, MAX(ts) as last_ts, (? - MAX(ts)) as age_s
            FROM gpu_telemetry
            WHERE cluster_id = ?
            GROUP BY node_id
            HAVING age_s > 60
        """, (time.time(), cluster_id)).fetchall()

        if stale_nodes:
            for sn in stale_nodes:
                s.checks.append(Check(
                    f"Node {sn['node_id']}", Status.ALERT,
                    f"{sn['age_s']:.0f}s", "since last push",
                    "agent may have crashed or lost network",
                    source="db",
                ))
        else:
            active_nodes = conn.execute(
                "SELECT COUNT(DISTINCT node_id) FROM gpu_telemetry WHERE cluster_id=? AND ts > ?",
                (cluster_id, time.time() - 60)
            ).fetchone()[0]
            s.checks.append(Check(
                "Node heartbeats", Status.OK,
                str(active_nodes), "nodes live",
                "all nodes pushing within 60s", source="db",
            ))

        # ECC alerts in DB
        ecc_gpus = conn.execute("""
            SELECT DISTINCT gpu_uuid, node_id, ecc_dbe_aggregate
            FROM gpu_telemetry
            WHERE cluster_id = ?
              AND ecc_dbe_aggregate > 0
              AND ts > ?
        """, (cluster_id, time.time() - 300)).fetchall()

        if ecc_gpus:
            for eg in ecc_gpus:
                s.checks.append(Check(
                    f"ECC: {eg['gpu_uuid'][:16]}…",
                    Status.CRITICAL,
                    str(eg['ecc_dbe_aggregate']),
                    "double-bit errors",
                    f"node={eg['node_id']} — GPU requires hardware inspection",
                    source="db",
                ))

        # Disk space
        stat = os.statvfs(db_path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        used_pct = 100 * (1 - stat.f_bavail / stat.f_blocks) if stat.f_blocks else 0

        if used_pct > 90:
            disk_status = Status.CRITICAL
            disk_detail = "disk nearly full — data loss risk"
        elif used_pct > 75:
            disk_status = Status.WARN
            disk_detail = "disk filling — schedule cleanup"
        else:
            disk_status = Status.OK
            disk_detail = f"{free_gb:.1f} GB free of {total_gb:.1f} GB"
        s.checks.append(Check(
            "Disk space", disk_status,
            f"{used_pct:.1f}", "%",
            disk_detail, source="local",
        ))

        # Fallback DB
        fallback_path = db_path.replace("telemetry.db", "fallback.db")
        if os.path.exists(fallback_path):
            fb_size = os.path.getsize(fallback_path)
            if fb_size > 1024 * 100:   # >100KB means undelivered buffered data
                fb_conn = sqlite3.connect(fallback_path)
                fb_rows = fb_conn.execute("SELECT COUNT(*) FROM fallback_queue").fetchone()
                fb_count = fb_rows[0] if fb_rows else 0
                fb_conn.close()
                if fb_count > 0:
                    s.checks.append(Check(
                        "Agent fallback buffer", Status.WARN,
                        str(fb_count), "buffered payloads",
                        "agent has unsent data — check control plane connectivity",
                        source="local",
                    ))
                else:
                    s.checks.append(Check(
                        "Agent fallback buffer", Status.OK, "empty",
                        "", "no buffered payloads", source="local",
                    ))
            else:
                s.checks.append(Check(
                    "Agent fallback buffer", Status.OK,
                    "empty", "", source="local",
                ))

        conn.close()

    except sqlite3.Error as e:
        s.checks.append(Check(
            "Database access", Status.CRITICAL, "ERROR",
            str(e), source="db",
        ))

    return s


def check_gpu_health(db_path: str, cluster_id: str) -> Section:
    s = Section("GPU Hardware Health", "⬢")

    if not os.path.exists(db_path):
        s.checks.append(Check("Database", Status.UNKNOWN, "N/A",
                               "Database not found", source="local"))
        return s

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Latest row per (node, gpu)
        gpus = conn.execute("""
            SELECT g.node_id, g.rack_id, g.gpu_name, g.gpu_uuid,
                   g.temp_c, g.power_draw_w, g.power_limit_w, g.power_util_pct,
                   g.util_gpu_pct, g.util_mem_pct,
                   g.clock_sm_mhz, g.throttle_bitmask,
                   g.throttle_power_capped, g.throttle_hw_thermal,
                   g.ecc_dbe_aggregate, g.ecc_dbe_volatile,
                   g.mem_used_mb, g.mem_total_mb,
                   (? - g.ts) AS data_age_s
            FROM gpu_telemetry g
            WHERE g.cluster_id = ?
              AND g.ts = (
                  SELECT MAX(ts) FROM gpu_telemetry g2
                  WHERE g2.cluster_id = g.cluster_id
                    AND g2.node_id    = g.node_id
                    AND g2.gpu_uuid   = g.gpu_uuid
              )
            ORDER BY g.rack_id, g.node_id, g.gpu_uuid
        """, (time.time(), cluster_id)).fetchall()

        if not gpus:
            s.checks.append(Check(
                "GPU records", Status.CRITICAL, "NONE",
                f"No GPU data for cluster '{cluster_id}'",
                source="db",
            ))
            conn.close()
            return s

        s.checks.append(Check(
            "GPU records found", Status.OK,
            str(len(gpus)), "GPUs",
            source="db",
        ))

        for gpu in gpus:
            label = f"{gpu['node_id']} · GPU {gpu['gpu_uuid'][:8]}…"
            model = (gpu['gpu_name'] or 'Unknown')[:28]

            # Temperature
            temp = gpu['temp_c']
            if temp is not None:
                if temp >= 90:
                    t_st = Status.CRITICAL
                    t_dt = f"critical thermal — throttling likely, check cooling [{model}]"
                elif temp >= 83:
                    t_st = Status.ALERT
                    t_dt = f"high temp — monitor closely [{model}]"
                elif temp >= 75:
                    t_st = Status.WARN
                    t_dt = f"elevated — ensure airflow [{model}]"
                else:
                    t_st = Status.OK
                    t_dt = model
                s.checks.append(Check(
                    f"Temp · {label}", t_st,
                    f"{temp:.0f}", "°C", t_dt, source="db",
                ))

            # Power utilisation
            putil = gpu['power_util_pct']
            if putil is not None:
                if putil >= 98:
                    p_st = Status.WARN
                    p_dt = f"at power limit — throttle risk [{model}]"
                else:
                    p_st = Status.OK
                    p_dt = (f"{gpu['power_draw_w']:.1f}W / "
                            f"{gpu['power_limit_w']:.0f}W limit" if gpu['power_limit_w'] else "")
                s.checks.append(Check(
                    f"Power · {label}", p_st,
                    f"{putil:.1f}", "%", p_dt, source="db",
                ))

            # Memory utilisation
            if gpu['mem_total_mb'] and gpu['mem_used_mb'] is not None:
                mem_pct = gpu['mem_used_mb'] / gpu['mem_total_mb'] * 100
                if mem_pct > 95:
                    m_st = Status.ALERT
                    m_dt = "near OOM — jobs may crash"
                elif mem_pct > 85:
                    m_st = Status.WARN
                    m_dt = "high memory pressure"
                else:
                    m_st = Status.OK
                    m_dt = (f"{gpu['mem_used_mb']:,} / "
                            f"{gpu['mem_total_mb']:,} MB")
                s.checks.append(Check(
                    f"VRAM · {label}", m_st,
                    f"{mem_pct:.1f}", "%", m_dt, source="db",
                ))

            # Throttle flags
            throttle = gpu['throttle_bitmask']
            if throttle is not None and throttle not in (0, 1):  # 1 = GpuIdle, not a problem
                active_flags = []
                flag_map = {
                    0x04:  "SwPowerCap",
                    0x08:  "HwSlowdown",
                    0x40:  "SwThermal",
                    0x80:  "HwThermal",
                    0x100: "HwPowerBrake",
                }
                for mask, name in flag_map.items():
                    if throttle & mask:
                        active_flags.append(name)
                if active_flags:
                    th_st = Status.CRITICAL if any(
                        f in active_flags for f in ("HwThermal", "HwSlowdown", "HwPowerBrake")
                    ) else Status.WARN
                    s.checks.append(Check(
                        f"Throttle · {label}", th_st,
                        " | ".join(active_flags), "",
                        f"performance degraded — bitmask=0x{throttle:X} [{model}]",
                        source="db",
                    ))

            # ECC errors
            ecc_agg = gpu['ecc_dbe_aggregate']
            ecc_vol = gpu['ecc_dbe_volatile']
            if ecc_agg and ecc_agg > 0:
                s.checks.append(Check(
                    f"ECC · {label}", Status.CRITICAL,
                    str(ecc_agg), "double-bit errors",
                    f"aggregate={ecc_agg}  volatile={ecc_vol}  "
                    f"— GPU requires hardware inspection, consider draining [{model}]",
                    source="db",
                ))
            elif ecc_vol and ecc_vol > 0:
                s.checks.append(Check(
                    f"ECC volatile · {label}", Status.WARN,
                    str(ecc_vol), "corrected errors",
                    f"single-bit errors corrected — monitor for escalation [{model}]",
                    source="db",
                ))

        conn.close()

    except sqlite3.Error as e:
        s.checks.append(Check(
            "GPU health query", Status.CRITICAL, "ERROR",
            str(e), source="db",
        ))

    return s


def check_rack_summary(api_url: str, cluster_id: str) -> Section:
    s = Section("Rack Summary", "▦")

    data = api_get(
        f"{api_url}/api/v1/telemetry/fleet/racks?cluster_id={cluster_id}"
    )

    if data is None:
        s.checks.append(Check(
            "Rack endpoint", Status.WARN, "UNAVAILABLE",
            "/api/v1/telemetry/fleet/racks not reachable",
            source="api",
        ))
        return s

    racks = data if isinstance(data, list) else data.get("racks", [])

    if not racks:
        s.checks.append(Check(
            "Rack data", Status.INFO, "NONE",
            "No rack data returned — agent RACK_ID may not be set",
            source="api",
        ))
        return s

    s.checks.append(Check(
        "Racks reporting", Status.OK,
        str(len(racks)), "racks", source="api",
    ))

    for rack in racks:
        rid   = rack.get("rack_id", "unknown")
        pw    = rack.get("rack_power_w") or rack.get("total_power_w", 0)
        util  = rack.get("avg_util_pct", 0)
        gpus  = rack.get("gpu_count") or rack.get("total_gpus", 0)
        ecc   = rack.get("ecc_error_gpus") or rack.get("ecc_alert_count", 0)
        pcap  = rack.get("power_capped_gpus") or rack.get("power_capped_count", 0)
        age   = rack.get("data_age_s")

        rack_status = Status.OK
        rack_detail = f"{gpus} GPUs  ·  {util:.0f}% util"

        if ecc and ecc > 0:
            rack_status = Status.CRITICAL
            rack_detail += f"  ·  {ecc} ECC ERROR(S)"
        elif pcap and pcap > 0:
            rack_status = Status.WARN
            rack_detail += f"  ·  {pcap} power-capped"
        elif age and age > 60:
            rack_status = Status.ALERT
            rack_detail += f"  ·  stale ({age:.0f}s)"

        s.checks.append(Check(
            f"Rack {rid}", rack_status,
            f"{pw:.1f}", "W", rack_detail, source="api",
        ))

    return s


def check_host_reachable(url: str, timeout: int = 5) -> tuple[bool, str]:
    """Probe a URL — any HTTP response means reachable, only network errors mean down."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, "reachable"
    except urllib.error.HTTPError as e:
        return True, f"reachable (HTTP {e.code})"
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "timed out" in reason.lower():
            return False, "timeout — check network/firewall"
        return False, f"unreachable — {reason}"
    except OSError as e:
        return False, f"connection error — {e}"


def check_environment(api_url: str, db_path: str) -> Section:
    s = Section("Environment", "◉")

    # Python version
    pv = sys.version_info
    py_ok = pv.major == 3 and pv.minor >= 9
    s.checks.append(Check(
        "Python version",
        Status.OK if py_ok else Status.WARN,
        f"{pv.major}.{pv.minor}.{pv.micro}",
        "",
        "" if py_ok else "Python 3.9+ recommended",
        source="local",
    ))

    # pynvml availability
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        count  = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        s.checks.append(Check(
            "pynvml / NVML", Status.OK,
            f"{count} GPU(s)",
            "",
            f"driver {driver}",
            source="local",
        ))
    except ImportError:
        s.checks.append(Check(
            "pynvml / NVML", Status.WARN,
            "NOT INSTALLED",
            "",
            "pip install pynvml — required for GPU telemetry collection",
            source="local",
        ))
    except Exception as e:
        s.checks.append(Check(
            "pynvml / NVML", Status.WARN,
            "UNAVAILABLE",
            "",
            str(e)[:60],
            source="local",
        ))

    # EIA API reachability

    eia_ok, eia_detail = check_host_reachable("https://api.eia.gov/v2/", timeout=5)
    s.checks.append(Check(
        "EIA API (grid prices)",
        Status.OK if eia_ok else Status.WARN,
        "api.eia.gov", "",
        eia_detail if eia_ok else f"{eia_detail} — grid price data will not update",
        source="local",
    ))

    # Electricity Maps API reachability
    em_ok, em_detail = check_host_reachable("https://api.electricitymap.org", timeout=5)
    s.checks.append(Check(
        "Electricity Maps (carbon)",
        Status.OK if em_ok else Status.WARN,
        "api.electricitymap.org", "",
        em_detail if em_ok else f"{em_detail} — carbon intensity data will not update",
        source="local",
    ))

    # DB path writable
    db_dir = os.path.dirname(db_path)
    if os.path.exists(db_dir) and os.access(db_dir, os.W_OK):
        s.checks.append(Check(
            "DB directory writable", Status.OK,
            db_dir, "", source="local",
        ))
    else:
        s.checks.append(Check(
            "DB directory writable", Status.CRITICAL,
            db_dir, "",
            "Cannot write to DB directory — check permissions",
            source="local",
        ))

    return s


# ── JSON output ────────────────────────────────────────────────────────────────
def to_json(sections: list[Section]) -> dict:
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": []
    }
    for s in sections:
        out["sections"].append({
            "title":        s.title,
            "worst_status": s.worst_status,
            "checks": [
                {
                    "name":   c.name,
                    "status": c.status,
                    "value":  c.value,
                    "unit":   c.unit,
                    "detail": c.detail,
                    "source": c.source,
                }
                for c in s.checks
            ]
        })
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def run_diagnostics(api_url: str, cluster_id: str, db_path: str,
                    json_out: bool = False) -> list[Section]:
    t0 = time.time()
    ts = datetime.now(timezone.utc)

    sections = [
        check_environment(api_url, db_path),
        check_api_reachability(api_url),
        check_fleet_telemetry(api_url, cluster_id),
        check_database(db_path, cluster_id),
        check_gpu_health(db_path, cluster_id),
        check_rack_summary(api_url, cluster_id),
    ]

    elapsed = time.time() - t0

    if json_out:
        result = to_json(sections)
        result["elapsed_s"] = round(elapsed, 3)
        print(json.dumps(result, indent=2))
        return sections

    render_header(cluster_id, api_url, ts)
    for section in sections:
        render_section(section)
    render_summary(sections, elapsed)

    return sections


def main():
    parser = argparse.ArgumentParser(
        description="HexaGrid Fleet Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hg_diag.py
  python3 hg_diag.py --cluster production --api http://hexagrid-host:8000
  python3 hg_diag.py --json | jq '.sections[] | select(.worst_status != "OK")'
  python3 hg_diag.py --watch --interval 30
        """,
    )
    parser.add_argument("--api",      default="http://localhost:8000",
                        help="HexaGrid API base URL (default: http://localhost:8000)")
    parser.add_argument("--cluster",  default="local-dev",
                        help="Cluster ID to inspect (default: local-dev)")
    parser.add_argument("--db",       default="/var/lib/hexagrid/telemetry.db",
                        help="Path to telemetry SQLite DB")
    parser.add_argument("--json",     action="store_true",
                        help="Output JSON instead of formatted terminal output")
    parser.add_argument("--watch",    action="store_true",
                        help="Re-run diagnostics on an interval (like watch)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Refresh interval in seconds for --watch mode (default: 30)")

    args = parser.parse_args()

    if args.watch and not args.json:
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")
                sections = run_diagnostics(args.api, args.cluster, args.db)
                print(f"  {C.MUTED}Refreshing every {args.interval}s  ·  "
                      f"Ctrl+C to exit{C.RESET}\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{C.MUTED}Diagnostic watch stopped.{C.RESET}\n")
    else:
        sections = run_diagnostics(args.api, args.cluster, args.db,
                                   json_out=args.json)
        # Exit code reflects worst status
        all_checks = [c for s in sections for c in s.checks]
        if any(c.status == Status.CRITICAL for c in all_checks):
            sys.exit(2)
        elif any(c.status in (Status.ALERT, Status.WARN) for c in all_checks):
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
