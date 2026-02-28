"""
collector_agent.py — HexaGrid GPU Telemetry Collector
======================================================
Deploys on every GPU node as a systemd service or Docker sidecar.
Pushes structured GPU telemetry to the HexaGrid control plane every
POLL_INTERVAL seconds.

MIG-AWARE: Detects Multi-Instance GPU mode per device and collects
both physical GPU metrics and per-MIG-instance metrics in one payload.
The control plane (and dashboard) can toggle between two views:
  • Physical view  — one record per physical GPU (8 GPUs in an H100 DGX)
  • MIG view       — one record per MIG compute instance (up to 56 per node)

Environment variables
─────────────────────
HEXAGRID_ENDPOINT   URL of the HexaGrid telemetry receiver
                    Default: http://hexagrid-control-plane:8000/api/v1/telemetry/gpu
NODE_ID             Node identifier. Default: hostname
RACK_ID             Rack identifier. Set in deployment config. Default: "unknown"
CLUSTER_ID          Cluster identifier. Default: "default"
POLL_INTERVAL_S     Collection interval in seconds. Default: 10
HEXAGRID_TOKEN      Bearer token for receiver authentication
SQLITE_FALLBACK     Path to local SQLite fallback DB.
                    Default: /var/lib/hexagrid/fallback.db
                    Set to "" to disable local fallback.
COLLECT_MIG_MODE    "auto"     — detect per-device, collect both (default)
                    "physical" — physical GPUs only, ignore MIG slices
                    "mig"      — MIG instances only (skip non-MIG GPUs)
LOG_LEVEL           DEBUG / INFO / WARNING. Default: INFO
"""

import os
import sys
import time
import json
import socket
import random
import logging
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import requests

# ── pynvml import with graceful degradation ─────────────────────────────────
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not installed. Run: pip install nvidia-ml-py3")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

HEXAGRID_ENDPOINT = os.environ.get(
    "HEXAGRID_ENDPOINT",
    "http://hexagrid-control-plane:8000/api/v1/telemetry/gpu"
)
NODE_ID         = os.environ.get("NODE_ID",        socket.gethostname())
RACK_ID         = os.environ.get("RACK_ID",        "unknown")
CLUSTER_ID      = os.environ.get("CLUSTER_ID",     "default")
POLL_INTERVAL   = int(os.environ.get("POLL_INTERVAL_S", "10"))
AUTH_TOKEN      = os.environ.get("HEXAGRID_TOKEN", "")
SQLITE_PATH     = os.environ.get("SQLITE_FALLBACK", "/var/lib/hexagrid/fallback.db")
COLLECT_MIG     = os.environ.get("COLLECT_MIG_MODE", "auto").lower()
LOG_LEVEL       = os.environ.get("LOG_LEVEL", "INFO").upper()

# Throttle bitmask constants — documented for clarity
THROTTLE_GPU_IDLE           = 0x0000000000000001
THROTTLE_APP_CLOCK          = 0x0000000000000002
THROTTLE_SW_POWER_CAP       = 0x0000000000000004  # ← HexaGrid RL reward signal
THROTTLE_HW_SLOWDOWN        = 0x0000000000000008
THROTTLE_SYNC_BOOST         = 0x0000000000000020
THROTTLE_SW_THERMAL         = 0x0000000000000040
THROTTLE_HW_THERMAL         = 0x0000000000000080  # ← critical: GPU overheating
THROTTLE_HW_POWER_BRAKE     = 0x0000000000000100

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] hg-agent %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("hg-agent")


# ════════════════════════════════════════════════════════════════════════════
# SQLITE FALLBACK — local ring buffer when control plane is unreachable
# ════════════════════════════════════════════════════════════════════════════

def _init_sqlite(path: str) -> Optional[sqlite3.Connection]:
    """
    Create local SQLite DB for fallback buffering.
    Stores up to FALLBACK_RETAIN_HOURS hours of readings.
    Returns None if path is empty (fallback disabled).
    """
    if not path:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_buffer (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                payload_json TEXT   NOT NULL,
                sent        INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_unsent ON telemetry_buffer(sent, ts)")
        conn.commit()
        log.info(f"SQLite fallback ready at {path}")
        return conn
    except Exception as e:
        log.warning(f"SQLite fallback unavailable: {e}")
        return None


def _sqlite_write(conn: sqlite3.Connection, payload: dict) -> None:
    """Write one payload to the fallback buffer."""
    try:
        conn.execute(
            "INSERT INTO telemetry_buffer (ts, payload_json) VALUES (?, ?)",
            (payload["timestamp"], json.dumps(payload))
        )
        conn.commit()
        # Purge entries older than 24 hours to cap disk usage
        conn.execute(
            "DELETE FROM telemetry_buffer WHERE ts < ? AND sent = 1",
            (time.time() - 86400,)
        )
        conn.commit()
    except Exception as e:
        log.warning(f"SQLite write failed: {e}")


def _sqlite_drain(conn: sqlite3.Connection, endpoint: str, headers: dict) -> int:
    """
    Replay unsent buffered payloads to the control plane.
    Called when connectivity is restored. Returns count sent.
    """
    try:
        rows = conn.execute(
            "SELECT id, payload_json FROM telemetry_buffer WHERE sent = 0 ORDER BY ts LIMIT 100"
        ).fetchall()
        sent = 0
        for row_id, payload_json in rows:
            try:
                r = requests.post(endpoint, data=payload_json,
                                  headers=headers, timeout=5)
                r.raise_for_status()
                conn.execute("UPDATE telemetry_buffer SET sent = 1 WHERE id = ?", (row_id,))
                sent += 1
            except Exception:
                break  # Stop replaying on first failure — don't flood
        conn.commit()
        if sent:
            log.info(f"Drained {sent} buffered payloads")
        return sent
    except Exception as e:
        log.warning(f"SQLite drain failed: {e}")
        return 0


# ════════════════════════════════════════════════════════════════════════════
# MIG DETECTION AND ENUMERATION
# ════════════════════════════════════════════════════════════════════════════

def _is_mig_enabled(handle) -> bool:
    """
    Check whether MIG mode is active on this physical device.
    Returns False if the GPU doesn't support MIG or MIG is disabled.
    """
    try:
        current_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
        return current_mode == pynvml.NVML_DEVICE_MIG_ENABLE
    except pynvml.NVMLError:
        return False


def _collect_mig_instances(handle, physical_index: int) -> List[Dict[str, Any]]:
    """
    Enumerate all active MIG compute instances on a physical GPU handle.

    MIG hierarchy on H100/A100:
      Physical GPU
        └── GPU Instance (GI)  — slice of memory + SM partition
              └── Compute Instance (CI) — schedulable unit

    pynvml exposes MIG devices as "child" handles accessible via
    nvmlDeviceGetMigDeviceHandleByIndex(). Each MIG device maps to one
    Compute Instance and has its own UUID, memory partition, and utilization.

    Power and temperature are ONLY available at the physical device level —
    they are pro-rated across MIG instances by memory fraction.
    """
    mig_instances = []

    # How many MIG device slots exist (not all may be populated)
    try:
        max_mig = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
    except pynvml.NVMLError:
        return mig_instances

    # Physical-level power and temp — shared across all instances
    try:
        phys_power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        phys_temp_c  = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        phys_mem      = pynvml.nvmlDeviceGetMemoryInfo(handle)
        phys_total_mb = phys_mem.total // (1024 * 1024)
    except pynvml.NVMLError:
        phys_power_w  = None
        phys_temp_c   = None
        phys_total_mb = None

    # Physical-level throttle and ECC — applies to all MIG instances
    phys_throttle = _safe_throttle(handle)
    phys_ecc      = _safe_ecc(handle)

    active_count = 0
    for mig_idx in range(max_mig):
        try:
            mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, mig_idx)
        except pynvml.NVMLError_NotFound:
            continue  # This slot is not populated — normal
        except pynvml.NVMLError:
            continue

        active_count += 1

        # ── MIG instance identity ─────────────────────────────────────────
        try:
            mig_uuid = pynvml.nvmlDeviceGetUUID(mig_handle)
            # UUID format: "MIG-GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/N/M"
            # The /N/M suffix identifies GI index / CI index
        except pynvml.NVMLError:
            mig_uuid = f"MIG-{physical_index}-{mig_idx}-unknown"

        try:
            mig_name = pynvml.nvmlDeviceGetName(mig_handle)
            # e.g. "NVIDIA A100-SXM4-40GB MIG 3g.20gb"
        except pynvml.NVMLError:
            mig_name = "MIG instance"

        # ── GPU Instance and Compute Instance IDs ────────────────────────
        try:
            gi_id = pynvml.nvmlDeviceGetGpuInstanceId(mig_handle)
        except pynvml.NVMLError:
            gi_id = None

        try:
            ci_id = pynvml.nvmlDeviceGetComputeInstanceId(mig_handle)
        except pynvml.NVMLError:
            ci_id = None

        # ── MIG instance memory ───────────────────────────────────────────
        try:
            mig_mem        = pynvml.nvmlDeviceGetMemoryInfo(mig_handle)
            mig_mem_total  = mig_mem.total  // (1024 * 1024)
            mig_mem_used   = mig_mem.used   // (1024 * 1024)
            mig_mem_free   = mig_mem.free   // (1024 * 1024)
            mig_mem_pct    = round(mig_mem.used / mig_mem.total * 100, 1) if mig_mem.total else 0
            # Memory fraction of the physical GPU — used to prorate power
            mem_fraction   = mig_mem.total / phys_mem.total if phys_total_mb else None
        except pynvml.NVMLError:
            mig_mem_total = mig_mem_used = mig_mem_free = mig_mem_pct = None
            mem_fraction  = None

        # ── MIG instance utilization ──────────────────────────────────────
        try:
            mig_util     = pynvml.nvmlDeviceGetUtilizationRates(mig_handle)
            util_gpu_pct = mig_util.gpu
            util_mem_pct = mig_util.memory
        except pynvml.NVMLError:
            util_gpu_pct = util_mem_pct = None

        # ── MIG instance clocks ───────────────────────────────────────────
        try:
            clock_sm  = pynvml.nvmlDeviceGetClockInfo(mig_handle, pynvml.NVML_CLOCK_SM)
            max_clock = pynvml.nvmlDeviceGetMaxClockInfo(mig_handle, pynvml.NVML_CLOCK_SM)
            clock_throttle_pct = round(clock_sm / max_clock * 100, 1) if max_clock else None
        except pynvml.NVMLError:
            clock_sm = max_clock = clock_throttle_pct = None

        # ── Running processes on this MIG instance ────────────────────────
        procs = _safe_processes(mig_handle)

        # ── Pro-rated power estimate ──────────────────────────────────────
        # Power is not available per-MIG — distribute by memory fraction.
        # This is the standard approach used by DCGM and nvidia-smi.
        prorated_power_w = None
        if phys_power_w is not None and mem_fraction is not None:
            prorated_power_w = round(phys_power_w * mem_fraction, 2)

        mig_instances.append({
            "mig_index":          mig_idx,
            "mig_uuid":           mig_uuid,
            "mig_name":           mig_name,
            "gpu_instance_id":    gi_id,
            "compute_instance_id": ci_id,
            "physical_gpu_index": physical_index,
            "memory": {
                "total_mb":       mig_mem_total,
                "used_mb":        mig_mem_used,
                "free_mb":        mig_mem_free,
                "used_pct":       mig_mem_pct,
                "phys_fraction":  round(mem_fraction, 4) if mem_fraction else None,
            },
            "utilisation": {
                "gpu_pct":        util_gpu_pct,
                "memory_bw_pct":  util_mem_pct,
            },
            "clocks": {
                "sm_mhz":             clock_sm,
                "max_sm_mhz":         max_clock,
                "sm_throttle_pct":    clock_throttle_pct,
            },
            # Physical GPU metrics — same for all instances on this GPU
            "physical": {
                "temp_c":             phys_temp_c,
                "power_draw_w":       phys_power_w,
                "prorated_power_w":   prorated_power_w,
                "throttle_bitmask":   phys_throttle,
                "ecc_dbe_aggregate":  phys_ecc,
            },
            "processes":          procs,
        })

    log.debug(f"GPU[{physical_index}]: {active_count} MIG instances active (of {max_mig} slots)")
    return mig_instances


# ════════════════════════════════════════════════════════════════════════════
# PHYSICAL GPU COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def _safe_call(fn, *args, default=None):
    """Execute an NVML call, return default on any NVMLError."""
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return default


def _safe_throttle(handle) -> Optional[int]:
    return _safe_call(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle)


def _safe_ecc(handle) -> Optional[int]:
    return _safe_call(
        pynvml.nvmlDeviceGetTotalEccErrors, handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
        pynvml.NVML_AGGREGATE_ECC,
    )


def _safe_processes(handle) -> List[dict]:
    try:
        return [
            {
                "pid":     p.pid,
                "mem_mb":  p.usedGpuMemory // (1024 * 1024) if p.usedGpuMemory else None,
            }
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ]
    except pynvml.NVMLError:
        return []


def _collect_physical_gpu(handle, index: int) -> dict:
    """
    Collect full telemetry for one physical GPU.
    Called for both non-MIG GPUs and as the parent record for MIG GPUs.
    """
    # ── Identity ────────────────────────────────────────────────────────────
    uuid   = _safe_call(pynvml.nvmlDeviceGetUUID,    handle, default=f"gpu-{index}")
    name   = _safe_call(pynvml.nvmlDeviceGetName,    handle, default="Unknown GPU")
    pci    = _safe_call(pynvml.nvmlDeviceGetPciInfo, handle)
    bus_id = pci.busId if pci else None

    # serial() requires root on many systems — always fall back gracefully
    serial = _safe_call(pynvml.nvmlDeviceGetSerial,  handle, default=None)

    # ── Utilisation ─────────────────────────────────────────────────────────
    util = _safe_call(pynvml.nvmlDeviceGetUtilizationRates, handle)
    util_gpu_pct = util.gpu    if util else None
    util_mem_pct = util.memory if util else None

    # ── Memory ──────────────────────────────────────────────────────────────
    mem = _safe_call(pynvml.nvmlDeviceGetMemoryInfo, handle)
    if mem:
        mem_total_mb = mem.total // (1024 * 1024)
        mem_used_mb  = mem.used  // (1024 * 1024)
        mem_free_mb  = mem.free  // (1024 * 1024)
        mem_used_pct = round(mem.used / mem.total * 100, 1) if mem.total else 0
    else:
        mem_total_mb = mem_used_mb = mem_free_mb = mem_used_pct = None

    # ── Temperature ─────────────────────────────────────────────────────────
    temp_c = _safe_call(
        pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
    )

    fan_pct = _safe_call(pynvml.nvmlDeviceGetFanSpeed, handle, default=None)

    # ── Clocks ──────────────────────────────────────────────────────────────
    clock_sm  = _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM)
    clock_mem = _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM)
    clock_gr  = _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS)
    max_sm    = _safe_call(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM)
    max_mem   = _safe_call(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM)
    sm_throttle_pct = round(clock_sm / max_sm * 100, 1) if (clock_sm and max_sm) else None

    # ── Power ───────────────────────────────────────────────────────────────
    power_draw_mw    = _safe_call(pynvml.nvmlDeviceGetPowerUsage,                 handle)
    power_limit_mw   = _safe_call(pynvml.nvmlDeviceGetPowerManagementLimit,       handle)
    power_default_mw = _safe_call(pynvml.nvmlDeviceGetPowerManagementDefaultLimit,handle)
    # energy_total is mJ since driver load — store delta, not absolute
    energy_mj        = _safe_call(pynvml.nvmlDeviceGetTotalEnergyConsumption,     handle)

    power_draw_w    = power_draw_mw    / 1000 if power_draw_mw    is not None else None
    power_limit_w   = power_limit_mw   / 1000 if power_limit_mw   is not None else None
    power_default_w = power_default_mw / 1000 if power_default_mw is not None else None
    power_util_pct  = round(power_draw_mw / power_limit_mw * 100, 1) \
                      if (power_draw_mw and power_limit_mw) else None
    energy_kj       = energy_mj / 1_000_000 if energy_mj is not None else None

    # ── PCIe ────────────────────────────────────────────────────────────────
    pcie_tx = _safe_call(
        pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
    )
    pcie_rx = _safe_call(
        pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
    )

    # ── Throttle bitmask ────────────────────────────────────────────────────
    throttle = _safe_throttle(handle)
    throttle_decoded = _decode_throttle(throttle) if throttle is not None else {}

    # ── ECC ─────────────────────────────────────────────────────────────────
    ecc_sbe_volatile = _safe_call(
        pynvml.nvmlDeviceGetTotalEccErrors, handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_VOLATILE_ECC,
    )
    ecc_dbe_volatile = _safe_call(
        pynvml.nvmlDeviceGetTotalEccErrors, handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC,
    )
    ecc_dbe_aggregate = _safe_ecc(handle)

    # ── NVLink ──────────────────────────────────────────────────────────────
    nvlink_active = []
    try:
        for lnk in range(pynvml.NVML_NVLINK_MAX_LINKS):
            if _safe_call(pynvml.nvmlDeviceGetNvLinkState, handle, lnk):
                nvlink_active.append(lnk)
    except Exception:
        pass

    # ── Processes ───────────────────────────────────────────────────────────
    processes = _safe_processes(handle)

    return {
        "index":              index,
        "uuid":               uuid,
        "name":               name,
        "serial":             serial,
        "pci_bus_id":         bus_id,

        "utilisation": {
            "gpu_pct":        util_gpu_pct,
            "memory_bw_pct":  util_mem_pct,
        },
        "memory": {
            "total_mb":       mem_total_mb,
            "used_mb":        mem_used_mb,
            "free_mb":        mem_free_mb,
            "used_pct":       mem_used_pct,
        },
        "temperature_c":      temp_c,
        "fan_speed_pct":      fan_pct,
        "clocks": {
            "sm_mhz":             clock_sm,
            "mem_mhz":            clock_mem,
            "graphics_mhz":       clock_gr,
            "max_sm_mhz":         max_sm,
            "max_mem_mhz":        max_mem,
            "sm_throttle_pct":    sm_throttle_pct,
        },
        "power": {
            "draw_w":             power_draw_w,
            "limit_w":            power_limit_w,
            "default_limit_w":    power_default_w,
            "util_pct":           power_util_pct,
            "energy_total_kj":    energy_kj,     # cumulative since driver load — use delta
        },
        "pcie": {
            "tx_kb_s":        pcie_tx,
            "rx_kb_s":        pcie_rx,
        },
        "ecc": {
            "sbe_volatile":       ecc_sbe_volatile,
            "dbe_volatile":       ecc_dbe_volatile,
            "dbe_aggregate":      ecc_dbe_aggregate,   # primary alert signal
        },
        "throttle": {
            "bitmask":            throttle,
            "decoded":            throttle_decoded,     # human-readable flags
            "power_capped":       bool(throttle & THROTTLE_SW_POWER_CAP)   if throttle else False,
            "hw_thermal":         bool(throttle & THROTTLE_HW_THERMAL)     if throttle else False,
            "hw_slowdown":        bool(throttle & THROTTLE_HW_SLOWDOWN)    if throttle else False,
        },
        "nvlink_active_links":    nvlink_active,
        "processes":              processes,
    }


def _decode_throttle(bitmask: int) -> dict:
    """
    Convert the throttle reason bitmask to named boolean flags.
    Makes API responses human-readable without bit manipulation on the client.
    """
    return {
        "gpu_idle":         bool(bitmask & THROTTLE_GPU_IDLE),
        "app_clock":        bool(bitmask & THROTTLE_APP_CLOCK),
        "power_cap":        bool(bitmask & THROTTLE_SW_POWER_CAP),   # RL reward signal
        "hw_slowdown":      bool(bitmask & THROTTLE_HW_SLOWDOWN),
        "sync_boost":       bool(bitmask & THROTTLE_SYNC_BOOST),
        "sw_thermal":       bool(bitmask & THROTTLE_SW_THERMAL),
        "hw_thermal":       bool(bitmask & THROTTLE_HW_THERMAL),     # alert: too hot
        "hw_power_brake":   bool(bitmask & THROTTLE_HW_POWER_BRAKE),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION — assembles the full node payload
# ════════════════════════════════════════════════════════════════════════════

def collect_node() -> dict:
    """
    Collect GPU telemetry for the entire node.

    Returns a structured payload with:
      - node-level metadata (driver version, CUDA version)
      - per physical GPU: full telemetry + MIG mode flag + MIG instance list
      - summary counters for the view toggle:
          physical_gpu_count   — total physical GPUs on node
          mig_enabled_count    — GPUs with MIG active
          total_mig_instances  — total active MIG compute instances
          effective_gpu_count  — what the scheduler sees (mig instances OR physical)

    COLLECT_MIG_MODE behaviour:
      "auto"     → include both physical records and mig_instances arrays
      "physical" → physical records only, mig_instances always []
      "mig"      → skip non-MIG GPUs, return only MIG instance records
    """
    pynvml.nvmlInit()

    try:
        driver_ver  = pynvml.nvmlSystemGetDriverVersion()
        cuda_ver_raw = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_ver    = f"{cuda_ver_raw // 1000}.{(cuda_ver_raw % 1000) // 10}"
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as e:
        pynvml.nvmlShutdown()
        raise RuntimeError(f"NVML init query failed: {e}")

    gpus          = []
    mig_enabled   = 0
    total_mig_inst = 0
    total_power_w  = 0.0

    for i in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        except pynvml.NVMLError as e:
            log.error(f"Cannot get handle for GPU {i}: {e}")
            continue

        is_mig = _is_mig_enabled(handle)

        # ── Skip logic based on COLLECT_MIG_MODE ──────────────────────────
        if COLLECT_MIG == "mig" and not is_mig:
            log.debug(f"GPU[{i}]: skipping non-MIG device (mode=mig)")
            continue
        if COLLECT_MIG == "physical" and is_mig:
            log.debug(f"GPU[{i}]: collecting physical record only (mode=physical)")

        # ── Physical GPU record ────────────────────────────────────────────
        gpu_rec = _collect_physical_gpu(handle, i)
        gpu_rec["mig_enabled"] = is_mig

        # Accumulate node-level power
        if gpu_rec["power"]["draw_w"] is not None:
            total_power_w += gpu_rec["power"]["draw_w"]

        # ── MIG instance records ───────────────────────────────────────────
        mig_instances = []
        if is_mig and COLLECT_MIG in ("auto", "mig"):
            mig_instances = _collect_mig_instances(handle, i)
            mig_enabled  += 1
            total_mig_inst += len(mig_instances)
            gpu_rec["mig_instance_count"] = len(mig_instances)

            # Annotate each MIG instance with this GPU's identity fields
            # so the MIG view doesn't need to join back to the parent
            for m in mig_instances:
                m["physical_gpu_uuid"] = gpu_rec["uuid"]
                m["physical_gpu_name"] = gpu_rec["name"]
        else:
            gpu_rec["mig_instance_count"] = 0

        gpu_rec["mig_instances"] = mig_instances
        gpus.append(gpu_rec)

    pynvml.nvmlShutdown()

    # ── Effective GPU count — what the workload scheduler actually sees ────
    # If ALL GPUs are MIG-enabled → effective = total MIG instances
    # If NONE are MIG-enabled    → effective = physical GPU count
    # If MIXED                   → physical (non-MIG) + MIG instances
    non_mig_gpus = len([g for g in gpus if not g["mig_enabled"]])
    effective_count = non_mig_gpus + total_mig_inst

    return {
        "schema_version":      "2.0",
        "timestamp":           time.time(),
        "collected_at":        datetime.now(timezone.utc).isoformat(),
        "cluster_id":          CLUSTER_ID,
        "rack_id":             RACK_ID,
        "node_id":             NODE_ID,
        "driver_version":      driver_ver,
        "cuda_version":        cuda_ver,

        # ── Counters for the dashboard view toggle ────────────────────────
        "summary": {
            "physical_gpu_count":    len(gpus),
            "mig_enabled_count":     mig_enabled,
            "total_mig_instances":   total_mig_inst,
            "effective_gpu_count":   effective_count,
            "node_power_draw_w":     round(total_power_w, 2),
            "collect_mig_mode":      COLLECT_MIG,
        },

        "gpus": gpus,
    }


# ════════════════════════════════════════════════════════════════════════════
# ALERT EVALUATION — local fast-path before sending to control plane
# ════════════════════════════════════════════════════════════════════════════

# Track previous ECC counts to detect rising values (resets on driver reload)
_prev_ecc: Dict[str, int] = {}

def _check_local_alerts(payload: dict) -> List[str]:
    """
    Evaluate hard alert conditions locally.
    Critical alerts are logged immediately — don't wait for the dashboard.
    Returns list of alert strings for inclusion in the payload.
    """
    alerts = []
    for gpu in payload.get("gpus", []):
        uuid = gpu.get("uuid", "unknown")
        name = gpu.get("name", "GPU")

        # ── ECC double-bit errors — hardware failure signal ───────────────
        dbe = gpu.get("ecc", {}).get("dbe_aggregate")
        if dbe is not None:
            prev = _prev_ecc.get(uuid, 0)
            if dbe > prev:
                msg = (
                    f"CRITICAL ECC: {name} (GPU {gpu['index']}) "
                    f"dbe_aggregate={dbe} (+{dbe - prev} since last poll) — "
                    f"GPU may require replacement"
                )
                log.critical(msg)
                alerts.append(msg)
            _prev_ecc[uuid] = dbe

        # Check MIG instances for ECC too
        for mig in gpu.get("mig_instances", []):
            phys = mig.get("physical", {})
            mig_dbe = phys.get("ecc_dbe_aggregate")
            # MIG shares the physical ECC counter — already caught above

        # ── Hardware thermal throttling ───────────────────────────────────
        if gpu.get("throttle", {}).get("hw_thermal"):
            temp = gpu.get("temperature_c", "?")
            msg  = (
                f"ALERT HW_THERMAL: {name} (GPU {gpu['index']}) "
                f"is hardware-thermal-throttling at {temp}°C"
            )
            log.error(msg)
            alerts.append(msg)

        # ── Temperature threshold ─────────────────────────────────────────
        temp_c = gpu.get("temperature_c")
        if temp_c is not None and temp_c > 90:
            msg = f"ALERT TEMP: {name} (GPU {gpu['index']}) temperature {temp_c}°C exceeds 90°C"
            log.error(msg)
            alerts.append(msg)

        # ── Power cap throttling — RL reward signal ───────────────────────
        # Not a critical alert but logged at WARNING for RL awareness
        # Skip if power_limit unavailable (laptop GPUs with locked TDP report N/A)
        power_limit_available = gpu.get("power", {}).get("limit_w") is not None
        if gpu.get("throttle", {}).get("power_capped") and power_limit_available:
            log.warning(
                f"POWER_CAP: {name} (GPU {gpu['index']}) hitting power limit — "
                f"RL dispatch efficiency is impacted"
            )

    return alerts


# ════════════════════════════════════════════════════════════════════════════
# PUSH LOOP
# ════════════════════════════════════════════════════════════════════════════

def _build_headers() -> dict:
    return {
        "X-API-Key":       AUTH_TOKEN,
        "Content-Type":    "application/json",
        "X-HexaGrid-Node": NODE_ID,
        "X-HexaGrid-Cluster": CLUSTER_ID,
    }


def push_forever():
    """
    Main collection and push loop.

    Reliability features:
    • Jitter on first sleep — prevents thundering herd when many agents start together
    • Exponential backoff with cap on HTTP failures
    • Local SQLite fallback — no telemetry lost during network partitions
    • Automatic drain of buffered payloads on reconnect
    • NVML errors don't crash the loop (driver may be reloading)
    • energy_total is tracked as delta to handle driver-reload resets
    """
    headers     = _build_headers()
    db_conn     = _init_sqlite(SQLITE_PATH) if SQLITE_PATH else None
    failures    = 0
    backoff_s   = POLL_INTERVAL
    max_backoff = POLL_INTERVAL * 30        # cap at 5 minutes
    connected   = True

    # Previous energy totals for delta calculation
    prev_energy: Dict[str, float] = {}

    # ── Jitter — spread agent wake-ups across the poll interval ──────────
    jitter = random.uniform(0, POLL_INTERVAL)
    log.info(
        f"HexaGrid collector starting — node={NODE_ID} rack={RACK_ID} "
        f"cluster={CLUSTER_ID} mig_mode={COLLECT_MIG} "
        f"jitter={jitter:.1f}s interval={POLL_INTERVAL}s"
    )
    time.sleep(jitter)

    while True:
        cycle_start = time.monotonic()

        # ── Collect ────────────────────────────────────────────────────────
        try:
            payload = collect_node()
        except RuntimeError as e:
            log.error(f"Collection failed: {e}. Retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue
        except Exception as e:
            log.error(f"Unexpected collection error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)
            continue

        # ── Energy delta calculation (reset-safe) ─────────────────────────
        for gpu in payload.get("gpus", []):
            uuid       = gpu.get("uuid", "")
            energy_abs = gpu.get("power", {}).get("energy_total_kj")
            if energy_abs is not None:
                delta = None
                if uuid in prev_energy:
                    raw_delta = energy_abs - prev_energy[uuid]
                    # Negative delta = driver reloaded, counter reset — discard
                    delta = max(0.0, raw_delta) if raw_delta >= 0 else None
                gpu["power"]["energy_delta_kj"] = delta
                prev_energy[uuid] = energy_abs

        # ── Local alert evaluation ────────────────────────────────────────
        alerts = _check_local_alerts(payload)
        if alerts:
            payload["local_alerts"] = alerts

        # ── Push to control plane ─────────────────────────────────────────
        try:
            body = json.dumps(payload)
            r    = requests.post(
                HEXAGRID_ENDPOINT, data=body,
                headers=headers, timeout=8,
            )
            r.raise_for_status()

            if not connected:
                log.info("Reconnected to control plane")
                # Drain buffered payloads
                if db_conn:
                    _sqlite_drain(db_conn, HEXAGRID_ENDPOINT, headers)

            connected = True
            failures  = 0
            backoff_s = POLL_INTERVAL

            phys  = payload["summary"]["physical_gpu_count"]
            mig_i = payload["summary"]["total_mig_instances"]
            pwr   = payload["summary"]["node_power_draw_w"]
            log.debug(
                f"Push OK — {phys} GPUs / {mig_i} MIG instances / "
                f"{pwr:.1f}W node power"
            )

        except requests.RequestException as e:
            failures  += 1
            connected  = False
            backoff_s  = min(backoff_s * 2, max_backoff)
            log.warning(
                f"Push failed ({failures}): {e}. "
                f"Backoff {backoff_s}s. Buffering locally."
            )
            if db_conn:
                _sqlite_write(db_conn, payload)

        # ── Sleep for remainder of interval (account for collection time) ──
        elapsed  = time.monotonic() - cycle_start
        sleep_for = max(0, POLL_INTERVAL - elapsed)

        if not connected:
            sleep_for = backoff_s

        time.sleep(sleep_for)


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST MODE
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(payload: dict, verbose: bool = False):
    """Pretty-print a collected payload for local testing."""
    s = payload["summary"]
    print(f"\n{'='*60}")
    print(f"  HexaGrid GPU Telemetry — {payload['node_id']}")
    print(f"  {payload['collected_at']}")
    print(f"{'='*60}")
    print(f"  Driver: {payload['driver_version']}  CUDA: {payload['cuda_version']}")
    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Physical GPUs:      {s['physical_gpu_count']}")
    print(f"  MIG-enabled GPUs:   {s['mig_enabled_count']}")
    print(f"  Total MIG instances:{s['total_mig_instances']}")
    print(f"  Effective GPU count:{s['effective_gpu_count']}")
    print(f"  Node power draw:    {s['node_power_draw_w']:.1f} W")
    print(f"  Collection mode:    {s['collect_mig_mode']}")

    for gpu in payload["gpus"]:
        print(f"\n  ── GPU {gpu['index']}: {gpu['name']} {'[MIG]' if gpu['mig_enabled'] else ''}")
        if not gpu["mig_enabled"]:
            u = gpu["utilisation"]
            m = gpu["memory"]
            p = gpu["power"]
            t = gpu["throttle"]
            print(f"     Util:  GPU {u['gpu_pct']}% | Mem BW {u['memory_bw_pct']}%")
            print(f"     Mem:   {m['used_mb']} / {m['total_mb']} MB ({m['used_pct']}%)")
            fan = f"{gpu['fan_speed_pct']}%" if gpu['fan_speed_pct'] is not None else "N/A"
            print(f"     Temp:  {gpu['temperature_c']}°C | Fan: {fan}")
            draw  = f"{p['draw_w']:.1f}W"  if p['draw_w']  is not None else "N/A"
            limit = f"{p['limit_w']:.1f}W" if p['limit_w'] is not None else "N/A"
            util  = f"{p['util_pct']}%"   if p['util_pct'] is not None else "N/A"
            print(f"     Power: {draw} / {limit} ({util})")
            if t["power_capped"]:
                print(f"     ⚠️  POWER CAP THROTTLE (SwPowerCap)")
            if t["hw_thermal"]:
                print(f"     🔴 HW THERMAL THROTTLE")
            ecc = gpu.get("ecc", {})
            if ecc.get("dbe_aggregate"):
                print(f"     🔴 ECC DBE AGGREGATE: {ecc['dbe_aggregate']}")
        else:
            pdraw  = f"{gpu['power']['draw_w']:.1f}W"  if gpu['power']['draw_w']  is not None else "N/A"
            plimit = f"{gpu['power']['limit_w']:.1f}W" if gpu['power']['limit_w'] is not None else "N/A"
            print(f"     Physical power:    {pdraw} / {plimit}")
            print(f"     Physical temp:     {gpu['temperature_c']}°C")
            print(f"     MIG instances:     {gpu['mig_instance_count']}")
            for mig in gpu.get("mig_instances", []):
                m  = mig["memory"]
                u  = mig["utilisation"]
                ph = mig["physical"]
                print(
                    f"\n       MIG[{mig['mig_index']}] {mig['mig_name']}")
                print(
                    f"         UUID:     {mig['mig_uuid']}")
                print(
                    f"         GI/CI:    {mig['gpu_instance_id']}/{mig['compute_instance_id']}")
                print(
                    f"         Mem:      {m['used_mb']}/{m['total_mb']} MB "
                    f"({m['used_pct']}%) — {m['phys_fraction']*100:.0f}% of physical")
                print(
                    f"         Util:     GPU {u['gpu_pct']}%  Mem BW {u['memory_bw_pct']}%")
                mig_pwr = f"{ph['prorated_power_w']:.1f}W" if ph['prorated_power_w'] is not None else "N/A"
                print(
                    f"         Power:    {mig_pwr} (pro-rated)")
                if ph.get("ecc_dbe_aggregate"):
                    print(f"         🔴 ECC DBE: {ph['ecc_dbe_aggregate']}")

    if "local_alerts" in payload and payload["local_alerts"]:
        print(f"\n  ── Alerts ──────────────────────────────────")
        for alert in payload["local_alerts"]:
            print(f"  🔴 {alert}")

    if verbose:
        print(f"\n  ── Full JSON ────────────────────────────────")
        print(json.dumps(payload, indent=2))

    print()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HexaGrid GPU Telemetry Collector Agent"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one collection cycle, print results, and exit (no push)"
    )
    parser.add_argument(
        "--push-once",
        action="store_true",
        help="Collect, print, push once to the endpoint, then exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In --test mode, also print full JSON payload"
    )
    parser.add_argument(
        "--mig-mode",
        choices=["auto", "physical", "mig"],
        default=None,
        help="Override COLLECT_MIG_MODE env var"
    )
    args = parser.parse_args()

    if args.mig_mode:
        COLLECT_MIG = args.mig_mode
        log.info(f"MIG mode overridden to: {COLLECT_MIG}")

    if args.test:
        log.info("Test mode — single collection, no push")
        try:
            payload = collect_node()
            _print_summary(payload, verbose=args.verbose)
        except Exception as e:
            print(f"Collection error: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.push_once:
        log.info("Push-once mode — collecting and pushing one payload")
        try:
            payload = collect_node()
            _print_summary(payload, verbose=args.verbose)
            headers = _build_headers()
            r = requests.post(
                HEXAGRID_ENDPOINT,
                data=json.dumps(payload),
                headers=headers,
                timeout=8,
            )
            r.raise_for_status()
            print(f"\n  ✓ Pushed to {HEXAGRID_ENDPOINT}")
            print(f"  Response: {r.json()}")
        except Exception as e:
            print(f"Push error: {e}")
            sys.exit(1)
        sys.exit(0)

    push_forever()
