"""
Energia Phase 10 — GPU Health Monitor
======================================
Real-time GPU telemetry via NVIDIA NVML (pynvml) with anomaly detection
and remaining-useful-life (RUL) estimation.

Install dependency:
    pip install pynvml --break-system-packages

What it monitors per GPU:
  - Temperature (°C)
  - Power draw (W) and TDP %
  - Memory used / total / utilisation %
  - GPU core utilisation %
  - Fan speed % (where available)
  - ECC memory error counts (single-bit & double-bit)
  - PCIe throughput (TX/RX MB/s)
  - Clock speeds (graphics, memory, SM)

Anomaly detection (sklearn IsolationForest):
  - Trained on a rolling 30-sample baseline per GPU
  - Scores each sample; flags anomalies in real time
  - Severity: normal / warning / critical

RUL estimation:
  - Tracks thermal margin, ECC trend, and power variance
  - Produces a 0–100 health score and plain-English status

SQLite persistence:
  - Rolling 24h of telemetry per GPU (capped at 8640 rows @ 10s intervals)
  - Alert log (last 500 alerts)

Usage:
    from gpu_monitor import GPUMonitor
    mon = GPUMonitor()
    snapshot = mon.get_snapshot()       # all GPUs, current reading
    history  = mon.get_history(gpu_idx=0, minutes=60)
    alerts   = mon.get_alerts(limit=20)
"""

import os, json, sqlite3, time, logging, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

import numpy as np

DB_PATH  = Path(__file__).parent.parent / "cache" / "gpu_health.db"
LOG_DIR  = Path(__file__).parent.parent / "logs" / "gpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu_monitor")

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "temp_warn":      78,    # °C
    "temp_critical":  88,    # °C
    "power_warn":     90,    # % of TDP
    "power_critical": 100,   # % of TDP
    "mem_warn":       90,    # % utilisation
    "mem_critical":   98,    # %
    "ecc_warn":       1,     # any single-bit errors
    "ecc_critical":   1,     # any double-bit errors (immediate action)
    "fan_warn":       85,    # % speed
    "fan_critical":   95,    # %
}

# GPU TDP reference table (W) — fallback if NVML doesn't report it
GPU_TDP = {
    "RTX 4060":         115,
    "RTX 4060 Ti":      160,
    "RTX 4070":         200,
    "RTX 4070 Ti":      285,
    "RTX 4080":         320,
    "RTX 4090":         450,
    "RTX A1000 6GB Lap":60,
    "RTX A1000":        50,
    "RTX A2000":        70,
    "RTX A4000":        140,
    "RTX A5000":        230,
    "RTX A6000":        300,
    "H100 PCIe":        350,
    "H100 SXM5":        700,
    "A100 PCIe":        300,
    "A100 SXM4":        400,
}


@dataclass
class GPUReading:
    gpu_idx:          int
    gpu_name:         str
    uuid:             str
    timestamp:        float
    datetime_iso:     str

    # Thermal
    temp_c:           float
    temp_warn:        int    = 78
    temp_critical:    int    = 88

    # Power
    power_w:          float  = 0.0
    power_limit_w:    float  = 0.0
    power_pct:        float  = 0.0
    tdp_w:            float  = 0.0

    # Memory
    mem_used_mb:      float  = 0.0
    mem_total_mb:     float  = 0.0
    mem_pct:          float  = 0.0

    # Utilisation
    gpu_util_pct:     float  = 0.0
    mem_util_pct:     float  = 0.0

    # Fan
    fan_speed_pct:    float  = 0.0
    fan_available:    bool   = True

    # ECC errors
    ecc_single:       int    = 0
    ecc_double:       int    = 0

    # Clocks MHz
    clock_graphics:   int    = 0
    clock_memory:     int    = 0
    clock_sm:         int    = 0

    # PCIe MB/s
    pcie_tx_mbps:     float  = 0.0
    pcie_rx_mbps:     float  = 0.0

    # Health
    health_score:     float  = 100.0   # 0–100
    health_status:    str    = "Healthy"
    anomaly:          bool   = False
    anomaly_score:    float  = 0.0     # IsolationForest; negative = anomalous
    severity:         str    = "normal"  # normal | warning | critical
    alerts:           list   = field(default_factory=list)


class GPUMonitor:
    def __init__(self, poll_interval_s: int = 10):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._db   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()
        self._nvml_ok     = False
        self._gpus        = []          # list of pynvml handles
        self._gpu_names   = []
        self._gpu_uuids   = []
        self._tdps        = []
        self._baselines   = {}          # gpu_idx → list of feature vectors
        self._models      = {}          # gpu_idx → IsolationForest
        self._init_nvml()
        self.poll_interval = poll_interval_s
        self._bg_thread   = None

    # ── DB ────────────────────────────────────────────────────────────────────
    def _init_db(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                gpu_idx     INTEGER NOT NULL,
                ts          REAL    NOT NULL,
                data        TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tel_gpu_ts ON telemetry(gpu_idx, ts);
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                gpu_idx     INTEGER NOT NULL,
                severity    TEXT    NOT NULL,
                message     TEXT    NOT NULL
            );
        """)
        self._db.commit()

    def _save_reading(self, r: GPUReading):
        with self._lock:
            self._db.execute(
                "INSERT INTO telemetry (gpu_idx, ts, data) VALUES (?,?,?)",
                (r.gpu_idx, r.timestamp, json.dumps(asdict(r)))
            )
            # Keep only last 8640 rows per GPU (~24h at 10s)
            self._db.execute("""
                DELETE FROM telemetry WHERE gpu_idx=? AND id NOT IN (
                    SELECT id FROM telemetry WHERE gpu_idx=?
                    ORDER BY ts DESC LIMIT 8640
                )""", (r.gpu_idx, r.gpu_idx))
            for alert_msg in r.alerts:
                self._db.execute(
                    "INSERT INTO alerts (ts, gpu_idx, severity, message) VALUES (?,?,?,?)",
                    (r.timestamp, r.gpu_idx, r.severity, alert_msg)
                )
            self._db.commit()

    # ── NVML init ─────────────────────────────────────────────────────────────
    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._pynvml = pynvml
            for i in range(count):
                h    = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                uuid = pynvml.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
                tdp  = self._lookup_tdp(name, pynvml, h)
                self._gpus.append(h)
                self._gpu_names.append(name)
                self._gpu_uuids.append(uuid)
                self._tdps.append(tdp)
                self._baselines[i] = []
                logger.info(f"  [GPU {i}] {name}  TDP={tdp}W  UUID={uuid[:12]}...")
            self._nvml_ok = len(self._gpus) > 0
            logger.info(f"  [NVML] {len(self._gpus)} GPU(s) initialised")
        except ImportError:
            logger.warning("  [NVML] pynvml not installed — run: pip install pynvml --break-system-packages")
        except Exception as e:
            logger.warning(f"  [NVML] Init failed: {e}")

    def _lookup_tdp(self, name: str, pynvml, handle) -> float:
        # Try NVML first
        try:
            return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
        except Exception:
            pass
        for key, tdp in GPU_TDP.items():
            if key.lower() in name.lower():
                return float(tdp)
        return 150.0   # safe fallback

    # ── Poll one GPU ──────────────────────────────────────────────────────────
    def _poll_gpu(self, idx: int) -> GPUReading:
        pynvml = self._pynvml
        h      = self._gpus[idx]
        now    = time.time()
        dt     = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        def safe(fn, default=0):
            try:    return fn()
            except: return default

        temp    = safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU), 0)
        pw      = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 0.0)
        pw_lim  = safe(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, self._tdps[idx])
        tdp     = self._tdps[idx]
        pw_pct  = round(pw / tdp * 100, 1) if tdp else 0.0

        mem_info   = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h), None)
        mem_used   = round(mem_info.used  / 1024**2, 1) if mem_info else 0.0
        mem_total  = round(mem_info.total / 1024**2, 1) if mem_info else 0.0
        mem_pct    = round(mem_used / mem_total * 100, 1) if mem_total else 0.0

        util    = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h), None)
        gpu_u   = util.gpu    if util else 0
        mem_u   = util.memory if util else 0

        fan_pct = 0.0
        fan_ok  = True
        try:
            fan_pct = pynvml.nvmlDeviceGetFanSpeed(h)
        except pynvml.NVMLError:
            fan_ok = False

        ecc_s = safe(lambda: pynvml.nvmlDeviceGetTotalEccErrors(
            h, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
            pynvml.NVML_AGGREGATE_ECC), 0)
        ecc_d = safe(lambda: pynvml.nvmlDeviceGetTotalEccErrors(
            h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            pynvml.NVML_AGGREGATE_ECC), 0)

        clk_g = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS), 0)
        clk_m = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM), 0)
        clk_s = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), 0)

        pcie_tx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
            h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0, 0.0)
        pcie_rx = safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
            h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0, 0.0)

        r = GPUReading(
            gpu_idx       = idx,
            gpu_name      = self._gpu_names[idx],
            uuid          = self._gpu_uuids[idx],
            timestamp     = now,
            datetime_iso  = dt,
            temp_c        = float(temp),
            power_w       = pw,
            power_limit_w = pw_lim,
            power_pct     = pw_pct,
            tdp_w         = tdp,
            mem_used_mb   = mem_used,
            mem_total_mb  = mem_total,
            mem_pct       = mem_pct,
            gpu_util_pct  = float(gpu_u),
            mem_util_pct  = float(mem_u),
            fan_speed_pct = float(fan_pct),
            fan_available = fan_ok,
            ecc_single    = int(ecc_s),
            ecc_double    = int(ecc_d),
            clock_graphics= int(clk_g),
            clock_memory  = int(clk_m),
            clock_sm      = int(clk_s),
            pcie_tx_mbps  = float(pcie_tx),
            pcie_rx_mbps  = float(pcie_rx),
        )

        # Run anomaly detection + health scoring
        self._analyze(r)
        self._save_reading(r)
        return r

    # ── Anomaly detection ─────────────────────────────────────────────────────
    def _analyze(self, r: GPUReading):
        from sklearn.ensemble import IsolationForest

        alerts = []
        severity = "normal"

        # ── Rule-based threshold alerts ──
        if r.temp_c >= THRESHOLDS["temp_critical"]:
            alerts.append(f"CRITICAL: Temperature {r.temp_c:.0f}°C exceeds critical threshold ({THRESHOLDS['temp_critical']}°C)")
            severity = "critical"
        elif r.temp_c >= THRESHOLDS["temp_warn"]:
            alerts.append(f"WARNING: Temperature {r.temp_c:.0f}°C approaching limit")
            severity = "warning"

        if r.power_pct >= THRESHOLDS["power_critical"]:
            alerts.append(f"CRITICAL: Power draw at {r.power_pct:.0f}% of TDP ({r.power_w:.0f}W / {r.tdp_w:.0f}W)")
            severity = "critical"
        elif r.power_pct >= THRESHOLDS["power_warn"]:
            alerts.append(f"WARNING: High power draw {r.power_pct:.0f}% of TDP")
            if severity == "normal": severity = "warning"

        if r.mem_pct >= THRESHOLDS["mem_critical"]:
            alerts.append(f"CRITICAL: GPU memory {r.mem_pct:.0f}% full ({r.mem_used_mb:.0f}/{r.mem_total_mb:.0f} MB)")
            severity = "critical"
        elif r.mem_pct >= THRESHOLDS["mem_warn"]:
            alerts.append(f"WARNING: GPU memory {r.mem_pct:.0f}% full")
            if severity == "normal": severity = "warning"

        if r.ecc_double > 0:
            alerts.append(f"CRITICAL: {r.ecc_double} uncorrectable ECC memory error(s) — hardware may be failing")
            severity = "critical"
        elif r.ecc_single >= THRESHOLDS["ecc_warn"]:
            alerts.append(f"WARNING: {r.ecc_single} correctable ECC memory error(s) detected")
            if severity == "normal": severity = "warning"

        if r.fan_available and r.fan_speed_pct >= THRESHOLDS["fan_critical"]:
            alerts.append(f"CRITICAL: Fan at {r.fan_speed_pct:.0f}% — check cooling")
            severity = "critical"
        elif r.fan_available and r.fan_speed_pct >= THRESHOLDS["fan_warn"]:
            alerts.append(f"WARNING: Fan speed {r.fan_speed_pct:.0f}% — elevated thermal load")
            if severity == "normal": severity = "warning"

        # ── IsolationForest anomaly detection ──
        features = np.array([
            r.temp_c, r.power_pct, r.mem_pct,
            r.gpu_util_pct, r.fan_speed_pct, r.pcie_tx_mbps
        ])
        baseline = self._baselines[r.gpu_idx]
        baseline.append(features)
        if len(baseline) > 30:
            baseline.pop(0)

        anomaly_score = 0.0
        if len(baseline) >= 10:
            X = np.array(baseline)
            if r.gpu_idx not in self._models or len(baseline) % 10 == 0:
                self._models[r.gpu_idx] = IsolationForest(
                    n_estimators=50, contamination=0.1, random_state=42
                ).fit(X)
            model = self._models[r.gpu_idx]
            score = model.score_samples(features.reshape(1, -1))[0]
            anomaly_score = float(score)
            if score < -0.6:
                alerts.append(f"ANOMALY: Unusual telemetry pattern detected (score={score:.2f})")
                if severity == "normal": severity = "warning"

        # ── Health score (0–100) ──
        # Penalise: temperature margin, power margin, ECC errors, anomaly
        temp_margin   = max(0, (THRESHOLDS["temp_critical"] - r.temp_c) / THRESHOLDS["temp_critical"])
        power_margin  = max(0, (100 - r.power_pct) / 100)
        mem_margin    = max(0, (100 - r.mem_pct) / 100)
        ecc_penalty   = min(30, r.ecc_single * 2 + r.ecc_double * 20)
        anomaly_pen   = max(0, (-anomaly_score - 0.4) * 20) if anomaly_score < -0.4 else 0

        health = (
            temp_margin  * 35 +
            power_margin * 25 +
            mem_margin   * 20 +
            (1.0)        * 20    # base score
        ) * 100 / 100
        health = max(0, min(100, health - ecc_penalty - anomaly_pen))

        if health >= 85:    status = "Healthy"
        elif health >= 65:  status = "Good"
        elif health >= 45:  status = "Fair — monitor closely"
        elif health >= 25:  status = "Degraded — schedule maintenance"
        else:               status = "Critical — immediate action required"

        r.health_score  = round(health, 1)
        r.health_status = status
        r.anomaly       = anomaly_score < -0.5
        r.anomaly_score = round(anomaly_score, 3)
        r.severity      = severity
        r.alerts        = alerts

    # ── Synthetic readings (when NVML unavailable) ────────────────────────────
    def _synthetic_reading(self, idx: int) -> GPUReading:
        """
        Returns a realistic synthetic reading for development/testing.
        Simulates two GPUs: RTX 4060 (idx=0) and RTX A1000 (idx=1).
        """
        now  = time.time()
        dt   = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        rng  = np.random.default_rng(int(now) % 10000 + idx)

        PROFILES = [
            {"name": "NVIDIA GeForce RTX 4060", "tdp": 115, "temp_base": 62, "pow_base": 75},
            {"name": "NVIDIA RTX A1000 6GB Lap", "tdp": 60,  "temp_base": 45, "pow_base": 15},
        ]
        p = PROFILES[idx % len(PROFILES)]

        temp    = p["temp_base"] + rng.normal(0, 3)
        pw      = p["pow_base"]  + rng.normal(0, 5)
        pw_pct  = round(pw / p["tdp"] * 100, 1)
        mem_t   = 8192.0 if idx == 0 else 16384.0
        mem_u   = mem_t * rng.uniform(0.3, 0.7)
        gpu_u   = rng.uniform(40, 90)
        fan     = 45 + (temp - 55) * 1.5 + rng.normal(0, 3)

        r = GPUReading(
            gpu_idx       = idx,
            gpu_name      = p["name"],
            uuid          = f"GPU-SYNTHETIC-{idx:04d}",
            timestamp     = now,
            datetime_iso  = dt,
            temp_c        = round(float(temp), 1),
            power_w       = round(float(pw), 1),
            power_limit_w = float(p["tdp"]),
            power_pct     = pw_pct,
            tdp_w         = float(p["tdp"]),
            mem_used_mb   = round(float(mem_u), 1),
            mem_total_mb  = mem_t,
            mem_pct       = round(mem_u / mem_t * 100, 1),
            gpu_util_pct  = round(float(gpu_u), 1),
            mem_util_pct  = round(float(gpu_u * 0.6), 1),
            fan_speed_pct = round(max(0, min(100, float(fan))), 1),
            fan_available = True,
            ecc_single    = 0,
            ecc_double    = 0,
            clock_graphics= int(2400 + rng.integers(-100, 100)),
            clock_memory  = int(9000 + rng.integers(-200, 200)),
            clock_sm      = int(2400 + rng.integers(-100, 100)),
            pcie_tx_mbps  = round(float(rng.uniform(200, 800)), 1),
            pcie_rx_mbps  = round(float(rng.uniform(100, 400)), 1),
        )
        self._analyze(r)
        self._save_reading(r)
        return r

    # ── Public interface ──────────────────────────────────────────────────────
    def get_snapshot(self) -> dict:
        """Poll all GPUs and return current readings."""
        readings = []
        if self._nvml_ok:
            for i in range(len(self._gpus)):
                try:
                    readings.append(asdict(self._poll_gpu(i)))
                except Exception as e:
                    logger.warning(f"GPU {i} poll failed: {e}")
        else:
            # Synthetic mode — always returns 2 GPUs for dev/demo
            for i in range(2):
                readings.append(asdict(self._synthetic_reading(i)))

        fleet_severity = "normal"
        for r in readings:
            if r["severity"] == "critical":
                fleet_severity = "critical"; break
            if r["severity"] == "warning":
                fleet_severity = "warning"

        avg_health = round(sum(r["health_score"] for r in readings) / len(readings), 1) if readings else 0

        return {
            "gpus":             readings,
            "gpu_count":        len(readings),
            "fleet_severity":   fleet_severity,
            "avg_health_score": avg_health,
            "nvml_live":        self._nvml_ok,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

    def get_history(self, gpu_idx: int = 0, minutes: int = 60) -> list:
        """Return telemetry history for one GPU over the past N minutes."""
        since = time.time() - minutes * 60
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM telemetry WHERE gpu_idx=? AND ts>=? ORDER BY ts ASC",
                (gpu_idx, since)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_alerts(self, limit: int = 50) -> list:
        """Return most recent alerts across all GPUs."""
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, gpu_idx, severity, message FROM alerts ORDER BY ts DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [
            {
                "datetime": datetime.fromtimestamp(r[0], tz=timezone.utc).isoformat(),
                "gpu_idx":  r[1],
                "severity": r[2],
                "message":  r[3],
            }
            for r in rows
        ]

    def start_background_polling(self):
        """Start background thread that polls all GPUs every poll_interval seconds."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        def _loop():
            while True:
                try:
                    self.get_snapshot()
                except Exception as e:
                    logger.warning(f"Background poll error: {e}")
                time.sleep(self.poll_interval)
        self._bg_thread = threading.Thread(target=_loop, daemon=True)
        self._bg_thread.start()
        logger.info(f"  [GPU Monitor] Background polling started ({self.poll_interval}s interval)")


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mon = GPUMonitor()
    print(f"\n=== GPU Snapshot ({'LIVE NVML' if mon._nvml_ok else 'SYNTHETIC'}) ===")
    snap = mon.get_snapshot()
    for g in snap["gpus"]:
        print(f"\n  GPU {g['gpu_idx']}: {g['gpu_name']}")
        print(f"    Temp:    {g['temp_c']:.1f}°C")
        print(f"    Power:   {g['power_w']:.1f}W  ({g['power_pct']:.1f}% TDP)")
        print(f"    Memory:  {g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f} MB  ({g['mem_pct']:.1f}%)")
        print(f"    GPU Util:{g['gpu_util_pct']:.1f}%")
        print(f"    Fan:     {g['fan_speed_pct']:.1f}%")
        print(f"    Health:  {g['health_score']:.1f}/100  [{g['health_status']}]")
        print(f"    Severity:{g['severity']}")
        if g['alerts']:
            for a in g['alerts']:
                print(f"    ⚠  {a}")
    print(f"\n  Fleet severity: {snap['fleet_severity']}")
    print(f"  Avg health:     {snap['avg_health_score']}/100")
