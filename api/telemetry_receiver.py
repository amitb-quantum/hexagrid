"""
telemetry_receiver.py — HexaGrid Telemetry Receiver & Fleet API
================================================================
FastAPI router that:
  1. Accepts GPU telemetry pushes from all collector agents
  2. Stores to SQLite (pilot) or TimescaleDB (production) transparently
  3. Exposes fleet query endpoints with MIG view toggle

Mount this router in your existing api.py:
    from telemetry_receiver import router as telemetry_router
    app.include_router(telemetry_router)

MIG view toggle
───────────────
Every fleet query accepts a ?view= parameter:
  ?view=physical   — one record per physical GPU
                     (traditional view, matches nvidia-smi device list)
  ?view=mig        — one record per MIG compute instance
                     (scheduler view, matches what Slurm/K8s sees)
  ?view=auto       — physical for non-MIG nodes, MIG instances for MIG nodes
                     (default — most useful for mixed fleets)

Database backends
─────────────────
Set DB_BACKEND env var:
  "sqlite"      — local SQLite, no extra dependencies (default for pilots)
  "timescaledb" — PostgreSQL + TimescaleDB (production at scale)

Set SQLITE_PATH or DB_DSN accordingly.
"""

import os
import time
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, List, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger("hg-telemetry")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

DB_BACKEND   = os.environ.get("DB_BACKEND",   "sqlite").lower()
SQLITE_PATH  = os.environ.get("SQLITE_PATH",  "/var/lib/hexagrid/telemetry.db")
DB_DSN       = os.environ.get("DB_DSN",       "postgresql://hexagrid:password@localhost/hexagrid")
API_TOKEN    = os.environ.get("HEXAGRID_TOKEN", "")    # "" = auth disabled (dev)

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])

# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS — mirrors collector_agent.py payload structure
# ════════════════════════════════════════════════════════════════════════════

class ThrottleInfo(BaseModel):
    bitmask:          Optional[int]   = None
    power_capped:     bool            = False
    hw_thermal:       bool            = False
    hw_slowdown:      bool            = False

class PowerInfo(BaseModel):
    draw_w:           Optional[float] = None
    limit_w:          Optional[float] = None
    util_pct:         Optional[float] = None
    energy_total_kj:  Optional[float] = None
    energy_delta_kj:  Optional[float] = None

class MemInfo(BaseModel):
    total_mb:         Optional[int]   = None
    used_mb:          Optional[int]   = None
    free_mb:          Optional[int]   = None
    used_pct:         Optional[float] = None
    phys_fraction:    Optional[float] = None   # MIG only

class UtilInfo(BaseModel):
    gpu_pct:          Optional[float] = None
    memory_bw_pct:    Optional[float] = None

class EccInfo(BaseModel):
    sbe_volatile:     Optional[int]   = None
    dbe_volatile:     Optional[int]   = None
    dbe_aggregate:    Optional[int]   = None

class MigPhysical(BaseModel):
    temp_c:              Optional[float] = None
    power_draw_w:        Optional[float] = None
    prorated_power_w:    Optional[float] = None
    throttle_bitmask:    Optional[int]   = None
    ecc_dbe_aggregate:   Optional[int]   = None

class MigInstance(BaseModel):
    mig_index:            int
    mig_uuid:             str
    mig_name:             str
    gpu_instance_id:      Optional[int] = None
    compute_instance_id:  Optional[int] = None
    physical_gpu_index:   int
    physical_gpu_uuid:    Optional[str] = None
    physical_gpu_name:    Optional[str] = None
    memory:               MemInfo       = Field(default_factory=MemInfo)
    utilisation:          UtilInfo      = Field(default_factory=UtilInfo)
    physical:             MigPhysical   = Field(default_factory=MigPhysical)
    processes:            List[dict]    = []

class GpuRecord(BaseModel):
    index:                int
    uuid:                 str
    name:                 str
    serial:               Optional[str]  = None
    pci_bus_id:           Optional[str]  = None
    mig_enabled:          bool           = False
    mig_instance_count:   int            = 0
    utilisation:          UtilInfo       = Field(default_factory=UtilInfo)
    memory:               MemInfo        = Field(default_factory=MemInfo)
    temperature_c:        Optional[float] = None
    fan_speed_pct:        Optional[float] = None
    power:                PowerInfo      = Field(default_factory=PowerInfo)
    ecc:                  EccInfo        = Field(default_factory=EccInfo)
    throttle:             ThrottleInfo   = Field(default_factory=ThrottleInfo)
    processes:            List[dict]     = []
    mig_instances:        List[MigInstance] = []

class NodeSummary(BaseModel):
    physical_gpu_count:   int   = 0
    mig_enabled_count:    int   = 0
    total_mig_instances:  int   = 0
    effective_gpu_count:  int   = 0
    node_power_draw_w:    float = 0.0

class NodeTelemetry(BaseModel):
    schema_version:   str         = "2.0"
    timestamp:        float
    collected_at:     str
    cluster_id:       str
    rack_id:          str
    node_id:          str
    driver_version:   Optional[str] = None
    cuda_version:     Optional[str] = None
    summary:          NodeSummary   = Field(default_factory=NodeSummary)
    gpus:             List[GpuRecord] = []
    local_alerts:     List[str]    = []


# ════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER — SQLite (pilot) + TimescaleDB (production) via same interface
# ════════════════════════════════════════════════════════════════════════════

class TelemetryDB:
    """
    Thin abstraction over SQLite and TimescaleDB.
    Both use identical SQL — TimescaleDB is just partitioned PostgreSQL.
    Switch backends with DB_BACKEND env var, zero application code changes.
    """

    def __init__(self):
        self.backend = DB_BACKEND
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._pg_pool = None   # asyncpg pool, created on first use

    def init_sqlite(self):
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # ── Physical GPU telemetry ──────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_telemetry (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL    NOT NULL,
                cluster_id        TEXT    NOT NULL,
                rack_id           TEXT    NOT NULL,
                node_id           TEXT    NOT NULL,
                gpu_index         INTEGER NOT NULL,
                gpu_uuid          TEXT    NOT NULL,
                gpu_name          TEXT,
                mig_enabled       INTEGER NOT NULL DEFAULT 0,
                mig_instance_count INTEGER NOT NULL DEFAULT 0,
                util_gpu_pct      REAL,
                util_mem_pct      REAL,
                mem_used_mb       INTEGER,
                mem_total_mb      INTEGER,
                temp_c            REAL,
                fan_pct           REAL,
                power_draw_w      REAL,
                power_limit_w     REAL,
                power_util_pct    REAL,
                energy_delta_kj   REAL,
                clock_sm_mhz      INTEGER,
                throttle_bitmask  INTEGER,
                throttle_power_capped  INTEGER DEFAULT 0,
                throttle_hw_thermal    INTEGER DEFAULT 0,
                ecc_dbe_aggregate INTEGER,
                ecc_dbe_volatile  INTEGER,
                processes_json    TEXT
            )
        """)

        # ── MIG instance telemetry ──────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mig_telemetry (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                    REAL    NOT NULL,
                cluster_id            TEXT    NOT NULL,
                rack_id               TEXT    NOT NULL,
                node_id               TEXT    NOT NULL,
                physical_gpu_index    INTEGER NOT NULL,
                physical_gpu_uuid     TEXT    NOT NULL,
                physical_gpu_name     TEXT,
                mig_index             INTEGER NOT NULL,
                mig_uuid              TEXT    NOT NULL,
                mig_name              TEXT,
                gpu_instance_id       INTEGER,
                compute_instance_id   INTEGER,
                mem_used_mb           INTEGER,
                mem_total_mb          INTEGER,
                mem_used_pct          REAL,
                phys_mem_fraction     REAL,
                util_gpu_pct          REAL,
                util_mem_bw_pct       REAL,
                phys_temp_c           REAL,
                phys_power_draw_w     REAL,
                prorated_power_w      REAL,
                phys_throttle_bitmask INTEGER,
                phys_ecc_dbe_aggregate INTEGER,
                processes_json        TEXT
            )
        """)

        # ── Node-level summary (for fast fleet queries) ────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_summary (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                    REAL    NOT NULL,
                cluster_id            TEXT    NOT NULL,
                rack_id               TEXT    NOT NULL,
                node_id               TEXT    NOT NULL,
                physical_gpu_count    INTEGER,
                mig_enabled_count     INTEGER,
                total_mig_instances   INTEGER,
                effective_gpu_count   INTEGER,
                node_power_draw_w     REAL,
                driver_version        TEXT,
                cuda_version          TEXT
            )
        """)

        # ── Indexes ────────────────────────────────────────────────────────
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_gpu_cluster_ts   ON gpu_telemetry(cluster_id, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gpu_uuid_ts      ON gpu_telemetry(gpu_uuid, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gpu_rack_ts      ON gpu_telemetry(rack_id, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gpu_ecc          ON gpu_telemetry(ecc_dbe_aggregate) WHERE ecc_dbe_aggregate > 0",
            "CREATE INDEX IF NOT EXISTS idx_mig_cluster_ts   ON mig_telemetry(cluster_id, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_mig_uuid_ts      ON mig_telemetry(mig_uuid, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_node_cluster_ts  ON node_summary(cluster_id, ts DESC)",
        ]:
            conn.execute(stmt)

        conn.commit()
        self._sqlite_conn = conn
        log.info(f"SQLite telemetry DB ready at {SQLITE_PATH}")

    def write_telemetry(self, payload: NodeTelemetry):
        """Write one node's telemetry push to the database."""
        ts  = payload.timestamp
        cid = payload.cluster_id
        rid = payload.rack_id
        nid = payload.node_id

        if self.backend == "sqlite":
            conn = self._sqlite_conn

            # ── Node summary ────────────────────────────────────────────────
            s = payload.summary
            conn.execute("""
                INSERT INTO node_summary
                    (ts, cluster_id, rack_id, node_id,
                     physical_gpu_count, mig_enabled_count, total_mig_instances,
                     effective_gpu_count, node_power_draw_w,
                     driver_version, cuda_version)
                VALUES (?,?,?,?, ?,?,?, ?,?, ?,?)
            """, (ts, cid, rid, nid,
                  s.physical_gpu_count, s.mig_enabled_count, s.total_mig_instances,
                  s.effective_gpu_count, s.node_power_draw_w,
                  payload.driver_version, payload.cuda_version))

            # ── Physical GPU records ────────────────────────────────────────
            for g in payload.gpus:
                conn.execute("""
                    INSERT INTO gpu_telemetry
                        (ts, cluster_id, rack_id, node_id,
                         gpu_index, gpu_uuid, gpu_name,
                         mig_enabled, mig_instance_count,
                         util_gpu_pct, util_mem_pct,
                         mem_used_mb, mem_total_mb,
                         temp_c, fan_pct,
                         power_draw_w, power_limit_w, power_util_pct, energy_delta_kj,
                         throttle_bitmask, throttle_power_capped, throttle_hw_thermal,
                         ecc_dbe_aggregate, ecc_dbe_volatile,
                         processes_json)
                    VALUES (?,?,?,?, ?,?,?, ?,?, ?,?, ?,?, ?,?, ?,?,?,?, ?,?,?, ?,?, ?)
                """, (
                    ts, cid, rid, nid,
                    g.index, g.uuid, g.name,
                    int(g.mig_enabled), g.mig_instance_count,
                    g.utilisation.gpu_pct, g.utilisation.memory_bw_pct,
                    g.memory.used_mb, g.memory.total_mb,
                    g.temperature_c, g.fan_speed_pct,
                    g.power.draw_w, g.power.limit_w, g.power.util_pct,
                    g.power.energy_delta_kj,
                    g.throttle.bitmask,
                    int(g.throttle.power_capped), int(g.throttle.hw_thermal),
                    g.ecc.dbe_aggregate, g.ecc.dbe_volatile,
                    json.dumps(g.processes),
                ))

                # ── MIG instance records ────────────────────────────────────
                for m in g.mig_instances:
                    ph = m.physical
                    conn.execute("""
                        INSERT INTO mig_telemetry
                            (ts, cluster_id, rack_id, node_id,
                             physical_gpu_index, physical_gpu_uuid, physical_gpu_name,
                             mig_index, mig_uuid, mig_name,
                             gpu_instance_id, compute_instance_id,
                             mem_used_mb, mem_total_mb, mem_used_pct, phys_mem_fraction,
                             util_gpu_pct, util_mem_bw_pct,
                             phys_temp_c, phys_power_draw_w, prorated_power_w,
                             phys_throttle_bitmask, phys_ecc_dbe_aggregate,
                             processes_json)
                        VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?, ?,?,?,?, ?,?, ?,?,?, ?,?, ?)
                    """, (
                        ts, cid, rid, nid,
                        m.physical_gpu_index, m.physical_gpu_uuid or g.uuid, g.name,
                        m.mig_index, m.mig_uuid, m.mig_name,
                        m.gpu_instance_id, m.compute_instance_id,
                        m.memory.used_mb, m.memory.total_mb,
                        m.memory.used_pct, m.memory.phys_fraction,
                        m.utilisation.gpu_pct, m.utilisation.memory_bw_pct,
                        ph.temp_c, ph.power_draw_w, ph.prorated_power_w,
                        ph.throttle_bitmask, ph.ecc_dbe_aggregate,
                        json.dumps(m.processes),
                    ))

            conn.commit()

        else:
            raise NotImplementedError(
                "TimescaleDB backend: use telemetry_receiver_pg.py "
                "which uses asyncpg. SQLite backend is active for pilot."
            )

    def query_fleet_now(self, cluster_id: str,
                        view: str = "auto",
                        window_s: int = 300) -> dict:
        """
        Current fleet state — used by the dashboard Fleet tab and RL agent.

        Always returns the most recent reading regardless of age.
        Adds a "stale" flag and "data_age_s" field so the caller knows
        how fresh the data is. window_s controls the staleness threshold
        (default 300s = 5 minutes).

        view="physical" → aggregate over gpu_telemetry
        view="mig"      → aggregate over mig_telemetry (prorated power)
        view="auto"     → mix: physical for non-MIG, MIG instances for MIG nodes
        """
        conn = self._sqlite_conn
        now  = time.time()

        if view == "physical" or view == "auto":
            row = conn.execute("""
                SELECT
                    SUM(power_draw_w)        AS total_power_w,
                    COUNT(DISTINCT gpu_uuid) AS total_gpus,
                    COUNT(DISTINCT node_id)  AS total_nodes,
                    AVG(util_gpu_pct)        AS avg_util_pct,
                    SUM(CASE WHEN mig_enabled=1 THEN 1 ELSE 0 END) AS mig_gpu_count,
                    SUM(mig_instance_count)  AS total_mig_instances,
                    SUM(CASE WHEN throttle_power_capped=1 THEN 1 ELSE 0 END) AS power_capped_count,
                    SUM(CASE WHEN ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END)   AS ecc_alert_count,
                    MIN(ts)                  AS oldest_ts,
                    MAX(ts)                  AS newest_ts
                FROM (
                    SELECT * FROM gpu_telemetry
                    WHERE cluster_id = ?
                    GROUP BY gpu_uuid
                    HAVING ts = MAX(ts)
                )
            """, (cluster_id,)).fetchone()

            result = dict(row) if row else {}
            oldest_ts = result.pop("oldest_ts", None)
            newest_ts = result.pop("newest_ts", None)
            # Use oldest node heartbeat — a silently dead node must not be masked
            data_age_s = round(now - oldest_ts, 1) if oldest_ts else None
            result["view"]          = "physical"
            result["cluster_id"]    = cluster_id
            result["data_age_s"]    = data_age_s        # worst-case node age
            result["newest_age_s"]  = round(now - newest_ts, 1) if newest_ts else None
            result["stale"]         = (data_age_s is None or data_age_s > window_s)
            return result

        elif view == "mig":
            row = conn.execute("""
                SELECT
                    SUM(prorated_power_w)             AS total_power_w,
                    COUNT(DISTINCT mig_uuid)          AS total_mig_instances,
                    COUNT(DISTINCT node_id)           AS total_nodes,
                    COUNT(DISTINCT physical_gpu_uuid) AS total_physical_gpus,
                    AVG(util_gpu_pct)                 AS avg_util_pct,
                    SUM(CASE WHEN phys_ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END) AS ecc_alert_count,
                    MIN(ts)                           AS oldest_ts,
                    MAX(ts)                           AS newest_ts
                FROM (
                    SELECT * FROM mig_telemetry
                    WHERE cluster_id = ?
                    GROUP BY mig_uuid
                    HAVING ts = MAX(ts)
                )
            """, (cluster_id,)).fetchone()

            result = dict(row) if row else {}
            oldest_ts = result.pop("oldest_ts", None)
            newest_ts = result.pop("newest_ts", None)
            data_age_s = round(now - oldest_ts, 1) if oldest_ts else None
            result["view"]          = "mig"
            result["cluster_id"]    = cluster_id
            result["data_age_s"]    = data_age_s
            result["newest_age_s"]  = round(now - newest_ts, 1) if newest_ts else None
            result["stale"]         = (data_age_s is None or data_age_s > window_s)
            return result

    def query_rack_summary(self, cluster_id: str,
                           view: str = "auto",
                           window_s: int = 300) -> List[dict]:
        """Per-rack rollup — Fleet tab grid display. Always returns latest reading per GPU."""
        conn = self._sqlite_conn

        if view in ("physical", "auto"):
            rows = conn.execute("""
                SELECT
                    rack_id,
                    SUM(power_draw_w)              AS rack_power_w,
                    AVG(util_gpu_pct)              AS avg_util_pct,
                    MAX(temp_c)                    AS max_temp_c,
                    COUNT(DISTINCT gpu_uuid)       AS gpu_count,
                    COUNT(DISTINCT node_id)        AS node_count,
                    SUM(mig_instance_count)        AS mig_instance_count,
                    SUM(throttle_power_capped)     AS power_capped_gpus,
                    SUM(CASE WHEN ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END) AS ecc_error_gpus
                FROM (
                    SELECT * FROM gpu_telemetry
                    WHERE cluster_id = ?
                    GROUP BY gpu_uuid
                    HAVING ts = MAX(ts)
                )
                GROUP BY rack_id
                ORDER BY rack_power_w DESC
            """, (cluster_id,)).fetchall()
            return [dict(r) for r in rows]

        elif view == "mig":
            rows = conn.execute("""
                SELECT
                    rack_id,
                    SUM(prorated_power_w)               AS rack_power_w,
                    AVG(util_gpu_pct)                   AS avg_util_pct,
                    MAX(phys_temp_c)                    AS max_temp_c,
                    COUNT(DISTINCT mig_uuid)            AS mig_instance_count,
                    COUNT(DISTINCT physical_gpu_uuid)   AS physical_gpu_count,
                    COUNT(DISTINCT node_id)             AS node_count,
                    SUM(CASE WHEN phys_ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END) AS ecc_error_instances
                FROM (
                    SELECT * FROM mig_telemetry
                    WHERE cluster_id = ?
                    GROUP BY mig_uuid
                    HAVING ts = MAX(ts)
                )
                GROUP BY rack_id
                ORDER BY rack_power_w DESC
            """, (cluster_id,)).fetchall()
            return [dict(r) for r in rows]

        return []

    def query_unhealthy(self, cluster_id: str, window_s: int = 120) -> List[dict]:
        """
        GPUs or MIG instances flagged for hardware issues.
        Always returns physical-GPU-level records since alerts are hardware events.
        ECC errors, thermal throttling, and high temps are all physical signals.
        """
        conn  = self._sqlite_conn
        since = time.time() - window_s

        rows = conn.execute("""
            SELECT DISTINCT
                gpu_uuid, gpu_name, node_id, rack_id,
                temp_c, power_draw_w,
                ecc_dbe_aggregate, ecc_dbe_volatile,
                throttle_bitmask, throttle_power_capped, throttle_hw_thermal,
                ts AS last_seen
            FROM (
                SELECT * FROM gpu_telemetry
                WHERE cluster_id = ?
                  AND ts > ?
                GROUP BY gpu_uuid
                HAVING ts = MAX(ts)
            )
            WHERE (
                temp_c > 85
                OR ecc_dbe_aggregate > 0
                OR throttle_hw_thermal = 1
            )
            ORDER BY ecc_dbe_aggregate DESC, temp_c DESC
        """, (cluster_id, since)).fetchall()
        return [dict(r) for r in rows]

    def query_node_mig_summary(self, cluster_id: str,
                               node_id: Optional[str] = None) -> List[dict]:
        """
        Per-node MIG configuration summary.
        Used by the dashboard Hardware tab MIG toggle.
        Returns: for each node, how many physical GPUs, how many are MIG,
        and how many MIG instances total.
        """
        conn  = self._sqlite_conn
        since = time.time() - 60

        filter_node = "AND node_id = ?" if node_id else ""
        params      = [cluster_id, since] + ([node_id] if node_id else [])

        rows = conn.execute(f"""
            SELECT
                node_id, rack_id,
                COUNT(DISTINCT gpu_uuid)                           AS physical_gpus,
                SUM(mig_enabled)                                   AS mig_enabled_gpus,
                SUM(mig_instance_count)                            AS total_mig_instances,
                SUM(power_draw_w)                                  AS node_power_w,
                AVG(util_gpu_pct)                                  AS avg_util_pct,
                SUM(CASE WHEN ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END) AS ecc_alerts
            FROM (
                SELECT * FROM gpu_telemetry
                WHERE cluster_id = ?
                  AND ts > ?
                  {filter_node}
                GROUP BY gpu_uuid
                HAVING ts = MAX(ts)
            )
            GROUP BY node_id, rack_id
            ORDER BY node_id
        """, params).fetchall()
        return [dict(r) for r in rows]

    def query_mig_instances_for_gpu(self, cluster_id: str,
                                    gpu_uuid: str,
                                    window_s: int = 60) -> List[dict]:
        """
        All current MIG instances for a specific physical GPU.
        Used by the Hardware tab GPU detail drill-down.
        """
        conn  = self._sqlite_conn
        since = time.time() - window_s

        rows = conn.execute("""
            SELECT
                mig_uuid, mig_name,
                gpu_instance_id, compute_instance_id,
                mig_index,
                mem_used_mb, mem_total_mb, mem_used_pct, phys_mem_fraction,
                util_gpu_pct, util_mem_bw_pct,
                prorated_power_w,
                phys_temp_c, phys_ecc_dbe_aggregate,
                processes_json,
                ts
            FROM (
                SELECT * FROM mig_telemetry
                WHERE cluster_id = ?
                  AND physical_gpu_uuid = ?
                  AND ts > ?
                GROUP BY mig_uuid
                HAVING ts = MAX(ts)
            )
            ORDER BY mig_index
        """, (cluster_id, gpu_uuid, since)).fetchall()

        result = [dict(r) for r in rows]
        for r in result:
            try:
                r["processes"] = json.loads(r.pop("processes_json") or "[]")
            except Exception:
                r["processes"] = []
        return result



    def query_nodes(self, cluster_id: str) -> List[dict]:
        """
        Per-node status — which nodes are online, which are stale, which
        have ECC alerts. Uses MIN heartbeat across all GPUs on each node so
        a silently dead GPU surfaces immediately rather than being masked
        by healthy siblings on the same node.
        """
        conn = self._sqlite_conn
        now  = time.time()

        rows = conn.execute("""
            SELECT
                node_id,
                rack_id,
                COUNT(DISTINCT gpu_uuid)                               AS gpu_count,
                SUM(mig_instance_count)                                AS mig_instance_count,
                SUM(CASE WHEN mig_enabled=1 THEN 1 ELSE 0 END)        AS mig_enabled_gpus,
                SUM(power_draw_w)                                      AS node_power_w,
                AVG(util_gpu_pct)                                      AS avg_util_pct,
                MAX(temp_c)                                            AS max_temp_c,
                -- Worst-case heartbeat: oldest GPU on this node determines node health
                MIN(ts)                                                AS oldest_gpu_ts,
                MAX(ts)                                                AS newest_gpu_ts,
                SUM(CASE WHEN ecc_dbe_aggregate > 0 THEN 1 ELSE 0 END) AS ecc_alerts,
                SUM(CASE WHEN throttle_power_capped=1 THEN 1 ELSE 0 END) AS power_capped_gpus,
                SUM(CASE WHEN throttle_hw_thermal=1  THEN 1 ELSE 0 END)  AS hw_thermal_gpus
            FROM (
                -- Latest reading per GPU
                SELECT * FROM gpu_telemetry
                WHERE cluster_id = ?
                GROUP BY gpu_uuid
                HAVING ts = MAX(ts)
            )
            GROUP BY node_id, rack_id
            ORDER BY rack_id, node_id
        """, (cluster_id,)).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            oldest = d.pop("oldest_gpu_ts")
            newest = d.pop("newest_gpu_ts")
            d["data_age_s"]   = round(now - oldest, 1) if oldest else None
            d["newest_age_s"] = round(now - newest, 1) if newest else None
            d["stale"]        = (d["data_age_s"] is None or d["data_age_s"] > 300)
            d["status"]       = (
                "ecc_alert"  if d["ecc_alerts"] > 0          else
                "thermal"    if d["hw_thermal_gpus"] > 0     else
                "stale"      if d["stale"]                   else
                "degraded"   if d["power_capped_gpus"] > 0  else
                "healthy"
            )
            result.append(d)
        return result

# ── Singleton DB instance ──────────────────────────────────────────────────
_db = TelemetryDB()


def init_telemetry_db():
    """Call from api.py startup event."""
    if DB_BACKEND == "sqlite":
        _db.init_sqlite()
    else:
        log.info("TimescaleDB backend configured — asyncpg pool init deferred")


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

def _check_auth(authorization: str):
    if API_TOKEN and not authorization.startswith(f"Bearer {API_TOKEN}"):
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/gpu")
def receive_gpu_telemetry(
    payload: NodeTelemetry,
    authorization: str = Header(default=""),
):
    """
    Accept a telemetry push from a collector agent.
    Called every POLL_INTERVAL seconds from every GPU node.
    """
    _check_auth(authorization)

    try:
        _db.write_telemetry(payload)
    except Exception as e:
        log.error(f"DB write error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Storage error")

    s = payload.summary
    log.debug(
        f"Received: {payload.node_id} | "
        f"{s.physical_gpu_count} GPUs | "
        f"{s.total_mig_instances} MIG instances | "
        f"{s.node_power_draw_w:.1f}W"
    )

    # Immediate critical alert forwarding
    if payload.local_alerts:
        for alert in payload.local_alerts:
            log.critical(f"NODE ALERT [{payload.node_id}]: {alert}")

    return {
        "status": "ok",
        "gpus_received":    len(payload.gpus),
        "mig_received":     s.total_mig_instances,
        "node_power_draw_w": s.node_power_draw_w,
    }


@router.get("/fleet/now")
def get_fleet_now(
    cluster_id:  str   = Query(default="default"),
    view:        str   = Query(default="auto",
                               description="physical | mig | auto"),
    window_s:    int   = Query(default=300, ge=5, le=3600),
):
    """
    Current fleet-wide aggregate.

    view=physical  → total physical GPUs, total node power
    view=mig       → total MIG instances, pro-rated power per instance
    view=auto      → physical where not MIG, MIG instances where MIG active
    """
    return _db.query_fleet_now(cluster_id, view=view, window_s=window_s)


@router.get("/fleet/racks")
def get_rack_summary(
    cluster_id:  str = Query(default="default"),
    view:        str = Query(default="auto"),
    window_s:    int = Query(default=300, ge=5, le=3600),
):
    """
    Per-rack summary — used by Fleet tab grid.

    Returns one record per rack with power, utilization, GPU count,
    MIG instance count, and alert flags.
    """
    return _db.query_rack_summary(cluster_id, view=view, window_s=window_s)


@router.get("/fleet/unhealthy")
def get_unhealthy_gpus(
    cluster_id: str = Query(default="default"),
    window_s:   int = Query(default=120, ge=30, le=600),
):
    """
    GPUs with active hardware alerts — ECC errors, thermal throttling, high temps.
    Always returns physical GPU records since alerts are hardware-level events.
    Used by the Alerts tab and anomaly detection pipeline.
    """
    return _db.query_unhealthy(cluster_id, window_s=window_s)


@router.get("/fleet/mig")
def get_mig_summary(
    cluster_id: str            = Query(default="default"),
    node_id:    Optional[str]  = Query(default=None),
):
    """
    MIG configuration summary per node.
    Dashboard Hardware tab toggle: shows physical vs MIG view of each node.

    Response fields per node:
      physical_gpus         — total physical GPU count
      mig_enabled_gpus      — how many are running MIG mode
      total_mig_instances   — total active compute instances
      node_power_w          — total node power draw
      avg_util_pct          — average GPU utilization
      ecc_alerts            — count of GPUs with ECC errors
    """
    return _db.query_node_mig_summary(cluster_id, node_id=node_id)


@router.get("/fleet/mig/gpu/{gpu_uuid}")
def get_mig_instances_for_gpu(
    gpu_uuid:   str,
    cluster_id: str = Query(default="default"),
    window_s:   int = Query(default=60),
):
    """
    All active MIG instances for one physical GPU.
    Hardware tab drill-down: click a GPU card → see MIG slice breakdown.

    Response fields per instance:
      mig_uuid, mig_name     — MIG device identifier and profile name
                                e.g. "NVIDIA A100 MIG 3g.20gb"
      gpu_instance_id        — GI index (NVIDIA partition identifier)
      compute_instance_id    — CI index within the GI
      mem_total_mb           — memory allocated to this instance
      phys_mem_fraction      — fraction of physical GPU memory (e.g. 0.25 = 25%)
      util_gpu_pct           — compute utilization of this instance
      prorated_power_w       — estimated power consumption (mem_fraction × phys_power)
      processes              — PIDs running on this instance (for job attribution)
    """
    return _db.query_mig_instances_for_gpu(cluster_id, gpu_uuid, window_s=window_s)



@router.get("/fleet/nodes")
def get_node_status(
    cluster_id: str = Query(default="default"),
):
    """
    Per-node status for the fleet.

    Returns one record per node with:
      node_id, rack_id          — node identity
      gpu_count                 — physical GPUs on this node
      mig_instance_count        — total MIG instances (0 if no MIG)
      node_power_w              — total node power draw (sum of all GPUs)
      avg_util_pct              — average GPU utilization across node
      max_temp_c                — hottest GPU on this node
      data_age_s                — seconds since OLDEST GPU heartbeat (worst-case)
      newest_age_s              — seconds since NEWEST GPU heartbeat
      stale                     — true if data_age_s > 300 seconds (node may be dead)
      ecc_alerts                — count of GPUs with double-bit ECC errors
      power_capped_gpus         — count of GPUs hitting software power cap
      hw_thermal_gpus           — count of GPUs in hardware thermal throttle
      status                    — "healthy" | "degraded" | "stale" | "thermal" | "ecc_alert"

    Status priority: ecc_alert > thermal > stale > degraded > healthy
    Use this endpoint to drive the per-node health grid on the Fleet tab.
    """
    return _db.query_nodes(cluster_id)


@router.get("/fleet/power")
def get_fleet_power_timeline(
    cluster_id:  str = Query(default="default"),
    hours:       int = Query(default=1, ge=1, le=168),
    granularity: int = Query(default=60, description="Bucket size in seconds"),
):
    """
    Node power draw over time — feeds the dashboard Overview power chart.
    Returns time-bucketed totals for the requested window.
    """
    conn  = _db._sqlite_conn
    since = time.time() - (hours * 3600)

    # SQLite: manual time bucketing by rounding ts to granularity seconds
    rows = conn.execute("""
        SELECT
            CAST(ts / ? AS INTEGER) * ?  AS bucket,
            cluster_id,
            rack_id,
            SUM(power_draw_w)            AS total_power_w,
            AVG(util_gpu_pct)            AS avg_util_pct,
            COUNT(DISTINCT gpu_uuid)     AS gpu_count
        FROM gpu_telemetry
        WHERE cluster_id = ?
          AND ts > ?
        GROUP BY bucket, cluster_id, rack_id
        ORDER BY bucket ASC
    """, (granularity, granularity, cluster_id, since)).fetchall()

    return {
        "cluster_id":    cluster_id,
        "hours":         hours,
        "granularity_s": granularity,
        "points": [
            {
                "time":       datetime.fromtimestamp(r["bucket"], tz=timezone.utc).isoformat(),
                "rack_id":    r["rack_id"],
                "power_w":    r["total_power_w"],
                "util_pct":   r["avg_util_pct"],
                "gpu_count":  r["gpu_count"],
            }
            for r in rows
        ],
    }
