"""
tpu_receiver.py — HexaGrid TPU Telemetry Receiver
==================================================
Receives and stores TPU telemetry from GCP (Cloud TPU v4/v5)
and AWS (Trainium/Inferentia) collector agents.

Schema mirrors gpu_telemetry but uses TPU-appropriate fields:
  - matrix_util_pct   instead of sm_util_pct
  - hbm_used_mib      instead of memory_used_mib
  - ici_bw_gbps       instead of nvlink_tx_gbps
  - No ECC (not exposed by vendors), no thermal (not exposed)

Usage:
    from tpu_receiver import router as tpu_router, init_tpu_db
    app.include_router(tpu_router)
    init_tpu_db()
"""

import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/telemetry/tpu", tags=["TPU Telemetry"])

SQLITE_PATH = os.environ.get("HEXAGRID_DB",
    os.path.join(os.path.expanduser("~"), "hexagrid", "data", "telemetry.db"))
if not os.path.exists(os.path.dirname(SQLITE_PATH)):
    SQLITE_PATH = "/var/lib/hexagrid/telemetry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tpu_telemetry (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    cluster_id          TEXT    NOT NULL DEFAULT 'default',
    provider            TEXT    NOT NULL DEFAULT 'gcp',
    node_id             TEXT    NOT NULL,
    zone                TEXT    NOT NULL DEFAULT 'unknown',
    chip_id             TEXT    NOT NULL,
    chip_type           TEXT    NOT NULL DEFAULT 'unknown',
    accelerator_type    TEXT    NOT NULL DEFAULT 'unknown',

    -- Utilization
    matrix_util_pct     REAL,
    memory_util_pct     REAL,

    -- Memory (HBM)
    hbm_used_mib        REAL,
    hbm_total_mib       REAL,

    -- Throughput
    ici_bw_gbps         REAL,
    host_bw_gbps        REAL,

    -- AWS Neuron-specific
    neuroncore_util_pct REAL,
    runtime_errors      INTEGER DEFAULT 0,

    -- Identity
    instance_type       TEXT,
    runtime_version     TEXT
);

CREATE INDEX IF NOT EXISTS idx_tpu_cluster_ts  ON tpu_telemetry(cluster_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_tpu_node_ts     ON tpu_telemetry(node_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_tpu_provider    ON tpu_telemetry(provider, ts DESC);

CREATE TABLE IF NOT EXISTS tpu_node_summary (
    node_id          TEXT PRIMARY KEY,
    cluster_id       TEXT,
    provider         TEXT,
    zone             TEXT,
    chip_type        TEXT,
    accelerator_type TEXT,
    chip_count       INTEGER,
    last_seen        TEXT,
    avg_util_pct     REAL,
    avg_hbm_used_pct REAL,
    total_errors     INTEGER DEFAULT 0
);
"""


def _conn():
    os.makedirs(os.path.dirname(os.path.abspath(SQLITE_PATH)), exist_ok=True)
    c = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_tpu_db():
    with _conn() as c:
        c.executescript(_SCHEMA)
        c.commit()
    print("  ✓  TPU telemetry DB ready")


# ── Pydantic models ───────────────────────────────────────────────────────────

class TPUChipPayload(BaseModel):
    chip_id:             str
    chip_type:           str   = "unknown"       # tpu-v4, tpu-v5e, trainium, inferentia2
    matrix_util_pct:     Optional[float] = None
    memory_util_pct:     Optional[float] = None
    hbm_used_mib:        Optional[float] = None
    hbm_total_mib:       Optional[float] = None
    ici_bw_gbps:         Optional[float] = None
    host_bw_gbps:        Optional[float] = None
    neuroncore_util_pct: Optional[float] = None
    runtime_errors:      int   = 0


class TPUNodePayload(BaseModel):
    node_id:          str
    cluster_id:       str   = "default"
    provider:         str   = "gcp"              # gcp | aws
    zone:             str   = "unknown"
    accelerator_type: str   = "unknown"          # v4-8, v5e-256, trn1.32xlarge etc.
    instance_type:    Optional[str] = None
    runtime_version:  Optional[str] = None
    chips:            List[TPUChipPayload] = Field(default_factory=list)


# ── Ingest endpoint ───────────────────────────────────────────────────────────

@router.post("/", summary="Ingest TPU telemetry from collector agent")
def receive_tpu_telemetry(payload: TPUNodePayload):
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for chip in payload.chips:
        rows.append((
            ts, payload.cluster_id, payload.provider,
            payload.node_id, payload.zone,
            chip.chip_id, chip.chip_type, payload.accelerator_type,
            chip.matrix_util_pct, chip.memory_util_pct,
            chip.hbm_used_mib, chip.hbm_total_mib,
            chip.ici_bw_gbps, chip.host_bw_gbps,
            chip.neuroncore_util_pct, chip.runtime_errors,
            payload.instance_type, payload.runtime_version,
        ))

    with _conn() as c:
        c.executemany("""
            INSERT INTO tpu_telemetry
              (ts, cluster_id, provider, node_id, zone, chip_id, chip_type,
               accelerator_type, matrix_util_pct, memory_util_pct,
               hbm_used_mib, hbm_total_mib, ici_bw_gbps, host_bw_gbps,
               neuroncore_util_pct, runtime_errors, instance_type, runtime_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

        # Upsert node summary
        chip_count = len(payload.chips)
        utils = [ch.matrix_util_pct for ch in payload.chips if ch.matrix_util_pct is not None]
        avg_util = sum(utils) / len(utils) if utils else None
        hbm_pcts = []
        for ch in payload.chips:
            if ch.hbm_used_mib and ch.hbm_total_mib:
                hbm_pcts.append(100 * ch.hbm_used_mib / ch.hbm_total_mib)
        avg_hbm = sum(hbm_pcts) / len(hbm_pcts) if hbm_pcts else None
        total_errors = sum(ch.runtime_errors for ch in payload.chips)

        c.execute("""
            INSERT INTO tpu_node_summary
              (node_id, cluster_id, provider, zone, chip_type, accelerator_type,
               chip_count, last_seen, avg_util_pct, avg_hbm_used_pct, total_errors)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(node_id) DO UPDATE SET
              last_seen=excluded.last_seen,
              avg_util_pct=excluded.avg_util_pct,
              avg_hbm_used_pct=excluded.avg_hbm_used_pct,
              total_errors=excluded.total_errors,
              chip_count=excluded.chip_count
        """, (
            payload.node_id, payload.cluster_id, payload.provider,
            payload.zone, payload.chips[0].chip_type if payload.chips else "unknown",
            payload.accelerator_type, chip_count, ts,
            avg_util, avg_hbm, total_errors,
        ))
        c.commit()

    return {"status": "ok", "chips_recorded": len(rows), "ts": ts}


# ── Query endpoints ───────────────────────────────────────────────────────────

@router.get("/fleet", summary="TPU fleet overview — all nodes")
def tpu_fleet(
    cluster_id: Optional[str] = Query(None),
    provider:   Optional[str] = Query(None, description="gcp | aws"),
    stale_secs: int           = Query(120),
):
    """Return current TPU node summaries with staleness detection."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_secs)).isoformat()
    wheres, params = [], []
    if cluster_id:
        wheres.append("cluster_id=?"); params.append(cluster_id)
    if provider:
        wheres.append("provider=?"); params.append(provider)
    where = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    with _conn() as c:
        nodes = [dict(r) for r in c.execute(
            f"SELECT * FROM tpu_node_summary {where} ORDER BY last_seen DESC",
            params
        ).fetchall()]

    # Annotate staleness
    for n in nodes:
        n["stale"] = (n.get("last_seen") or "") < cutoff

    total_chips = sum(n.get("chip_count", 0) for n in nodes)
    utils = [n["avg_util_pct"] for n in nodes if n.get("avg_util_pct") is not None]
    fleet_util = round(sum(utils) / len(utils), 1) if utils else None

    return {
        "node_count":      len(nodes),
        "total_chips":     total_chips,
        "fleet_util_pct":  fleet_util,
        "providers":       list({n["provider"] for n in nodes}),
        "nodes":           nodes,
    }


@router.get("/history", summary="Recent TPU telemetry for a node")
def tpu_history(
    node_id:    str,
    minutes:    int = Query(30, ge=1, le=1440),
    cluster_id: Optional[str] = None,
):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    params = [node_id, cutoff]
    extra = ""
    if cluster_id:
        extra = " AND cluster_id=?"; params.append(cluster_id)
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM tpu_telemetry WHERE node_id=? AND ts>?{extra} ORDER BY ts DESC LIMIT 500",
            params
        ).fetchall()
    return {"node_id": node_id, "count": len(rows), "rows": [dict(r) for r in rows]}


@router.get("/summary", summary="Fleet-wide TPU aggregate stats")
def tpu_summary(cluster_id: Optional[str] = Query(None)):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    params = [cutoff]
    extra = ""
    if cluster_id:
        extra = " AND cluster_id=?"; params.append(cluster_id)
    with _conn() as c:
        row = c.execute(f"""
            SELECT
                COUNT(DISTINCT node_id)     AS nodes,
                COUNT(DISTINCT chip_id)     AS chips,
                AVG(matrix_util_pct)        AS avg_util,
                AVG(memory_util_pct)        AS avg_mem_util,
                SUM(hbm_used_mib)           AS total_hbm_used_mib,
                SUM(runtime_errors)         AS total_errors,
                COUNT(DISTINCT provider)    AS providers
            FROM tpu_telemetry
            WHERE ts>?{extra}
        """, params).fetchone()
    return {k: round(v, 2) if isinstance(v, float) else v
            for k, v in dict(row).items()} if row else {}
