"""
chargeback_ledger.py — HexaGrid Cost Allocation & Chargeback
=============================================================
Tracks every scheduled job's energy cost attributed to a cost center,
team, and project. Produces monthly/custom-period reports exportable
as JSON or CSV.

The ledger is populated automatically when jobs are submitted through
the scheduler with cost_center/team/project tags on JobSpec.
It can also be populated manually via the API for jobs run outside
the HexaGrid scheduler.

Schema:
    chargeback_entries   — one row per job run
    chargeback_periods   — cached monthly roll-ups (rebuilt on demand)

Usage:
    from chargeback_ledger import ChargebackLedger
    ledger = ChargebackLedger()
    ledger.record(
        job_id="train_042", job_name="ResNet Training",
        cost_center="ML-Platform", team="infra", project="ImageNet-2026",
        energy_kwh=12.4, cost_usd=0.87, duration_min=45,
        region="CAISO", node_id="PF57VBJL", scheduler="qaoa"
    )
"""

import csv
import io
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.environ.get("HEXAGRID_DB", os.path.expanduser("~/hexagrid/hexagrid.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chargeback_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at     TEXT    NOT NULL,
    job_id          TEXT    NOT NULL,
    job_name        TEXT    NOT NULL DEFAULT '',
    cost_center     TEXT    NOT NULL DEFAULT 'untagged',
    team            TEXT    NOT NULL DEFAULT 'untagged',
    project         TEXT    NOT NULL DEFAULT 'untagged',
    region          TEXT    NOT NULL DEFAULT 'unknown',
    node_id         TEXT    NOT NULL DEFAULT 'unknown',
    scheduler       TEXT    NOT NULL DEFAULT 'unknown',
    duration_min    REAL    NOT NULL DEFAULT 0,
    power_kw        REAL    NOT NULL DEFAULT 0,
    energy_kwh      REAL    NOT NULL DEFAULT 0,
    price_per_kwh   REAL    NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0,
    naive_cost_usd  REAL    NOT NULL DEFAULT 0,
    saved_usd       REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cb_cost_center ON chargeback_entries(cost_center);
CREATE INDEX IF NOT EXISTS idx_cb_team        ON chargeback_entries(team);
CREATE INDEX IF NOT EXISTS idx_cb_project     ON chargeback_entries(project);
CREATE INDEX IF NOT EXISTS idx_cb_recorded_at ON chargeback_entries(recorded_at);
CREATE INDEX IF NOT EXISTS idx_cb_region      ON chargeback_entries(region);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c


class ChargebackLedger:

    def record(
        self,
        job_id:         str,
        job_name:       str        = "",
        cost_center:    str        = "untagged",
        team:           str        = "untagged",
        project:        str        = "untagged",
        region:         str        = "unknown",
        node_id:        str        = "unknown",
        scheduler:      str        = "unknown",
        duration_min:   float      = 0.0,
        power_kw:       float      = 0.0,
        energy_kwh:     float      = 0.0,
        price_per_kwh:  float      = 0.0,
        cost_usd:       float      = 0.0,
        naive_cost_usd: float      = 0.0,
    ) -> int:
        """Record a job's energy cost. Returns the new row id."""
        saved_usd = max(0.0, naive_cost_usd - cost_usd)
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            cur = c.execute(
                """INSERT INTO chargeback_entries
                   (recorded_at, job_id, job_name, cost_center, team, project,
                    region, node_id, scheduler, duration_min, power_kw,
                    energy_kwh, price_per_kwh, cost_usd, naive_cost_usd, saved_usd)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now, str(job_id), job_name, cost_center, team, project,
                 region, node_id, scheduler, duration_min, power_kw,
                 energy_kwh, price_per_kwh, cost_usd, naive_cost_usd, saved_usd)
            )
            c.commit()
            return cur.lastrowid

    def report(
        self,
        period_start:   Optional[str] = None,
        period_end:     Optional[str] = None,
        cost_center:    Optional[str] = None,
        team:           Optional[str] = None,
        project:        Optional[str] = None,
        group_by:       str           = "cost_center",
    ) -> dict:
        """
        Return a chargeback report for the given period and filters.
        group_by: 'cost_center' | 'team' | 'project' | 'region' | 'job_id'
        """
        valid_groups = {"cost_center", "team", "project", "region", "job_id", "scheduler"}
        if group_by not in valid_groups:
            group_by = "cost_center"

        where, params = self._build_where(period_start, period_end,
                                          cost_center, team, project)
        q = f"""
            SELECT
                {group_by}                          AS group_key,
                COUNT(*)                            AS job_count,
                SUM(duration_min)                   AS total_duration_min,
                SUM(energy_kwh)                     AS total_energy_kwh,
                SUM(cost_usd)                       AS total_cost_usd,
                SUM(naive_cost_usd)                 AS total_naive_cost_usd,
                SUM(saved_usd)                      AS total_saved_usd,
                AVG(price_per_kwh)                  AS avg_price_per_kwh,
                MIN(recorded_at)                    AS first_job,
                MAX(recorded_at)                    AS last_job
            FROM chargeback_entries
            {where}
            GROUP BY {group_by}
            ORDER BY total_cost_usd DESC
        """
        with _conn() as c:
            rows = c.execute(q, params).fetchall()

        # Totals row
        tq = f"""
            SELECT
                COUNT(*)        AS job_count,
                SUM(energy_kwh) AS total_energy_kwh,
                SUM(cost_usd)   AS total_cost_usd,
                SUM(saved_usd)  AS total_saved_usd
            FROM chargeback_entries {where}
        """
        with _conn() as c:
            tot = dict(c.execute(tq, params).fetchone())

        return {
            "period_start":  period_start or "all-time",
            "period_end":    period_end   or "all-time",
            "group_by":      group_by,
            "filters":       {"cost_center": cost_center, "team": team, "project": project},
            "totals":        {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in tot.items()},
            "breakdown":     [
                {
                    "group":               row["group_key"],
                    "job_count":           row["job_count"],
                    "total_duration_min":  round(row["total_duration_min"] or 0, 1),
                    "total_energy_kwh":    round(row["total_energy_kwh"]   or 0, 3),
                    "total_cost_usd":      round(row["total_cost_usd"]     or 0, 4),
                    "total_naive_cost_usd":round(row["total_naive_cost_usd"] or 0, 4),
                    "total_saved_usd":     round(row["total_saved_usd"]    or 0, 4),
                    "avg_price_per_kwh":   round(row["avg_price_per_kwh"]  or 0, 5),
                    "first_job":           row["first_job"],
                    "last_job":            row["last_job"],
                }
                for row in rows
            ],
        }

    def report_csv(self, **kwargs) -> str:
        """Return the report as a CSV string."""
        data = self.report(**kwargs)
        buf = io.StringIO()
        if not data["breakdown"]:
            return "No data for the selected period and filters.\n"
        writer = csv.DictWriter(buf, fieldnames=data["breakdown"][0].keys())
        writer.writeheader()
        writer.writerows(data["breakdown"])
        # Totals row
        buf.write("\n")
        buf.write(f"Period,{data['period_start']} → {data['period_end']}\n")
        buf.write(f"Total jobs,{data['totals'].get('job_count','')}\n")
        buf.write(f"Total energy (kWh),{data['totals'].get('total_energy_kwh','')}\n")
        buf.write(f"Total cost (USD),{data['totals'].get('total_cost_usd','')}\n")
        buf.write(f"Total saved (USD),{data['totals'].get('total_saved_usd','')}\n")
        return buf.getvalue()

    def entries(
        self,
        period_start: Optional[str] = None,
        period_end:   Optional[str] = None,
        cost_center:  Optional[str] = None,
        team:         Optional[str] = None,
        project:      Optional[str] = None,
        limit:        int           = 200,
    ) -> list:
        """Return raw entries (paginated) for audit/detail views."""
        where, params = self._build_where(period_start, period_end,
                                          cost_center, team, project)
        params.append(min(limit, 1000))
        with _conn() as c:
            rows = c.execute(
                f"SELECT * FROM chargeback_entries {where} ORDER BY recorded_at DESC LIMIT ?",
                params
            ).fetchall()
        return [dict(r) for r in rows]

    def dimensions(self) -> dict:
        """Return all known cost_center / team / project values for filter dropdowns."""
        with _conn() as c:
            def _vals(col):
                return [r[0] for r in c.execute(
                    f"SELECT DISTINCT {col} FROM chargeback_entries ORDER BY {col}"
                ).fetchall()]
            return {
                "cost_centers": _vals("cost_center"),
                "teams":        _vals("team"),
                "projects":     _vals("project"),
                "regions":      _vals("region"),
            }

    @staticmethod
    def _build_where(start, end, cost_center, team, project):
        clauses, params = [], []
        if start:
            clauses.append("recorded_at >= ?"); params.append(start)
        if end:
            clauses.append("recorded_at <= ?"); params.append(end)
        if cost_center:
            clauses.append("cost_center = ?"); params.append(cost_center)
        if team:
            clauses.append("team = ?"); params.append(team)
        if project:
            clauses.append("project = ?"); params.append(project)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params
