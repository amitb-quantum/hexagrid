"""
savings_ledger.py — HexaGrid Cost Savings Ledger
=================================================
Tracks every dispatch decision made by the RL agent or QAOA scheduler,
comparing actual (optimized) cost vs naive flat-rate cost.
Persists to SQLite. Exposes cumulative and windowed savings summaries.

Usage:
    from savings_ledger import SavingsLedger
    ledger = SavingsLedger()
    ledger.record(
        region="CAISO",
        job_id="train_job_042",
        naive_price=0.142,       # $/kWh — what we would have paid immediately
        actual_price=0.031,      # $/kWh — what we actually paid after deferral
        power_kw=4.2,            # GPU rack power during job
        duration_minutes=45,
        decision="defer_60",
        source="rl_agent"        # or "qaoa_scheduler"
    )
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.environ.get("HEXAGRID_DB", os.path.expanduser("~/hexagrid/hexagrid.db"))


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS savings_ledger (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at       TEXT    NOT NULL,
                region            TEXT    NOT NULL,
                job_id            TEXT,
                naive_price_kwh   REAL    NOT NULL,
                actual_price_kwh  REAL    NOT NULL,
                power_kw          REAL    NOT NULL,
                duration_minutes  REAL    NOT NULL,
                naive_cost_usd    REAL    NOT NULL,
                actual_cost_usd   REAL    NOT NULL,
                saved_usd         REAL    NOT NULL,
                savings_pct       REAL    NOT NULL,
                decision          TEXT    NOT NULL,
                source            TEXT    NOT NULL DEFAULT 'rl_agent'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_savings_recorded_at
            ON savings_ledger (recorded_at)
        """)
        conn.commit()


_init_db()


class SavingsLedger:
    """Records and retrieves cost savings decisions."""

    def record(
        self,
        region: str,
        naive_price: float,
        actual_price: float,
        power_kw: float,
        duration_minutes: float,
        decision: str,
        job_id: Optional[str] = None,
        source: str = "rl_agent",
    ) -> dict:
        """
        Record a single dispatch decision and compute savings.

        Args:
            region:           ISO region (e.g. "CAISO")
            naive_price:      $/kWh price if job ran immediately
            actual_price:     $/kWh price after optimized deferral
            power_kw:         Average GPU rack power draw during job
            duration_minutes: Job duration in minutes
            decision:         "run_now" | "defer_15" | "defer_30" | "defer_60"
            job_id:           Optional job identifier
            source:           "rl_agent" | "qaoa_scheduler" | "manual"

        Returns:
            dict with saved_usd and savings_pct
        """
        duration_hours = duration_minutes / 60.0
        naive_cost = naive_price * power_kw * duration_hours
        actual_cost = actual_price * power_kw * duration_hours
        saved = naive_cost - actual_cost
        savings_pct = (saved / naive_cost * 100) if naive_cost > 0 else 0.0

        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO savings_ledger
                  (recorded_at, region, job_id, naive_price_kwh, actual_price_kwh,
                   power_kw, duration_minutes, naive_cost_usd, actual_cost_usd,
                   saved_usd, savings_pct, decision, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                region, job_id,
                naive_price, actual_price,
                power_kw, duration_minutes,
                round(naive_cost, 6), round(actual_cost, 6),
                round(saved, 6), round(savings_pct, 2),
                decision, source
            ))
            conn.commit()

        return {
            "naive_cost_usd": round(naive_cost, 4),
            "actual_cost_usd": round(actual_cost, 4),
            "saved_usd": round(saved, 4),
            "savings_pct": round(savings_pct, 2),
        }

    def summary(self) -> dict:
        """
        Returns cumulative savings summary plus windowed breakdowns.
        """
        now = datetime.utcnow()
        windows = {
            "today":    (now - timedelta(days=1)).isoformat(),
            "last_7d":  (now - timedelta(days=7)).isoformat(),
            "last_30d": (now - timedelta(days=30)).isoformat(),
            "all_time": "1970-01-01T00:00:00",
        }

        result = {}
        with _get_conn() as conn:
            for label, since in windows.items():
                row = conn.execute("""
                    SELECT
                        COUNT(*)            AS decisions,
                        COALESCE(SUM(saved_usd), 0)       AS total_saved,
                        COALESCE(SUM(naive_cost_usd), 0)  AS total_naive,
                        COALESCE(SUM(actual_cost_usd), 0) AS total_actual,
                        COALESCE(AVG(savings_pct), 0)     AS avg_savings_pct
                    FROM savings_ledger
                    WHERE recorded_at >= ?
                """, (since,)).fetchone()

                total_naive = row["total_naive"]
                total_saved = row["total_saved"]
                overall_pct = (total_saved / total_naive * 100) if total_naive > 0 else 0.0

                result[label] = {
                    "decisions":       row["decisions"],
                    "total_saved_usd": round(total_saved, 2),
                    "total_naive_usd": round(total_naive, 2),
                    "total_actual_usd":round(row["total_actual"], 2),
                    "savings_pct":     round(overall_pct, 1),
                    "avg_savings_pct": round(row["avg_savings_pct"], 1),
                }

        return result

    def ledger(self, limit: int = 100, region: Optional[str] = None) -> list:
        """
        Returns recent ledger entries, newest first.
        """
        with _get_conn() as conn:
            if region:
                rows = conn.execute("""
                    SELECT * FROM savings_ledger
                    WHERE region = ?
                    ORDER BY recorded_at DESC LIMIT ?
                """, (region, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM savings_ledger
                    ORDER BY recorded_at DESC LIMIT ?
                """, (limit,)).fetchall()

        return [dict(r) for r in rows]

    def by_region(self) -> list:
        """
        Returns savings broken down by ISO region.
        """
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    region,
                    COUNT(*)                          AS decisions,
                    COALESCE(SUM(saved_usd), 0)       AS total_saved,
                    COALESCE(SUM(naive_cost_usd), 0)  AS total_naive,
                    COALESCE(AVG(savings_pct), 0)     AS avg_savings_pct
                FROM savings_ledger
                GROUP BY region
                ORDER BY total_saved DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def inject_demo_data(self, n: int = 50):
        """
        Inject realistic demo entries for dashboard testing.
        Run once: python -c "from savings_ledger import SavingsLedger; SavingsLedger().inject_demo_data()"
        """
        import random
        regions = ["CAISO", "ERCOT", "NYISO", "ISONE", "PJM"]
        decisions = ["defer_15", "defer_30", "defer_60", "run_now"]
        sources = ["rl_agent", "qaoa_scheduler"]

        now = datetime.utcnow()
        for i in range(n):
            region = random.choice(regions)
            # Realistic wholesale price range
            naive_price = round(random.uniform(0.08, 0.22), 4)
            # Optimized price is 20–80% lower
            discount = random.uniform(0.20, 0.80)
            actual_price = round(naive_price * (1 - discount), 4)
            power_kw = round(random.uniform(2.5, 8.0), 2)
            duration = round(random.uniform(15, 120), 1)
            decision = random.choice(decisions)
            if decision == "run_now":
                actual_price = naive_price  # No savings on immediate dispatch

            duration_hours = duration / 60.0
            naive_cost = naive_price * power_kw * duration_hours
            actual_cost = actual_price * power_kw * duration_hours
            saved = naive_cost - actual_cost
            savings_pct = (saved / naive_cost * 100) if naive_cost > 0 else 0.0

            # Spread over last 30 days
            offset_minutes = random.randint(0, 43200)
            recorded_at = (now - timedelta(minutes=offset_minutes)).isoformat()

            with _get_conn() as conn:
                conn.execute("""
                    INSERT INTO savings_ledger
                      (recorded_at, region, job_id, naive_price_kwh, actual_price_kwh,
                       power_kw, duration_minutes, naive_cost_usd, actual_cost_usd,
                       saved_usd, savings_pct, decision, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recorded_at, region, f"demo_job_{i:04d}",
                    naive_price, actual_price,
                    power_kw, duration,
                    round(naive_cost, 6), round(actual_cost, 6),
                    round(saved, 6), round(savings_pct, 2),
                    decision, random.choice(sources)
                ))
                conn.commit()

        print(f"Injected {n} demo savings entries.")


if __name__ == "__main__":
    ledger = SavingsLedger()
    ledger.inject_demo_data(50)
    summary = ledger.summary()
    print("\n── Savings Summary ──")
    for window, data in summary.items():
        print(f"  {window:12s}: ${data['total_saved_usd']:.2f} saved "
              f"across {data['decisions']} decisions "
              f"({data['savings_pct']}%)")
