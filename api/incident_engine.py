"""
incident_engine.py — HexaGrid Incident Response Automation
===========================================================
State machine for automated node incident response.

Node states:
    healthy   → Normal operation
    degraded  → Health score low or ECC errors detected — monitoring closely
    draining  → Actively migrating jobs off the node
    drained   → Node empty, awaiting maintenance/recovery
    recovering→ Node back online, health score recovering

Triggers (from alert_manager events):
    gpu_temperature  → degraded (if temp > critical threshold)
    gpu_health       → degraded (if health score < threshold)
    gpu_anomaly      → degraded
    ecc_error        → draining (immediate, ECC DBE is data-corrupting)

Outbound notifications:
    PagerDuty  — via Events API v2 (requires HEXAGRID_PAGERDUTY_KEY)
    OpsGenie   — via Alerts API   (requires HEXAGRID_OPSGENIE_KEY)
    Existing AlertManager webhook — always fires

Usage:
    from incident_engine import IncidentEngine
    engine = IncidentEngine()
    engine.handle_alert("gpu_temperature", node_id="PF57VBJL",
                        gpu_uuid="GPU-abc123", detail={"temp_c": 91})
"""

import json
import logging
import os
import sqlite3
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("hexagrid.incident")

DB_PATH = os.environ.get("HEXAGRID_DB", os.path.expanduser("~/hexagrid/hexagrid.db"))

PAGERDUTY_KEY    = os.environ.get("HEXAGRID_PAGERDUTY_KEY", "")
OPSGENIE_KEY     = os.environ.get("HEXAGRID_OPSGENIE_KEY",  "")
DRAIN_THRESHOLD  = int(os.environ.get("HEXAGRID_DRAIN_THRESHOLD_TEMP_C", "91"))
DEGRADE_THRESHOLD= int(os.environ.get("HEXAGRID_DEGRADE_THRESHOLD_TEMP_C", "85"))
HEALTH_THRESHOLD = int(os.environ.get("HEXAGRID_DEGRADE_HEALTH_SCORE", "60"))

# Node states
HEALTHY    = "healthy"
DEGRADED   = "degraded"
DRAINING   = "draining"
DRAINED    = "drained"
RECOVERING = "recovering"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incident_nodes (
    node_id     TEXT PRIMARY KEY,
    state       TEXT NOT NULL DEFAULT 'healthy',
    updated_at  TEXT NOT NULL,
    reason      TEXT,
    gpu_uuid    TEXT
);
CREATE TABLE IF NOT EXISTS incident_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    node_id     TEXT    NOT NULL,
    from_state  TEXT    NOT NULL,
    to_state    TEXT    NOT NULL,
    trigger     TEXT    NOT NULL,
    detail      TEXT,
    notified    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inc_node  ON incident_log(node_id);
CREATE INDEX IF NOT EXISTS idx_inc_ts    ON incident_log(ts DESC);
"""

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c


class IncidentEngine:

    # ── State management ──────────────────────────────────────────────────────

    def get_node_state(self, node_id: str) -> str:
        with _conn() as c:
            row = c.execute(
                "SELECT state FROM incident_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
        return row["state"] if row else HEALTHY

    def _set_state(self, node_id: str, state: str, reason: str, gpu_uuid: str = "") -> str:
        """Transition node to new state, log it, return previous state."""
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            row = c.execute(
                "SELECT state FROM incident_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            from_state = row["state"] if row else HEALTHY
            c.execute(
                """INSERT INTO incident_nodes(node_id, state, updated_at, reason, gpu_uuid)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(node_id) DO UPDATE SET
                   state=excluded.state, updated_at=excluded.updated_at,
                   reason=excluded.reason, gpu_uuid=excluded.gpu_uuid""",
                (node_id, state, now, reason, gpu_uuid)
            )
            c.execute(
                """INSERT INTO incident_log(ts, node_id, from_state, to_state, trigger, detail)
                   VALUES(?,?,?,?,?,?)""",
                (now, node_id, from_state, state, reason, gpu_uuid)
            )
            c.commit()
        log.warning("Incident: node=%s %s→%s reason=%s", node_id, from_state, state, reason)
        return from_state

    def recover_node(self, node_id: str) -> dict:
        """Mark a drained/degraded node as recovering."""
        from_state = self._set_state(node_id, RECOVERING, "manual_recovery")
        return {"node_id": node_id, "from": from_state, "to": RECOVERING}

    def clear_node(self, node_id: str) -> dict:
        """Mark a recovering node as healthy."""
        from_state = self._set_state(node_id, HEALTHY, "cleared")
        return {"node_id": node_id, "from": from_state, "to": HEALTHY}

    # ── Main alert handler ────────────────────────────────────────────────────

    def handle_alert(
        self,
        event_type: str,
        node_id:    str,
        gpu_uuid:   str  = "",
        detail:     dict = None,
    ) -> Optional[dict]:
        """
        Called by alert_manager when a relevant event fires.
        Returns the state transition dict, or None if no action taken.
        """
        detail = detail or {}
        current = self.get_node_state(node_id)

        # ECC double-bit error → immediate drain (data corruption risk)
        if event_type == "ecc_error":
            if current not in (DRAINING, DRAINED):
                return self._transition_to_draining(
                    node_id, gpu_uuid,
                    reason=f"ECC double-bit error on {gpu_uuid or node_id}",
                    detail=detail, urgency="critical"
                )

        # Temperature events
        elif event_type == "gpu_temperature":
            temp_c = detail.get("temp_c", 0)
            if temp_c >= DRAIN_THRESHOLD and current not in (DRAINING, DRAINED):
                return self._transition_to_draining(
                    node_id, gpu_uuid,
                    reason=f"GPU temperature critical: {temp_c}°C ≥ {DRAIN_THRESHOLD}°C",
                    detail=detail, urgency="critical"
                )
            elif temp_c >= DEGRADE_THRESHOLD and current == HEALTHY:
                return self._transition_to_degraded(
                    node_id, gpu_uuid,
                    reason=f"GPU temperature elevated: {temp_c}°C ≥ {DEGRADE_THRESHOLD}°C",
                    detail=detail, urgency="warning"
                )

        # Health score drop
        elif event_type in ("gpu_health", "gpu_anomaly"):
            health_score = detail.get("health_score", 100)
            if health_score < HEALTH_THRESHOLD and current == HEALTHY:
                return self._transition_to_degraded(
                    node_id, gpu_uuid,
                    reason=f"GPU health score low: {health_score} < {HEALTH_THRESHOLD}",
                    detail=detail, urgency="warning"
                )

        return None

    # ── State transitions ─────────────────────────────────────────────────────

    def _transition_to_degraded(self, node_id, gpu_uuid, reason, detail, urgency):
        with _lock:
            from_state = self._set_state(node_id, DEGRADED, reason, gpu_uuid)
        transition = {
            "node_id": node_id, "from": from_state, "to": DEGRADED,
            "reason": reason, "action": "monitoring",
        }
        self._notify(node_id, DEGRADED, reason, urgency, detail)
        return transition

    def _transition_to_draining(self, node_id, gpu_uuid, reason, detail, urgency):
        with _lock:
            from_state = self._set_state(node_id, DRAINING, reason, gpu_uuid)

        # Attempt job drain via scheduler
        drain_result = self._drain_node(node_id)

        transition = {
            "node_id":    node_id,
            "from":       from_state,
            "to":         DRAINING,
            "reason":     reason,
            "action":     "draining",
            "drain":      drain_result,
        }
        self._notify(node_id, DRAINING, reason, urgency, {**detail, **drain_result})

        # If drain completes synchronously, mark drained
        if drain_result.get("jobs_remaining", 1) == 0:
            self._set_state(node_id, DRAINED, "drain_complete", gpu_uuid)
            transition["to"] = DRAINED

        return transition

    def _drain_node(self, node_id: str) -> dict:
        """
        Attempt to drain jobs from the node.
        In a full k8s integration this would cordon + evict pods.
        Here we query telemetry for active jobs and log the drain intent.
        """
        try:
            with _conn() as c:
                # Count recent GPU activity on this node (proxy for active jobs)
                row = c.execute("""
                    SELECT COUNT(DISTINCT gpu_uuid) as active_gpus
                    FROM gpu_telemetry
                    WHERE node_id=? AND ts > datetime('now', '-2 minutes')
                """, (node_id,)).fetchone()
                active_gpus = row["active_gpus"] if row else 0
        except Exception:
            active_gpus = 0

        log.warning("DRAIN: node=%s active_gpus=%d — flagging for job migration",
                    node_id, active_gpus)

        return {
            "node_id":       node_id,
            "active_gpus":   active_gpus,
            "jobs_remaining": active_gpus,  # proxy — real drain would track job IDs
            "action":        "drain_initiated",
            "note":          "Node flagged. New jobs will not be scheduled here. "
                             "Existing jobs should complete or be migrated by the operator.",
        }

    # ── Outbound notifications ────────────────────────────────────────────────

    def _notify(self, node_id, state, reason, urgency, detail):
        """Fire all configured notification channels."""
        threading.Thread(
            target=self._notify_all,
            args=(node_id, state, reason, urgency, detail),
            daemon=True
        ).start()

    def _notify_all(self, node_id, state, reason, urgency, detail):
        if PAGERDUTY_KEY:
            self._notify_pagerduty(node_id, state, reason, urgency, detail)
        if OPSGENIE_KEY:
            self._notify_opsgenie(node_id, state, reason, urgency, detail)
        self._notify_alert_manager(node_id, state, reason, detail)

    def _notify_pagerduty(self, node_id, state, reason, urgency, detail):
        """PagerDuty Events API v2."""
        severity_map = {"critical": "critical", "warning": "warning", "info": "info"}
        payload = {
            "routing_key":  PAGERDUTY_KEY,
            "event_action": "trigger" if state in (DRAINING, DEGRADED) else "resolve",
            "dedup_key":    f"hexagrid-node-{node_id}",
            "payload": {
                "summary":   f"HexaGrid: {node_id} → {state} — {reason}",
                "severity":  severity_map.get(urgency, "warning"),
                "source":    f"hexagrid/{node_id}",
                "component": "gpu-node",
                "group":     "hexagrid-fleet",
                "custom_details": {**detail, "state": state, "node_id": node_id},
            },
        }
        self._http_post(
            url="https://events.pagerduty.com/v2/enqueue",
            data=payload,
            headers={"Content-Type": "application/json"},
            label="PagerDuty",
        )

    def _notify_opsgenie(self, node_id, state, reason, urgency, detail):
        """OpsGenie Alerts API."""
        priority_map = {"critical": "P1", "warning": "P2", "info": "P3"}
        if state in (DRAINING, DEGRADED):
            payload = {
                "message":     f"HexaGrid: {node_id} → {state}",
                "description": reason,
                "priority":    priority_map.get(urgency, "P2"),
                "alias":       f"hexagrid-node-{node_id}",
                "tags":        ["hexagrid", "gpu", state],
                "details":     {str(k): str(v) for k, v in {**detail, "node_id": node_id}.items()},
                "source":      "HexaGrid",
            }
            url = "https://api.opsgenie.com/v2/alerts"
        else:
            # Close the alert on recovery
            payload = {"user": "hexagrid-automation", "note": f"Node {node_id} recovered"}
            url = f"https://api.opsgenie.com/v2/alerts/hexagrid-node-{node_id}/close?identifierType=alias"

        self._http_post(
            url=url,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"GenieKey {OPSGENIE_KEY}",
            },
            label="OpsGenie",
        )

    def _notify_alert_manager(self, node_id, state, reason, detail):
        """Also fire through HexaGrid's own alert_manager for email/webhook."""
        try:
            sys_path = os.path.dirname(os.path.abspath(__file__))
            import sys
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from alert_manager import AlertManager
            am = AlertManager()
            level = "critical" if state in (DRAINING, DRAINED) else "warning"
            am.fire(
                event_type="gpu_health",
                title=f"Node {node_id} → {state.upper()}",
                message=reason,
                level=level,
                metadata={"node_id": node_id, "state": state, **detail},
            )
        except Exception as e:
            log.error("AlertManager notify failed: %s", e)

    @staticmethod
    def _http_post(url, data, headers, label):
        try:
            body = json.dumps(data).encode()
            req  = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as r:
                log.info("%s notified: HTTP %d", label, r.status)
        except urllib.error.HTTPError as e:
            log.error("%s HTTP error: %d %s", label, e.code, e.reason)
        except Exception as e:
            log.error("%s notify failed: %s", label, e)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def fleet_status(self) -> list:
        """Return current state of all nodes that have ever had an incident."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM incident_nodes ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def incident_history(self, node_id: str = None, limit: int = 100) -> list:
        """Return incident timeline for a node or the whole fleet."""
        with _conn() as c:
            if node_id:
                rows = c.execute(
                    "SELECT * FROM incident_log WHERE node_id=? ORDER BY ts DESC LIMIT ?",
                    (node_id, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM incident_log ORDER BY ts DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [dict(r) for r in rows]
