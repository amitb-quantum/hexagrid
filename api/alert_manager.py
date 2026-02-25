"""
alert_manager.py — HexaGrid Alerting & Notifications
=====================================================
Handles email (SMTP) and webhook (HTTP POST) delivery for all
platform alert types. Includes deduplication to prevent spam,
SQLite-persisted alert history, and config loading from alerts.json.

Supported alert types:
    price_spike       — Grid price spike detected by forecast
    gpu_health        — GPU health score dropped below threshold
    gpu_anomaly       — IsolationForest anomaly detected
    gpu_temperature   — GPU temperature hit critical threshold (88°C+)
    fleet_offline     — Fleet site went offline or hit capacity critical

Usage:
    from alert_manager import AlertManager
    am = AlertManager()
    am.fire("price_spike", region="CAISO", message="2.3× spike in 45 min", level="critical")
"""

import json
import os
import smtplib
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.expanduser("~/hexagrid")
DB_PATH     = os.environ.get("HEXAGRID_DB", os.path.join(BASE_DIR, "hexagrid.db"))
CONFIG_PATH = os.environ.get("HEXAGRID_ALERTS_CONFIG", os.path.join(BASE_DIR, "alerts.json"))

# ── Default config (written to alerts.json on first run) ──────────────────────
DEFAULT_CONFIG = {
    "enabled": True,
    "dedup_window_minutes": 30,

    "email": {
        "enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_password": "",
        "from_address": "hexagrid@yourdomain.com",
        "to_addresses": []
    },

    "webhook": {
        "enabled": False,
        "url": "",
        "headers": {
            "Content-Type": "application/json"
        },
        "include_metadata": True
    },

    "thresholds": {
        "price_spike_levels":       ["critical", "warning"],
        "gpu_health_min_score":     60,
        "gpu_temp_critical_c":      88,
        "fleet_capacity_critical_pct": 90
    },

    "events": {
        "price_spike":     True,
        "gpu_health":      True,
        "gpu_anomaly":     True,
        "gpu_temperature": True,
        "fleet_offline":   True
    }
}

# ── HTML email template ───────────────────────────────────────────────────────
EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: 'Inter', Arial, sans-serif; background: #0a0e1a; color: #e8edf5; margin: 0; padding: 0; }}
  .wrap {{ max-width: 560px; margin: 40px auto; background: #0f1425; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }}
  .header {{ background: linear-gradient(135deg, #0f1240, #1a0e35); padding: 28px 32px; border-bottom: 1px solid rgba(0,255,136,0.15); }}
  .header-logo {{ font-size: 20px; font-weight: 700; letter-spacing: 3px; color: #00ff88; }}
  .header-sub {{ font-size: 12px; color: rgba(255,255,255,0.4); margin-top: 4px; letter-spacing: 0.5px; }}
  .alert-badge {{ display: inline-block; margin-top: 16px; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }}
  .badge-critical {{ background: rgba(255,68,68,0.15);  color: #ff4444; border: 1px solid rgba(255,68,68,0.4); }}
  .badge-warning  {{ background: rgba(255,170,0,0.15);  color: #ffaa00; border: 1px solid rgba(255,170,0,0.4); }}
  .badge-elevated {{ background: rgba(0,204,255,0.12);  color: #00ccff; border: 1px solid rgba(0,204,255,0.4); }}
  .badge-info     {{ background: rgba(0,255,136,0.1);   color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }}
  .body {{ padding: 28px 32px; }}
  .title {{ font-size: 22px; font-weight: 700; margin-bottom: 10px; line-height: 1.3; }}
  .message {{ font-size: 15px; color: rgba(255,255,255,0.7); line-height: 1.7; margin-bottom: 24px; }}
  .meta {{ background: rgba(255,255,255,0.04); border-radius: 8px; padding: 16px 18px; }}
  .meta-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; }}
  .meta-row:last-child {{ border-bottom: none; }}
  .meta-key {{ color: rgba(255,255,255,0.4); }}
  .meta-val {{ color: rgba(255,255,255,0.85); font-family: monospace; }}
  .footer {{ padding: 18px 32px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; color: rgba(255,255,255,0.25); }}
  .action-btn {{ display: inline-block; margin-top: 20px; padding: 12px 24px; background: rgba(0,255,136,0.1); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); border-radius: 8px; text-decoration: none; font-size: 13px; font-weight: 600; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="header-logo">HEXAGRID</div>
    <div class="header-sub">AI Infrastructure Optimization Control Plane</div>
    <div class="alert-badge badge-{level}">{level_label}</div>
  </div>
  <div class="body">
    <div class="title">{title}</div>
    <div class="message">{message}</div>
    <div class="meta">
      <div class="meta-row"><span class="meta-key">Event Type</span><span class="meta-val">{event_type}</span></div>
      <div class="meta-row"><span class="meta-key">Severity</span><span class="meta-val">{level}</span></div>
      <div class="meta-row"><span class="meta-key">Time</span><span class="meta-val">{timestamp}</span></div>
      {extra_meta}
    </div>
    <a href="http://localhost:8000" class="action-btn">Open HexaGrid Dashboard →</a>
  </div>
  <div class="footer">
    HexaGrid™ · Quantum Clarity LLC · Alert ID: {alert_id}
    <br>To adjust alert settings, visit the Alerts tab in your dashboard.
  </div>
</div>
</body>
</html>
"""

# ── DB init ───────────────────────────────────────────────────────────────────
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id     TEXT    NOT NULL,
                fired_at     TEXT    NOT NULL,
                event_type   TEXT    NOT NULL,
                level        TEXT    NOT NULL,
                title        TEXT    NOT NULL,
                message      TEXT    NOT NULL,
                metadata     TEXT,
                email_sent   INTEGER DEFAULT 0,
                webhook_sent INTEGER DEFAULT 0,
                error        TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alert_history_fired_at
            ON alert_history (fired_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alert_history_event_type
            ON alert_history (event_type, fired_at)
        """)
        conn.commit()

_init_db()


# ── AlertManager ──────────────────────────────────────────────────────────────
class AlertManager:

    def __init__(self):
        self.config = self._load_config()

    # ── Config ────────────────────────────────────────────────────────────────
    def _load_config(self) -> dict:
        if not os.path.exists(CONFIG_PATH):
            self._write_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
        try:
            with open(CONFIG_PATH) as f:
                stored = json.load(f)
            # Merge with defaults so new keys always exist
            merged = DEFAULT_CONFIG.copy()
            merged.update(stored)
            return merged
        except Exception:
            return DEFAULT_CONFIG.copy()

    def _write_config(self, config: dict):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

    def reload_config(self):
        self.config = self._load_config()

    def get_config(self) -> dict:
        self.reload_config()
        # Never return smtp password in API responses
        safe = json.loads(json.dumps(self.config))
        safe["email"]["smtp_password"] = "••••••••" if self.config["email"]["smtp_password"] else ""
        return safe

    def update_config(self, updates: dict) -> dict:
        self.reload_config()
        # Deep merge
        def deep_merge(base, overrides):
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
        deep_merge(self.config, updates)
        # Don't overwrite password if placeholder sent
        if updates.get("email", {}).get("smtp_password") == "••••••••":
            # reload original password
            try:
                with open(CONFIG_PATH) as f:
                    original = json.load(f)
                self.config["email"]["smtp_password"] = original["email"].get("smtp_password", "")
            except Exception:
                pass
        self._write_config(self.config)
        return self.get_config()

    # ── Deduplication ─────────────────────────────────────────────────────────
    def _is_duplicate(self, event_type: str, level: str) -> bool:
        window_mins = self.config.get("dedup_window_minutes", 30)
        since = (datetime.utcnow() - timedelta(minutes=window_mins)).isoformat()
        with _get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM alert_history
                WHERE event_type = ? AND level = ? AND fired_at >= ?
            """, (event_type, level, since)).fetchone()
        return row["cnt"] > 0

    # ── Fire ──────────────────────────────────────────────────────────────────
    def fire(
        self,
        event_type: str,
        message: str,
        level: str = "warning",
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
        force: bool = False,
    ) -> dict:
        """
        Fire an alert. Checks config, deduplication, then delivers.

        Args:
            event_type: price_spike | gpu_health | gpu_anomaly | gpu_temperature | fleet_offline
            message:    Human-readable description
            level:      critical | warning | elevated | info
            title:      Short alert title (auto-generated if None)
            metadata:   Extra key-value pairs included in delivery
            force:      Skip deduplication check

        Returns:
            dict with email_sent, webhook_sent, skipped, reason
        """
        self.reload_config()

        # Check global enabled
        if not self.config.get("enabled", True):
            return {"skipped": True, "reason": "alerts_disabled"}

        # Check event type enabled
        if not self.config["events"].get(event_type, True):
            return {"skipped": True, "reason": f"event_{event_type}_disabled"}

        # Deduplication
        if not force and self._is_duplicate(event_type, level):
            return {"skipped": True, "reason": "deduplicated"}

        # Build alert record
        alert_id  = f"{event_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        fired_at  = datetime.utcnow().isoformat()
        auto_titles = {
            "price_spike":     "⚡ Price Spike Alert",
            "gpu_health":      "🖥️ GPU Health Warning",
            "gpu_anomaly":     "🔍 GPU Anomaly Detected",
            "gpu_temperature": "🌡️ GPU Temperature Critical",
            "fleet_offline":   "🌐 Fleet Site Alert",
        }
        title = title or auto_titles.get(event_type, "HexaGrid Alert")

        # Persist to history
        meta_json = json.dumps(metadata or {})
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO alert_history
                  (alert_id, fired_at, event_type, level, title, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, fired_at, event_type, level, title, message, meta_json))
            conn.commit()

        result = {
            "alert_id":     alert_id,
            "skipped":      False,
            "email_sent":   False,
            "webhook_sent": False,
            "errors":       []
        }

        # ── Email delivery ────────────────────────────────────────────────────
        if self.config["email"]["enabled"] and self.config["email"]["to_addresses"]:
            try:
                self._send_email(alert_id, fired_at, event_type, level, title, message, metadata or {})
                result["email_sent"] = True
                with _get_conn() as conn:
                    conn.execute("UPDATE alert_history SET email_sent=1 WHERE alert_id=?", (alert_id,))
                    conn.commit()
            except Exception as e:
                err = f"Email error: {str(e)}"
                result["errors"].append(err)
                with _get_conn() as conn:
                    conn.execute("UPDATE alert_history SET error=? WHERE alert_id=?", (err, alert_id))
                    conn.commit()

        # ── Webhook delivery ──────────────────────────────────────────────────
        if self.config["webhook"]["enabled"] and self.config["webhook"]["url"]:
            try:
                self._send_webhook(alert_id, fired_at, event_type, level, title, message, metadata or {})
                result["webhook_sent"] = True
                with _get_conn() as conn:
                    conn.execute("UPDATE alert_history SET webhook_sent=1 WHERE alert_id=?", (alert_id,))
                    conn.commit()
            except Exception as e:
                err = f"Webhook error: {str(e)}"
                result["errors"].append(err)
                with _get_conn() as conn:
                    conn.execute("UPDATE alert_history SET error=? WHERE alert_id=?", (err, alert_id))
                    conn.commit()

        return result

    # ── Email ─────────────────────────────────────────────────────────────────
    def _send_email(self, alert_id, fired_at, event_type, level, title, message, metadata):
        cfg = self.config["email"]
        level_labels = {"critical": "CRITICAL", "warning": "WARNING", "elevated": "ELEVATED", "info": "INFO"}

        extra_meta = ""
        for k, v in metadata.items():
            extra_meta += f'<div class="meta-row"><span class="meta-key">{k}</span><span class="meta-val">{v}</span></div>'

        html_body = EMAIL_TEMPLATE.format(
            level=level,
            level_label=level_labels.get(level, level.upper()),
            title=title,
            message=message,
            event_type=event_type,
            timestamp=datetime.fromisoformat(fired_at).strftime("%Y-%m-%d %H:%M:%S UTC"),
            extra_meta=extra_meta,
            alert_id=alert_id,
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[HexaGrid {level.upper()}] {title}"
        msg["From"]    = cfg["from_address"]
        msg["To"]      = ", ".join(cfg["to_addresses"])
        msg.attach(MIMEText(f"HexaGrid Alert\n\n{title}\n\n{message}\n\nEvent: {event_type}\nLevel: {level}\nTime: {fired_at}", "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(cfg["from_address"], cfg["to_addresses"], msg.as_string())

    # ── Webhook ───────────────────────────────────────────────────────────────
    def _send_webhook(self, alert_id, fired_at, event_type, level, title, message, metadata):
        cfg = self.config["webhook"]

        payload = {
            "alert_id":   alert_id,
            "event_type": event_type,
            "level":      level,
            "title":      title,
            "message":    message,
            "fired_at":   fired_at,
            "source":     "hexagrid",
        }
        if cfg.get("include_metadata"):
            payload["metadata"] = metadata

        # Slack-compatible format (works if URL is a Slack webhook)
        level_emoji = {"critical": "🔴", "warning": "🟡", "elevated": "🔵", "info": "🟢"}
        payload["text"] = f"{level_emoji.get(level, '⚡')} *{title}*\n{message}"
        payload["attachments"] = [{
            "color": {"critical": "#ff4444", "warning": "#ffaa00", "elevated": "#00ccff", "info": "#00ff88"}.get(level, "#00ff88"),
            "fields": [{"title": k, "value": str(v), "short": True} for k, v in metadata.items()]
        }]

        data    = json.dumps(payload).encode("utf-8")
        headers = cfg.get("headers", {"Content-Type": "application/json"})
        req     = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 201, 202, 204):
                raise ValueError(f"Webhook returned HTTP {resp.status}")

    # ── History ───────────────────────────────────────────────────────────────
    def history(self, limit: int = 100, event_type: Optional[str] = None) -> list:
        with _get_conn() as conn:
            if event_type:
                rows = conn.execute("""
                    SELECT * FROM alert_history WHERE event_type = ?
                    ORDER BY fired_at DESC LIMIT ?
                """, (event_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM alert_history
                    ORDER BY fired_at DESC LIMIT ?
                """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def history_summary(self) -> dict:
        with _get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM alert_history").fetchone()["c"]
            by_type = conn.execute("""
                SELECT event_type, COUNT(*) as cnt,
                       MAX(fired_at) as last_fired
                FROM alert_history GROUP BY event_type
            """).fetchall()
            last_24h = conn.execute("""
                SELECT COUNT(*) as c FROM alert_history
                WHERE fired_at >= ?
            """, ((datetime.utcnow() - timedelta(hours=24)).isoformat(),)).fetchone()["c"]
        return {
            "total":   total,
            "last_24h": last_24h,
            "by_type": [dict(r) for r in by_type],
        }

    # ── Test ──────────────────────────────────────────────────────────────────
    def send_test(self, channel: str = "both") -> dict:
        """Send a test alert to verify delivery configuration."""
        return self.fire(
            event_type="price_spike",
            title="✅ HexaGrid Test Alert",
            message="This is a test alert from HexaGrid. Your notification channel is configured correctly.",
            level="info",
            metadata={"channel": channel, "test": "true"},
            force=True,
        )


if __name__ == "__main__":
    am = AlertManager()
    print("Config path:", CONFIG_PATH)
    print("Config:", json.dumps(am.get_config(), indent=2))
    print("\nFiring test alert (no channels enabled by default)...")
    result = am.fire(
        event_type="price_spike",
        message="CAISO price spike 2.3× current price predicted in 45 minutes.",
        level="critical",
        metadata={"region": "CAISO", "ratio": "2.3x", "eta_minutes": 45},
    )
    print("Result:", json.dumps(result, indent=2))
    print("\nAlert history:")
    for entry in am.history(limit=5):
        print(f"  [{entry['level'].upper()}] {entry['event_type']} — {entry['title']}")
