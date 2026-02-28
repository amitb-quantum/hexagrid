#!/usr/bin/env python3
"""
HexaGrid — Hotfix: wire alert_manager.fire() → incident_engine.handle_alert()
Run from ~/hexagrid/api/:  python hotfix_wire_incidents.py
"""
from pathlib import Path

ALERT_MGR = Path.home() / "hexagrid" / "api" / "alert_manager.py"
content = ALERT_MGR.read_text()

OLD = '''\
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO alert_history
                  (alert_id, fired_at, event_type, level, title, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, fired_at, event_type, level, title, message, meta_json))
            conn.commit()'''

NEW = '''\
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO alert_history
                  (alert_id, fired_at, event_type, level, title, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, fired_at, event_type, level, title, message, meta_json))
            conn.commit()

        # ── Incident engine hook ──────────────────────────────────────────────
        if event_type in ("gpu_temperature", "gpu_health", "gpu_anomaly", "ecc_error"):
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
                from incident_engine import IncidentEngine as _IE
                _node = (metadata or {}).get("node_id", "unknown")
                _gpu  = (metadata or {}).get("gpu_uuid", "")
                _IE().handle_alert(event_type, node_id=_node, gpu_uuid=_gpu,
                                   detail=metadata or {})
            except Exception as _ie_err:
                pass  # Never let incident engine crash alert delivery'''

if NEW.strip() in content:
    print("✓ Already wired — alert_manager → incident_engine")
elif OLD in content:
    # Backup
    import shutil
    from datetime import datetime
    bak = Path(str(ALERT_MGR) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(ALERT_MGR, bak)
    print(f"✓ Backed up → {bak.name}")
    ALERT_MGR.write_text(content.replace(OLD, NEW, 1))
    print("✓ Wired alert_manager.fire() → incident_engine.handle_alert()")
    print()
    print("  Restart the API to activate:")
    print("  ./hexagrid.sh restart")
else:
    print("✗ Could not find anchor — check alert_manager.py manually")
    print("  Looking for the conn.commit() block after INSERT INTO alert_history")
