#!/usr/bin/env python3
"""
HexaGrid Collector — Hotfix: send API key as X-API-Key instead of Authorization: Bearer
API keys and JWTs are different credential classes and should use different headers.

Run from ~/hexagrid/deploy/:  python hotfix_collector_header.py
"""
from pathlib import Path

COLLECTOR = Path.home() / "hexagrid" / "deploy" / "collector_agent.py"
content = COLLECTOR.read_text()

OLD = '''\
def _build_headers() -> dict:
    return {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type":  "application/json",
        "X-HexaGrid-Node": NODE_ID,
        "X-HexaGrid-Cluster": CLUSTER_ID,
    }'''

NEW = '''\
def _build_headers() -> dict:
    return {
        "X-API-Key":       AUTH_TOKEN,
        "Content-Type":    "application/json",
        "X-HexaGrid-Node": NODE_ID,
        "X-HexaGrid-Cluster": CLUSTER_ID,
    }'''

if NEW in content:
    print("✓ Already fixed — collector is already using X-API-Key")
elif OLD in content:
    COLLECTOR.write_text(content.replace(OLD, NEW, 1))
    print("✓ Fixed — collector now sends X-API-Key instead of Authorization: Bearer")
    print()
    print("  Make sure HEXAGRID_TOKEN is set in the collector's environment:")
    print("  export HEXAGRID_TOKEN=hg_RriAS1rrWE90Ab4RtaTMiDBcIqOOm6gey4hVBneo89s")
else:
    print("✗ Could not find _build_headers block — check collector_agent.py manually")
    print("  Change:  \"Authorization\": f\"Bearer {AUTH_TOKEN}\"")
    print("  To:      \"X-API-Key\": AUTH_TOKEN")
