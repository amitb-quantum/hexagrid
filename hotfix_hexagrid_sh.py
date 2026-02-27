#!/usr/bin/env python3
"""
HexaGrid — Hotfix: auto-source .env.auth in hexagrid.sh on every start/restart
Run from ~/hexagrid/:  python hotfix_hexagrid_sh.py
"""
from pathlib import Path

HEXAGRID_SH = Path.home() / "hexagrid" / "hexagrid.sh"
content = HEXAGRID_SH.read_text()

OLD = '''    cd "$APP_DIR" || { ERR "Cannot cd to $APP_DIR"; return 1; }'''

NEW = '''    # ── Source auth environment variables ─────────────────────────────────────
    local env_auth="$HOME/hexagrid/.env.auth"
    if [[ -f "$env_auth" ]]; then
        # shellcheck source=/dev/null
        source "$env_auth"
        INFO "Auth env: $env_auth"
    else
        WARN ".env.auth not found at $env_auth — JWT auth may not work"
        WARN "Run setup_credentials.py to create it"
    fi

    cd "$APP_DIR" || { ERR "Cannot cd to $APP_DIR"; return 1; }'''

if NEW.strip() in content:
    print("✓ Already patched — .env.auth is already sourced in hexagrid.sh")
elif OLD in content:
    HEXAGRID_SH.write_text(content.replace(OLD, NEW, 1))
    print("✓ Patched hexagrid.sh — .env.auth will be sourced automatically on every start/restart")
    print()
    print("  Test it:")
    print("  ./hexagrid.sh restart")
    print()
    print("  You should see:")
    print("  → Auth env: /home/manager/hexagrid/.env.auth")
else:
    print("✗ Could not find anchor line — check hexagrid.sh manually")
    print(f"  Looking for: {OLD.strip()}")
