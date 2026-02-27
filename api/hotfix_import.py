#!/usr/bin/env python3
"""
HexaGrid Auth — Hotfix: remove unused get_user_by_id import from auth_routes.py
Run from ~/hexagrid/api/:  python hotfix_import.py
"""
from pathlib import Path

AUTH_ROUTES = Path.home() / "hexagrid" / "api" / "auth_routes.py"

content = AUTH_ROUTES.read_text()

OLD = """from auth.users import (
    authenticate_local, create_user, get_user_by_id,
    list_users, update_user_role, deactivate_user, bootstrap_superadmin
)"""

NEW = """from auth.users import (
    authenticate_local, create_user,
    list_users, update_user_role, deactivate_user, bootstrap_superadmin
)"""

if NEW in content:
    print("✓ Already fixed")
elif OLD in content:
    AUTH_ROUTES.write_text(content.replace(OLD, NEW, 1))
    print("✓ Fixed — removed unused get_user_by_id from auth_routes.py imports")
else:
    print("✗ Could not find import block — check auth_routes.py manually")
    print(f"  Looking for: get_user_by_id")
