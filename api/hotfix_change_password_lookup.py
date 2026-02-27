#!/usr/bin/env python3
"""
HexaGrid Auth — Hotfix: change_password() looks up user by email not UUID.
The JWT sub claim contains the email, not the UUID, so the lookup was failing.

Run from ~/hexagrid/api/:  python hotfix_change_password_lookup.py
"""
from pathlib import Path

USERS_PY = Path.home() / "hexagrid" / "api" / "auth" / "users.py"
content = USERS_PY.read_text()

OLD = '''\
def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """Verify current password then store new bcrypt hash. Returns False if current password wrong."""
    _ensure_schema()
    with _conn() as c:
        row = c.execute(
            "SELECT password_hash FROM users WHERE id=? AND active=1", (user_id,)
        ).fetchone()
    if not row or not row["password_hash"]:
        return False
    if not verify_password(current_password, row["password_hash"]):
        return False
    new_hash = hash_password(new_password)
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id)
        )
        c.commit()
    return True'''

NEW = '''\
def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """Verify current password then store new bcrypt hash. Returns False if current password wrong.
    user_id may be either a UUID or an email address (JWT sub claim contains email)."""
    _ensure_schema()
    with _conn() as c:
        # Try email first (JWT sub = email), fall back to UUID
        row = c.execute(
            "SELECT id, password_hash FROM users WHERE (email=? OR id=?) AND active=1",
            (user_id, user_id)
        ).fetchone()
    if not row or not row["password_hash"]:
        return False
    if not verify_password(current_password, row["password_hash"]):
        return False
    new_hash = hash_password(new_password)
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (new_hash, row["id"])
        )
        c.commit()
    return True'''

if NEW in content:
    print("✓ Already fixed")
elif OLD in content:
    USERS_PY.write_text(content.replace(OLD, NEW, 1))
    print("✓ Fixed — change_password() now looks up by email OR UUID")
    print()
    print("  No restart needed — retry your curl immediately:")
    print("  curl -s -X POST http://localhost:8000/api/v1/auth/me/password \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -H 'Authorization: Bearer <token>' \\")
    print("    -d '{\"current_password\":\"gpuadmin\",\"new_password\":\"yournewpassword\"}'")
else:
    print("✗ Could not find change_password() — check auth/users.py manually")
