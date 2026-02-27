#!/usr/bin/env python3
"""
HexaGrid Auth — Hotfix: add POST /api/v1/auth/me/password endpoint
Allows authenticated users to change their own password.

Run from ~/hexagrid/api/:  python hotfix_password_change.py
"""
from pathlib import Path

USERS_PY      = Path.home() / "hexagrid" / "api" / "auth" / "users.py"
AUTH_ROUTES   = Path.home() / "hexagrid" / "api" / "auth_routes.py"

# ── 1. Add change_password() to users.py ─────────────────────────────────────

USERS_NEW_FN = '''
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
    return True
'''

users_content = USERS_PY.read_text()

if "def change_password" in users_content:
    print("✓ change_password() already in users.py")
else:
    # Append before the final newline
    USERS_PY.write_text(users_content.rstrip() + "\n" + USERS_NEW_FN)
    print("✓ Added change_password() to auth/users.py")


# ── 2. Add the endpoint to auth_routes.py ────────────────────────────────────

routes_content = AUTH_ROUTES.read_text()

# Add ChangePasswordRequest model after the other models
MODEL_ANCHOR = "class CreateAPIKeyRequest(BaseModel):"
NEW_MODEL = '''\
class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=8,
                                  description="Minimum 8 characters")

    model_config = {"json_schema_extra": {
        "example": {"current_password": "old-password", "new_password": "new-password-123"}
    }}


'''

# Add the endpoint after the /me endpoint
ME_ENDPOINT_ANCHOR = '''\
@router.get("/me", summary="Current user profile")
async def me(user: CurrentUser = Depends(get_current_user)):
    """Return the authenticated user's profile and role."""
    return user.to_dict()'''

NEW_ENDPOINT = '''\
@router.get("/me", summary="Current user profile")
async def me(user: CurrentUser = Depends(get_current_user)):
    """Return the authenticated user's profile and role."""
    return user.to_dict()


@router.post("/me/password", summary="Change own password")
async def change_password_endpoint(
    req: ChangePasswordRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Change the authenticated user's password.
    Requires the current password for verification.
    SSO-only accounts (no local password) cannot use this endpoint.
    """
    from auth.users import change_password
    ok = change_password(user.user_id, req.current_password, req.new_password)
    if not ok:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = {"error": "WRONG_PASSWORD",
                           "message": "Current password is incorrect, or this account "
                                      "uses SSO and has no local password."},
        )
    audit_log(user, "change_password", "self")
    return {"status": "ok", "message": "Password updated successfully."}'''

changed = 0

if "class ChangePasswordRequest" not in routes_content:
    if MODEL_ANCHOR in routes_content:
        routes_content = routes_content.replace(MODEL_ANCHOR, NEW_MODEL + MODEL_ANCHOR, 1)
        changed += 1
        print("✓ Added ChangePasswordRequest model to auth_routes.py")
    else:
        print("⚠  Could not find model anchor — add ChangePasswordRequest manually")
else:
    print("✓ ChangePasswordRequest already in auth_routes.py")
    changed += 1

if "/me/password" not in routes_content:
    if ME_ENDPOINT_ANCHOR in routes_content:
        routes_content = routes_content.replace(ME_ENDPOINT_ANCHOR, NEW_ENDPOINT, 1)
        changed += 1
        print("✓ Added POST /me/password endpoint to auth_routes.py")
    else:
        print("⚠  Could not find /me endpoint anchor — add password endpoint manually")
else:
    print("✓ /me/password endpoint already in auth_routes.py")
    changed += 1

AUTH_ROUTES.write_text(routes_content)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if changed >= 2:
    print("✓ Password change endpoint ready")
    print()
    print("  Restart the API then test with:")
    print("  curl -s -X POST http://localhost:8000/api/v1/auth/me/password \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -H 'Authorization: Bearer <your_token>' \\")
    print("    -d '{\"current_password\":\"old\",\"new_password\":\"new-password-123\"}'")
    print()
    print("  ./hexagrid.sh restart")
else:
    print("⚠  Some changes may need to be applied manually — check output above")
