#!/usr/bin/env python3
"""
HexaGrid Auth — Fill in .env.auth credentials
Run from anywhere:  python setup_credentials.py
"""
import secrets
import getpass
from pathlib import Path

ENV_AUTH = Path.home() / "hexagrid" / ".env.auth"

if not ENV_AUTH.exists():
    print(f"✗  {ENV_AUTH} not found — run apply_auth_patch.py first")
    raise SystemExit(1)

print("\n  ⚡ HexaGrid — Credential Setup")
print("  " + "─" * 40)

# ── Generate JWT secret automatically ────────────────────────────────────────
jwt_secret = secrets.token_hex(32)
print(f"\n  ✓  JWT secret generated (64-char hex)")

# ── Prompt for email ──────────────────────────────────────────────────────────
print()
email = input("  Admin email [admin@hexagrid.local]: ").strip()
if not email:
    email = "admin@hexagrid.local"

# ── Prompt for password (hidden input, confirmed) ─────────────────────────────
while True:
    password = getpass.getpass("  Admin password: ")
    if len(password) < 8:
        print("  ✗  Password must be at least 8 characters")
        continue
    confirm = getpass.getpass("  Confirm password: ")
    if password != confirm:
        print("  ✗  Passwords do not match — try again")
        continue
    break

# ── Write into .env.auth ──────────────────────────────────────────────────────
content = ENV_AUTH.read_text()

replacements = {
    "export HEXAGRID_JWT_SECRET=REPLACE_WITH_GENERATED_SECRET":
        f"export HEXAGRID_JWT_SECRET={jwt_secret}",
    "export HEXAGRID_ADMIN_EMAIL=admin@yourcompany.com":
        f"export HEXAGRID_ADMIN_EMAIL={email}",
    "export HEXAGRID_ADMIN_PASSWORD=REPLACE_WITH_STRONG_PASSWORD":
        f"export HEXAGRID_ADMIN_PASSWORD={password}",
}

changed = 0
for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new, 1)
        changed += 1
    elif new in content:
        changed += 1  # already set

ENV_AUTH.write_text(content)

print()
print("  " + "─" * 40)
if changed == 3:
    print(f"  ✓  Written to {ENV_AUTH}")
    print(f"     Email:   {email}")
    print(f"     Secret:  {jwt_secret[:8]}...  (64 chars)")
    print(f"     Password: {'*' * len(password)}")
else:
    print(f"  ⚠  Only {changed}/3 values updated — check .env.auth manually")

print()
print("  Next steps:")
print(f"    source ~/hexagrid/.env.auth")
print(f"    ./hexagrid.sh start")
print()
