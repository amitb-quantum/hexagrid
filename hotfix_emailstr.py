#!/usr/bin/env python3
"""
HexaGrid Auth — Hotfix: relax email validation in LoginRequest
The email-validator library rejects .local domains used in dev.
Switching LoginRequest.email from EmailStr to str fixes this.

Run from ~/hexagrid/api/:  python hotfix_emailstr.py
"""
from pathlib import Path

AUTH_ROUTES = Path.home() / "hexagrid" / "api" / "auth_routes.py"
content = AUTH_ROUTES.read_text()

fixes = [
    # Remove EmailStr from the pydantic import line
    (
        "from pydantic import BaseModel, Field, EmailStr",
        "from pydantic import BaseModel, Field"
    ),
    # Change LoginRequest.email field type
    (
        "    email:    EmailStr\n    password: str = Field(..., min_length=1)",
        "    email:    str\n    password: str = Field(..., min_length=1)"
    ),
    # Change CreateUserRequest.email field type
    (
        "    email:        EmailStr\n    role:         str",
        "    email:        str\n    role:         str"
    ),
]

changed = 0
for old, new in fixes:
    if old in content:
        content = content.replace(old, new, 1)
        changed += 1
    elif new in content:
        changed += 1  # already applied

AUTH_ROUTES.write_text(content)
print(f"✓ Fixed {changed}/3 — EmailStr replaced with str in auth_routes.py")
print("  Restart the API to apply:")
print("  ./hexagrid.sh restart")
