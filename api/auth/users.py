"""
HexaGrid Auth — Local User Store
SQLite-backed local accounts for dev + emergency fallback.
Production users authenticate via SSO (OIDC/SAML).

Env vars:
  HEXAGRID_ADMIN_EMAIL     Initial superadmin email (default: admin@hexagrid.local)
  HEXAGRID_ADMIN_PASSWORD  Initial superadmin password (required for bootstrap)
"""
import os, uuid, logging
from datetime import datetime, timezone
from typing import Optional
from auth.api_keys import _conn
from auth.jwt_handler import hash_password, verify_password, issue_token_pair
from auth.rbac import VALID_ROLES

log = logging.getLogger("hexagrid.auth.users")

_CREATE_USERS="""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, display_name TEXT,
    password_hash TEXT, role TEXT NOT NULL, tenant TEXT NOT NULL DEFAULT 'default',
    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
    last_login TEXT, sso_provider TEXT, sso_sub TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_email  ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant);
"""

def _ensure_schema():
    with _conn() as c: c.executescript(_CREATE_USERS); c.commit()

class UserRecord:
    __slots__=("id","email","display_name","role","tenant","active",
               "created_at","last_login","sso_provider","sso_sub","_has_password")
    def __init__(self,row):
        self.id=row["id"]; self.email=row["email"]; self.display_name=row["display_name"] or row["email"]
        self.role=row["role"]; self.tenant=row["tenant"]; self.active=bool(row["active"])
        self.created_at=row["created_at"]; self.last_login=row["last_login"]
        self.sso_provider=row["sso_provider"]; self.sso_sub=row["sso_sub"]
        self._has_password=bool(row["password_hash"])
    def to_dict(self):
        return {"id":self.id,"email":self.email,"display_name":self.display_name,"role":self.role,
                "tenant":self.tenant,"active":self.active,"created_at":self.created_at,
                "last_login":self.last_login,"sso_provider":self.sso_provider,"has_password":self._has_password}

def create_user(email,role,tenant="default",display_name=None,password=None,
                sso_provider=None,sso_sub=None,created_by="system"):
    _ensure_schema()
    if role not in VALID_ROLES: raise ValueError(f"Invalid role '{role}'")
    email=email.lower().strip(); uid=str(uuid.uuid4()); now=datetime.now(timezone.utc).isoformat()
    pw=hash_password(password) if password else None
    try:
        with _conn() as c:
            c.execute("INSERT INTO users(id,email,display_name,password_hash,role,tenant,active,created_at,sso_provider,sso_sub) VALUES(?,?,?,?,?,?,1,?,?,?)",
                      (uid,email,display_name,pw,role,tenant,now,sso_provider,sso_sub)); c.commit()
    except Exception as e:
        if "UNIQUE" in str(e): raise ValueError(f"User already exists: {email}")
        raise
    return get_user_by_email(email)

def get_user_by_email(email:str) -> Optional[UserRecord]:
    _ensure_schema()
    with _conn() as c:
        row=c.execute("SELECT * FROM users WHERE email=? AND active=1",(email.lower().strip(),)).fetchone()
    return UserRecord(row) if row else None

def authenticate_local(email:str, password:str) -> Optional[dict]:
    _ensure_schema(); email=email.lower().strip()
    with _conn() as c:
        row=c.execute("SELECT * FROM users WHERE email=? AND active=1",(email,)).fetchone()
    if not row: return None
    if not row["password_hash"]: return None
    if not verify_password(password,row["password_hash"]): return None
    now=datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE users SET last_login=? WHERE email=?",(now,email)); c.commit()
    return issue_token_pair(row["email"],row["role"],row["tenant"])

def update_user_role(user_id:str, new_role:str, updated_by:str="system") -> bool:
    if new_role not in VALID_ROLES: raise ValueError(f"Invalid role '{new_role}'")
    _ensure_schema()
    with _conn() as c:
        r=c.execute("UPDATE users SET role=? WHERE id=? AND active=1",(new_role,user_id)); c.commit()
        return r.rowcount>0

def deactivate_user(user_id:str, deactivated_by:str="system") -> bool:
    _ensure_schema()
    with _conn() as c:
        r=c.execute("UPDATE users SET active=0 WHERE id=? AND active=1",(user_id,)); c.commit()
        return r.rowcount>0

def list_users(tenant=None, include_inactive=False) -> list:
    _ensure_schema(); q="SELECT * FROM users"; p=[]; w=[]
    if tenant: w.append("tenant=?"); p.append(tenant)
    if not include_inactive: w.append("active=1")
    if w: q+=" WHERE "+" AND ".join(w)
    q+=" ORDER BY created_at DESC"
    with _conn() as c: rows=c.execute(q,p).fetchall()
    return [UserRecord(r).to_dict() for r in rows]

def bootstrap_superadmin() -> Optional[dict]:
    _ensure_schema()
    with _conn() as c:
        count=c.execute("SELECT COUNT(*) FROM users WHERE active=1").fetchone()[0]
    if count>0: return None
    email=os.environ.get("HEXAGRID_ADMIN_EMAIL","admin@hexagrid.local")
    password=os.environ.get("HEXAGRID_ADMIN_PASSWORD","")
    if not password:
        log.warning("No users exist and HEXAGRID_ADMIN_PASSWORD not set. Set this env var to bootstrap superadmin.")
        return None
    user=create_user(email=email,role="superadmin",tenant="default",
                     display_name="HexaGrid Admin",password=password,created_by="bootstrap")
    print("\n"+"═"*60)
    print("  ⚡ HexaGrid — Superadmin Account Created")
    print(f"  Email: {email}  |  Role: superadmin")
    print("  Change this password after first login!")
    print("═"*60+"\n")
    return user.to_dict() if user else None

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
    return True
