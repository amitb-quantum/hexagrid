#!/usr/bin/env python3
"""
HexaGrid RBAC + SSO — Automated Deployment Script
===================================================
Run this ONCE from your WSL2 terminal inside the hexagrid conda env:

    conda activate hexagrid
    python3 apply_auth_patch.py

What it does:
  1. Prints the exact directory layout of every file being deployed
  2. Creates ~/hexagrid/api/auth/ with all 6 auth modules
  3. Creates ~/hexagrid/api/auth_routes.py
  4. Patches ~/hexagrid/api/api.py  (6 surgical insertions)
  5. Patches ~/hexagrid/dashboard/index.html  (3 surgical insertions)
  6. Installs Python dependencies into the hexagrid conda env
  7. Writes a .env.auth template to ~/hexagrid/.env.auth

All existing files are backed up before modification.
Run with --dry-run to preview changes without writing anything.
"""

import os, sys, shutil, textwrap, argparse, subprocess
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

HEXAGRID_ROOT = Path.home() / "hexagrid"
API_DIR       = HEXAGRID_ROOT / "api"
AUTH_DIR      = API_DIR / "auth"
DASHBOARD_DIR = HEXAGRID_ROOT / "dashboard"
API_PY        = API_DIR / "api.py"
INDEX_HTML    = DASHBOARD_DIR / "index.html"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg):
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")

def step(msg):
    print(f"\n  ➤  {msg}")

def ok(msg):
    print(f"     ✓  {msg}")

def warn(msg):
    print(f"     ⚠  {msg}")

def fail(msg):
    print(f"     ✗  {msg}")
    sys.exit(1)

def backup(path: Path):
    bak = Path(str(path) + BACKUP_SUFFIX)
    shutil.copy2(path, bak)
    ok(f"Backed up → {bak.name}")
    return bak

def write_file(path: Path, content: str, dry_run: bool):
    if dry_run:
        print(f"     [DRY-RUN] Would write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    ok(f"Written: {path.relative_to(HEXAGRID_ROOT)}")

def patch_file(path: Path, old: str, new: str, description: str, dry_run: bool):
    content = path.read_text(encoding="utf-8")
    if old not in content:
        warn(f"Patch anchor not found — skipping: {description}")
        return False
    if new.strip() in content:
        ok(f"Already patched — skipping: {description}")
        return True
    patched = content.replace(old, new, 1)
    if dry_run:
        print(f"     [DRY-RUN] Would patch: {description}")
        return True
    path.write_text(patched, encoding="utf-8")
    ok(f"Patched: {description}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Print directory layout
# ══════════════════════════════════════════════════════════════════════════════

def print_layout():
    banner("FILE DEPLOYMENT LAYOUT")
    print("""
  ~/hexagrid/                          ← HexaGrid root
  │
  ├── api/                             ← FastAPI backend
  │   ├── api.py                       ← PATCHED (6 changes)
  │   │
  │   ├── auth/                        ← NEW directory
  │   │   ├── __init__.py              ← Package exports
  │   │   ├── jwt_handler.py           ← JWT issue/verify/refresh
  │   │   ├── api_keys.py              ← Service account API keys (SQLite)
  │   │   ├── rbac.py                  ← Roles, permissions, FastAPI Depends()
  │   │   ├── oidc.py                  ← Okta / Azure AD / Google (OIDC)
  │   │   ├── saml.py                  ← Generic SAML 2.0 fallback
  │   │   └── users.py                 ← Local user store + bootstrap
  │   │
  │   └── auth_routes.py               ← NEW: all /api/v1/auth/* endpoints
  │                                       Includes built-in login page HTML
  │
  ├── dashboard/
  │   └── index.html                   ← PATCHED (3 changes)
  │                                       + apiFetch() token injection
  │                                       + user chip + logout in header
  │                                       + _initAuth() on load
  │
  └── .env.auth                        ← NEW: env var template (fill in + source)

  /var/lib/hexagrid/
  ├── telemetry.db                     ← UNCHANGED (existing)
  └── auth.db                          ← NEW: users, API keys, audit log
""")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Auth module file contents
# ══════════════════════════════════════════════════════════════════════════════

AUTH_INIT = '''\
# HexaGrid Auth Package
from auth.rbac import get_current_user, require_role, require_any_role, CurrentUser, VALID_ROLES
from auth.jwt_handler import issue_token_pair, verify_token, InvalidTokenError, ExpiredTokenError
from auth.api_keys import create_api_key, verify_api_key, revoke_api_key, bootstrap_collector_key
from auth.users import authenticate_local, create_user, bootstrap_superadmin

__all__ = [
    "get_current_user", "require_role", "require_any_role", "CurrentUser", "VALID_ROLES",
    "issue_token_pair", "verify_token", "InvalidTokenError", "ExpiredTokenError",
    "create_api_key", "verify_api_key", "revoke_api_key", "bootstrap_collector_key",
    "authenticate_local", "create_user", "bootstrap_superadmin",
]
'''

AUTH_JWT = '''\
"""
HexaGrid Auth — JWT Handler
Issues, verifies, and refreshes JWT access + refresh tokens.

Access token:  15 min, stateless, carries user + role claims
Refresh token: 7 days, HttpOnly cookie, rotated on use

Env vars:
  HEXAGRID_JWT_SECRET       HS256 signing secret (min 32 chars, required)
  HEXAGRID_JWT_ALGORITHM    default HS256
  HEXAGRID_ACCESS_TTL_MIN   access token lifetime minutes  (default 15)
  HEXAGRID_REFRESH_TTL_DAYS refresh token lifetime days    (default 7)
  HEXAGRID_ISSUER           token issuer claim             (default hexagrid)
"""
import os, uuid, logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

log = logging.getLogger("hexagrid.auth.jwt")

JWT_SECRET    = os.environ.get("HEXAGRID_JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("HEXAGRID_JWT_ALGORITHM", "HS256")
ACCESS_TTL    = int(os.environ.get("HEXAGRID_ACCESS_TTL_MIN",   "15"))
REFRESH_TTL   = int(os.environ.get("HEXAGRID_REFRESH_TTL_DAYS", "7"))
ISSUER        = os.environ.get("HEXAGRID_ISSUER", "hexagrid")

_SECRET_MISSING = not JWT_SECRET or len(JWT_SECRET) < 32
if _SECRET_MISSING:
    log.warning("HEXAGRID_JWT_SECRET is not set or < 32 chars. Token issuance will be blocked.")

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)

class TokenPayload:
    __slots__ = ("sub","role","tenant","jti","exp","iss","token_type")
    def __init__(self,sub,role,tenant,jti,exp,iss,token_type):
        self.sub=sub; self.role=role; self.tenant=tenant; self.jti=jti
        self.exp=exp; self.iss=iss; self.token_type=token_type
    def is_expired(self): return datetime.now(timezone.utc) >= self.exp

class InvalidTokenError(Exception): pass
class ExpiredTokenError(InvalidTokenError): pass

def _assert_secret():
    if _SECRET_MISSING:
        raise RuntimeError("HEXAGRID_JWT_SECRET not set or < 32 chars.")

def issue_access_token(user_id:str, role:str, tenant:str="default", extra_claims:Optional[dict]=None) -> str:
    _assert_secret()
    from auth.rbac import VALID_ROLES
    if role not in VALID_ROLES: raise ValueError(f"Unknown role \'{role}\'")
    now = datetime.now(timezone.utc)
    payload = {"sub":user_id,"role":role,"tenant":tenant,"jti":str(uuid.uuid4()),
               "iat":now,"exp":now+timedelta(minutes=ACCESS_TTL),"iss":ISSUER,"token_type":"access"}
    if extra_claims: payload.update(extra_claims)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def issue_refresh_token(user_id:str, role:str, tenant:str="default") -> str:
    _assert_secret()
    now = datetime.now(timezone.utc)
    payload = {"sub":user_id,"role":role,"tenant":tenant,"jti":str(uuid.uuid4()),
               "iat":now,"exp":now+timedelta(days=REFRESH_TTL),"iss":ISSUER,"token_type":"refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def issue_token_pair(user_id:str, role:str, tenant:str="default") -> dict:
    return {"access_token":issue_access_token(user_id,role,tenant),
            "refresh_token":issue_refresh_token(user_id,role,tenant),
            "token_type":"bearer","expires_in":ACCESS_TTL*60,"role":role,"tenant":tenant}

def verify_token(token:str, expected_type:str="access") -> TokenPayload:
    if not token: raise InvalidTokenError("No token provided")
    try:
        raw = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp":True})
    except jwt.ExpiredSignatureError:
        raise ExpiredTokenError("Token has expired")
    except JWTError as e:
        raise InvalidTokenError(f"Token validation failed: {e}")
    if raw.get("token_type") != expected_type:
        raise InvalidTokenError(f"Wrong token type. Expected \'{expected_type}\', got \'{raw.get(\'token_type\',\'unknown\')}\'")
    if raw.get("iss") != ISSUER:
        raise InvalidTokenError(f"Unknown issuer: {raw.get(\'iss\')}")
    return TokenPayload(raw["sub"],raw["role"],raw.get("tenant","default"),raw["jti"],
                        datetime.fromtimestamp(raw["exp"],tz=timezone.utc),raw["iss"],raw["token_type"])

def rotate_refresh_token(refresh_token:str) -> dict:
    payload = verify_token(refresh_token, expected_type="refresh")
    return issue_token_pair(payload.sub, payload.role, payload.tenant)
'''

AUTH_APIKEYS = '''\
"""
HexaGrid Auth — API Key Store
Service account keys for collector agents, k8s operator, etc.
Keys are SHA-256 hashed in SQLite — never stored in plaintext.
Raw key shown ONCE on creation.

Env vars:
  HEXAGRID_AUTH_DB   path to auth SQLite db (default /var/lib/hexagrid/auth.db)
"""
import os, uuid, json, hashlib, sqlite3, secrets, logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("hexagrid.auth.apikeys")

_AUTH_DB = os.environ.get("HEXAGRID_AUTH_DB", "/var/lib/hexagrid/auth.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    key_hash     TEXT NOT NULL UNIQUE,
    key_prefix   TEXT NOT NULL,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL,
    tenant       TEXT NOT NULL DEFAULT \'default\',
    scopes       TEXT NOT NULL DEFAULT \'[]\',
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    last_used_at TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0,
    created_by   TEXT NOT NULL DEFAULT \'system\'
);
CREATE INDEX IF NOT EXISTS idx_apk_hash    ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_apk_tenant  ON api_keys(tenant);
CREATE INDEX IF NOT EXISTS idx_apk_revoked ON api_keys(revoked);
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, user_id TEXT NOT NULL,
    role TEXT NOT NULL, tenant TEXT NOT NULL, action TEXT NOT NULL,
    resource TEXT NOT NULL, detail TEXT, auth_method TEXT
);
"""

def _conn() -> sqlite3.Connection:
    db_dir = os.path.dirname(_AUTH_DB)
    if db_dir: os.makedirs(db_dir, exist_ok=True)
    c = sqlite3.connect(_AUTH_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c

_KEY_PREFIX = "hg_"

def _generate_raw_key() -> str:
    return f"{_KEY_PREFIX}{secrets.token_urlsafe(32)}"

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()

class APIKeyRecord:
    __slots__ = ("id","key_prefix","name","role","tenant","scopes",
                 "created_at","expires_at","last_used_at","revoked","created_by")
    def __init__(self, row):
        self.id=row["id"]; self.key_prefix=row["key_prefix"]; self.name=row["name"]
        self.role=row["role"]; self.tenant=row["tenant"]; self.scopes=json.loads(row["scopes"])
        self.created_at=row["created_at"]; self.expires_at=row["expires_at"]
        self.last_used_at=row["last_used_at"]; self.revoked=bool(row["revoked"])
        self.created_by=row["created_by"]
    def is_expired(self):
        if not self.expires_at: return False
        return datetime.now(timezone.utc) >= datetime.fromisoformat(self.expires_at).replace(tzinfo=timezone.utc)
    def is_valid(self): return not self.revoked and not self.is_expired()
    def allows_path(self, path):
        if not self.scopes: return True
        return any(path.startswith(s) for s in self.scopes)
    def to_dict(self):
        return {"id":self.id,"key_prefix":self.key_prefix,"name":self.name,"role":self.role,
                "tenant":self.tenant,"scopes":self.scopes,"created_at":self.created_at,
                "expires_at":self.expires_at,"last_used_at":self.last_used_at,
                "revoked":self.revoked,"created_by":self.created_by}

def create_api_key(name,role,tenant="default",scopes=None,expires_at=None,created_by="system") -> dict:
    from auth.rbac import VALID_ROLES
    if role not in VALID_ROLES: raise ValueError(f"Invalid role \'{role}\'")
    raw=_generate_raw_key(); kh=_hash_key(raw); kp=raw[:12]+"..."; kid=str(uuid.uuid4())
    now=datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("INSERT INTO api_keys(id,key_hash,key_prefix,name,role,tenant,scopes,created_at,expires_at,revoked,created_by) VALUES(?,?,?,?,?,?,?,?,?,0,?)",
                  (kid,kh,kp,name,role,tenant,json.dumps(scopes or []),now,expires_at,created_by))
        c.commit()
    log.info("API key created: id=%s name=%s role=%s", kid, name, role)
    return {"id":kid,"raw_key":raw,"key_prefix":kp,"name":name,"role":role,"tenant":tenant,
            "scopes":scopes or [],"created_at":now,"expires_at":expires_at,
            "warning":"Save the raw_key now — it cannot be retrieved again."}

def verify_api_key(raw_key:str, request_path:str="/") -> Optional[APIKeyRecord]:
    if not raw_key or not raw_key.startswith(_KEY_PREFIX): return None
    kh = _hash_key(raw_key)
    with _conn() as c:
        row = c.execute("SELECT * FROM api_keys WHERE key_hash=?",(kh,)).fetchone()
    if not row: return None
    rec = APIKeyRecord(row)
    if not rec.is_valid(): return None
    if not rec.allows_path(request_path): return None
    try:
        now=datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            c.execute("UPDATE api_keys SET last_used_at=? WHERE id=?",(now,rec.id)); c.commit()
    except Exception: pass
    return rec

def revoke_api_key(key_id:str, revoked_by:str="system") -> bool:
    with _conn() as c:
        r=c.execute("UPDATE api_keys SET revoked=1 WHERE id=? AND revoked=0",(key_id,)); c.commit()
        return r.rowcount > 0

def list_api_keys(tenant=None, include_revoked=False) -> list:
    q="SELECT * FROM api_keys"; p=[]; w=[]
    if tenant: w.append("tenant=?"); p.append(tenant)
    if not include_revoked: w.append("revoked=0")
    if w: q+=" WHERE "+" AND ".join(w)
    q+=" ORDER BY created_at DESC"
    with _conn() as c: rows=c.execute(q,p).fetchall()
    return [APIKeyRecord(r).to_dict() for r in rows]

def get_api_key(key_id:str) -> Optional[dict]:
    with _conn() as c:
        row=c.execute("SELECT * FROM api_keys WHERE id=?",(key_id,)).fetchone()
    return APIKeyRecord(row).to_dict() if row else None

def bootstrap_collector_key() -> Optional[dict]:
    with _conn() as c:
        count=c.execute("SELECT COUNT(*) FROM api_keys WHERE revoked=0").fetchone()[0]
    if count > 0: return None
    result=create_api_key(name="local-dev-collector",role="collector",tenant="default",
                          scopes=["/api/v1/telemetry/"],created_by="bootstrap")
    print("\\n"+"═"*60)
    print("  ⚡ HexaGrid — Bootstrap Collector API Key")
    print(f"  HEXAGRID_TOKEN={result[\'raw_key\']}")
    print("  This key will NOT be shown again.")
    print("═"*60+"\\n")
    return result
'''

AUTH_RBAC = '''\
"""
HexaGrid Auth — RBAC
Roles, permissions matrix, and FastAPI Depends() guards.

Roles (least → most privileged):
  collector, viewer, finance, operator, scheduler_admin, superadmin

Usage:
  from auth.rbac import require_role, get_current_user
  @app.post("/api/v1/schedule")
  async def schedule(user=Depends(require_role("operator"))): ...
"""
import logging
from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.jwt_handler import verify_token, InvalidTokenError, ExpiredTokenError

log = logging.getLogger("hexagrid.auth.rbac")

VALID_ROLES = {"collector","viewer","finance","operator","scheduler_admin","superadmin"}

_ROLE_RANK = {"collector":0,"viewer":1,"finance":1,"operator":2,"scheduler_admin":3,"superadmin":4}

def role_rank(role:str) -> int: return _ROLE_RANK.get(role,-1)

PERMISSION_MATRIX = {
    "telemetry_push":"collector","health":"viewer","pricefeed":"viewer",
    "carbon":"viewer","hardware_read":"viewer","fleet_read":"viewer",
    "savings_read":"finance","benchmark_read":"finance",
    "simulate":"operator","schedule":"operator","fleet_route":"operator","rl_recommend":"operator",
    "rl_train":"scheduler_admin","alerts_config":"scheduler_admin",
    "user_management":"superadmin","api_key_management":"superadmin","audit_log_read":"superadmin",
}

class CurrentUser:
    __slots__ = ("user_id","role","tenant","jti","auth_method","display_name")
    def __init__(self,user_id,role,tenant,jti,auth_method,display_name=None):
        self.user_id=user_id; self.role=role; self.tenant=tenant; self.jti=jti
        self.auth_method=auth_method; self.display_name=display_name or user_id
    def has_role(self, required:str) -> bool:
        if required=="finance": return self.role in ("finance","superadmin")
        return role_rank(self.role) >= role_rank(required)
    def to_dict(self):
        return {"user_id":self.user_id,"role":self.role,"tenant":self.tenant,
                "auth_method":self.auth_method,"display_name":self.display_name}

_bearer = HTTPBearer(auto_error=False)
_API_KEY_HEADER = "x-api-key"

def _raise_401(detail:str, code:str="UNAUTHORIZED"):
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error":code,"message":detail},headers={"WWW-Authenticate":"Bearer"})

def _raise_403(detail:str):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
        detail={"error":"FORBIDDEN","message":detail})

async def get_current_user(
    request:Request,
    credentials:Optional[HTTPAuthorizationCredentials]=Depends(_bearer)
) -> CurrentUser:
    api_key_header = request.headers.get(_API_KEY_HEADER)
    if api_key_header:
        from auth.api_keys import verify_api_key
        record = verify_api_key(api_key_header, request_path=request.url.path)
        if record:
            return CurrentUser(f"svc:{record.name}",record.role,record.tenant,
                               record.id,"api_key",record.name)
        _raise_401("Invalid or revoked API key")
    if credentials and credentials.credentials:
        try:
            payload = verify_token(credentials.credentials, expected_type="access")
            return CurrentUser(payload.sub,payload.role,payload.tenant,payload.jti,"jwt")
        except ExpiredTokenError:
            _raise_401("Access token expired — please refresh","TOKEN_EXPIRED")
        except InvalidTokenError as e:
            _raise_401(str(e))
    _raise_401("Authentication required. Provide Bearer token or X-API-Key header.")

def require_role(minimum_role:str):
    async def _guard(user:CurrentUser=Depends(get_current_user)) -> CurrentUser:
        if not user.has_role(minimum_role):
            log.warning("Access denied: user=%s role=%s required=%s",user.user_id,user.role,minimum_role)
            _raise_403(f"Role \'{user.role}\' is not permitted. Required: \'{minimum_role}\' or higher.")
        return user
    return _guard

def require_any_role(allowed_roles:list):
    role_set=set(allowed_roles)
    async def _guard(user:CurrentUser=Depends(get_current_user)) -> CurrentUser:
        if user.role not in role_set: _raise_403(f"Role \'{user.role}\' not in allowed set.")
        return user
    return _guard

def audit_log(user:CurrentUser, action:str, resource:str, detail:str=""):
    try:
        import uuid as _uuid
        from datetime import datetime as _dt, timezone as _tz
        from auth.api_keys import _conn
        now=_dt.now(_tz.utc).isoformat()
        with _conn() as c:
            c.execute("INSERT INTO audit_log(id,ts,user_id,role,tenant,action,resource,detail,auth_method) VALUES(?,?,?,?,?,?,?,?,?)",
                      (str(_uuid.uuid4()),now,user.user_id,user.role,user.tenant,action,resource,detail,user.auth_method))
            c.commit()
    except Exception as e:
        log.error("Audit log write failed: %s", e)

def apply_security_to_openapi(app):
    from fastapi.openapi.utils import get_openapi
    def custom_openapi():
        if app.openapi_schema: return app.openapi_schema
        schema=get_openapi(title=app.title,version=app.version,description=app.description,routes=app.routes)
        schema.setdefault("components",{}).setdefault("securitySchemes",{}).update({
            "BearerAuth":{"type":"http","scheme":"bearer","bearerFormat":"JWT",
                          "description":"JWT access token. Obtain via POST /api/v1/auth/login"},
            "ApiKeyAuth":{"type":"apiKey","in":"header","name":"X-API-Key",
                          "description":"Service account API key for machine-to-machine auth."},
        })
        schema["security"]=[{"BearerAuth":[]},{"ApiKeyAuth":[]}]
        app.openapi_schema=schema
        return schema
    app.openapi=custom_openapi
'''

AUTH_OIDC = '''\
"""
HexaGrid Auth — OIDC Provider Handler
Handles OpenID Connect SSO for Okta, Azure AD, and Google Workspace.
All three are OIDC-compliant — one implementation covers all.

Env vars (per provider, replace {P} with OKTA | AZURE | GOOGLE):
  HEXAGRID_OIDC_{P}_CLIENT_ID
  HEXAGRID_OIDC_{P}_CLIENT_SECRET
  HEXAGRID_OIDC_{P}_DISCOVERY_URL
  HEXAGRID_OIDC_{P}_ROLE_CLAIM    (default: groups)
  HEXAGRID_OIDC_{P}_ROLE_MAP      (JSON: {"idp-group": "hexagrid-role"})
"""
import os, json, hashlib, secrets, logging, urllib.parse
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime, timezone
import httpx
from jose import jwt as jose_jwt, JWTError

log = logging.getLogger("hexagrid.auth.oidc")

_state_store: Dict[str, dict] = {}
_STATE_TTL_S = 600

def _clean_state_store():
    now=datetime.now(timezone.utc).timestamp()
    expired=[k for k,v in _state_store.items() if v.get("ts",0)+_STATE_TTL_S < now]
    for k in expired: del _state_store[k]

@dataclass
class OIDCProvider:
    name:str; display_name:str; client_id:str; client_secret:str; discovery_url:str
    role_claim:str="groups"; role_map:dict=field(default_factory=dict)
    default_role:str="viewer"; scopes:list=field(default_factory=lambda:["openid","email","profile"])
    _authorization_endpoint:str=field(default="",init=False,repr=False)
    _token_endpoint:str=field(default="",init=False,repr=False)
    _jwks_uri:str=field(default="",init=False,repr=False)
    _discovery_fetched:bool=field(default=False,init=False,repr=False)
    @property
    def is_configured(self): return bool(self.client_id and self.client_secret and self.discovery_url)
    async def fetch_discovery(self):
        if self._discovery_fetched: return
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(self.discovery_url); r.raise_for_status(); doc=r.json()
        self._authorization_endpoint=doc["authorization_endpoint"]
        self._token_endpoint=doc["token_endpoint"]; self._jwks_uri=doc["jwks_uri"]
        self._discovery_fetched=True
    async def get_jwks(self):
        await self.fetch_discovery()
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(self._jwks_uri); r.raise_for_status(); return r.json()

def _load_provider(name:str) -> Optional[OIDCProvider]:
    p=f"HEXAGRID_OIDC_{name.upper()}_"
    cid=os.environ.get(f"{p}CLIENT_ID",""); cs=os.environ.get(f"{p}CLIENT_SECRET","")
    du=os.environ.get(f"{p}DISCOVERY_URL","")
    if not (cid and cs and du): return None
    rc=os.environ.get(f"{p}ROLE_CLAIM","groups")
    try: rm=json.loads(os.environ.get(f"{p}ROLE_MAP","{}"))
    except: rm={}
    dn={"okta":"Okta","azure":"Microsoft Entra ID","google":"Google Workspace"}.get(name,name.title())
    return OIDCProvider(name=name,display_name=dn,client_id=cid,client_secret=cs,
                        discovery_url=du,role_claim=rc,role_map=rm)

_providers: Optional[Dict[str,OIDCProvider]] = None
def providers() -> Dict[str,OIDCProvider]:
    global _providers
    if _providers is None:
        _providers={n:p for n in("okta","azure","google") if (p:=_load_provider(n))}
    return _providers

def _pkce_pair():
    import base64
    v=secrets.token_urlsafe(64)
    c=base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    return v,c

async def build_authorize_url(provider_name:str, redirect_uri:str, tenant:str="default"):
    pvd=providers().get(provider_name)
    if not pvd: raise ValueError(f"OIDC provider \'{provider_name}\' not configured")
    await pvd.fetch_discovery()
    state=secrets.token_urlsafe(32); verifier,challenge=_pkce_pair(); _clean_state_store()
    _state_store[state]={"provider":provider_name,"code_verifier":verifier,"tenant":tenant,
                         "ts":datetime.now(timezone.utc).timestamp()}
    params={"response_type":"code","client_id":pvd.client_id,"redirect_uri":redirect_uri,
            "scope":" ".join(pvd.scopes),"state":state,"code_challenge":challenge,
            "code_challenge_method":"S256","access_type":"offline"}
    if provider_name=="azure": params["response_mode"]="query"
    return pvd._authorization_endpoint+"?"+urllib.parse.urlencode(params), state

async def handle_callback(code:str, state:str, redirect_uri:str) -> dict:
    stored=_state_store.pop(state,None)
    if not stored: raise ValueError("Invalid or expired state token — possible CSRF attempt")
    if datetime.now(timezone.utc).timestamp()-stored["ts"]>_STATE_TTL_S: raise ValueError("State expired")
    pvd=providers().get(stored["provider"])
    if not pvd: raise ValueError("Provider disappeared from config")
    async with httpx.AsyncClient(timeout=15) as c:
        r=await c.post(pvd._token_endpoint,data={"grant_type":"authorization_code","code":code,
            "redirect_uri":redirect_uri,"client_id":pvd.client_id,"client_secret":pvd.client_secret,
            "code_verifier":stored["code_verifier"]},headers={"Accept":"application/json"})
    if not r.is_success: raise ValueError(f"Token exchange failed: {r.status_code}")
    tokens=r.json(); id_token=tokens.get("id_token")
    if not id_token: raise ValueError("IdP did not return id_token")
    jwks=await pvd.get_jwks()
    try:
        claims=jose_jwt.decode(id_token,jwks,algorithms=["RS256","RS384","RS512"],
                               audience=pvd.client_id,options={"verify_at_hash":False,"leeway":30})
    except JWTError as e: raise ValueError(f"id_token verification failed: {e}")
    user_id=claims.get("email") or claims.get("sub","unknown")
    raw_groups=claims.get(pvd.role_claim,[])
    if isinstance(raw_groups,str): raw_groups=[raw_groups]
    priority=["superadmin","scheduler_admin","operator","finance","viewer"]
    mapped={pvd.role_map[g] for g in raw_groups if g in pvd.role_map}
    role=next((r for r in priority if r in mapped),pvd.default_role)
    from auth.jwt_handler import issue_token_pair
    return issue_token_pair(user_id, role, stored["tenant"])

def get_provider_list() -> list:
    icons={"okta":"🔐","azure":"🪟","google":"🔵"}
    return [{"name":n,"display_name":p.display_name,"icon":icons.get(n,"🔑"),
             "login_url":f"/api/v1/auth/sso/{n}"} for n,p in providers().items()]
'''

AUTH_SAML = '''\
"""
HexaGrid Auth — SAML 2.0 Handler
Generic SAML 2.0 for enterprise IdPs (ADFS, Ping Identity, OneLogin, etc.)

Requires:
  pip install python3-saml --break-system-packages
  sudo apt-get install xmlsec1 libxmlsec1-dev

Env vars:
  HEXAGRID_SAML_IDP_METADATA_URL   IdP metadata URL
  HEXAGRID_SAML_IDP_METADATA_XML   Or inline XML
  HEXAGRID_SAML_SP_ENTITY_ID       Our SP entity ID
  HEXAGRID_SAML_SP_ACS_URL         Our callback URL
  HEXAGRID_SAML_ROLE_ATTRIBUTE     SAML attribute for groups (default: groups)
  HEXAGRID_SAML_ROLE_MAP           JSON map of IdP groups to HexaGrid roles
  HEXAGRID_SAML_DEFAULT_ROLE       Fallback role (default: viewer)
"""
import os, json, logging
from typing import Optional

log = logging.getLogger("hexagrid.auth.saml")

_IDP_METADATA_URL = os.environ.get("HEXAGRID_SAML_IDP_METADATA_URL","")
_IDP_METADATA_XML = os.environ.get("HEXAGRID_SAML_IDP_METADATA_XML","")
_SP_ENTITY_ID     = os.environ.get("HEXAGRID_SAML_SP_ENTITY_ID","https://hexagrid.ai")
_SP_ACS_URL       = os.environ.get("HEXAGRID_SAML_SP_ACS_URL","")
_ROLE_ATTRIBUTE   = os.environ.get("HEXAGRID_SAML_ROLE_ATTRIBUTE","groups")
_DEFAULT_ROLE     = os.environ.get("HEXAGRID_SAML_DEFAULT_ROLE","viewer")
_WANT_ENCRYPTED   = os.environ.get("HEXAGRID_SAML_WANT_ENCRYPTED","false").lower()=="true"
try: _ROLE_MAP=json.loads(os.environ.get("HEXAGRID_SAML_ROLE_MAP","{}"))
except: _ROLE_MAP={}

def is_configured(): return bool(_SP_ACS_URL and (_IDP_METADATA_URL or _IDP_METADATA_XML))

def _build_settings():
    return {"strict":True,"debug":False,
            "sp":{"entityId":_SP_ENTITY_ID,
                  "assertionConsumerService":{"url":_SP_ACS_URL,
                      "binding":"urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"},
                  "NameIDFormat":"urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"},
            "idp":{},"security":{"authnRequestsSigned":False,"wantAssertionsSigned":True,
                "wantAssertionsEncrypted":_WANT_ENCRYPTED,"wantNameIdEncrypted":False,
                "wantAttributeStatement":True,"requestedAuthnContext":False}}

async def _fetch_idp_metadata():
    if _IDP_METADATA_XML: return _IDP_METADATA_XML
    if _IDP_METADATA_URL:
        import httpx
        async with httpx.AsyncClient(timeout=15) as c:
            r=await c.get(_IDP_METADATA_URL); r.raise_for_status(); return r.text
    raise ValueError("No SAML IdP metadata configured")

def _mock_request(settings):
    from urllib.parse import urlparse
    acs=urlparse(_SP_ACS_URL)
    return {"https":"on" if acs.scheme=="https" else "off","http_host":acs.netloc,
            "server_port":str(acs.port or(443 if acs.scheme=="https" else 80)),
            "script_name":acs.path,"get_data":{},"post_data":{}}

async def build_authn_request(relay_state:str=""):
    if not is_configured(): raise ValueError("SAML not configured")
    try: from onelogin.saml2.auth import OneLogin_Saml2_Auth
    except ImportError: raise ImportError("python3-saml not installed. pip install python3-saml --break-system-packages")
    from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser
    xml=await _fetch_idp_metadata()
    idp=OneLogin_Saml2_IdPMetadataParser.parse(xml)
    s=_build_settings(); s["idp"]=idp.get("idp",{})
    req=_mock_request(s); auth=OneLogin_Saml2_Auth(req,s)
    return auth.login(return_to=relay_state), auth.get_last_request_id()

async def handle_saml_response(saml_response_b64:str, relay_state:str="", tenant:str="default"):
    if not is_configured(): raise ValueError("SAML not configured")
    try: from onelogin.saml2.auth import OneLogin_Saml2_Auth
    except ImportError: raise ImportError("python3-saml not installed")
    from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser
    xml=await _fetch_idp_metadata(); idp=OneLogin_Saml2_IdPMetadataParser.parse(xml)
    s=_build_settings(); s["idp"]=idp.get("idp",{})
    req=_mock_request(s); req["post_data"]={"SAMLResponse":saml_response_b64}
    auth=OneLogin_Saml2_Auth(req,s); auth.process_response()
    errors=auth.get_errors()
    if errors: raise ValueError(f"SAML validation failed: {auth.get_last_error_reason() or errors}")
    if not auth.is_authenticated(): raise ValueError("SAML: not authenticated")
    user_id=auth.get_nameid() or "unknown"; attrs=auth.get_attributes()
    raw=attrs.get(_ROLE_ATTRIBUTE,[]); raw=[raw] if isinstance(raw,str) else raw
    priority=["superadmin","scheduler_admin","operator","finance","viewer"]
    mapped={_ROLE_MAP[g] for g in raw if g in _ROLE_MAP}
    role=next((r for r in priority if r in mapped),_DEFAULT_ROLE)
    from auth.jwt_handler import issue_token_pair
    return issue_token_pair(user_id, role, tenant)

async def get_sp_metadata():
    try: from onelogin.saml2.settings import OneLogin_Saml2_Settings
    except ImportError: return "<error>python3-saml not installed</error>"
    s=_build_settings(); s["idp"]={}
    ss=OneLogin_Saml2_Settings(s,sp_validation_only=True)
    m=ss.get_sp_metadata()
    return m.decode("utf-8") if isinstance(m,bytes) else m

def get_saml_status():
    return {"configured":is_configured(),"sp_entity_id":_SP_ENTITY_ID,"acs_url":_SP_ACS_URL,
            "login_url":"/api/v1/auth/saml/login" if is_configured() else None,
            "metadata_url":"/api/v1/auth/saml/metadata" if is_configured() else None}
'''

AUTH_USERS = '''\
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
    password_hash TEXT, role TEXT NOT NULL, tenant TEXT NOT NULL DEFAULT \'default\',
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
    if role not in VALID_ROLES: raise ValueError(f"Invalid role \'{role}\'")
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
    if new_role not in VALID_ROLES: raise ValueError(f"Invalid role \'{new_role}\'")
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
    print("\\n"+"═"*60)
    print("  ⚡ HexaGrid — Superadmin Account Created")
    print(f"  Email: {email}  |  Role: superadmin")
    print("  Change this password after first login!")
    print("═"*60+"\\n")
    return user.to_dict() if user else None
'''

AUTH_ROUTES = '"""\nHexaGrid Auth — FastAPI Router\n================================\nAll authentication endpoints mounted at /api/v1/auth/*.\n\nEndpoints:\n  POST /api/v1/auth/login                  Local email+password login\n  POST /api/v1/auth/refresh                Rotate refresh token → new access token\n  POST /api/v1/auth/logout                 Invalidate session (clear refresh cookie)\n  GET  /api/v1/auth/me                     Current user profile\n  GET  /api/v1/auth/providers              List configured SSO providers\n\n  GET  /api/v1/auth/sso/{provider}         Initiate OIDC SSO redirect\n  GET  /api/v1/auth/sso/{provider}/callback OIDC callback handler\n\n  GET  /api/v1/auth/saml/login             Initiate SAML AuthnRequest\n  POST /api/v1/auth/saml/callback          SAML ACS (Assertion Consumer Service)\n  GET  /api/v1/auth/saml/metadata          SP metadata XML for IdP configuration\n\n  GET  /api/v1/auth/users                  List users          [superadmin]\n  POST /api/v1/auth/users                  Create user         [superadmin]\n  PUT  /api/v1/auth/users/{id}/role        Change user role    [superadmin]\n  DELETE /api/v1/auth/users/{id}           Deactivate user     [superadmin]\n\n  GET  /api/v1/auth/apikeys                List API keys       [superadmin]\n  POST /api/v1/auth/apikeys                Create API key      [superadmin]\n  DELETE /api/v1/auth/apikeys/{id}         Revoke API key      [superadmin]\n\n  GET  /api/v1/auth/audit                  Audit log           [superadmin]\n\nMount in api.py:\n  from auth_routes import router as auth_router\n  app.include_router(auth_router)\n"""\n\nimport logging\nfrom typing import Optional\n\nfrom fastapi import APIRouter, HTTPException, Request, Response, Depends, status\nfrom fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse\nfrom pydantic import BaseModel, Field, EmailStr\n\nfrom auth.rbac import (\n    get_current_user, require_role, require_any_role,\n    CurrentUser, audit_log, apply_security_to_openapi\n)\nfrom auth.jwt_handler import (\n    issue_token_pair, verify_token,\n    InvalidTokenError, ExpiredTokenError,\n)\nfrom auth.api_keys import (\n    create_api_key, revoke_api_key, list_api_keys, get_api_key\n)\nfrom auth.users import (\n    authenticate_local, create_user, get_user_by_id,\n    list_users, update_user_role, deactivate_user, bootstrap_superadmin\n)\n\nlog = logging.getLogger("hexagrid.auth.routes")\n\nrouter = APIRouter(prefix="/api/v1/auth", tags=["Auth"])\n\n# Cookie name for refresh token (HttpOnly, Secure, SameSite=Lax)\n_REFRESH_COOKIE = "hg_refresh"\n_COOKIE_MAX_AGE = 7 * 24 * 3600   # 7 days in seconds\n\n\n# ── Request/Response models ────────────────────────────────────────────────────\n\nclass LoginRequest(BaseModel):\n    email:    EmailStr\n    password: str = Field(..., min_length=1)\n\n    model_config = {"json_schema_extra": {\n        "example": {"email": "admin@hexagrid.local", "password": "changeme"}\n    }}\n\n\nclass CreateUserRequest(BaseModel):\n    email:        EmailStr\n    role:         str\n    tenant:       str      = "default"\n    display_name: Optional[str] = None\n    password:     Optional[str] = None\n    sso_provider: Optional[str] = None\n\n\nclass UpdateRoleRequest(BaseModel):\n    role: str\n\n\nclass CreateAPIKeyRequest(BaseModel):\n    name:       str\n    role:       str\n    tenant:     str            = "default"\n    scopes:     Optional[list] = None\n    expires_at: Optional[str]  = None\n\n\ndef _set_refresh_cookie(response: Response, refresh_token: str) -> None:\n    """Set the HttpOnly refresh token cookie."""\n    response.set_cookie(\n        key      = _REFRESH_COOKIE,\n        value    = refresh_token,\n        max_age  = _COOKIE_MAX_AGE,\n        httponly = True,\n        secure   = True,        # HTTPS only in production\n        samesite = "lax",\n    )\n\n\ndef _clear_refresh_cookie(response: Response) -> None:\n    response.delete_cookie(_REFRESH_COOKIE)\n\n\n# ── Auth endpoints ─────────────────────────────────────────────────────────────\n\n@router.post("/login", summary="Local email + password login")\nasync def login(req: LoginRequest, response: Response):\n    """\n    Authenticate with email + password (local account).\n    Returns access token in response body.\n    Sets refresh token as HttpOnly cookie.\n    """\n    tokens = authenticate_local(req.email, req.password)\n    if not tokens:\n        raise HTTPException(\n            status_code = status.HTTP_401_UNAUTHORIZED,\n            detail      = {"error": "INVALID_CREDENTIALS",\n                           "message": "Invalid email or password"},\n            headers     = {"WWW-Authenticate": "Bearer"},\n        )\n\n    _set_refresh_cookie(response, tokens["refresh_token"])\n\n    return {\n        "access_token": tokens["access_token"],\n        "token_type":   "bearer",\n        "expires_in":   tokens["expires_in"],\n        "role":         tokens["role"],\n        "tenant":       tokens["tenant"],\n    }\n\n\n@router.post("/refresh", summary="Refresh access token using refresh cookie")\nasync def refresh_token(request: Request, response: Response):\n    """\n    Issue a new access token using the refresh token stored in the HttpOnly cookie.\n    The refresh token is rotated (old token invalidated, new cookie set).\n    """\n    refresh_tk = request.cookies.get(_REFRESH_COOKIE)\n    if not refresh_tk:\n        raise HTTPException(\n            status_code = status.HTTP_401_UNAUTHORIZED,\n            detail      = {"error": "NO_REFRESH_TOKEN",\n                           "message": "No refresh token found. Please log in again."},\n        )\n\n    try:\n        from auth.jwt_handler import rotate_refresh_token\n        tokens = rotate_refresh_token(refresh_tk)\n    except ExpiredTokenError:\n        _clear_refresh_cookie(response)\n        raise HTTPException(\n            status_code = status.HTTP_401_UNAUTHORIZED,\n            detail      = {"error": "REFRESH_EXPIRED",\n                           "message": "Session expired. Please log in again."},\n        )\n    except InvalidTokenError as e:\n        _clear_refresh_cookie(response)\n        raise HTTPException(\n            status_code = status.HTTP_401_UNAUTHORIZED,\n            detail      = {"error": "INVALID_REFRESH", "message": str(e)},\n        )\n\n    _set_refresh_cookie(response, tokens["refresh_token"])\n    return {\n        "access_token": tokens["access_token"],\n        "token_type":   "bearer",\n        "expires_in":   tokens["expires_in"],\n        "role":         tokens["role"],\n        "tenant":       tokens["tenant"],\n    }\n\n\n@router.post("/logout", summary="Invalidate session")\nasync def logout(\n    response: Response,\n    user: CurrentUser = Depends(get_current_user),\n):\n    """\n    Log out — clears the refresh token cookie.\n    The access token remains valid until it expires (15 min max).\n    For immediate invalidation, add the JTI to a revocation list (future Phase 12).\n    """\n    _clear_refresh_cookie(response)\n    audit_log(user, "logout", "session")\n    return {"status": "ok", "message": "Logged out successfully."}\n\n\n@router.get("/me", summary="Current user profile")\nasync def me(user: CurrentUser = Depends(get_current_user)):\n    """Return the authenticated user\'s profile and role."""\n    return user.to_dict()\n\n\n@router.get("/providers", summary="List configured SSO providers")\nasync def list_providers():\n    """\n    Return configured SSO providers for the login page to display buttons.\n    Does not require authentication.\n    """\n    from auth.oidc import get_provider_list\n    from auth.saml import get_saml_status\n\n    oidc_providers = get_provider_list()\n    saml_status    = get_saml_status()\n\n    return {\n        "local_auth": True,    # local email/password always available\n        "oidc":       oidc_providers,\n        "saml":       saml_status if saml_status["configured"] else None,\n    }\n\n\n# ── OIDC SSO endpoints ─────────────────────────────────────────────────────────\n\n@router.get("/sso/{provider}", summary="Initiate OIDC SSO login",\n            include_in_schema=False)\nasync def sso_initiate(provider: str, request: Request):\n    """Redirect browser to the IdP authorization endpoint."""\n    from auth.oidc import build_authorize_url\n\n    # Determine redirect_uri (our callback URL)\n    base_url     = str(request.base_url).rstrip("/")\n    redirect_uri = f"{base_url}/api/v1/auth/sso/{provider}/callback"\n    tenant       = request.query_params.get("tenant", "default")\n\n    try:\n        authorize_url, _state = await build_authorize_url(\n            provider_name = provider,\n            redirect_uri  = redirect_uri,\n            tenant        = tenant,\n        )\n    except ValueError as e:\n        raise HTTPException(status_code=400, detail=str(e))\n\n    return RedirectResponse(url=authorize_url, status_code=302)\n\n\n@router.get("/sso/{provider}/callback", summary="OIDC callback handler",\n            include_in_schema=False)\nasync def sso_callback(provider: str, request: Request, response: Response):\n    """\n    Receive authorization code from IdP, exchange for tokens,\n    verify id_token, issue HexaGrid JWT pair, redirect to dashboard.\n    """\n    from auth.oidc import handle_callback\n\n    code  = request.query_params.get("code")\n    state = request.query_params.get("state")\n    error = request.query_params.get("error")\n\n    if error:\n        error_desc = request.query_params.get("error_description", error)\n        log.warning("SSO error from IdP: %s — %s", provider, error_desc)\n        return RedirectResponse(\n            url = f"/login?error={error_desc}",\n            status_code = 302,\n        )\n\n    if not code or not state:\n        raise HTTPException(status_code=400, detail="Missing code or state parameter")\n\n    base_url     = str(request.base_url).rstrip("/")\n    redirect_uri = f"{base_url}/api/v1/auth/sso/{provider}/callback"\n\n    try:\n        tokens = await handle_callback(code, state, redirect_uri)\n    except ValueError as e:\n        log.error("SSO callback error: %s", e)\n        return RedirectResponse(url=f"/login?error={str(e)}", status_code=302)\n\n    # Store access token in session (JS will pick it up) and set refresh cookie\n    _set_refresh_cookie(response, tokens["refresh_token"])\n\n    # Redirect to dashboard with access token in URL fragment\n    # (fragment is never sent to server — safe short-term transport)\n    return RedirectResponse(\n        url         = f"/#sso_token={tokens[\'access_token\']}&role={tokens[\'role\']}",\n        status_code = 302,\n    )\n\n\n# ── SAML endpoints ─────────────────────────────────────────────────────────────\n\n@router.get("/saml/login", summary="Initiate SAML login",\n            include_in_schema=False)\nasync def saml_login(request: Request):\n    """Generate SAML AuthnRequest and redirect to IdP."""\n    from auth.saml import build_authn_request, is_configured\n\n    if not is_configured():\n        raise HTTPException(status_code=503, detail="SAML is not configured")\n\n    relay_state = str(request.base_url).rstrip("/") + "/"\n    try:\n        sso_url, _req_id = await build_authn_request(relay_state=relay_state)\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))\n\n    return RedirectResponse(url=sso_url, status_code=302)\n\n\n@router.post("/saml/callback", summary="SAML ACS endpoint",\n             include_in_schema=False)\nasync def saml_callback(request: Request, response: Response):\n    """Receive and verify SAMLResponse, issue HexaGrid JWT pair."""\n    from auth.saml import handle_saml_response\n\n    form = await request.form()\n    saml_response = form.get("SAMLResponse", "")\n    relay_state   = form.get("RelayState", "")\n    tenant        = request.query_params.get("tenant", "default")\n\n    if not saml_response:\n        raise HTTPException(status_code=400, detail="Missing SAMLResponse")\n\n    try:\n        tokens = await handle_saml_response(saml_response, relay_state, tenant)\n    except Exception as e:\n        log.error("SAML callback error: %s", e)\n        return RedirectResponse(url=f"/login?error={str(e)}", status_code=302)\n\n    _set_refresh_cookie(response, tokens["refresh_token"])\n    return RedirectResponse(\n        url         = f"/#sso_token={tokens[\'access_token\']}&role={tokens[\'role\']}",\n        status_code = 302,\n    )\n\n\n@router.get("/saml/metadata", summary="SAML SP metadata XML",\n            response_class=HTMLResponse)\nasync def saml_metadata():\n    """\n    Serve our SAML SP metadata XML.\n    Enterprise customers upload this to their IdP to configure the integration.\n    """\n    from auth.saml import get_sp_metadata, is_configured\n    if not is_configured():\n        raise HTTPException(status_code=503, detail="SAML is not configured")\n    try:\n        xml = await get_sp_metadata()\n        return HTMLResponse(content=xml, media_type="application/xml")\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))\n\n\n# ── User management (superadmin only) ─────────────────────────────────────────\n\n@router.get("/users", summary="List users [superadmin]")\nasync def list_users_endpoint(\n    tenant: Optional[str] = None,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    return {"users": list_users(tenant=tenant or user.tenant)}\n\n\n@router.post("/users", summary="Create user [superadmin]", status_code=201)\nasync def create_user_endpoint(\n    req: CreateUserRequest,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    try:\n        new_user = create_user(\n            email        = str(req.email),\n            role         = req.role,\n            tenant       = req.tenant or user.tenant,\n            display_name = req.display_name,\n            password     = req.password,\n            sso_provider = req.sso_provider,\n            created_by   = user.user_id,\n        )\n        audit_log(user, "create_user", str(req.email), f"role={req.role}")\n        return new_user.to_dict()\n    except ValueError as e:\n        raise HTTPException(status_code=422, detail=str(e))\n\n\n@router.put("/users/{user_id}/role", summary="Update user role [superadmin]")\nasync def update_role_endpoint(\n    user_id: str,\n    req: UpdateRoleRequest,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    ok = update_user_role(user_id, req.role, updated_by=user.user_id)\n    if not ok:\n        raise HTTPException(status_code=404, detail=f"User {user_id} not found")\n    audit_log(user, "update_role", user_id, f"new_role={req.role}")\n    return {"status": "ok", "user_id": user_id, "new_role": req.role}\n\n\n@router.delete("/users/{user_id}", summary="Deactivate user [superadmin]")\nasync def deactivate_user_endpoint(\n    user_id: str,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    if user_id == user.user_id:\n        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")\n    ok = deactivate_user(user_id, deactivated_by=user.user_id)\n    if not ok:\n        raise HTTPException(status_code=404, detail=f"User {user_id} not found")\n    audit_log(user, "deactivate_user", user_id)\n    return {"status": "ok", "user_id": user_id}\n\n\n# ── API key management (superadmin only) ──────────────────────────────────────\n\n@router.get("/apikeys", summary="List API keys [superadmin]")\nasync def list_apikeys_endpoint(\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    return {"api_keys": list_api_keys(tenant=user.tenant)}\n\n\n@router.post("/apikeys", summary="Create API key [superadmin]", status_code=201)\nasync def create_apikey_endpoint(\n    req: CreateAPIKeyRequest,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    try:\n        result = create_api_key(\n            name       = req.name,\n            role       = req.role,\n            tenant     = req.tenant or user.tenant,\n            scopes     = req.scopes,\n            expires_at = req.expires_at,\n            created_by = user.user_id,\n        )\n        audit_log(user, "create_api_key", req.name, f"role={req.role}")\n        return result   # includes raw_key — shown ONCE\n    except ValueError as e:\n        raise HTTPException(status_code=422, detail=str(e))\n\n\n@router.delete("/apikeys/{key_id}", summary="Revoke API key [superadmin]")\nasync def revoke_apikey_endpoint(\n    key_id: str,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    ok = revoke_api_key(key_id, revoked_by=user.user_id)\n    if not ok:\n        raise HTTPException(status_code=404, detail=f"API key {key_id} not found")\n    audit_log(user, "revoke_api_key", key_id)\n    return {"status": "ok", "key_id": key_id}\n\n\n# ── Audit log (superadmin only) ────────────────────────────────────────────────\n\n@router.get("/audit", summary="Audit log [superadmin]")\nasync def audit_endpoint(\n    limit: int = 100,\n    user: CurrentUser = Depends(require_role("superadmin")),\n):\n    """Return the last N audit log entries."""\n    from auth.api_keys import _conn\n    with _conn() as c:\n        try:\n            rows = c.execute(\n                """SELECT * FROM audit_log\n                   ORDER BY ts DESC LIMIT ?""",\n                (min(limit, 1000),)\n            ).fetchall()\n            return {\n                "entries": [dict(r) for r in rows],\n                "count":   len(rows),\n            }\n        except Exception:\n            return {"entries": [], "count": 0, "note": "Audit log table not yet created"}\n\n\n# ── Login page (served at /login) ──────────────────────────────────────────────\n\n_LOGIN_PAGE = """<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8"/>\n<meta name="viewport" content="width=device-width,initial-scale=1.0"/>\n<title>HexaGrid — Sign In</title>\n<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">\n<style>\n  :root{--bg:#0a0e1a;--bg2:#0f1425;--green:#00ff88;--cyan:#00ccff;--text:#e8edf5;--muted:#6b7a99;--border:rgba(0,255,136,0.15);--red:#ff4444}\n  *{box-sizing:border-box;margin:0;padding:0}\n  body{background:linear-gradient(135deg,#0a0e1a 0%,#0d1528 50%,#0a1020 100%);color:var(--text);font-family:\'Inter\',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}\n  .card{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:16px;padding:48px 40px;width:100%;max-width:420px;box-shadow:0 24px 80px rgba(0,0,0,0.6)}\n  .logo{display:flex;align-items:center;gap:14px;margin-bottom:36px;justify-content:center}\n  .logo-mark{width:44px;height:44px;background:linear-gradient(135deg,var(--green),var(--cyan));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;color:#000}\n  .logo-text{font-size:22px;font-weight:700;letter-spacing:3px;background:linear-gradient(135deg,var(--green),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}\n  h2{font-size:20px;font-weight:600;margin-bottom:8px;text-align:center}\n  .sub{color:var(--muted);font-size:14px;text-align:center;margin-bottom:32px}\n  label{display:block;font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;margin-top:20px}\n  input{width:100%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:12px 16px;color:var(--text);font-size:15px;font-family:\'Inter\',sans-serif;outline:none;transition:border 0.2s}\n  input:focus{border-color:rgba(0,255,136,0.4)}\n  .btn{width:100%;margin-top:28px;padding:14px;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:all 0.2s;letter-spacing:0.3px}\n  .btn-primary{background:linear-gradient(135deg,var(--green),var(--cyan));color:#000}\n  .btn-primary:hover{opacity:0.9;transform:translateY(-1px)}\n  .btn-sso{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);color:var(--text);display:flex;align-items:center;justify-content:center;gap:10px;margin-top:12px;font-size:14px}\n  .btn-sso:hover{background:rgba(255,255,255,0.1);border-color:rgba(0,204,255,0.3)}\n  .divider{display:flex;align-items:center;gap:12px;margin:24px 0;color:var(--muted);font-size:12px}\n  .divider::before,.divider::after{content:\'\';flex:1;height:1px;background:rgba(255,255,255,0.08)}\n  .error{color:var(--red);font-size:13px;margin-top:12px;padding:10px 14px;background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.2);border-radius:8px;display:none}\n  .footer{margin-top:32px;text-align:center;font-size:12px;color:var(--muted)}\n  #spinner{display:none;width:18px;height:18px;border:2px solid rgba(0,0,0,0.3);border-top-color:#000;border-radius:50%;animation:spin 0.7s linear infinite;margin:0 auto}\n  @keyframes spin{to{transform:rotate(360deg)}}\n</style>\n</head>\n<body>\n<div class="card">\n  <div class="logo">\n    <div class="logo-mark">⚡</div>\n    <div class="logo-text">hexagrid</div>\n  </div>\n  <h2>Welcome back</h2>\n  <p class="sub">AI Data Center Energy Intelligence</p>\n\n  <!-- Error message -->\n  <div class="error" id="error-msg"></div>\n\n  <!-- SSO buttons (populated from /api/v1/auth/providers) -->\n  <div id="sso-buttons"></div>\n\n  <!-- Divider -->\n  <div class="divider" id="sso-divider" style="display:none">or sign in with email</div>\n\n  <!-- Local login form -->\n  <div>\n    <label>Email</label>\n    <input type="email" id="email" placeholder="admin@company.com" autocomplete="email"/>\n    <label>Password</label>\n    <input type="password" id="password" placeholder="••••••••" autocomplete="current-password"/>\n    <button class="btn btn-primary" onclick="doLogin()">\n      <span id="btn-text">Sign In</span>\n      <div id="spinner"></div>\n    </button>\n  </div>\n\n  <div class="footer">HexaGrid™ · Quantum Clarity LLC · hexagrid.ai</div>\n</div>\n\n<script>\nconst API = \'/api/v1\';\n\n// ── Check if already logged in ──────────────────────────────────────────────\nasync function checkSession() {\n  const token = sessionStorage.getItem(\'hg_access_token\');\n  if (!token) return;\n  try {\n    const r = await fetch(API + \'/auth/me\', {\n      headers: {\'Authorization\': \'Bearer \' + token}\n    });\n    if (r.ok) window.location.replace(\'/\');\n  } catch(e) {}\n}\n\n// ── Handle SSO token in URL fragment (after OIDC/SAML redirect) ────────────\nfunction checkFragmentToken() {\n  const hash   = window.location.hash.substring(1);\n  const params = new URLSearchParams(hash);\n  const token  = params.get(\'sso_token\');\n  const role   = params.get(\'role\');\n  if (token) {\n    sessionStorage.setItem(\'hg_access_token\', token);\n    if (role) sessionStorage.setItem(\'hg_role\', role);\n    history.replaceState(null, \'\', window.location.pathname);\n    window.location.replace(\'/\');\n  }\n}\n\n// ── Load SSO provider buttons ───────────────────────────────────────────────\nasync function loadProviders() {\n  try {\n    const r = await fetch(API + \'/auth/providers\');\n    const d = await r.json();\n\n    const container = document.getElementById(\'sso-buttons\');\n    const divider   = document.getElementById(\'sso-divider\');\n    const providers = [...(d.oidc || [])];\n    if (d.saml && d.saml.login_url) {\n      providers.push({\n        name: \'saml\', display_name: \'Enterprise SSO (SAML)\',\n        icon: \'🔐\', login_url: d.saml.login_url\n      });\n    }\n\n    if (providers.length > 0) {\n      divider.style.display = \'flex\';\n      container.innerHTML = providers.map(p =>\n        `<button class="btn btn-sso" onclick="window.location=\'${p.login_url}\'">\n           <span>${p.icon}</span> Continue with ${p.display_name}\n         </button>`\n      ).join(\'\');\n    }\n\n    // Check URL for error param\n    const urlParams = new URLSearchParams(window.location.search);\n    const err = urlParams.get(\'error\');\n    if (err) showError(\'SSO error: \' + decodeURIComponent(err));\n  } catch(e) {\n    console.warn(\'Could not load SSO providers:\', e);\n  }\n}\n\n// ── Local login ─────────────────────────────────────────────────────────────\nasync function doLogin() {\n  const email    = document.getElementById(\'email\').value.trim();\n  const password = document.getElementById(\'password\').value;\n  if (!email || !password) { showError(\'Email and password are required\'); return; }\n\n  setLoading(true);\n  try {\n    const r = await fetch(API + \'/auth/login\', {\n      method:  \'POST\',\n      headers: {\'Content-Type\': \'application/json\'},\n      body:    JSON.stringify({email, password}),\n      credentials: \'include\',   // send/receive cookies\n    });\n\n    const d = await r.json();\n    if (!r.ok) {\n      showError(d.detail?.message || \'Login failed\');\n      return;\n    }\n\n    sessionStorage.setItem(\'hg_access_token\', d.access_token);\n    sessionStorage.setItem(\'hg_role\',         d.role);\n    window.location.replace(\'/\');\n\n  } catch(e) {\n    showError(\'Network error — is the API running?\');\n  } finally {\n    setLoading(false);\n  }\n}\n\nfunction showError(msg) {\n  const el = document.getElementById(\'error-msg\');\n  el.textContent = msg;\n  el.style.display = \'block\';\n}\n\nfunction setLoading(on) {\n  document.getElementById(\'btn-text\').style.display    = on ? \'none\' : \'block\';\n  document.getElementById(\'spinner\').style.display     = on ? \'block\' : \'none\';\n}\n\n// Enter key on password field triggers login\ndocument.addEventListener(\'DOMContentLoaded\', () => {\n  checkFragmentToken();\n  checkSession();\n  loadProviders();\n  document.getElementById(\'password\').addEventListener(\'keydown\', e => {\n    if (e.key === \'Enter\') doLogin();\n  });\n  document.getElementById(\'email\').addEventListener(\'keydown\', e => {\n    if (e.key === \'Enter\') document.getElementById(\'password\').focus();\n  });\n});\n</script>\n</body>\n</html>"""\n\n\n# ── Login page route ───────────────────────────────────────────────────────────\n\n@router.get("/login", include_in_schema=False, response_class=HTMLResponse)\nasync def login_page():\n    """Serve the login page."""\n    return HTMLResponse(content=_LOGIN_PAGE)\n'


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: api.py patches
# ══════════════════════════════════════════════════════════════════════════════

API_PATCHES = [
    {
        "description": "Add auth imports after uvicorn import",
        "old": "import uvicorn\n",
        "new": (
            "import uvicorn\n"
            "\n"
            "# ── Auth layer ────────────────────────────────────────────────────────────\n"
            "import sys as _sys_auth\n"
            "_sys_auth.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "from auth.rbac import get_current_user, require_role, apply_security_to_openapi\n"
            "from auth.users import bootstrap_superadmin\n"
            "from auth.api_keys import bootstrap_collector_key\n"
        ),
    },
    {
        "description": "Bootstrap auth on startup inside lifespan()",
        "old": "    await loop.run_in_executor(None, _load_engines)\n    yield\n",
        "new": (
            "    await loop.run_in_executor(None, _load_engines)\n"
            "    # Bootstrap initial superadmin + collector key on first run\n"
            "    bootstrap_superadmin()\n"
            "    bootstrap_collector_key()\n"
            "    yield\n"
        ),
    },
    {
        "description": "Lock down CORS — replace wildcard origins",
        "old": (
            "app.add_middleware(\n"
            "    CORSMiddleware,\n"
            "    allow_origins  = [\"*\"],\n"
            "    allow_methods  = [\"*\"],\n"
            "    allow_headers  = [\"*\"],\n"
            ")"
        ),
        "new": (
            "import json as _json_cors\n"
            "_ALLOWED_ORIGINS = _json_cors.loads(\n"
            "    os.environ.get(\n"
            "        \"HEXAGRID_CORS_ORIGINS\",\n"
            "        '[\"http://localhost:8000\",\"http://localhost:3000\"]'\n"
            "    )\n"
            ")\n"
            "app.add_middleware(\n"
            "    CORSMiddleware,\n"
            "    allow_origins     = _ALLOWED_ORIGINS,\n"
            "    allow_methods     = [\"GET\",\"POST\",\"PUT\",\"DELETE\",\"OPTIONS\"],\n"
            "    allow_headers     = [\"Authorization\",\"Content-Type\",\"X-API-Key\"],\n"
            "    allow_credentials = True,\n"
            ")"
        ),
    },
    {
        "description": "Add /login route that serves the login page",
        "old": "# ── In-memory job store (replace with Redis in production) ────────────────────",
        "new": (
            "@app.get(\"/login\", response_class=HTMLResponse, include_in_schema=False)\n"
            "async def serve_login():\n"
            "    \"\"\"Serve the HexaGrid login page.\"\"\"\n"
            "    from auth_routes import _LOGIN_PAGE\n"
            "    return HTMLResponse(content=_LOGIN_PAGE)\n"
            "\n"
            "# ── In-memory job store (replace with Redis in production) ────────────────────"
        ),
    },
    {
        "description": "Protect /api/v1/simulate with operator role",
        "old": "async def simulate(req: SimulateRequest, background_tasks: BackgroundTasks):",
        "new": "async def simulate(req: SimulateRequest, background_tasks: BackgroundTasks,\n                   _user=Depends(require_role(\"operator\"))):",
    },
    {
        "description": "Protect /api/v1/schedule with operator role",
        "old": "async def schedule(req: ScheduleRequest, background_tasks: BackgroundTasks):",
        "new": "async def schedule(req: ScheduleRequest, background_tasks: BackgroundTasks,\n                   _user=Depends(require_role(\"operator\"))):",
    },
    {
        "description": "Protect /api/v1/rl/train with scheduler_admin role",
        "old": "async def rl_train(req: RLTrainRequest, background_tasks: BackgroundTasks):",
        "new": "async def rl_train(req: RLTrainRequest, background_tasks: BackgroundTasks,\n                   _user=Depends(require_role(\"scheduler_admin\"))):",
    },
    {
        "description": "Mount auth router and apply OpenAPI security at end of file",
        "old": "from telemetry_receiver import router as telemetry_router, init_telemetry_db\napp.include_router(telemetry_router)",
        "new": (
            "from telemetry_receiver import router as telemetry_router, init_telemetry_db\n"
            "app.include_router(telemetry_router)\n"
            "from auth_routes import router as auth_router\n"
            "app.include_router(auth_router)\n"
            "apply_security_to_openapi(app)\n"
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: index.html patches
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_PATCHES = [
    {
        "description": "Replace apiFetch() with auth-aware version + refresh logic",
        "old": (
            "async function apiFetch(path, opts={}) {\n"
            "  const r = await fetch(API + path, opts);\n"
            "  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);\n"
            "  return r.json();\n"
            "}"
        ),
        "new": """\
async function apiFetch(path, opts={}) {
  const token = sessionStorage.getItem('hg_access_token');
  if (token) {
    opts.headers = Object.assign({}, opts.headers || {}, {
      'Authorization': `Bearer ${token}`
    });
  }
  opts.credentials = opts.credentials || 'include';
  const r = await fetch(API + path, opts);
  if (r.status === 401) {
    const body = await r.json().catch(() => ({}));
    if (body?.detail?.error === 'TOKEN_EXPIRED') {
      const refreshed = await _tryRefreshToken();
      if (refreshed) {
        const newToken = sessionStorage.getItem('hg_access_token');
        opts.headers['Authorization'] = `Bearer ${newToken}`;
        const r2 = await fetch(API + path, opts);
        if (!r2.ok) throw new Error(`HTTP ${r2.status}: ${await r2.text()}`);
        return r2.json();
      }
    }
    _redirectToLogin();
    throw new Error('Authentication required');
  }
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

async function _tryRefreshToken() {
  try {
    const r = await fetch(API + '/auth/refresh', {method:'POST',credentials:'include'});
    if (!r.ok) return false;
    const d = await r.json();
    sessionStorage.setItem('hg_access_token', d.access_token);
    sessionStorage.setItem('hg_role', d.role);
    _updateUserChip(d.role);
    return true;
  } catch(e) { return false; }
}

function _redirectToLogin() {
  sessionStorage.removeItem('hg_access_token');
  sessionStorage.removeItem('hg_role');
  window.location.replace('/login');
}

async function doLogout() {
  try {
    await fetch(API + '/auth/logout', {method:'POST',credentials:'include',
      headers:{'Authorization':`Bearer ${sessionStorage.getItem('hg_access_token')||''}`}});
  } catch(e) {}
  _redirectToLogin();
}

function _updateUserChip(role) {
  const chip = document.getElementById('user-role-chip');
  if (!chip) return;
  const colors = {superadmin:'#bf7fff',scheduler_admin:'#00ccff',
    operator:'#00ff88',finance:'#ffaa00',viewer:'#6b7a99',collector:'#6b7a99'};
  chip.textContent = role || 'viewer';
  chip.style.color = colors[role] || '#6b7a99';
}

async function _initAuth() {
  const token = sessionStorage.getItem('hg_access_token');
  if (!token) {
    const refreshed = await _tryRefreshToken();
    if (!refreshed) { _redirectToLogin(); return; }
  } else {
    try {
      const r = await fetch(API + '/auth/me', {
        headers:{'Authorization':`Bearer ${token}`}, credentials:'include'});
      if (r.ok) {
        const me = await r.json();
        _updateUserChip(me.role);
        const emailEl = document.getElementById('user-email-display');
        if (emailEl) emailEl.textContent = me.display_name || me.user_id;
      } else if (r.status === 401) {
        const refreshed = await _tryRefreshToken();
        if (!refreshed) { _redirectToLogin(); return; }
      }
    } catch(e) {}
  }
}""",
    },
    {
        "description": "Add user chip + logout button to header-right",
        "old": '    <div class="header-right">',
        "new": '''\
    <div class="header-right">
      <!-- ── Auth: user chip + logout ───────────────────────── -->
      <div class="badge teal" style="gap:8px;cursor:default" title="Signed in as">
        <i class="fa-solid fa-user-shield" style="font-size:11px"></i>
        <span id="user-email-display" style="max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px">—</span>
        <span id="user-role-chip" style="font-family:var(--mono);font-size:10px;opacity:0.7">—</span>
      </div>
      <button class="help-btn" onclick="doLogout()" title="Sign out"
        style="color:var(--muted);border-color:rgba(255,255,255,0.1)">
        <i class="fa-solid fa-right-from-bracket"></i> Sign Out
      </button>
      <!-- ── End auth ─────────────────────────────────────────── -->''',
    },
    {
        "description": "Call _initAuth() before init() on page load",
        "old": "init();\ninitSavingsAndSpike();",
        "new": (
            "(async function() {\n"
            "  await _initAuth();\n"
            "  init();\n"
            "  initSavingsAndSpike();\n"
            "})();"
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: .env.auth template
# ══════════════════════════════════════════════════════════════════════════════

ENV_TEMPLATE = """\
# HexaGrid Auth — Environment Variables
# ========================================
# Source this file before starting the API:
#   source ~/hexagrid/.env.auth
# Or add to your hexagrid.sh / systemd unit.

# ── REQUIRED ─────────────────────────────────────────────────────────────────
# Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
export HEXAGRID_JWT_SECRET=REPLACE_WITH_GENERATED_SECRET

# Initial superadmin account (created on first startup, skip if users already exist)
export HEXAGRID_ADMIN_EMAIL=admin@yourcompany.com
export HEXAGRID_ADMIN_PASSWORD=REPLACE_WITH_STRONG_PASSWORD

# ── OPTIONAL ─────────────────────────────────────────────────────────────────
# CORS: add your actual dashboard origin(s)
export HEXAGRID_CORS_ORIGINS='["http://localhost:8000","https://hexagrid.yourcompany.com"]'

# Auth DB location (separate from telemetry.db)
export HEXAGRID_AUTH_DB=/var/lib/hexagrid/auth.db

# Access token lifetime (default 15 min)
export HEXAGRID_ACCESS_TTL_MIN=15

# ── OIDC: Okta ───────────────────────────────────────────────────────────────
# export HEXAGRID_OIDC_OKTA_CLIENT_ID=0oa...
# export HEXAGRID_OIDC_OKTA_CLIENT_SECRET=...
# export HEXAGRID_OIDC_OKTA_DISCOVERY_URL=https://dev-xxxxx.okta.com/.well-known/openid-configuration
# export HEXAGRID_OIDC_OKTA_ROLE_CLAIM=groups
# export HEXAGRID_OIDC_OKTA_ROLE_MAP='{"hexagrid-admins":"superadmin","hexagrid-ops":"operator","hexagrid-finance":"finance"}'

# ── OIDC: Azure AD / Microsoft Entra ID ──────────────────────────────────────
# export HEXAGRID_OIDC_AZURE_CLIENT_ID=...
# export HEXAGRID_OIDC_AZURE_CLIENT_SECRET=...
# export HEXAGRID_OIDC_AZURE_DISCOVERY_URL=https://login.microsoftonline.com/{YOUR_TENANT_ID}/v2.0/.well-known/openid-configuration
# export HEXAGRID_OIDC_AZURE_ROLE_CLAIM=roles
# export HEXAGRID_OIDC_AZURE_ROLE_MAP='{"HexaGrid.Admin":"superadmin","HexaGrid.Operator":"operator"}'

# ── OIDC: Google Workspace ───────────────────────────────────────────────────
# export HEXAGRID_OIDC_GOOGLE_CLIENT_ID=...
# export HEXAGRID_OIDC_GOOGLE_CLIENT_SECRET=...
# export HEXAGRID_OIDC_GOOGLE_DISCOVERY_URL=https://accounts.google.com/.well-known/openid-configuration
# export HEXAGRID_OIDC_GOOGLE_ROLE_MAP='{"admin@yourcompany.com":"superadmin"}'

# ── SAML 2.0 (generic enterprise IdP) ────────────────────────────────────────
# export HEXAGRID_SAML_SP_ENTITY_ID=https://hexagrid.yourcompany.com
# export HEXAGRID_SAML_SP_ACS_URL=https://hexagrid.yourcompany.com/api/v1/auth/saml/callback
# export HEXAGRID_SAML_IDP_METADATA_URL=https://your-idp.com/saml/metadata
# export HEXAGRID_SAML_ROLE_ATTRIBUTE=groups
# export HEXAGRID_SAML_ROLE_MAP='{"DataCenter-Admins":"superadmin","DC-Operators":"operator"}'
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HexaGrid RBAC+SSO deployment script")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview all changes without writing anything")
    parser.add_argument("--skip-pip", action="store_true",
                        help="Skip pip install step")
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        banner("DRY RUN — no files will be modified")

    # ── 1. Layout ─────────────────────────────────────────────────────────────
    print_layout()

    # ── 2. Validate paths ─────────────────────────────────────────────────────
    banner("VALIDATING PATHS")
    for p, label in [(API_PY, "api.py"), (INDEX_HTML, "index.html"), (API_DIR, "api/")]:
        if p.exists():
            ok(f"Found: {p}")
        else:
            fail(f"Not found: {p}  — is HEXAGRID_ROOT correct? ({HEXAGRID_ROOT})")

    # ── 3. Install deps ───────────────────────────────────────────────────────
    if not args.skip_pip:
        banner("INSTALLING PYTHON DEPENDENCIES")
        step("pip install python-jose[cryptography] passlib[bcrypt] httpx")
        if not dry:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "python-jose[cryptography]", "passlib[bcrypt]", "httpx",
                 "--break-system-packages", "-q"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                warn(f"pip install had issues:\n{result.stderr[-500:]}")
            else:
                ok("Dependencies installed")
        else:
            ok("[DRY-RUN] Would install: python-jose passlib httpx")

    # ── 4. Write auth/ package ────────────────────────────────────────────────
    banner("WRITING AUTH PACKAGE")
    step(f"Creating: {AUTH_DIR}")

    files = {
        AUTH_DIR / "__init__.py":   AUTH_INIT,
        AUTH_DIR / "jwt_handler.py": AUTH_JWT,
        AUTH_DIR / "api_keys.py":   AUTH_APIKEYS,
        AUTH_DIR / "rbac.py":       AUTH_RBAC,
        AUTH_DIR / "oidc.py":       AUTH_OIDC,
        AUTH_DIR / "saml.py":       AUTH_SAML,
        AUTH_DIR / "users.py":      AUTH_USERS,
        API_DIR  / "auth_routes.py": AUTH_ROUTES,
    }

    for path, content in files.items():
        write_file(path, content, dry)

    # ── 5. Patch api.py ────────────────────────────────────────────────────────
    banner("PATCHING api/api.py")
    if not dry:
        backup(API_PY)
    for patch in API_PATCHES:
        step(patch["description"])
        patch_file(API_PY, patch["old"], patch["new"], patch["description"], dry)

    # ── 6. Patch index.html ────────────────────────────────────────────────────
    banner("PATCHING dashboard/index.html")
    if not dry:
        backup(INDEX_HTML)
    for patch in DASHBOARD_PATCHES:
        step(patch["description"])
        patch_file(INDEX_HTML, patch["old"], patch["new"], patch["description"], dry)

    # ── 7. Write .env.auth ─────────────────────────────────────────────────────
    banner("WRITING .env.auth TEMPLATE")
    env_path = HEXAGRID_ROOT / ".env.auth"
    if env_path.exists() and not dry:
        warn(f".env.auth already exists — skipping (not overwriting existing secrets)")
    else:
        write_file(env_path, ENV_TEMPLATE, dry)

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    banner("DEPLOYMENT COMPLETE")
    print(f"""
  Next steps:

  1. Fill in your JWT secret and admin password:
       nano ~/hexagrid/.env.auth

  2. Source the env vars:
       source ~/hexagrid/.env.auth

  3. Restart the API:
       hexagrid restart

  4. Open the dashboard — you'll see the login page at:
       http://localhost:8000/login

  5. Log in with:
       Email:    $HEXAGRID_ADMIN_EMAIL
       Password: $HEXAGRID_ADMIN_PASSWORD

  6. (Optional) Add SSO providers by filling in the OIDC/SAML
     sections of .env.auth and restarting.

  Backups of modified files are in the same directory
  with suffix: {BACKUP_SUFFIX}
""")

if __name__ == "__main__":
    main()
