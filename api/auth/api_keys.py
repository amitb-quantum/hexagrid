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
    tenant       TEXT NOT NULL DEFAULT 'default',
    scopes       TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    last_used_at TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0,
    created_by   TEXT NOT NULL DEFAULT 'system'
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
    if role not in VALID_ROLES: raise ValueError(f"Invalid role '{role}'")
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
    print("\n"+"═"*60)
    print("  ⚡ HexaGrid — Bootstrap Collector API Key")
    print(f"  HEXAGRID_TOKEN={result['raw_key']}")
    print("  This key will NOT be shown again.")
    print("═"*60+"\n")
    return result
