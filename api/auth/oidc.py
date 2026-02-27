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
    if not pvd: raise ValueError(f"OIDC provider '{provider_name}' not configured")
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
