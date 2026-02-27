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
            _raise_403(f"Role '{user.role}' is not permitted. Required: '{minimum_role}' or higher.")
        return user
    return _guard

def require_any_role(allowed_roles:list):
    role_set=set(allowed_roles)
    async def _guard(user:CurrentUser=Depends(get_current_user)) -> CurrentUser:
        if user.role not in role_set: _raise_403(f"Role '{user.role}' not in allowed set.")
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
