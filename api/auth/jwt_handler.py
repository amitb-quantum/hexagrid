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
    if role not in VALID_ROLES: raise ValueError(f"Unknown role '{role}'")
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
        raise InvalidTokenError(f"Wrong token type. Expected '{expected_type}', got '{raw.get('token_type','unknown')}'")
    if raw.get("iss") != ISSUER:
        raise InvalidTokenError(f"Unknown issuer: {raw.get('iss')}")
    return TokenPayload(raw["sub"],raw["role"],raw.get("tenant","default"),raw["jti"],
                        datetime.fromtimestamp(raw["exp"],tz=timezone.utc),raw["iss"],raw["token_type"])

def rotate_refresh_token(refresh_token:str) -> dict:
    payload = verify_token(refresh_token, expected_type="refresh")
    return issue_token_pair(payload.sub, payload.role, payload.tenant)
