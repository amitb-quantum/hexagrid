"""
HexaGrid Auth — FastAPI Router
================================
All authentication endpoints mounted at /api/v1/auth/*.

Endpoints:
  POST /api/v1/auth/login                  Local email+password login
  POST /api/v1/auth/refresh                Rotate refresh token → new access token
  POST /api/v1/auth/logout                 Invalidate session (clear refresh cookie)
  GET  /api/v1/auth/me                     Current user profile
  GET  /api/v1/auth/providers              List configured SSO providers

  GET  /api/v1/auth/sso/{provider}         Initiate OIDC SSO redirect
  GET  /api/v1/auth/sso/{provider}/callback OIDC callback handler

  GET  /api/v1/auth/saml/login             Initiate SAML AuthnRequest
  POST /api/v1/auth/saml/callback          SAML ACS (Assertion Consumer Service)
  GET  /api/v1/auth/saml/metadata          SP metadata XML for IdP configuration

  GET  /api/v1/auth/users                  List users          [superadmin]
  POST /api/v1/auth/users                  Create user         [superadmin]
  PUT  /api/v1/auth/users/{id}/role        Change user role    [superadmin]
  DELETE /api/v1/auth/users/{id}           Deactivate user     [superadmin]

  GET  /api/v1/auth/apikeys                List API keys       [superadmin]
  POST /api/v1/auth/apikeys                Create API key      [superadmin]
  DELETE /api/v1/auth/apikeys/{id}         Revoke API key      [superadmin]

  GET  /api/v1/auth/audit                  Audit log           [superadmin]

Mount in api.py:
  from auth_routes import router as auth_router
  app.include_router(auth_router)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, Depends, status
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field

from auth.rbac import (
    get_current_user, require_role, require_any_role,
    CurrentUser, audit_log, apply_security_to_openapi
)
from auth.jwt_handler import (
    issue_token_pair, verify_token,
    InvalidTokenError, ExpiredTokenError,
)
from auth.api_keys import (
    create_api_key, revoke_api_key, list_api_keys, get_api_key
)
from auth.users import (
    authenticate_local, create_user,
    list_users, update_user_role, deactivate_user, bootstrap_superadmin
)

log = logging.getLogger("hexagrid.auth.routes")

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

# Cookie name for refresh token (HttpOnly, Secure, SameSite=Lax)
_REFRESH_COOKIE = "hg_refresh"
_COOKIE_MAX_AGE = 7 * 24 * 3600   # 7 days in seconds


# ── Request/Response models ────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str = Field(..., min_length=1)

    model_config = {"json_schema_extra": {
        "example": {"email": "admin@hexagrid.local", "password": "changeme"}
    }}


class CreateUserRequest(BaseModel):
    email:        str
    role:         str
    tenant:       str      = "default"
    display_name: Optional[str] = None
    password:     Optional[str] = None
    sso_provider: Optional[str] = None


class UpdateRoleRequest(BaseModel):
    role: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=8,
                                  description="Minimum 8 characters")

    model_config = {"json_schema_extra": {
        "example": {"current_password": "old-password", "new_password": "new-password-123"}
    }}


class CreateAPIKeyRequest(BaseModel):
    name:       str
    role:       str
    tenant:     str            = "default"
    scopes:     Optional[list] = None
    expires_at: Optional[str]  = None


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Set the HttpOnly refresh token cookie."""
    response.set_cookie(
        key      = _REFRESH_COOKIE,
        value    = refresh_token,
        max_age  = _COOKIE_MAX_AGE,
        httponly = True,
        secure   = True,        # HTTPS only in production
        samesite = "lax",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(_REFRESH_COOKIE)


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@router.post("/login", summary="Local email + password login")
async def login(req: LoginRequest, response: Response):
    """
    Authenticate with email + password (local account).
    Returns access token in response body.
    Sets refresh token as HttpOnly cookie.
    """
    tokens = authenticate_local(req.email, req.password)
    if not tokens:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = {"error": "INVALID_CREDENTIALS",
                           "message": "Invalid email or password"},
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    _set_refresh_cookie(response, tokens["refresh_token"])

    return {
        "access_token": tokens["access_token"],
        "token_type":   "bearer",
        "expires_in":   tokens["expires_in"],
        "role":         tokens["role"],
        "tenant":       tokens["tenant"],
    }


@router.post("/refresh", summary="Refresh access token using refresh cookie")
async def refresh_token(request: Request, response: Response):
    """
    Issue a new access token using the refresh token stored in the HttpOnly cookie.
    The refresh token is rotated (old token invalidated, new cookie set).
    """
    refresh_tk = request.cookies.get(_REFRESH_COOKIE)
    if not refresh_tk:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = {"error": "NO_REFRESH_TOKEN",
                           "message": "No refresh token found. Please log in again."},
        )

    try:
        from auth.jwt_handler import rotate_refresh_token
        tokens = rotate_refresh_token(refresh_tk)
    except ExpiredTokenError:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = {"error": "REFRESH_EXPIRED",
                           "message": "Session expired. Please log in again."},
        )
    except InvalidTokenError as e:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = {"error": "INVALID_REFRESH", "message": str(e)},
        )

    _set_refresh_cookie(response, tokens["refresh_token"])
    return {
        "access_token": tokens["access_token"],
        "token_type":   "bearer",
        "expires_in":   tokens["expires_in"],
        "role":         tokens["role"],
        "tenant":       tokens["tenant"],
    }


@router.post("/logout", summary="Invalidate session")
async def logout(
    response: Response,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Log out — clears the refresh token cookie.
    The access token remains valid until it expires (15 min max).
    For immediate invalidation, add the JTI to a revocation list (future Phase 12).
    """
    _clear_refresh_cookie(response)
    audit_log(user, "logout", "session")
    return {"status": "ok", "message": "Logged out successfully."}


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
    return {"status": "ok", "message": "Password updated successfully."}


@router.get("/providers", summary="List configured SSO providers")
async def list_providers():
    """
    Return configured SSO providers for the login page to display buttons.
    Does not require authentication.
    """
    from auth.oidc import get_provider_list
    from auth.saml import get_saml_status

    oidc_providers = get_provider_list()
    saml_status    = get_saml_status()

    return {
        "local_auth": True,    # local email/password always available
        "oidc":       oidc_providers,
        "saml":       saml_status if saml_status["configured"] else None,
    }


# ── OIDC SSO endpoints ─────────────────────────────────────────────────────────

@router.get("/sso/{provider}", summary="Initiate OIDC SSO login",
            include_in_schema=False)
async def sso_initiate(provider: str, request: Request):
    """Redirect browser to the IdP authorization endpoint."""
    from auth.oidc import build_authorize_url

    # Determine redirect_uri (our callback URL)
    base_url     = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/api/v1/auth/sso/{provider}/callback"
    tenant       = request.query_params.get("tenant", "default")

    try:
        authorize_url, _state = await build_authorize_url(
            provider_name = provider,
            redirect_uri  = redirect_uri,
            tenant        = tenant,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RedirectResponse(url=authorize_url, status_code=302)


@router.get("/sso/{provider}/callback", summary="OIDC callback handler",
            include_in_schema=False)
async def sso_callback(provider: str, request: Request, response: Response):
    """
    Receive authorization code from IdP, exchange for tokens,
    verify id_token, issue HexaGrid JWT pair, redirect to dashboard.
    """
    from auth.oidc import handle_callback

    code  = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        error_desc = request.query_params.get("error_description", error)
        log.warning("SSO error from IdP: %s — %s", provider, error_desc)
        return RedirectResponse(
            url = f"/login?error={error_desc}",
            status_code = 302,
        )

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")

    base_url     = str(request.base_url).rstrip("/")
    redirect_uri = f"{base_url}/api/v1/auth/sso/{provider}/callback"

    try:
        tokens = await handle_callback(code, state, redirect_uri)
    except ValueError as e:
        log.error("SSO callback error: %s", e)
        return RedirectResponse(url=f"/login?error={str(e)}", status_code=302)

    # Store access token in session (JS will pick it up) and set refresh cookie
    _set_refresh_cookie(response, tokens["refresh_token"])

    # Redirect to dashboard with access token in URL fragment
    # (fragment is never sent to server — safe short-term transport)
    return RedirectResponse(
        url         = f"/#sso_token={tokens['access_token']}&role={tokens['role']}",
        status_code = 302,
    )


# ── SAML endpoints ─────────────────────────────────────────────────────────────

@router.get("/saml/login", summary="Initiate SAML login",
            include_in_schema=False)
async def saml_login(request: Request):
    """Generate SAML AuthnRequest and redirect to IdP."""
    from auth.saml import build_authn_request, is_configured

    if not is_configured():
        raise HTTPException(status_code=503, detail="SAML is not configured")

    relay_state = str(request.base_url).rstrip("/") + "/"
    try:
        sso_url, _req_id = await build_authn_request(relay_state=relay_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RedirectResponse(url=sso_url, status_code=302)


@router.post("/saml/callback", summary="SAML ACS endpoint",
             include_in_schema=False)
async def saml_callback(request: Request, response: Response):
    """Receive and verify SAMLResponse, issue HexaGrid JWT pair."""
    from auth.saml import handle_saml_response

    form = await request.form()
    saml_response = form.get("SAMLResponse", "")
    relay_state   = form.get("RelayState", "")
    tenant        = request.query_params.get("tenant", "default")

    if not saml_response:
        raise HTTPException(status_code=400, detail="Missing SAMLResponse")

    try:
        tokens = await handle_saml_response(saml_response, relay_state, tenant)
    except Exception as e:
        log.error("SAML callback error: %s", e)
        return RedirectResponse(url=f"/login?error={str(e)}", status_code=302)

    _set_refresh_cookie(response, tokens["refresh_token"])
    return RedirectResponse(
        url         = f"/#sso_token={tokens['access_token']}&role={tokens['role']}",
        status_code = 302,
    )


@router.get("/saml/metadata", summary="SAML SP metadata XML",
            response_class=HTMLResponse)
async def saml_metadata():
    """
    Serve our SAML SP metadata XML.
    Enterprise customers upload this to their IdP to configure the integration.
    """
    from auth.saml import get_sp_metadata, is_configured
    if not is_configured():
        raise HTTPException(status_code=503, detail="SAML is not configured")
    try:
        xml = await get_sp_metadata()
        return HTMLResponse(content=xml, media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── User management (superadmin only) ─────────────────────────────────────────

@router.get("/users", summary="List users [superadmin]")
async def list_users_endpoint(
    tenant: Optional[str] = None,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    return {"users": list_users(tenant=tenant or user.tenant)}


@router.post("/users", summary="Create user [superadmin]", status_code=201)
async def create_user_endpoint(
    req: CreateUserRequest,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    try:
        new_user = create_user(
            email        = str(req.email),
            role         = req.role,
            tenant       = req.tenant or user.tenant,
            display_name = req.display_name,
            password     = req.password,
            sso_provider = req.sso_provider,
            created_by   = user.user_id,
        )
        audit_log(user, "create_user", str(req.email), f"role={req.role}")
        return new_user.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.put("/users/{user_id}/role", summary="Update user role [superadmin]")
async def update_role_endpoint(
    user_id: str,
    req: UpdateRoleRequest,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    ok = update_user_role(user_id, req.role, updated_by=user.user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    audit_log(user, "update_role", user_id, f"new_role={req.role}")
    return {"status": "ok", "user_id": user_id, "new_role": req.role}


@router.delete("/users/{user_id}", summary="Deactivate user [superadmin]")
async def deactivate_user_endpoint(
    user_id: str,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    if user_id == user.user_id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    ok = deactivate_user(user_id, deactivated_by=user.user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    audit_log(user, "deactivate_user", user_id)
    return {"status": "ok", "user_id": user_id}


# ── API key management (superadmin only) ──────────────────────────────────────

@router.get("/apikeys", summary="List API keys [superadmin]")
async def list_apikeys_endpoint(
    user: CurrentUser = Depends(require_role("superadmin")),
):
    return {"api_keys": list_api_keys(tenant=user.tenant)}


@router.post("/apikeys", summary="Create API key [superadmin]", status_code=201)
async def create_apikey_endpoint(
    req: CreateAPIKeyRequest,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    try:
        result = create_api_key(
            name       = req.name,
            role       = req.role,
            tenant     = req.tenant or user.tenant,
            scopes     = req.scopes,
            expires_at = req.expires_at,
            created_by = user.user_id,
        )
        audit_log(user, "create_api_key", req.name, f"role={req.role}")
        return result   # includes raw_key — shown ONCE
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/apikeys/{key_id}", summary="Revoke API key [superadmin]")
async def revoke_apikey_endpoint(
    key_id: str,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    ok = revoke_api_key(key_id, revoked_by=user.user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"API key {key_id} not found")
    audit_log(user, "revoke_api_key", key_id)
    return {"status": "ok", "key_id": key_id}


# ── Audit log (superadmin only) ────────────────────────────────────────────────

@router.get("/audit", summary="Audit log [superadmin]")
async def audit_endpoint(
    limit: int = 100,
    user: CurrentUser = Depends(require_role("superadmin")),
):
    """Return the last N audit log entries."""
    from auth.api_keys import _conn
    with _conn() as c:
        try:
            rows = c.execute(
                """SELECT * FROM audit_log
                   ORDER BY ts DESC LIMIT ?""",
                (min(limit, 1000),)
            ).fetchall()
            return {
                "entries": [dict(r) for r in rows],
                "count":   len(rows),
            }
        except Exception:
            return {"entries": [], "count": 0, "note": "Audit log table not yet created"}


# ── Login page (served at /login) ──────────────────────────────────────────────

_LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>HexaGrid — Sign In</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{--bg:#0a0e1a;--bg2:#0f1425;--green:#00ff88;--cyan:#00ccff;--text:#e8edf5;--muted:#6b7a99;--border:rgba(0,255,136,0.15);--red:#ff4444}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:linear-gradient(135deg,#0a0e1a 0%,#0d1528 50%,#0a1020 100%);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
  .card{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:16px;padding:48px 40px;width:100%;max-width:420px;box-shadow:0 24px 80px rgba(0,0,0,0.6)}
  .logo{display:flex;align-items:center;gap:14px;margin-bottom:36px;justify-content:center}
  .logo-mark{width:44px;height:44px;background:linear-gradient(135deg,var(--green),var(--cyan));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;color:#000}
  .logo-text{font-size:22px;font-weight:700;letter-spacing:3px;background:linear-gradient(135deg,var(--green),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  h2{font-size:20px;font-weight:600;margin-bottom:8px;text-align:center}
  .sub{color:var(--muted);font-size:14px;text-align:center;margin-bottom:32px}
  label{display:block;font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;margin-top:20px}
  input{width:100%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:12px 16px;color:var(--text);font-size:15px;font-family:'Inter',sans-serif;outline:none;transition:border 0.2s}
  input:focus{border-color:rgba(0,255,136,0.4)}
  .btn{width:100%;margin-top:28px;padding:14px;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:all 0.2s;letter-spacing:0.3px}
  .btn-primary{background:linear-gradient(135deg,var(--green),var(--cyan));color:#000}
  .btn-primary:hover{opacity:0.9;transform:translateY(-1px)}
  .btn-sso{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);color:var(--text);display:flex;align-items:center;justify-content:center;gap:10px;margin-top:12px;font-size:14px}
  .btn-sso:hover{background:rgba(255,255,255,0.1);border-color:rgba(0,204,255,0.3)}
  .divider{display:flex;align-items:center;gap:12px;margin:24px 0;color:var(--muted);font-size:12px}
  .divider::before,.divider::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.08)}
  .error{color:var(--red);font-size:13px;margin-top:12px;padding:10px 14px;background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.2);border-radius:8px;display:none}
  .footer{margin-top:32px;text-align:center;font-size:12px;color:var(--muted)}
  #spinner{display:none;width:18px;height:18px;border:2px solid rgba(0,0,0,0.3);border-top-color:#000;border-radius:50%;animation:spin 0.7s linear infinite;margin:0 auto}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <div class="logo-mark">⚡</div>
    <div class="logo-text">hexagrid</div>
  </div>
  <h2>Welcome back</h2>
  <p class="sub">AI Data Center Energy Intelligence</p>

  <!-- Error message -->
  <div class="error" id="error-msg"></div>

  <!-- SSO buttons (populated from /api/v1/auth/providers) -->
  <div id="sso-buttons"></div>

  <!-- Divider -->
  <div class="divider" id="sso-divider" style="display:none">or sign in with email</div>

  <!-- Local login form -->
  <div>
    <label>Email</label>
    <input type="email" id="email" placeholder="admin@company.com" autocomplete="email"/>
    <label>Password</label>
    <input type="password" id="password" placeholder="••••••••" autocomplete="current-password"/>
    <button class="btn btn-primary" onclick="doLogin()">
      <span id="btn-text">Sign In</span>
      <div id="spinner"></div>
    </button>
  </div>

  <div class="footer">HexaGrid™ · Quantum Clarity LLC · hexagrid.ai</div>
</div>

<script>
const API = '/api/v1';

// ── Check if already logged in ──────────────────────────────────────────────
async function checkSession() {
  const token = sessionStorage.getItem('hg_access_token');
  if (!token) return;
  try {
    const r = await fetch(API + '/auth/me', {
      headers: {'Authorization': 'Bearer ' + token}
    });
    if (r.ok) window.location.replace('/');
  } catch(e) {}
}

// ── Handle SSO token in URL fragment (after OIDC/SAML redirect) ────────────
function checkFragmentToken() {
  const hash   = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const token  = params.get('sso_token');
  const role   = params.get('role');
  if (token) {
    sessionStorage.setItem('hg_access_token', token);
    if (role) sessionStorage.setItem('hg_role', role);
    history.replaceState(null, '', window.location.pathname);
    window.location.replace('/');
  }
}

// ── Load SSO provider buttons ───────────────────────────────────────────────
async function loadProviders() {
  try {
    const r = await fetch(API + '/auth/providers');
    const d = await r.json();

    const container = document.getElementById('sso-buttons');
    const divider   = document.getElementById('sso-divider');
    const providers = [...(d.oidc || [])];
    if (d.saml && d.saml.login_url) {
      providers.push({
        name: 'saml', display_name: 'Enterprise SSO (SAML)',
        icon: '🔐', login_url: d.saml.login_url
      });
    }

    if (providers.length > 0) {
      divider.style.display = 'flex';
      container.innerHTML = providers.map(p =>
        `<button class="btn btn-sso" onclick="window.location='${p.login_url}'">
           <span>${p.icon}</span> Continue with ${p.display_name}
         </button>`
      ).join('');
    }

    // Check URL for error param
    const urlParams = new URLSearchParams(window.location.search);
    const err = urlParams.get('error');
    if (err) showError('SSO error: ' + decodeURIComponent(err));
  } catch(e) {
    console.warn('Could not load SSO providers:', e);
  }
}

// ── Local login ─────────────────────────────────────────────────────────────
async function doLogin() {
  const email    = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  if (!email || !password) { showError('Email and password are required'); return; }

  setLoading(true);
  try {
    const r = await fetch(API + '/auth/login', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({email, password}),
      credentials: 'include',   // send/receive cookies
    });

    const d = await r.json();
    if (!r.ok) {
      showError(d.detail?.message || 'Login failed');
      return;
    }

    sessionStorage.setItem('hg_access_token', d.access_token);
    sessionStorage.setItem('hg_role',         d.role);
    window.location.replace('/');

  } catch(e) {
    showError('Network error — is the API running?');
  } finally {
    setLoading(false);
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function setLoading(on) {
  document.getElementById('btn-text').style.display    = on ? 'none' : 'block';
  document.getElementById('spinner').style.display     = on ? 'block' : 'none';
}

// Enter key on password field triggers login
document.addEventListener('DOMContentLoaded', () => {
  checkFragmentToken();
  checkSession();
  loadProviders();
  document.getElementById('password').addEventListener('keydown', e => {
    if (e.key === 'Enter') doLogin();
  });
  document.getElementById('email').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('password').focus();
  });
});
</script>
</body>
</html>"""


# ── Login page route ───────────────────────────────────────────────────────────

@router.get("/login", include_in_schema=False, response_class=HTMLResponse)
async def login_page():
    """Serve the login page."""
    return HTMLResponse(content=_LOGIN_PAGE)
