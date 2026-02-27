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
