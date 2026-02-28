"""
WebAuthn (Touch ID / passkey) authentication for Kismet Voice Agent.

Provides:
- Registration & login flows via py_webauthn
- Signed session cookies via itsdangerous
- FastAPI middleware protecting all routes
- JSON file-based credential storage
"""

import io
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from webauthn.registration.generate_registration_options import generate_registration_options
from webauthn.registration.verify_registration_response import verify_registration_response
from webauthn.authentication.generate_authentication_options import generate_authentication_options
from webauthn.authentication.verify_authentication_response import verify_authentication_response
from webauthn.helpers.options_to_json import options_to_json
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    ResidentKeyRequirement,
    UserVerificationRequirement,
    PublicKeyCredentialDescriptor,
)
from webauthn.helpers import bytes_to_base64url, base64url_to_bytes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUTH_SECRET = os.getenv("AUTH_SECRET", secrets.token_hex(32))
AUTH_SESSION_HOURS = int(os.getenv("AUTH_SESSION_HOURS", "24"))
RP_ID = os.getenv("WEBAUTHN_RP_ID", None)  # Resolved at runtime from request host
RP_NAME = os.getenv("WEBAUTHN_RP_NAME", "Kismet Voice Agent")
ORIGIN = os.getenv("WEBAUTHN_ORIGIN", None)  # Resolved at runtime

CREDENTIALS_FILE = Path(__file__).parent / "auth_credentials.json"
SESSION_COOKIE = "kismet_session"
SESSION_MAX_AGE = AUTH_SESSION_HOURS * 3600

_serializer = URLSafeTimedSerializer(AUTH_SECRET)

# In-memory challenge store (maps challenge -> timestamp, cleaned on use)
_pending_challenges: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Credential storage (JSON file)
# ---------------------------------------------------------------------------

def _load_credentials() -> list[dict]:
    if CREDENTIALS_FILE.exists():
        try:
            return json.loads(CREDENTIALS_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_credentials(creds: list[dict]):
    CREDENTIALS_FILE.write_text(json.dumps(creds, indent=2))


def _has_credentials() -> bool:
    return len(_load_credentials()) > 0


def _find_credential(credential_id_b64: str) -> Optional[dict]:
    for c in _load_credentials():
        if c["credential_id"] == credential_id_b64:
            return c
    return None


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session_cookie(user_id: str) -> str:
    return _serializer.dumps({"uid": user_id, "t": time.time()})


def verify_session_cookie(cookie_value: str) -> Optional[dict]:
    try:
        data = _serializer.loads(cookie_value, max_age=SESSION_MAX_AGE)
        return data
    except (BadSignature, SignatureExpired):
        return None


def _get_rp_id(request: Request = None, ws: WebSocket = None) -> str:
    if RP_ID:
        return RP_ID
    obj = request or ws
    host = obj.headers.get("host", "localhost")
    # Strip port
    return host.split(":")[0]


def _get_origin(request: Request = None, ws: WebSocket = None) -> str:
    if ORIGIN:
        return ORIGIN
    rp_id = _get_rp_id(request, ws)
    obj = request or ws
    # Detect scheme from headers
    scheme = "https"
    if obj.headers.get("x-forwarded-proto"):
        scheme = obj.headers["x-forwarded-proto"]
    host = obj.headers.get("host", f"{rp_id}:8765")
    return f"{scheme}://{host}"


# ---------------------------------------------------------------------------
# Login page HTML
# ---------------------------------------------------------------------------

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kismet — Authenticate</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #16213e; border-radius: 16px; padding: 48px 40px;
    text-align: center; max-width: 400px; width: 90%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  h1 { font-size: 1.6rem; margin-bottom: 8px; color: #fff; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 32px; }
  button {
    background: #00d97e; color: #1a1a2e; border: none;
    padding: 14px 32px; border-radius: 10px; font-size: 1rem;
    font-weight: 600; cursor: pointer; width: 100%;
    transition: background 0.2s;
  }
  button:hover { background: #00c06e; }
  button:disabled { background: #555; color: #999; cursor: not-allowed; }
  .error { color: #ff6b6b; margin-top: 16px; font-size: 0.85rem; }
  .icon { font-size: 3rem; margin-bottom: 16px; }
</style>
</head><body>
<div class="card">
  <div class="icon">&#x1F511;</div>
  <h1>Kismet Voice Agent</h1>
  <p class="sub">Secure access via Touch ID / Passkey</p>
  <button id="authBtn" onclick="doAuth()">Loading...</button>
  <p class="error" id="error"></p>
</div>
<script>
let isRegister = false;

async function init() {
  try {
    const res = await fetch('/auth/status');
    const data = await res.json();
    isRegister = !data.has_credentials;
    const btn = document.getElementById('authBtn');
    btn.textContent = isRegister ? 'Register Touch ID' : 'Authenticate with Touch ID';
    btn.disabled = false;
  } catch(e) {
    document.getElementById('error').textContent = 'Failed to connect to server';
  }
}

async function doAuth() {
  const btn = document.getElementById('authBtn');
  const err = document.getElementById('error');
  err.textContent = '';
  btn.disabled = true;
  btn.textContent = 'Waiting for Touch ID...';

  try {
    if (isRegister) {
      await register();
    } else {
      await authenticate();
    }
    window.location.href = '/';
  } catch(e) {
    err.textContent = e.message || 'Authentication failed';
    btn.textContent = isRegister ? 'Register Touch ID' : 'Authenticate with Touch ID';
    btn.disabled = false;
  }
}

async function register() {
  const optRes = await fetch('/auth/register-options', { method: 'POST' });
  if (!optRes.ok) throw new Error('Failed to get registration options');
  const options = await optRes.json();

  // Convert base64url fields to ArrayBuffer
  options.challenge = b64urlToBuffer(options.challenge);
  options.user.id = b64urlToBuffer(options.user.id);
  if (options.excludeCredentials) {
    options.excludeCredentials = options.excludeCredentials.map(c => ({
      ...c, id: b64urlToBuffer(c.id)
    }));
  }

  const credential = await navigator.credentials.create({ publicKey: options });
  const body = {
    id: credential.id,
    rawId: bufferToB64url(credential.rawId),
    type: credential.type,
    response: {
      attestationObject: bufferToB64url(credential.response.attestationObject),
      clientDataJSON: bufferToB64url(credential.response.clientDataJSON),
    }
  };

  const verifyRes = await fetch('/auth/register-verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!verifyRes.ok) {
    const d = await verifyRes.json();
    throw new Error(d.detail || 'Registration failed');
  }
}

async function authenticate() {
  const optRes = await fetch('/auth/login-options', { method: 'POST' });
  if (!optRes.ok) throw new Error('Failed to get login options');
  const options = await optRes.json();

  options.challenge = b64urlToBuffer(options.challenge);
  if (options.allowCredentials) {
    options.allowCredentials = options.allowCredentials.map(c => ({
      ...c, id: b64urlToBuffer(c.id)
    }));
  }

  const credential = await navigator.credentials.get({ publicKey: options });
  const body = {
    id: credential.id,
    rawId: bufferToB64url(credential.rawId),
    type: credential.type,
    response: {
      authenticatorData: bufferToB64url(credential.response.authenticatorData),
      clientDataJSON: bufferToB64url(credential.response.clientDataJSON),
      signature: bufferToB64url(credential.response.signature),
      userHandle: credential.response.userHandle
        ? bufferToB64url(credential.response.userHandle) : null,
    }
  };

  const verifyRes = await fetch('/auth/login-verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!verifyRes.ok) {
    const d = await verifyRes.json();
    throw new Error(d.detail || 'Authentication failed');
  }
}

// --- Base64URL helpers ---
function b64urlToBuffer(b64url) {
  const b64 = b64url.replace(/-/g, '+').replace(/_/g, '/');
  const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4));
  const bin = atob(b64 + pad);
  return Uint8Array.from(bin, c => c.charCodeAt(0)).buffer;
}

function bufferToB64url(buf) {
  const bytes = new Uint8Array(buf);
  let bin = '';
  bytes.forEach(b => bin += String.fromCharCode(b));
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

init();
</script>
</body></html>"""

def _generate_qr_svg(data: str) -> str:
    """Generate a QR code as an SVG string. Uses qrcode lib if available, falls back to text."""
    try:
        import qrcode
        import qrcode.image.svg
        factory = qrcode.image.svg.SvgPathImage
        img = qrcode.make(data, image_factory=factory, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return buf.getvalue().decode("utf-8")
    except ImportError:
        return f'<p style="color:#ff6b6b">Install qrcode: pip install qrcode</p>'


# One-time invite tokens: token -> expiry timestamp
_invite_tokens: dict[str, float] = {}
INVITE_TTL = 300  # 5 minutes

ADD_DEVICE_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kismet — Add Device</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #16213e; border-radius: 16px; padding: 48px 40px;
    text-align: center; max-width: 400px; width: 90%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  h1 { font-size: 1.4rem; margin-bottom: 8px; color: #fff; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 32px; }
  button {
    background: #00d97e; color: #1a1a2e; border: none;
    padding: 14px 32px; border-radius: 10px; font-size: 1rem;
    font-weight: 600; cursor: pointer; width: 100%;
    transition: background 0.2s; margin-bottom: 12px;
  }
  button:hover { background: #00c06e; }
  button:disabled { background: #555; color: #999; cursor: not-allowed; }
  .back { background: transparent; color: #888; font-size: 0.85rem; border: 1px solid #333; }
  .back:hover { background: #1a1a2e; color: #ccc; }
  .error { color: #ff6b6b; margin-top: 16px; font-size: 0.85rem; }
  .success { color: #00d97e; margin-top: 16px; font-size: 0.85rem; }
  .icon { font-size: 3rem; margin-bottom: 16px; }
</style>
</head><body>
<div class="card">
  <div class="icon">&#x1F4F1;</div>
  <h1>Add New Device</h1>
  <p class="sub">Register a passkey for this device</p>
  <button id="regBtn" onclick="doRegister()">Register This Device</button>
  <button id="inviteBtn" onclick="genInvite()">Generate Invite Link (5 min)</button>
  <button class="back" onclick="window.location.href='/'">Back to App</button>
  <p class="error" id="error"></p>
  <p class="success" id="success"></p>
  <p id="invite" style="margin-top:16px;word-break:break-all;font-size:0.8rem;color:#aaa;"></p>
</div>
<script>
async function doRegister() {
  const btn = document.getElementById('regBtn');
  const err = document.getElementById('error');
  const ok = document.getElementById('success');
  err.textContent = ''; ok.textContent = '';
  btn.disabled = true;
  btn.textContent = 'Waiting for biometric...';

  try {
    const optRes = await fetch('/auth/register-options', { method: 'POST' });
    if (!optRes.ok) throw new Error('Failed to get registration options');
    const options = await optRes.json();

    options.challenge = b64urlToBuffer(options.challenge);
    options.user.id = b64urlToBuffer(options.user.id);
    if (options.excludeCredentials) {
      options.excludeCredentials = options.excludeCredentials.map(c => ({
        ...c, id: b64urlToBuffer(c.id)
      }));
    }

    const credential = await navigator.credentials.create({ publicKey: options });
    const body = {
      id: credential.id,
      rawId: bufferToB64url(credential.rawId),
      type: credential.type,
      response: {
        attestationObject: bufferToB64url(credential.response.attestationObject),
        clientDataJSON: bufferToB64url(credential.response.clientDataJSON),
      }
    };

    const verifyRes = await fetch('/auth/register-verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!verifyRes.ok) {
      const d = await verifyRes.json();
      throw new Error(d.detail || 'Registration failed');
    }

    ok.textContent = 'Device registered! Redirecting...';
    setTimeout(() => window.location.href = '/', 1500);
  } catch(e) {
    err.textContent = e.message || 'Registration failed';
    btn.textContent = 'Register This Device';
    btn.disabled = false;
  }
}

function renderQR(svgMarkup) {
  return '<div style="margin:16px auto;display:inline-block;background:#fff;padding:12px;border-radius:8px">' + svgMarkup + '</div>';
}

async function genInvite() {
  const btn = document.getElementById('inviteBtn');
  const inv = document.getElementById('invite');
  btn.disabled = true;
  try {
    const res = await fetch('/auth/create-invite', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed');
    const qr = renderQR(data.qr_svg);
    inv.innerHTML = qr + '<br><a href="' + data.url + '" style="color:#555;font-size:0.7rem">' + data.url + '</a>';
  } catch(e) {
    document.getElementById('error').textContent = e.message;
  }
  btn.disabled = false;
}

function b64urlToBuffer(b64url) {
  const b64 = b64url.replace(/-/g, '+').replace(/_/g, '/');
  const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4));
  const bin = atob(b64 + pad);
  return Uint8Array.from(bin, c => c.charCodeAt(0)).buffer;
}
function bufferToB64url(buf) {
  const bytes = new Uint8Array(buf);
  let bin = '';
  bytes.forEach(b => bin += String.fromCharCode(b));
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# API Endpoints (to be mounted on the FastAPI app)
# ---------------------------------------------------------------------------

def mount_auth_routes(app):
    """Register all auth-related routes on the FastAPI app."""

    @app.get("/login")
    async def login_page():
        return HTMLResponse(LOGIN_PAGE)

    @app.get("/auth/status")
    async def auth_status():
        return JSONResponse({"has_credentials": _has_credentials()})

    @app.get("/auth/add-device")
    async def add_device_page(request: Request):
        """Register a new device — requires existing authenticated session."""
        if not is_authenticated(request):
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/login", status_code=302)
        return HTMLResponse(ADD_DEVICE_PAGE)

    @app.post("/auth/create-invite")
    async def create_invite(request: Request):
        """Generate a one-time invite link for registering a new device."""
        if not is_authenticated(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        token = secrets.token_urlsafe(32)
        _invite_tokens[token] = time.time() + INVITE_TTL
        # Clean expired
        now = time.time()
        for k in [k for k, v in _invite_tokens.items() if v < now]:
            del _invite_tokens[k]
        origin = _get_origin(request)
        url = f"{origin}/auth/invite?token={token}"
        print(f"[Auth] Invite link created, expires in {INVITE_TTL}s")
        # Generate QR code SVG
        qr_svg = _generate_qr_svg(url)
        return JSONResponse({"url": url, "expires_in": INVITE_TTL, "qr_svg": qr_svg})

    @app.get("/auth/invite")
    async def invite_page(request: Request):
        """One-time invite registration page for new devices."""
        token = request.query_params.get("token", "")
        expiry = _invite_tokens.get(token)
        if not expiry or time.time() > expiry:
            return HTMLResponse("<html><body style='background:#1a1a2e;color:#ff6b6b;display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui'><h1>Invalid or expired invite link</h1></body></html>", status_code=403)
        # Serve a registration page that passes the token
        page = LOGIN_PAGE.replace(
            "let isRegister = false;",
            "let isRegister = true; const inviteToken = '" + token + "';"
        ).replace(
            "const res = await fetch('/auth/status');",
            "// invite mode"
        ).replace(
            "const data = await res.json();",
            "const data = {has_credentials: false};"
        ).replace(
            "btn.textContent = isRegister ? 'Register Touch ID' : 'Authenticate with Touch ID';",
            "btn.textContent = 'Register This Device';"
        ).replace(
            "async function register() {",
            "async function register() { const tokenParam = inviteToken;"
        ).replace(
            "const optRes = await fetch('/auth/register-options', { method: 'POST' });",
            "const optRes = await fetch('/auth/register-options?invite=' + tokenParam, { method: 'POST' });"
        ).replace(
            "const verifyRes = await fetch('/auth/register-verify',",
            "const verifyRes = await fetch('/auth/register-verify?invite=' + tokenParam,"
        ).replace(
            "Kismet — Authenticate",
            "Kismet — Device Registration"
        )
        return HTMLResponse(page)

    @app.post("/auth/register-options")
    async def register_options(request: Request):
        rp_id = _get_rp_id(request)
        user_id = secrets.token_bytes(16)

        options = generate_registration_options(
            rp_id=rp_id,
            rp_name=RP_NAME,
            user_id=user_id,
            user_name="kismet-user",
            user_display_name="Kismet User",
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
        )

        # Store challenge for verification
        challenge_b64 = bytes_to_base64url(options.challenge)
        _pending_challenges[challenge_b64] = time.time()

        # Also store user_id for registration verification
        _pending_challenges[f"uid_{challenge_b64}"] = bytes_to_base64url(user_id)

        return JSONResponse(json.loads(options_to_json(options)))

    @app.post("/auth/register-verify")
    async def register_verify(request: Request):
        body = await request.json()
        rp_id = _get_rp_id(request)
        origin = _get_origin(request)

        # Find and consume the challenge
        # The challenge is embedded in clientDataJSON; we need to verify any pending challenge
        try:
            verification = verify_registration_response(
                credential=body,
                expected_challenge=_consume_challenge(),
                expected_rp_id=rp_id,
                expected_origin=origin,
            )
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=400)

        # Store credential
        cred_id_b64 = bytes_to_base64url(verification.credential_id)

        # Find the user_id from pending challenges
        user_id_b64 = "kismet-user"
        for k, v in list(_pending_challenges.items()):
            if k.startswith("uid_") and isinstance(v, str):
                user_id_b64 = v
                del _pending_challenges[k]
                break

        creds = _load_credentials()
        creds.append({
            "credential_id": cred_id_b64,
            "public_key": bytes_to_base64url(verification.credential_public_key),
            "sign_count": verification.sign_count,
            "user_id": user_id_b64,
        })
        _save_credentials(creds)
        print(f"[Auth] WebAuthn credential registered: {cred_id_b64[:16]}...")

        # Consume invite token if used
        invite = request.query_params.get("invite", "")
        if invite and invite in _invite_tokens:
            del _invite_tokens[invite]
            print("[Auth] Invite token consumed")

        # Set session cookie
        response = JSONResponse({"status": "ok"})
        response.set_cookie(
            SESSION_COOKIE,
            create_session_cookie(user_id_b64),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return response

    @app.post("/auth/login-options")
    async def login_options(request: Request):
        rp_id = _get_rp_id(request)
        creds = _load_credentials()

        allow_credentials = [
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(c["credential_id"]),
            )
            for c in creds
        ]

        options = generate_authentication_options(
            rp_id=rp_id,
            allow_credentials=allow_credentials,
            user_verification=UserVerificationRequirement.PREFERRED,
        )

        challenge_b64 = bytes_to_base64url(options.challenge)
        _pending_challenges[challenge_b64] = time.time()

        return JSONResponse(json.loads(options_to_json(options)))

    @app.post("/auth/login-verify")
    async def login_verify(request: Request):
        body = await request.json()
        rp_id = _get_rp_id(request)
        origin = _get_origin(request)

        cred_id_b64 = body.get("id", "")
        stored = _find_credential(cred_id_b64)
        if not stored:
            return JSONResponse({"detail": "Unknown credential"}, status_code=400)

        try:
            verification = verify_authentication_response(
                credential=body,
                expected_challenge=_consume_challenge(),
                expected_rp_id=rp_id,
                expected_origin=origin,
                credential_public_key=base64url_to_bytes(stored["public_key"]),
                credential_current_sign_count=stored["sign_count"],
            )
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=400)

        # Update sign count
        creds = _load_credentials()
        for c in creds:
            if c["credential_id"] == cred_id_b64:
                c["sign_count"] = verification.new_sign_count
                break
        _save_credentials(creds)

        print(f"[Auth] WebAuthn login successful: {cred_id_b64[:16]}...")

        response = JSONResponse({"status": "ok"})
        response.set_cookie(
            SESSION_COOKIE,
            create_session_cookie(stored["user_id"]),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return response


def _consume_challenge() -> bytes:
    """Return the most recent pending challenge and clean up old ones."""
    # Clean up challenges older than 5 minutes
    now = time.time()
    expired = [k for k, v in _pending_challenges.items()
               if isinstance(v, float) and now - v > 300]
    for k in expired:
        del _pending_challenges[k]

    # Return the most recent challenge
    challenges = [(k, v) for k, v in _pending_challenges.items()
                  if isinstance(v, float)]
    if not challenges:
        raise ValueError("No pending challenge found")

    challenges.sort(key=lambda x: x[1], reverse=True)
    challenge_b64 = challenges[0][0]
    del _pending_challenges[challenge_b64]

    return base64url_to_bytes(challenge_b64)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Paths that don't require authentication
_PUBLIC_PATHS = {"/login", "/auth/status", "/auth/login-options",
                 "/auth/login-verify", "/auth/add-device",
                 "/auth/create-invite", "/auth/invite"}
# Registration paths are public ONLY when no credentials exist (first-time setup)
_REGISTRATION_PATHS = {"/auth/register-options", "/auth/register-verify"}


def is_authenticated(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return False
    return verify_session_cookie(cookie) is not None


def is_ws_authenticated(ws: WebSocket) -> bool:
    cookie = ws.cookies.get(SESSION_COOKIE)
    if not cookie:
        return False
    return verify_session_cookie(cookie) is not None


async def auth_middleware(request: Request, call_next):
    """ASGI middleware that redirects unauthenticated requests to /login."""
    path = request.url.path

    # Allow public paths
    if path in _PUBLIC_PATHS:
        return await call_next(request)

    # Registration endpoints: allowed during first-time setup, with auth, or with valid invite token
    if path in _REGISTRATION_PATHS:
        invite = request.query_params.get("invite", "")
        invite_valid = invite and invite in _invite_tokens and time.time() < _invite_tokens.get(invite, 0)
        if not _has_credentials() or is_authenticated(request) or invite_valid:
            return await call_next(request)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # Check session
    if is_authenticated(request):
        return await call_next(request)

    # For API/asset requests, return 401
    if path.startswith("/assets/") or path.startswith("/api/"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # Redirect to login page
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/login", status_code=302)
