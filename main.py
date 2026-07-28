import os
import json
import uuid
import base64
import tempfile
import logging
import time
from typing import Optional, List
from urllib.parse import urljoin, quote
from datetime import datetime, timedelta, timezone

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow
import jupyter_kernel_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- OAuth config ----
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
CLIENT_CONFIG = {
    "installed": {
        "client_id": os.environ.get("OAUTH_CLIENT_ID"),
        "project_id": os.environ.get("OAUTH_PROJECT_ID", "your-project-id"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": os.environ.get("OAUTH_CLIENT_SECRET"),
        "redirect_uris": [REMOTE_REDIRECT_URI],
    }
}
if not CLIENT_CONFIG["installed"]["client_id"] or not CLIENT_CONFIG["installed"]["client_secret"]:
    raise RuntimeError("Missing OAuth credentials")

os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# ---- OAuth functions ----
def get_auth_url(code_challenge: str, code_challenge_method: str = "S256") -> str:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES, redirect_uri=REMOTE_REDIRECT_URI)
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote",
                                         code_challenge=code_challenge, code_challenge_method=code_challenge_method)
    return auth_url

def exchange_code(code: str, code_verifier: str) -> Credentials:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES, redirect_uri=REMOTE_REDIRECT_URI)
    flow.fetch_token(code=code, code_verifier=code_verifier)
    return flow.credentials

# ---- Credentials helpers ----
def creds_from_dict(data: dict) -> Credentials:
    return Credentials.from_authorized_user_info(data)

def refresh_if_needed(creds_data: dict) -> tuple[Credentials, Optional[dict]]:
    creds = creds_from_dict(creds_data)
    updated = None
    if creds.expiry and isinstance(creds.expiry, datetime):
        if creds.expiry.tzinfo is None:
            expiry = creds.expiry.replace(tzinfo=timezone.utc)
        else:
            expiry = creds.expiry.astimezone(timezone.utc)
        if expiry - datetime.now(timezone.utc) < timedelta(minutes=5):
            try:
                creds.refresh(Request())
                updated = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "expiry": creds.expiry.isoformat() if creds.expiry else None,
                    "scopes": creds.scopes,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                }
                logger.info("Token refreshed.")
            except Exception as e:
                logger.warning(f"Refresh failed: {e}")
    return creds, updated

# ---- FastAPI app ----
app = FastAPI(title="Colab API (clean)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---- Models ----
class CredentialsModel(BaseModel):
    token: str
    refresh_token: str
    expiry: Optional[str] = None
    scopes: List[str]
    token_uri: str
    client_id: str
    client_secret: str

class SessionContext(BaseModel):
    credentials: CredentialsModel
    endpoint: str
    token: str
    url: str
    kernel_id: Optional[str] = None
    session_id: Optional[str] = None

class ExecuteRequest(SessionContext):
    code: str
    timeout: Optional[float] = 30
    allow_stdin: bool = False

# ---- Colab tunnel client (standalone) ----
XSSI_PREFIX = ")]}'\n"
TUN_ENDPOINT = "/tun/m"

def uuid_to_web_safe_base64(uuid_val: uuid.UUID) -> str:
    return str(uuid_val).replace("-", "_") + "." * (44 - len(str(uuid_val)))

def strip_xssi(text: str) -> str:
    return text[len(XSSI_PREFIX):] if text.startswith(XSSI_PREFIX) else text

class ColabClient:
    def __init__(self, session: AuthorizedSession):
        self.session = session
        self.domain = "https://colab.research.google.com"

    def _request(self, endpoint: str, method="GET", headers=None, params=None, json_body=None) -> dict:
        url = urljoin(self.domain, endpoint)
        params = params or {}
        params["authuser"] = "0"
        req_headers = {"Accept": "application/json", "X-Colab-Client-Agent": "colab-api"}
        if headers:
            req_headers.update(headers)
        resp = self.session.request(method, url, headers=req_headers, params=params, json=json_body)
        if not resp.ok:
            raise RuntimeError(f"Colab API error {resp.status_code}: {resp.text}")
        body = strip_xssi(resp.text)
        return json.loads(body) if body else {}

    def assign(self, variant: str = None, accelerator: str = None) -> dict:
        params = {"nbh": uuid_to_web_safe_base64(uuid.uuid4())}
        if variant:
            params["variant"] = variant
        if accelerator:
            params["accelerator"] = accelerator
        get_resp = self._request(f"{TUN_ENDPOINT}/assign", params=params)
        if "endpoint" in get_resp:
            return get_resp
        token = get_resp.get("token")
        if not token:
            raise RuntimeError("No XSRF token")
        return self._request(f"{TUN_ENDPOINT}/assign", method="POST", params=params, headers={"X-Goog-Colab-Token": token})

    def unassign(self, endpoint: str):
        resp = self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}")
        token = resp.get("token")
        if token:
            self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}", method="POST", headers={"X-Goog-Colab-Token": token})

    def keep_alive(self, endpoint: str):
        url = f"{TUN_ENDPOINT}/{endpoint}/keep-alive/"
        try:
            self._request(url, headers={"X-Colab-Tunnel": "Google"}, method="GET")
        except requests.exceptions.ReadTimeout:
            pass  # normal

# ---- Jupyter runtime (standalone) ----
class ColabRuntime:
    def __init__(self, url: str, token: str, kernel_id: str = None, session_id: str = None):
        self.url = url
        self.token = token
        self.kernel_id = kernel_id
        self.session_id = session_id
        self._client = None

    @property
    def client(self):
        if not self._client:
            client_kwargs = {
                "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                "extra_params": {"colab-runtime-proxy-token": self.token},
            }
            if self.session_id:
                client_kwargs["session"] = self.session_id
            self._client = jupyter_kernel_client.KernelClient(
                server_url=self.url,
                token=self.token,
                kernel_id=self.kernel_id,
                client_kwargs=client_kwargs,
                headers={"X-Colab-Client-Agent": "colab-api", "X-Colab-Runtime-Proxy-Token": self.token},
            )
            self._client._own_kernel = False
            self._client.start()
        return self._client

    def execute(self, code: str, output_hook=None, timeout: float = 30, allow_stdin: bool = False) -> list:
        kwargs = {"allow_stdin": allow_stdin, "timeout": timeout}
        if output_hook:
            outputs = []
            def hook(msg):
                from jupyter_kernel_client.client import output_hook as default_hook
                new_idx = default_hook(outputs, msg)
                for i in new_idx:
                    if i < len(outputs):
                        output_hook(outputs[i])
            self.client.execute_interactive(code, output_hook=hook, **kwargs)
            return outputs
        else:
            return self.client.execute(code, **kwargs).get("outputs", [])

    def stop(self):
        if self._client:
            self._client._manager.client.stop_channels()

# ---- Endpoints ----
@app.get("/auth/url")
def auth_url(code_challenge: str, code_challenge_method: str = "S256"):
    return {"auth_url": get_auth_url(code_challenge, code_challenge_method)}

@app.post("/auth/token")
def auth_token(data: dict = Body(...)):
    code, verifier = data.get("code"), data.get("code_verifier")
    if not code or not verifier:
        raise HTTPException(400, "Missing code or code_verifier")
    creds = exchange_code(code, verifier)
    return {"credentials": json.loads(creds.to_json())}

@app.post("/sessions")
def create_session(credentials: CredentialsModel, gpu: Optional[str] = None, tpu: Optional[str] = None):
    creds_data = credentials.model_dump()
    creds, updated = refresh_if_needed(creds_data)
    sess = AuthorizedSession(creds)
    client = ColabClient(sess)

    variant = "DEFAULT"
    accelerator = "NONE"
    if tpu:
        variant = "TPU"
        accelerator = "V5E1" if tpu.lower() == "v5e1" else "V6E1"
    elif gpu:
        variant = "GPU"
        mapping = {"a100": "A100", "h100": "H100", "l4": "L4", "t4": "T4", "g4": "G4"}
        accelerator = mapping.get(gpu.lower(), "A100")

    res = client.assign(variant=variant, accelerator=accelerator)
    endpoint = res["endpoint"]
    token = res.get("runtimeProxyInfo", {}).get("token") or res.get("runtime_proxy_token")
    url = res.get("runtimeProxyInfo", {}).get("url") or res.get("url")
    response = {
        "endpoint": endpoint,
        "token": token,
        "url": url,
        "variant": variant,
        "accelerator": accelerator,
    }
    if updated:
        response["updated_credentials"] = updated
    return response

@app.post("/sessions/{endpoint}/execute")
def execute(endpoint: str, req: ExecuteRequest):
    creds_data = req.credentials.model_dump()
    creds, updated = refresh_if_needed(creds_data)
    if updated:
        req.credentials = CredentialsModel(**updated)  # update for runtime
    runtime = ColabRuntime(req.url, req.token, kernel_id=req.kernel_id, session_id=req.session_id)
    outputs = []
    def hook(out): outputs.append(out)
    try:
        runtime.execute(req.code, output_hook=hook, timeout=req.timeout, allow_stdin=req.allow_stdin)
    finally:
        runtime.stop()
    response = {"outputs": outputs}
    if updated:
        response["updated_credentials"] = updated
    return response

@app.delete("/sessions/{endpoint}")
def delete_session(endpoint: str, credentials: CredentialsModel, token: str, url: str):
    creds_data = credentials.model_dump()
    creds, _ = refresh_if_needed(creds_data)
    sess = AuthorizedSession(creds)
    client = ColabClient(sess)
    client.unassign(endpoint)
    return {"message": "Deleted"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
