import os
import json
import uuid
import logging
from typing import Optional, List
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone

import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow
import jupyter_kernel_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- OAuth Config ----
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

os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# ---- OAuth Helpers ----
def get_auth_url(code_challenge: str, code_challenge_method: str = "S256") -> str:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES, redirect_uri=REMOTE_REDIRECT_URI)
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        token_usage="remote",
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method
    )
    return auth_url

def exchange_code(code: str, code_verifier: str) -> Credentials:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES, redirect_uri=REMOTE_REDIRECT_URI)
    flow.fetch_token(code=code, code_verifier=code_verifier)
    return flow.credentials

def refresh_if_needed(creds_data: dict) -> tuple[Credentials, Optional[dict]]:
    creds = Credentials.from_authorized_user_info(creds_data)
    updated = None
    if creds.expiry and isinstance(creds.expiry, datetime):
        expiry = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry.tzinfo is None else creds.expiry.astimezone(timezone.utc)
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
                logger.info("Credentials successfully refreshed.")
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}")
    return creds, updated

# ---- Colab Tunnel Client ----
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
            raise RuntimeError(f"Colab API Error ({resp.status_code}): {resp.text}")
        
        body = strip_xssi(resp.text)
        return json.loads(body) if body else {}

    def assign(self, variant: str = "DEFAULT", accelerator: str = "NONE") -> dict:
        params = {
            "nbh": uuid_to_web_safe_base64(uuid.uuid4()),
            "nsa": "1",
            "variant": variant,
            "accelerator": accelerator,
        }
        get_resp = self._request(f"{TUN_ENDPOINT}/assign", params=params)
        if "endpoint" in get_resp:
            return get_resp
        
        token = get_resp.get("token")
        if not token:
            raise RuntimeError("Failed to retrieve XSRF token for session assignment.")
        
        return self._request(
            f"{TUN_ENDPOINT}/assign",
            method="POST",
            params=params,
            headers={"X-Goog-Colab-Token": token}
        )

    def unassign(self, endpoint: str):
        resp = self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}")
        token = resp.get("token")
        if token:
            self._request(
                f"{TUN_ENDPOINT}/unassign/{endpoint}",
                method="POST",
                headers={"X-Goog-Colab-Token": token}
            )

    def keep_alive(self, endpoint: str):
        url = f"{TUN_ENDPOINT}/{endpoint}/keep-alive/"
        try:
            self._request(url, method="GET", headers={"X-Colab-Tunnel": "Google"})
        except requests.exceptions.ReadTimeout:
            pass  # ReadTimeout is expected during long pings

# ---- Enhanced Jupyter Runtime Client ----
class ColabRuntime:
    def __init__(self, url: str, token: str, kernel_id: Optional[str] = None, session_id: Optional[str] = None):
        self.url = url
        self.token = token
        self.kernel_id = kernel_id
        self.session_id = session_id or str(uuid.uuid4())
        self._client: Optional[jupyter_kernel_client.ColabKernelClient] = None

    def _connect(self) -> jupyter_kernel_client.ColabKernelClient:
        if self._client:
            return self._client

        kwargs = {
            "server_url": self.url,
            "token": self.token,
            "session": self.session_id,
            "headers": {
                "X-Colab-Client-Agent": "colab-api",
                "X-Colab-Runtime-Proxy-Token": self.token,
            },
        }
        if self.kernel_id:
            kwargs["kernel_id"] = self.kernel_id

        client = jupyter_kernel_client.ColabKernelClient(**kwargs)
        client.start()

        if not self.kernel_id:
            self.kernel_id = getattr(client, "kernel_id", None)

        self._client = client
        return client

    def execute(self, code: str, timeout: Optional[float] = None) -> list:
        client = self._connect()
        outputs = []

        def _hook(msg: dict) -> None:
            mtype = msg.get("msg_type", "")
            content = msg.get("content", {})

            if mtype == "stream":
                outputs.append({"type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")})
            elif mtype == "execute_result":
                outputs.append({"type": "execute_result", "data": content.get("data", {})})
            elif mtype == "display_data":
                outputs.append({"type": "display_data", "data": content.get("data", {})})
            elif mtype == "error":
                outputs.append({"type": "error", "ename": content.get("ename", ""), "evalue": content.get("evalue", ""), "traceback": content.get("traceback", [])})

        client.execute_interactive(code, timeout=timeout, output_hook=_hook, allow_stdin=False)
        return outputs

    def disconnect(self):
        if not self._client:
            return
        try:
            self._client.stop()
        except Exception:
            try:
                self._client.stop_channels()
            except Exception:
                pass
        self._client = None

# ---- FastAPI App ----
app = FastAPI(title="Colab Standalone API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---- Pydantic Models ----
class CredentialsModel(BaseModel):
    token: str
    refresh_token: str
    expiry: Optional[str] = None
    scopes: List[str]
    token_uri: str
    client_id: str
    client_secret: str

class CreateSessionRequest(BaseModel):
    credentials: CredentialsModel

class ExecuteRequest(BaseModel):
    credentials: CredentialsModel
    url: str
    token: str
    kernel_id: Optional[str] = None
    session_id: Optional[str] = None
    code: str
    timeout: Optional[float] = None

class DeleteSessionRequest(BaseModel):
    credentials: CredentialsModel
    token: str
    url: str

class KeepAliveRequest(BaseModel):
    credentials: CredentialsModel

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
def create_session(req: CreateSessionRequest, gpu: Optional[str] = None, tpu: Optional[str] = None):
    creds_data = req.credentials.model_dump()
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
        accelerator = mapping.get(gpu.lower(), "T4")

    res = client.assign(variant=variant, accelerator=accelerator)
    endpoint = res["endpoint"]
    
    proxy_info = res.get("runtimeProxyInfo", {})
    token = proxy_info.get("token") or res.get("runtime_proxy_token")
    url = proxy_info.get("url") or res.get("url")

    response = {"endpoint": endpoint, "token": token, "url": url, "variant": variant, "accelerator": accelerator}
    if updated:
        response["updated_credentials"] = updated
    return response

@app.post("/sessions/{endpoint}/execute")
def execute(endpoint: str, req: ExecuteRequest):
    creds_data = req.credentials.model_dump()
    _, updated = refresh_if_needed(creds_data)

    runtime = ColabRuntime(url=req.url, token=req.token, kernel_id=req.kernel_id, session_id=req.session_id)
    
    outputs = []
    try:
        outputs = runtime.execute(req.code, timeout=req.timeout)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if not any(o.get("type") == "error" for o in outputs):
            outputs.append({"type": "error", "ename": "ExecutionException", "evalue": str(e), "traceback": []})
    finally:
        runtime.disconnect()

    response = {"outputs": outputs, "kernel_id": runtime.kernel_id, "session_id": runtime.session_id}
    if updated:
        response["updated_credentials"] = updated
    return response

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive(endpoint: str, req: KeepAliveRequest):
    creds_data = req.credentials.model_dump()
    creds, updated = refresh_if_needed(creds_data)
    sess = AuthorizedSession(creds)
    client = ColabClient(sess)
    client.keep_alive(endpoint)
    
    response = {"status": "ok"}
    if updated:
        response["updated_credentials"] = updated
    return response

@app.delete("/sessions/{endpoint}")
def delete_session(endpoint: str, req: DeleteSessionRequest):
    creds_data = req.credentials.model_dump()
    creds, _ = refresh_if_needed(creds_data)
    sess = AuthorizedSession(creds)
    client = ColabClient(sess)
    client.unassign(endpoint)
    return {"message": f"Session {endpoint} terminated successfully."}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
