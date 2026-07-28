import os
import json
import uuid
import base64
import tempfile
import urllib.request
import logging
import time
from typing import Optional, List
from urllib.parse import urljoin, urlparse, quote

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow
import jupyter_kernel_client

# ---- Setup logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# 1. Constants & Helpers
# --------------------------------------------------------------------

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
CLIENT_CONFIG = {
    "installed": {
        "client_id": os.environ.get("OAUTH_CLIENT_ID"),
        "project_id": os.environ.get("OAUTH_PROJECT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": os.environ.get("OAUTH_CLIENT_SECRET"),
        "redirect_uris": [REMOTE_REDIRECT_URI],
    }
}
logger.info("Using client_id: %s", CLIENT_CONFIG["installed"]["client_id"])
logger.info("Redirect URI configured: %s", REMOTE_REDIRECT_URI)

XSSI_PREFIX = ")]}'\n"
TUN_ENDPOINT = "/tun/m"
KEEP_ALIVE_TIMEOUT = 10
COLAB_TUNNEL_HEADER = {"X-Colab-Tunnel": "Google"}
COLAB_CLIENT_AGENT_HEADER = {"X-Colab-Client-Agent": "colab-api"}
ACCEPT_JSON_HEADER = {"Accept": "application/json"}

def uuid_to_web_safe_base64(uuid_val: uuid.UUID) -> str:
    s = str(uuid_val).replace("-", "_")
    return s + "." * (44 - len(s))

def strip_xssi(text: str) -> str:
    if text.startswith(XSSI_PREFIX):
        return text[len(XSSI_PREFIX):]
    return text

class Accelerator:
    NONE = "NONE"
    G4 = "G4"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    V5E1 = "V5E1"
    V6E1 = "V6E1"

class Variant:
    DEFAULT = "DEFAULT"
    GPU = "GPU"
    TPU = "TPU"

def resolve_accelerator(gpu: Optional[str], tpu: Optional[str]):
    logger.debug("resolve_accelerator(gpu=%s, tpu=%s)", gpu, tpu)
    if tpu:
        variant = Variant.TPU
        acc = Accelerator.V5E1 if tpu.lower() == "v5e1" else Accelerator.V6E1
        logger.info("Resolved to TPU variant %s, accelerator %s", variant, acc)
        return variant, acc
    if gpu:
        mapping = {
            "a100": Accelerator.A100,
            "h100": Accelerator.H100,
            "l4": Accelerator.L4,
            "t4": Accelerator.T4,
            "g4": Accelerator.G4,
        }
        variant = Variant.GPU
        acc = mapping.get(gpu.lower(), Accelerator.A100)
        logger.info("Resolved to GPU variant %s, accelerator %s", variant, acc)
        return variant, acc
    logger.info("No accelerator requested, using CPU (DEFAULT variant, NONE accelerator)")
    return Variant.DEFAULT, Accelerator.NONE

# --------------------------------------------------------------------
# 2. OAuth (with logging)
# --------------------------------------------------------------------

_flow: Optional[InstalledAppFlow] = None

def get_auth_url() -> str:
    global _flow
    logger.info("Creating OAuth flow with redirect_uri='%s'", REMOTE_REDIRECT_URI)
    _flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    _flow.redirect_uri = REMOTE_REDIRECT_URI
    logger.info("Generating authorization URL...")
    auth_url, _ = _flow.authorization_url(prompt="consent", token_usage="remote")
    logger.info("Auth URL generated (length: %d)", len(auth_url))
    logger.debug("Auth URL: %s", auth_url)
    return auth_url

def exchange_code(code: str) -> Credentials:
    global _flow
    if _flow is None:
        logger.error("No OAuth flow initiated. Call get_auth_url() first.")
        raise RuntimeError("No OAuth flow initiated. Call get_auth_url() first.")
    logger.info("Exchanging authorization code (first 10 chars: %s...)", code[:10])
    try:
        _flow.fetch_token(code=code)
        creds = _flow.credentials
        logger.info("Token exchange successful. Access token length: %d", len(creds.token))
        logger.debug("Token expiry: %s", creds.expiry)
        logger.debug("Scopes: %s", creds.scopes)
        return creds
    except Exception as e:
        logger.error("Token exchange failed: %s", str(e), exc_info=True)
        raise RuntimeError(f"Failed to exchange code: {e}")
    finally:
        _flow = None
        logger.info("Flow cleared after exchange attempt.")

# --------------------------------------------------------------------
# 3. Colab Tunnel Client
# --------------------------------------------------------------------

class ColabClient:
    def __init__(self, session: AuthorizedSession):
        self.session = session
        self.domain = "https://colab.research.google.com"
        self.api_domain = "https://colab.pa.googleapis.com"
        logger.info("ColabClient initialized with domain %s", self.domain)

    def _request(self, endpoint: str, method: str = "GET", headers: dict = None,
                 params: dict = None, json_body: dict = None) -> dict:
        url = urljoin(self.domain, endpoint)
        if params is None:
            params = {}
        params["authuser"] = "0"
        req_headers = ACCEPT_JSON_HEADER.copy()
        req_headers.update(COLAB_CLIENT_AGENT_HEADER)
        if headers:
            req_headers.update(headers)

        logger.info("ColabClient._request: %s %s", method, url)
        logger.debug("Params: %s", params)
        logger.debug("Headers: %s", req_headers)
        if json_body:
            logger.debug("JSON body: %s", json_body)

        resp = self.session.request(method, url, headers=req_headers,
                                    params=params, json=json_body)
        logger.info("ColabClient response status: %d", resp.status_code)
        if not resp.ok:
            logger.error("Colab API error %d: %s", resp.status_code, resp.text)
            raise RuntimeError(f"Colab API error {resp.status_code}: {resp.text}")
        body = strip_xssi(resp.text)
        if body:
            try:
                data = json.loads(body)
                logger.debug("Response JSON: %s", data)
                return data
            except json.JSONDecodeError:
                logger.warning("Response body not JSON: %s", body[:200])
                return {}
        logger.debug("Empty response body")
        return {}

    def list_assignments(self) -> list:
        logger.info("Listing assignments")
        data = self._request(f"{TUN_ENDPOINT}/assignments")
        assignments = data.get("assignments", [])
        logger.info("Found %d assignments", len(assignments))
        return assignments

    def assign(self, notebook_hash: uuid.UUID, variant: str = None, accelerator: str = None) -> dict:
        logger.info("Assigning notebook_hash=%s, variant=%s, accelerator=%s",
                    notebook_hash, variant, accelerator)
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant:
            params["variant"] = variant
        if accelerator:
            params["accelerator"] = accelerator

        # GET to get XSRF token and check if assignment already exists
        logger.info("GET /assign to get XSRF token")
        get_resp = self._request(f"{TUN_ENDPOINT}/assign", params=params)
        if "endpoint" in get_resp:
            logger.info("Assignment already exists: endpoint=%s", get_resp.get("endpoint"))
            return get_resp
        token = get_resp.get("token")
        if not token:
            logger.error("No XSRF token in GET response")
            raise RuntimeError("No XSRF token in GET response")
        logger.info("XSRF token obtained (length: %d)", len(token))
        headers = {"X-Goog-Colab-Token": token}
        logger.info("POST /assign to claim assignment")
        resp = self._request(f"{TUN_ENDPOINT}/assign", method="POST",
                             params=params, headers=headers)
        logger.info("Assignment response: endpoint=%s", resp.get("endpoint"))
        return resp

    def unassign(self, endpoint: str):
        logger.info("Unassigning endpoint: %s", endpoint)
        resp = self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}")
        token = resp.get("token")
        if not token:
            logger.warning("No token in unassign GET response, skipping POST")
            return
        headers = {"X-Goog-Colab-Token": token}
        self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}", method="POST",
                      headers=headers)
        logger.info("Unassign successful for endpoint %s", endpoint)

    def keep_alive(self, endpoint: str):
        url = f"{TUN_ENDPOINT}/{endpoint}/keep-alive/"
        headers = COLAB_TUNNEL_HEADER.copy()
        logger.info("Sending keep-alive for endpoint %s", endpoint)
        try:
            self._request(url, headers=headers, method="GET")
            logger.info("Keep-alive successful (or timeout handled)")
        except requests.exceptions.ReadTimeout:
            logger.info("Keep-alive request timed out, but activity was recorded (normal).")
        except Exception as e:
            logger.warning("Keep-alive failed: %s", str(e))

# --------------------------------------------------------------------
# 4. Colab Runtime (Jupyter Kernel)
# --------------------------------------------------------------------

class ColabRuntime:
    def __init__(self, url: str, token: str, kernel_id: str = None, session_id: str = None,
                 drive_hook: callable = None):
        self.url = url
        self.token = token
        self.kernel_id = kernel_id
        self.session_id = session_id
        self.drive_hook = drive_hook
        self._client = None
        logger.info("ColabRuntime initialized with url=%s, kernel_id=%s, session_id=%s",
                    url, kernel_id, session_id)

    @property
    def client(self):
        if self._client is None:
            logger.info("Creating Jupyter Kernel client")
            client_kwargs = {
                "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                "extra_params": {"colab-runtime-proxy-token": self.token},
            }
            if self.session_id:
                client_kwargs["session"] = self.session_id
                logger.info("Using existing session_id: %s", self.session_id)
            self._client = jupyter_kernel_client.KernelClient(
                server_url=self.url,
                token=self.token,
                kernel_id=self.kernel_id,
                client_kwargs=client_kwargs,
                headers={
                    "X-Colab-Client-Agent": "colab-api",
                    "X-Colab-Runtime-Proxy-Token": self.token,
                },
            )
            self._client._own_kernel = False
            self._client.start()
            logger.info("Kernel client started. Kernel ID: %s", self._client.id)
            if self.drive_hook:
                logger.info("Applying drive hook")
                self._apply_drive_hook()
        return self._client

    def _apply_drive_hook(self):
        wsclient = self._client._manager.client
        orig_on_message = wsclient.kernel_socket.on_message

        def hooked_on_message(ws, message):
            try:
                from jupyter_kernel_client.wsclient import deserialize_msg_from_ws_default, deserialize_msg_from_ws_v1
                if wsclient._subprotocol == jupyter_kernel_client.JupyterSubprotocol.DEFAULT:
                    msg = deserialize_msg_from_ws_default(message)
                else:
                    _, msg_list = deserialize_msg_from_ws_v1(message)
                    msg = wsclient.session.deserialize(msg_list)
                if msg and msg.get("msg_type") == "colab_request":
                    logger.debug("Intercepted colab_request message")
                    if self.drive_hook(msg, wsclient):
                        logger.info("Drive hook intercepted and handled colab_request")
                        return  # intercepted
            except Exception as e:
                logger.warning("Drive hook error: %s", str(e))
            orig_on_message(ws, message)

        wsclient.kernel_socket.on_message = hooked_on_message
        logger.info("Drive hook applied to WebSocket")

    def execute(self, code: str, output_hook: callable = None,
                timeout: float = 30, allow_stdin: bool = False) -> list:
        logger.info("Executing code (length: %d, timeout: %.1fs)", len(code), timeout)
        logger.debug("Code snippet: %s", code[:200] + ("..." if len(code) > 200 else ""))
        kwargs = {"allow_stdin": allow_stdin, "timeout": timeout}
        if output_hook:
            outputs = []
            def hook(msg):
                from jupyter_kernel_client.client import output_hook as default_hook
                new_idx = default_hook(outputs, msg)
                for i in new_idx:
                    if i < len(outputs):
                        output_hook(outputs[i])
            logger.info("Using interactive execution with output hook")
            self.client.execute_interactive(code, output_hook=hook, **kwargs)
            logger.info("Execution completed with %d outputs", len(outputs))
            return outputs
        else:
            logger.info("Using synchronous execution (no output hook)")
            reply = self.client.execute(code, **kwargs)
            outputs = reply.get("outputs", [])
            logger.info("Execution completed with %d outputs", len(outputs))
            return outputs

    def stop(self):
        if self._client:
            logger.info("Stopping kernel client")
            try:
                self._client._manager.client.stop_channels()
                logger.info("Kernel client stopped")
            except Exception as e:
                logger.warning("Error stopping kernel client: %s", str(e))

# --------------------------------------------------------------------
# 5. Contents Client (File Operations)
# --------------------------------------------------------------------

class ContentsClient:
    def __init__(self, url: str, token: str):
        self.base_url = url.rstrip("/")
        self.token = token
        logger.info("ContentsClient initialized with base_url=%s", self.base_url)

    def _request(self, method: str, path: str, json_body: dict = None) -> dict:
        quoted = quote(path.strip("/"), safe="/")
        url = f"{self.base_url}/api/contents/{quoted}"
        params = {"authuser": "0", "colab-runtime-proxy-token": self.token}
        logger.info("ContentsClient._request: %s %s", method, url)
        logger.debug("Params: %s", params)
        if json_body:
            logger.debug("JSON body: %s", json_body)

        resp = requests.request(method, url, params=params, json=json_body)
        logger.info("ContentsClient response status: %d", resp.status_code)
        if resp.status_code == 404:
            logger.warning("File/directory not found: %s", path)
            raise FileNotFoundError(path)
        if not resp.ok:
            logger.error("Contents API error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        if method == "DELETE":
            logger.info("Delete successful")
            return {}
        data = resp.json()
        logger.debug("Response data: %s", data)
        return data

    def list(self, path: str = "content") -> list:
        logger.info("Listing directory: %s", path)
        data = self._request("GET", path)
        files = data.get("content", [])
        logger.info("Found %d items", len(files))
        return files

    def upload(self, local_path: str, remote_path: str):
        logger.info("Uploading %s -> %s", local_path, remote_path)
        with open(local_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "name": os.path.basename(remote_path),
            "path": remote_path,
            "type": "file",
            "format": "base64",
            "content": content,
            "chunk": 1,
        }
        self._request("PUT", remote_path, json_body=payload)
        logger.info("Upload completed")

    def download(self, remote_path: str, local_path: str):
        logger.info("Downloading %s -> %s", remote_path, local_path)
        data = self._request("GET", remote_path)
        content = data.get("content", "")
        fmt = data.get("format", "text")
        if fmt == "base64":
            content_bytes = base64.b64decode(content)
        else:
            content_bytes = content.encode("utf-8")
        with open(local_path, "wb") as f:
            f.write(content_bytes)
        logger.info("Download completed (%d bytes)", len(content_bytes))

    def delete(self, remote_path: str):
        logger.info("Deleting %s", remote_path)
        self._request("DELETE", remote_path)
        logger.info("Delete completed")

# --------------------------------------------------------------------
# 6. Drive Auth Hook Factory
# --------------------------------------------------------------------

def make_drive_hook(credentials: Credentials, endpoint: str):
    session = AuthorizedSession(credentials)
    domain = "https://colab.research.google.com"
    logger.info("Creating drive hook for endpoint %s", endpoint)

    def hook(msg: dict, wsclient) -> bool:
        content = msg.get("content", {})
        req = content.get("request", {})
        if req.get("authType") != "dfs_ephemeral":
            logger.debug("Ignoring non-dfs_ephemeral request")
            return False
        logger.info("Intercepted dfs_ephemeral auth request")
        msg_id = msg.get("metadata", {}).get("colab_msg_id")
        url = f"{domain}/tun/m/credentials-propagation/{endpoint}"
        params = {
            "authuser": "0",
            "authtype": "dfs_ephemeral",
            "version": "2",
            "dryrun": "true",
            "propagate": "true",
            "record": "false",
        }
        logger.debug("Propagation GET URL: %s", url)
        resp = session.request("GET", url, params=params)
        if resp.status_code != 200:
            logger.error("Propagation GET failed: %d", resp.status_code)
            return False
        data = json.loads(strip_xssi(resp.text))
        token = data.get("token")
        if not token:
            logger.error("No token in propagation response")
            return False
        logger.info("Propagation token obtained (length: %d)", len(token))
        headers = {"x-goog-colab-token": token}
        params["dryrun"] = "false"
        resp = session.request("POST", url, params=params, headers=headers,
                               files={"file_id": (None, "empty.ipynb")})
        if resp.status_code != 200:
            logger.error("Propagation POST failed: %d", resp.status_code)
            return False
        logger.info("Credentials propagated successfully")
        # Send input_reply to resume
        reply = wsclient.session.msg(
            "input_reply",
            {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
        )
        if "header" in msg:
            reply["parent_header"] = msg["header"]
        wsclient.stdin_channel.send(reply)
        logger.info("input_reply sent to kernel")
        return True
    return hook

# --------------------------------------------------------------------
# 7. FastAPI Models
# --------------------------------------------------------------------

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

class AutomationRequest(SessionContext):
    timeout: Optional[float] = 600

class InstallRequest(SessionContext):
    packages: Optional[List[str]] = None
    requirement: Optional[str] = None
    timeout: Optional[float] = 600

# --------------------------------------------------------------------
# 8. FastAPI App with Middleware
# --------------------------------------------------------------------

app = FastAPI(title="Colab API (standalone)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        logger.info("Request: %s %s", request.method, request.url.path)
        if request.query_params:
            logger.info("Query params: %s", dict(request.query_params))
        try:
            body = await request.body()
            if body:
                logger.debug("Request body: %s", body[:500].decode(errors='ignore'))
        except Exception:
            pass
        response = await call_next(request)
        duration = time.time() - start
        logger.info("Response: %s (took %.3fs)", response.status_code, duration)
        return response

app.add_middleware(LoggingMiddleware)

# Helper to get session and runtime
def get_session_and_runtime(req: SessionContext, drive_hook_enabled=False):
    logger.info("Getting session and runtime for endpoint %s", req.endpoint)
    creds = Credentials.from_authorized_user_info(req.credentials.dict())
    sess = AuthorizedSession(creds)
    colab = ColabClient(sess)
    hook = None
    if drive_hook_enabled:
        hook = make_drive_hook(creds, req.endpoint)
    runtime = ColabRuntime(req.url, req.token,
                           kernel_id=req.kernel_id,
                           session_id=req.session_id,
                           drive_hook=hook)
    return colab, runtime

# --------------------------------------------------------------------
# 9. Endpoints
# --------------------------------------------------------------------

@app.get("/auth/url")
def auth_url_endpoint():
    logger.info("Handling /auth/url request")
    try:
        url = get_auth_url()
        logger.info("Returning auth URL")
        return {"auth_url": url}
    except Exception as e:
        logger.error("Failed to generate auth URL: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/auth/token")
def auth_token_endpoint(code: str):
    logger.info("Handling /auth/token request")
    try:
        creds = exchange_code(code)
        logger.info("Token exchange successful")
        return {"credentials": json.loads(creds.to_json())}
    except Exception as e:
        logger.error("Token exchange failed: %s", str(e), exc_info=True)
        raise HTTPException(400, str(e))

@app.post("/sessions")
def create_session_endpoint(
    credentials: CredentialsModel,
    name: Optional[str] = None,
    gpu: Optional[str] = None,
    tpu: Optional[str] = None,
):
    logger.info("Creating session with credentials (token length: %d), gpu=%s, tpu=%s",
                len(credentials.token), gpu, tpu)
    try:
        creds = Credentials.from_authorized_user_info(credentials.dict())
        sess = AuthorizedSession(creds)
        colab = ColabClient(sess)
        variant, accel = resolve_accelerator(gpu, tpu)
        logger.info("Resolved variant=%s, accelerator=%s", variant, accel)
        res = colab.assign(uuid.uuid4(), variant=variant, accelerator=accel)
        logger.info("Assignment response: %s", res)
        if "endpoint" in res:
            endpoint = res["endpoint"]
            token = res.get("runtime_proxy_info", {}).get("token")
            url = res.get("runtime_proxy_info", {}).get("url")
            if not token:
                token = res.get("runtime_proxy_token")
        else:
            endpoint = res.get("endpoint")
            token = res.get("runtime_proxy_token") or res.get("token")
            url = res.get("url")
        logger.info("Session created: endpoint=%s, token length=%d, url=%s",
                    endpoint, len(token) if token else 0, url)
        return {
            "endpoint": endpoint,
            "token": token,
            "url": url,
            "variant": variant,
            "accelerator": accel,
        }
    except Exception as e:
        logger.error("Session creation failed: %s", str(e), exc_info=True)
        raise HTTPException(400, str(e))

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive_endpoint(endpoint: str, credentials: CredentialsModel, token: str, url: str):
    logger.info("Keep-alive for endpoint %s", endpoint)
    try:
        creds = Credentials.from_authorized_user_info(credentials.dict())
        sess = AuthorizedSession(creds)
        colab = ColabClient(sess)
        colab.keep_alive(endpoint)
        logger.info("Keep-alive successful")
        return {"status": "ok"}
    except Exception as e:
        logger.error("Keep-alive failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/sessions/{endpoint}/execute")
def execute_endpoint(endpoint: str, req: ExecuteRequest):
    logger.info("Execute request for endpoint %s", endpoint)
    try:
        _, runtime = get_session_and_runtime(req)
        outputs = []
        def hook(out):
            outputs.append(out)
            logger.debug("Output hook received: %s", str(out)[:200])
        logger.info("Running code (timeout: %.1fs)", req.timeout)
        runtime.execute(req.code, output_hook=hook, timeout=req.timeout)
        logger.info("Execution completed with %d outputs", len(outputs))
        runtime.stop()
        return {"outputs": outputs}
    except Exception as e:
        logger.error("Execution failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.get("/sessions/{endpoint}/files")
def list_files_endpoint(endpoint: str, credentials: CredentialsModel, token: str, url: str, path: str = "content"):
    logger.info("List files endpoint for %s, path=%s", endpoint, path)
    try:
        contents = ContentsClient(url, token)
        files = contents.list(path)
        logger.info("Found %d files", len(files))
        return {"files": files}
    except Exception as e:
        logger.error("List files failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/sessions/{endpoint}/files")
def upload_file_endpoint(
    endpoint: str,
    credentials: CredentialsModel,
    token: str,
    url: str,
    remote_path: str = Form(...),
    file: UploadFile = File(...),
):
    logger.info("Upload file endpoint for %s, remote_path=%s", endpoint, remote_path)
    try:
        contents = ContentsClient(url, token)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = file.file.read()
            logger.info("Uploaded file size: %d bytes", len(content))
            tmp.write(content)
            local = tmp.name
        contents.upload(local, remote_path)
        os.unlink(local)
        logger.info("Upload successful")
        return {"message": "Uploaded"}
    except Exception as e:
        logger.error("Upload failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.get("/sessions/{endpoint}/files/{path:path}")
def download_file_endpoint(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    logger.info("Download file endpoint for %s, path=%s", endpoint, path)
    try:
        contents = ContentsClient(url, token)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local = tmp.name
        contents.download(path, local)
        logger.info("File downloaded to %s", local)
        return FileResponse(local, filename=os.path.basename(path))
    except Exception as e:
        logger.error("Download failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.delete("/sessions/{endpoint}/files/{path:path}")
def delete_file_endpoint(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    logger.info("Delete file endpoint for %s, path=%s", endpoint, path)
    try:
        contents = ContentsClient(url, token)
        contents.delete(path)
        logger.info("Delete successful")
        return {"message": "Deleted"}
    except Exception as e:
        logger.error("Delete failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

# ----- Automation endpoints -----
@app.post("/sessions/{endpoint}/automation/auth")
def run_auth_endpoint(endpoint: str, req: AutomationRequest):
    logger.info("Automation auth for endpoint %s", endpoint)
    code = (
        "import os\n"
        "os.environ['USE_AUTH_EPHEM'] = '0'\n"
        "from google.colab import auth\n"
        "auth.authenticate_user()\n"
    )
    try:
        _, runtime = get_session_and_runtime(req, drive_hook_enabled=True)
        outputs = []
        def hook(out):
            outputs.append(out)
            logger.debug("Output hook received: %s", str(out)[:200])
        logger.info("Running auth code with timeout %.1fs", req.timeout)
        runtime.execute(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
        logger.info("Auth automation completed with %d outputs", len(outputs))
        runtime.stop()
        return {"outputs": outputs}
    except Exception as e:
        logger.error("Auth automation failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/sessions/{endpoint}/automation/drivemount")
def run_drivemount_endpoint(endpoint: str, req: AutomationRequest, mount_path: str = "/content/drive"):
    logger.info("Automation drivemount for endpoint %s, mount_path=%s", endpoint, mount_path)
    code = f"from google.colab import drive\ndrive.mount('{mount_path}')"
    try:
        _, runtime = get_session_and_runtime(req, drive_hook_enabled=True)
        outputs = []
        def hook(out):
            outputs.append(out)
            logger.debug("Output hook received: %s", str(out)[:200])
        logger.info("Running drivemount code with timeout %.1fs", req.timeout)
        runtime.execute(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
        logger.info("Drivemount automation completed with %d outputs", len(outputs))
        runtime.stop()
        return {"outputs": outputs}
    except Exception as e:
        logger.error("Drivemount automation failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/sessions/{endpoint}/automation/install")
def run_install_endpoint(endpoint: str, req: InstallRequest):
    logger.info("Automation install for endpoint %s, packages=%s, requirement=%s",
                endpoint, req.packages, req.requirement)
    commands = []
    try:
        if req.requirement:
            contents = ContentsClient(req.url, req.token)
            if not os.path.isfile(req.requirement):
                logger.error("Requirements file not found: %s", req.requirement)
                raise HTTPException(400, "Requirements file not found locally")
            remote = f"content/{os.path.basename(req.requirement)}"
            logger.info("Uploading requirements file to %s", remote)
            contents.upload(req.requirement, remote)
            commands.extend(["-r", f"/{remote}"])
        if req.packages:
            commands.extend(req.packages)
        if not commands:
            logger.warning("No packages or requirements specified")
            raise HTTPException(400, "No packages specified")
        cmd_str = ", ".join(f"'{c}'" for c in commands)
        code = f"""
import subprocess, sys
def install():
    packages = [{cmd_str}]
    try:
        subprocess.check_call(['uv', 'pip', 'install', '--system'] + packages)
        print('Installation Complete (via uv)!')
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print('Installation Complete (via pip)!')
install()
"""
        _, runtime = get_session_and_runtime(req)
        outputs = []
        def hook(out):
            outputs.append(out)
            logger.debug("Output hook received: %s", str(out)[:200])
        logger.info("Running install code with timeout %.1fs", req.timeout)
        runtime.execute(code, output_hook=hook, timeout=req.timeout)
        logger.info("Install automation completed with %d outputs", len(outputs))
        runtime.stop()
        return {"outputs": outputs}
    except Exception as e:
        logger.error("Install automation failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.get("/version")
def version_endpoint():
    logger.info("Version endpoint called")
    return {"version": "0.1.0"}

@app.post("/whoami")
def whoami_endpoint(credentials: CredentialsModel):
    logger.info("Whoami endpoint called")
    try:
        creds = Credentials.from_authorized_user_info(credentials.dict())
        sess = AuthorizedSession(creds)
        if not creds.valid:
            logger.info("Refreshing credentials")
            creds.refresh(Request())
        url = f"https://oauth2.googleapis.com/tokeninfo?access_token={creds.token}"
        logger.info("Calling tokeninfo endpoint")
        resp = urllib.request.urlopen(url)
        info = json.loads(resp.read().decode())
        logger.info("Token info: email=%s, scopes=%d", info.get("email"), len(info.get("scope", "").split()))
        return {
            "email": info.get("email"),
            "scopes": info.get("scope", "").split(),
            "expires_in": info.get("expires_in"),
        }
    except Exception as e:
        logger.error("Whoami failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/sessions/{endpoint}/url")
def connect_url_endpoint(endpoint: str, credentials: CredentialsModel, token: str, url: str,
                         host: str = "https://colab.research.google.com"):
    logger.info("Connect URL endpoint for endpoint %s", endpoint)
    try:
        host_clean = host.rstrip("/")
        backend_path = f"/tun/m/{endpoint}"
        dbu = quote(backend_path, safe="")
        fragment = f"{host_clean}{backend_path}"
        full = f"{host_clean}/notebooks/empty.ipynb?dbu={dbu}#datalabBackendUrl={fragment}"
        logger.info("Connect URL generated: %s", full)
        return {"connect_url": full}
    except Exception as e:
        logger.error("Connect URL generation failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))

# --------------------------------------------------------------------
# 10. Health check
# --------------------------------------------------------------------
@app.get("/health")
def health_endpoint():
    logger.debug("Health check")
    return {"status": "ok"}

# --------------------------------------------------------------------
# 11. Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
