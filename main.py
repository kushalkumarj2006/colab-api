import os
import json
import uuid
import base64
import tempfile
import urllib.request
from typing import Optional, List
from urllib.parse import urljoin, urlparse, quote
from dataclasses import dataclass

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow
import jupyter_kernel_client

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
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "project_id": "cloud-sdk-platform",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "redirect_uris": [REMOTE_REDIRECT_URI],
    }
}

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
    if tpu:
        variant = Variant.TPU
        acc = Accelerator.V5E1 if tpu.lower() == "v5e1" else Accelerator.V6E1
        return variant, acc
    if gpu:
        mapping = {
            "a100": Accelerator.A100,
            "h100": Accelerator.H100,
            "l4": Accelerator.L4,
            "t4": Accelerator.T4,
            "g4": Accelerator.G4,
        }
        return Variant.GPU, mapping.get(gpu.lower(), Accelerator.A100)
    return Variant.DEFAULT, Accelerator.NONE

# --------------------------------------------------------------------
# 2. OAuth
# --------------------------------------------------------------------

def get_auth_url() -> str:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")
    return auth_url

def exchange_code(code: str) -> Credentials:
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.fetch_token(code=code)
    return flow.credentials

# --------------------------------------------------------------------
# 3. Colab Tunnel Client
# --------------------------------------------------------------------

class ColabClient:
    def __init__(self, session: AuthorizedSession):
        self.session = session
        self.domain = "https://colab.research.google.com"
        self.api_domain = "https://colab.pa.googleapis.com"

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

        resp = self.session.request(method, url, headers=req_headers,
                                    params=params, json=json_body)
        if not resp.ok:
            raise RuntimeError(f"Colab API error {resp.status_code}: {resp.text}")
        body = strip_xssi(resp.text)
        return json.loads(body) if body else {}

    def list_assignments(self) -> list:
        data = self._request(f"{TUN_ENDPOINT}/assignments")
        return data.get("assignments", [])

    def assign(self, notebook_hash: uuid.UUID, variant: str = None, accelerator: str = None) -> dict:
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant:
            params["variant"] = variant
        if accelerator:
            params["accelerator"] = accelerator
        # GET to get XSRF token and check if assignment already exists
        get_resp = self._request(f"{TUN_ENDPOINT}/assign", params=params)
        if "endpoint" in get_resp:
            return get_resp
        token = get_resp.get("token")
        if not token:
            raise RuntimeError("No XSRF token in GET response")
        headers = {"X-Goog-Colab-Token": token}
        return self._request(f"{TUN_ENDPOINT}/assign", method="POST",
                             params=params, headers=headers)

    def unassign(self, endpoint: str):
        resp = self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}")
        token = resp.get("token")
        if not token:
            return
        headers = {"X-Goog-Colab-Token": token}
        self._request(f"{TUN_ENDPOINT}/unassign/{endpoint}", method="POST",
                      headers=headers)

    def keep_alive(self, endpoint: str):
        url = f"{TUN_ENDPOINT}/{endpoint}/keep-alive/"
        headers = COLAB_TUNNEL_HEADER.copy()
        try:
            self._request(url, headers=headers, method="GET")
        except requests.exceptions.ReadTimeout:
            # Timeout is normal – activity was recorded
            pass

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

    @property
    def client(self):
        if self._client is None:
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
                headers={
                    "X-Colab-Client-Agent": "colab-api",
                    "X-Colab-Runtime-Proxy-Token": self.token,
                },
            )
            self._client._own_kernel = False
            self._client.start()
            if self.drive_hook:
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
                    if self.drive_hook(msg, wsclient):
                        return  # intercepted
            except Exception:
                pass
            orig_on_message(ws, message)

        wsclient.kernel_socket.on_message = hooked_on_message

    def execute(self, code: str, output_hook: callable = None,
                timeout: float = 30, allow_stdin: bool = False) -> list:
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
            reply = self.client.execute(code, **kwargs)
            return reply.get("outputs", [])

    def stop(self):
        if self._client:
            try:
                self._client._manager.client.stop_channels()
            except Exception:
                pass

# --------------------------------------------------------------------
# 5. Contents Client (File Operations)
# --------------------------------------------------------------------

class ContentsClient:
    def __init__(self, url: str, token: str):
        self.base_url = url.rstrip("/")
        self.token = token

    def _request(self, method: str, path: str, json_body: dict = None) -> dict:
        quoted = quote(path.strip("/"), safe="/")
        url = f"{self.base_url}/api/contents/{quoted}"
        params = {"authuser": "0", "colab-runtime-proxy-token": self.token}
        resp = requests.request(method, url, params=params, json=json_body)
        if resp.status_code == 404:
            raise FileNotFoundError(path)
        resp.raise_for_status()
        if method == "DELETE":
            return {}
        return resp.json()

    def list(self, path: str = "content") -> list:
        data = self._request("GET", path)
        return data.get("content", [])

    def upload(self, local_path: str, remote_path: str):
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

    def download(self, remote_path: str, local_path: str):
        data = self._request("GET", remote_path)
        content = data.get("content", "")
        fmt = data.get("format", "text")
        if fmt == "base64":
            content_bytes = base64.b64decode(content)
        else:
            content_bytes = content.encode("utf-8")
        with open(local_path, "wb") as f:
            f.write(content_bytes)

    def delete(self, remote_path: str):
        self._request("DELETE", remote_path)

# --------------------------------------------------------------------
# 6. Drive Auth Hook Factory
# --------------------------------------------------------------------

def make_drive_hook(credentials: Credentials, endpoint: str):
    session = AuthorizedSession(credentials)
    domain = "https://colab.research.google.com"

    def hook(msg: dict, wsclient) -> bool:
        content = msg.get("content", {})
        req = content.get("request", {})
        if req.get("authType") != "dfs_ephemeral":
            return False
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
        resp = session.request("GET", url, params=params)
        if resp.status_code != 200:
            return False
        data = json.loads(strip_xssi(resp.text))
        token = data.get("token")
        if not token:
            return False
        headers = {"x-goog-colab-token": token}
        params["dryrun"] = "false"
        resp = session.request("POST", url, params=params, headers=headers,
                               files={"file_id": (None, "empty.ipynb")})
        if resp.status_code != 200:
            return False
        reply = wsclient.session.msg(
            "input_reply",
            {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
        )
        if "header" in msg:
            reply["parent_header"] = msg["header"]
        wsclient.stdin_channel.send(reply)
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
# 8. FastAPI App
# --------------------------------------------------------------------

app = FastAPI(title="Colab API (standalone)")

def get_session_and_runtime(req: SessionContext, drive_hook_enabled=False):
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

@app.get("/auth/url")
def auth_url():
    return {"auth_url": get_auth_url()}

@app.post("/auth/token")
def auth_token(code: str):
    creds = exchange_code(code)
    return {"credentials": json.loads(creds.to_json())}

@app.post("/sessions")
def create_session(credentials: CredentialsModel,
                   name: Optional[str] = None,
                   gpu: Optional[str] = None,
                   tpu: Optional[str] = None):
    creds = Credentials.from_authorized_user_info(credentials.dict())
    sess = AuthorizedSession(creds)
    colab = ColabClient(sess)
    variant, accel = resolve_accelerator(gpu, tpu)
    try:
        res = colab.assign(uuid.uuid4(), variant=variant, accelerator=accel)
    except Exception as e:
        raise HTTPException(400, str(e))
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
    return {
        "endpoint": endpoint,
        "token": token,
        "url": url,
        "variant": variant,
        "accelerator": accel,
    }

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive(endpoint: str, credentials: CredentialsModel, token: str, url: str):
    creds = Credentials.from_authorized_user_info(credentials.dict())
    sess = AuthorizedSession(creds)
    colab = ColabClient(sess)
    colab.keep_alive(endpoint)
    return {"status": "ok"}

@app.post("/sessions/{endpoint}/execute")
def execute(endpoint: str, req: ExecuteRequest):
    _, runtime = get_session_and_runtime(req)
    outputs = []
    def hook(out):
        outputs.append(out)
    try:
        runtime.execute(req.code, output_hook=hook, timeout=req.timeout)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.get("/sessions/{endpoint}/files")
def list_files(endpoint: str, credentials: CredentialsModel, token: str, url: str, path: str = "content"):
    contents = ContentsClient(url, token)
    try:
        files = contents.list(path)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"files": files}

@app.post("/sessions/{endpoint}/files")
def upload_file(
    endpoint: str,
    credentials: CredentialsModel,
    token: str,
    url: str,
    remote_path: str = Form(...),
    file: UploadFile = File(...),
):
    contents = ContentsClient(url, token)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        local = tmp.name
    try:
        contents.upload(local, remote_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(local)
    return {"message": "Uploaded"}

@app.get("/sessions/{endpoint}/files/{path:path}")
def download_file(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    contents = ContentsClient(url, token)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        local = tmp.name
    try:
        contents.download(path, local)
        return FileResponse(local, filename=os.path.basename(path))
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(local)

@app.delete("/sessions/{endpoint}/files/{path:path}")
def delete_file(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    contents = ContentsClient(url, token)
    try:
        contents.delete(path)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"message": "Deleted"}

@app.post("/sessions/{endpoint}/automation/auth")
def run_auth(endpoint: str, req: AutomationRequest):
    code = (
        "import os\n"
        "os.environ['USE_AUTH_EPHEM'] = '0'\n"
        "from google.colab import auth\n"
        "auth.authenticate_user()\n"
    )
    _, runtime = get_session_and_runtime(req, drive_hook_enabled=True)
    outputs = []
    def hook(out):
        outputs.append(out)
    try:
        runtime.execute(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.post("/sessions/{endpoint}/automation/drivemount")
def run_drivemount(endpoint: str, req: AutomationRequest, mount_path: str = "/content/drive"):
    code = f"from google.colab import drive\ndrive.mount('{mount_path}')"
    _, runtime = get_session_and_runtime(req, drive_hook_enabled=True)
    outputs = []
    def hook(out):
        outputs.append(out)
    try:
        runtime.execute(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.post("/sessions/{endpoint}/automation/install")
def run_install(endpoint: str, req: InstallRequest):
    commands = []
    if req.requirement:
        contents = ContentsClient(req.url, req.token)
        if not os.path.isfile(req.requirement):
            raise HTTPException(400, "Requirements file not found locally")
        remote = f"content/{os.path.basename(req.requirement)}"
        contents.upload(req.requirement, remote)
        commands.extend(["-r", f"/{remote}"])
    if req.packages:
        commands.extend(req.packages)
    if not commands:
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
    try:
        runtime.execute(code, output_hook=hook, timeout=req.timeout)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.post("/whoami")
def whoami(credentials: CredentialsModel):
    creds = Credentials.from_authorized_user_info(credentials.dict())
    sess = AuthorizedSession(creds)
    if not creds.valid:
        creds.refresh(Request())
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={creds.token}"
    resp = urllib.request.urlopen(url)
    info = json.loads(resp.read().decode())
    return {
        "email": info.get("email"),
        "scopes": info.get("scope", "").split(),
        "expires_in": info.get("expires_in"),
    }

@app.post("/sessions/{endpoint}/url")
def connect_url(endpoint: str, credentials: CredentialsModel, token: str, url: str,
                host: str = "https://colab.research.google.com"):
    host_clean = host.rstrip("/")
    backend_path = f"/tun/m/{endpoint}"
    dbu = quote(backend_path, safe="")
    fragment = f"{host_clean}{backend_path}"
    full = f"{host_clean}/notebooks/empty.ipynb?dbu={dbu}#datalabBackendUrl={fragment}"
    return {"connect_url": full}

# --------------------------------------------------------------------
# 9. Run with uvicorn (if executed directly)
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
