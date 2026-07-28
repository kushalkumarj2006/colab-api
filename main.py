import os
import json
import uuid
import tempfile
import logging
import time
from typing import Optional, List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# Import from the official colab_cli library
from colab_cli.client import Client, Prod, ColabRequestError, PostAssignmentResponse
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient
from colab_cli.utils import get_status_code
from colab_cli.auth import PUBLIC_SCOPES

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow

# ---- Setup logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# 1. OAuth configuration (from environment)
# --------------------------------------------------------------------
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"

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
    logger.error("OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET must be set in environment.")
    raise RuntimeError("Missing OAuth credentials")

_flow: Optional[InstalledAppFlow] = None

def get_auth_url() -> str:
    """Generate the OAuth authorization URL (remote copy-paste flow)."""
    global _flow
    logger.info("Creating OAuth flow with redirect_uri='%s'", REMOTE_REDIRECT_URI)
    _flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    _flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = _flow.authorization_url(prompt="consent", token_usage="remote")
    logger.info("Auth URL generated (length: %d)", len(auth_url))
    return auth_url

def exchange_code(code: str) -> Credentials:
    """Exchange an authorization code for OAuth credentials."""
    global _flow
    if _flow is None:
        logger.error("No OAuth flow initiated. Call get_auth_url() first.")
        raise RuntimeError("No OAuth flow initiated. Call get_auth_url() first.")
    try:
        logger.info("Exchanging authorization code (first 10 chars: %s...)", code[:10])
        _flow.fetch_token(code=code)
        creds = _flow.credentials
        logger.info("Token exchange successful. Access token length: %d", len(creds.token))
        return creds
    except Exception as e:
        logger.exception("Token exchange failed")
        raise RuntimeError(f"Failed to exchange code: {e}")
    finally:
        _flow = None

# --------------------------------------------------------------------
# 2. FastAPI app with CORS and logging middleware
# --------------------------------------------------------------------
app = FastAPI(title="Colab API (library-based)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Logging middleware ----
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        logger.info("Request: %s %s", request.method, request.url.path)
        if request.query_params:
            logger.info("Query params: %s", dict(request.query_params))
        if request.method in ("POST", "PUT"):
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

# --------------------------------------------------------------------
# 3. Models
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

class CodeRequest(BaseModel):
    code: str

# --------------------------------------------------------------------
# 4. Helpers
# --------------------------------------------------------------------
def get_authorized_session(creds_dict: dict) -> AuthorizedSession:
    creds = Credentials.from_authorized_user_info(creds_dict)
    return AuthorizedSession(creds)

def make_drive_hook(credentials: Credentials, endpoint: str):
    """Return a hook that handles `dfs_ephemeral` colab_request messages."""
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

        # GET to obtain a propagation token
        resp = session.request("GET", url, params=params)
        if resp.status_code != 200:
            logger.error("Propagation GET failed: %d", resp.status_code)
            return False

        # Strip XSSI prefix
        text = resp.text
        if text.startswith(")]}'\n"):
            text = text[4:]
        data = json.loads(text)
        token = data.get("token")
        if not token:
            logger.error("No token in propagation response")
            return False

        # POST to propagate credentials
        headers = {"x-goog-colab-token": token}
        params["dryrun"] = "false"
        resp = session.request(
            "POST",
            url,
            params=params,
            headers=headers,
            files={"file_id": (None, "empty.ipynb")},
        )
        if resp.status_code != 200:
            logger.error("Propagation POST failed: %d", resp.status_code)
            return False

        # Send input_reply to resume the kernel
        reply = wsclient.session.msg(
            "input_reply",
            {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
        )
        if "header" in msg:
            reply["parent_header"] = msg["header"]
        wsclient.stdin_channel.send(reply)
        return True

    return hook

def get_session_and_runtime(req: SessionContext, drive_hook_enabled: bool = False):
    """Build a ColabClient and ColabRuntime from the request context."""
    creds = Credentials.from_authorized_user_info(req.credentials.dict())
    sess = AuthorizedSession(creds)
    colab = Client(Prod(), sess)
    hook = None
    if drive_hook_enabled:
        hook = make_drive_hook(creds, req.endpoint)
    runtime = ColabRuntime(
        req.url,
        req.token,
        kernel_id=req.kernel_id,
        session_id=req.session_id,
        drive_hook=hook,
    )
    return colab, runtime

# --------------------------------------------------------------------
# 5. Endpoints
# --------------------------------------------------------------------
@app.get("/auth/url")
def auth_url():
    logger.info("Handling /auth/url request")
    try:
        url = get_auth_url()
        return {"auth_url": url}
    except Exception as e:
        logger.exception("Failed to generate auth URL")
        raise HTTPException(500, str(e))

@app.post("/auth/token")
def auth_token(request: CodeRequest):
    code = request.code
    logger.info("Handling /auth/token request")
    try:
        creds = exchange_code(code)
        return {"credentials": json.loads(creds.to_json())}
    except Exception as e:
        logger.exception("Token exchange failed")
        raise HTTPException(400, str(e))

@app.post("/sessions")
def create_session(credentials: CredentialsModel, gpu: Optional[str] = None, tpu: Optional[str] = None):
    logger.info("Creating session with gpu=%s, tpu=%s", gpu, tpu)
    creds_obj = Credentials.from_authorized_user_info(credentials.dict())
    sess = AuthorizedSession(creds_obj)
    client = Client(Prod(), sess)

    from colab_cli.client import Variant, Accelerator
    variant = Variant.DEFAULT
    accelerator = Accelerator.NONE
    if tpu:
        variant = Variant.TPU
        accelerator = Accelerator.V5E1 if tpu.lower() == "v5e1" else Accelerator.V6E1
    elif gpu:
        variant = Variant.GPU
        mapping = {
            "a100": Accelerator.A100,
            "h100": Accelerator.H100,
            "l4": Accelerator.L4,
            "t4": Accelerator.T4,
            "g4": Accelerator.G4,
        }
        accelerator = mapping.get(gpu.lower(), Accelerator.A100)

    try:
        res = client.assign(uuid.uuid4(), variant=variant, accelerator=accelerator)
    except ColabRequestError as e:
        status = get_status_code(e)
        logger.error("Assignment failed with status %d: %s", status, str(e))
        if status == 412:
            raise HTTPException(429, "Too many active sessions. Stop an existing one first.")
        if status == 400 and accelerator != Accelerator.NONE:
            raise HTTPException(400, f"Accelerator '{accelerator.value}' not available.")
        raise HTTPException(500, str(e))

    if isinstance(res, PostAssignmentResponse):
        endpoint = res.endpoint
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        variant_val = res.variant.value
        accel_val = res.accelerator.value
    else:  # Assignment response
        endpoint = res.endpoint
        token = getattr(res, "runtime_proxy_token", None) or getattr(res, "token", None)
        url = getattr(res, "runtime_proxy_info", {}).get("url", "")
        variant_val = variant.value
        accel_val = accelerator.value

    logger.info("Session created: endpoint=%s", endpoint)
    return {
        "endpoint": endpoint,
        "token": token,
        "url": url,
        "variant": variant_val,
        "accelerator": accel_val,
    }

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive(endpoint: str, req: SessionContext):
    logger.info("Keep-alive for endpoint %s", endpoint)
    creds = Credentials.from_authorized_user_info(req.credentials.dict())
    sess = AuthorizedSession(creds)
    client = Client(Prod(), sess)
    try:
        client.keep_alive_assignment(endpoint)
    except Exception as e:
        logger.exception("Keep-alive failed")
        raise HTTPException(500, str(e))
    return {"status": "ok"}

@app.post("/sessions/{endpoint}/execute")
def execute(endpoint: str, req: ExecuteRequest):
    logger.info("Execute request for endpoint %s", endpoint)
    _, runtime = get_session_and_runtime(req)
    outputs = []
    def hook(out):
        outputs.append(out)

    try:
        runtime.execute_code(req.code, output_hook=hook, timeout=req.timeout)
    except Exception as e:
        logger.exception("Execution failed")
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()

    logger.info("Execution completed with %d outputs", len(outputs))
    return {"outputs": outputs}

@app.get("/sessions/{endpoint}/files")
def list_files(endpoint: str, credentials: CredentialsModel, token: str, url: str, path: str = "content"):
    logger.info("List files for endpoint %s, path=%s", endpoint, path)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)
    try:
        data = contents.list_dir(path)
    except Exception as e:
        logger.exception("List files failed")
        raise HTTPException(500, str(e))
    return {"files": data.get("content", [])}

@app.post("/sessions/{endpoint}/files")
def upload_file(
    endpoint: str,
    credentials: CredentialsModel,
    token: str,
    url: str,
    remote_path: str = Form(...),
    file: UploadFile = File(...),
):
    logger.info("Upload file to endpoint %s, remote_path=%s", endpoint, remote_path)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        local_path = tmp.name
    try:
        contents.upload(local_path, remote_path)
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(local_path)
    return {"message": "Uploaded"}

@app.get("/sessions/{endpoint}/files/{path:path}")
def download_file(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    logger.info("Download file from endpoint %s, path=%s", endpoint, path)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        local_path = tmp.name
    try:
        contents.download(path, local_path)
        return FileResponse(local_path, filename=os.path.basename(path))
    except Exception as e:
        logger.exception("Download failed")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(local_path)

@app.delete("/sessions/{endpoint}/files/{path:path}")
def delete_file(endpoint: str, path: str, credentials: CredentialsModel, token: str, url: str):
    logger.info("Delete file from endpoint %s, path=%s", endpoint, path)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)
    try:
        contents.rm(path)
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(500, str(e))
    return {"message": "Deleted"}

# ----- Automation endpoints -----
@app.post("/sessions/{endpoint}/automation/auth")
def run_auth(endpoint: str, req: AutomationRequest):
    logger.info("Automation auth for endpoint %s", endpoint)
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
        runtime.execute_code(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
    except Exception as e:
        logger.exception("Auth automation failed")
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.post("/sessions/{endpoint}/automation/drivemount")
def run_drivemount(endpoint: str, req: AutomationRequest, mount_path: str = "/content/drive"):
    logger.info("Automation drivemount for endpoint %s, mount_path=%s", endpoint, mount_path)
    code = f"from google.colab import drive\ndrive.mount('{mount_path}')"
    _, runtime = get_session_and_runtime(req, drive_hook_enabled=True)
    outputs = []
    def hook(out):
        outputs.append(out)
    try:
        runtime.execute_code(code, output_hook=hook, timeout=req.timeout, allow_stdin=True)
    except Exception as e:
        logger.exception("Drivemount automation failed")
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.post("/sessions/{endpoint}/automation/install")
def run_install(endpoint: str, req: InstallRequest):
    logger.info("Automation install for endpoint %s", endpoint)
    commands = []
    if req.requirement:
        class Dummy:
            pass
        dummy = Dummy()
        dummy.url = req.url
        dummy.token = req.token
        contents = ContentsClient(dummy)
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
        runtime.execute_code(code, output_hook=hook, timeout=req.timeout)
    except Exception as e:
        logger.exception("Install automation failed")
        raise HTTPException(500, str(e))
    finally:
        runtime.stop()
    return {"outputs": outputs}

@app.post("/sessions/{endpoint}/url")
def connect_url(endpoint: str, credentials: CredentialsModel, token: str, url: str,
                host: str = "https://colab.research.google.com"):
    logger.info("Connect URL for endpoint %s", endpoint)
    host_clean = host.rstrip("/")
    backend_path = f"/tun/m/{endpoint}"
    dbu = quote(backend_path, safe="")
    fragment = f"{host_clean}{backend_path}"
    full = f"{host_clean}/notebooks/empty.ipynb?dbu={dbu}#datalabBackendUrl={fragment}"
    return {"connect_url": full}

@app.get("/version")
def version():
    from colab_cli.auto_update import get_app_version
    return {"version": get_app_version()}

@app.post("/whoami")
def whoami(credentials: CredentialsModel):
    logger.info("Whoami called")
    creds = Credentials.from_authorized_user_info(credentials.dict())
    if not creds.valid:
        creds.refresh(Request())
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={creds.token}"
    try:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            info = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(400, str(e))
    return {
        "email": info.get("email"),
        "scopes": info.get("scope", "").split(),
        "expires_in": info.get("expires_in"),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------------------------
# 6. Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
