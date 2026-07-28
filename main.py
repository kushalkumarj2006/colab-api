import os
import json
import uuid
import tempfile
import logging
from typing import Optional, List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# colab_cli library imports
from colab_cli.client import Client, Prod, ColabRequestError, PostAssignmentResponse, Assignment
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient
from colab_cli.utils import get_status_code
from colab_cli.auth import PUBLIC_SCOPES  # reuse the standard scopes

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow

# ---------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# OAuth configuration from environment
# ---------------------------------------------------------------------
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"

# Build client config from environment variables (same as used by the CLI)
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
    raise RuntimeError("OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET must be set in environment")

_flow: Optional[InstalledAppFlow] = None

def get_auth_url() -> str:
    """Generate the OAuth authorization URL (remote copy-paste flow)."""
    global _flow
    _flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, PUBLIC_SCOPES)
    _flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = _flow.authorization_url(prompt="consent", token_usage="remote")
    return auth_url

def exchange_code(code: str) -> Credentials:
    """Exchange an authorization code for OAuth credentials."""
    global _flow
    if _flow is None:
        raise RuntimeError("No OAuth flow initiated. Call get_auth_url() first.")
    try:
        _flow.fetch_token(code=code)
        return _flow.credentials
    finally:
        _flow = None

# ---------------------------------------------------------------------
# FastAPI app with CORS
# ---------------------------------------------------------------------
app = FastAPI(title="Colab API (using colab_cli)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
            logger.error(f"Propagation GET failed: {resp.status_code}")
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
            logger.error(f"Propagation POST failed: {resp.status_code}")
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

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/auth/url")
def auth_url():
    try:
        url = get_auth_url()
        return {"auth_url": url}
    except Exception as e:
        logger.exception("Failed to generate auth URL")
        raise HTTPException(500, str(e))

@app.post("/auth/token")
def auth_token(code: str):
    try:
        creds = exchange_code(code)
        return {"credentials": json.loads(creds.to_json())}
    except Exception as e:
        logger.exception("Token exchange failed")
        raise HTTPException(400, str(e))

@app.post("/sessions")
def create_session(credentials: CredentialsModel, gpu: Optional[str] = None, tpu: Optional[str] = None):
    """Create a new Colab session (VM)."""
    creds_obj = Credentials.from_authorized_user_info(credentials.dict())
    sess = AuthorizedSession(creds_obj)
    client = Client(Prod(), sess)

    # Resolve accelerator flags (same logic as the CLI)
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

    return {
        "endpoint": endpoint,
        "token": token,
        "url": url,
        "variant": variant_val,
        "accelerator": accel_val,
    }

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive(endpoint: str, req: SessionContext):
    """Ping the session to prevent idle timeout."""
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
    """Run arbitrary Python code on the session."""
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

    return {"outputs": outputs}

@app.get("/sessions/{endpoint}/files")
def list_files(endpoint: str, credentials: CredentialsModel, token: str, url: str, path: str = "content"):
    """List remote directory contents."""
    contents = ContentsClient(type("State", (), {"url": url, "token": token})())
    try:
        files = contents.list_dir(path)
    except Exception as e:
        logger.exception("List files failed")
        raise HTTPException(500, str(e))
    return {"files": files.get("content", [])}

@app.post("/sessions/{endpoint}/files")
def upload_file(
    endpoint: str,
    credentials: CredentialsModel,
    token: str,
    url: str,
    remote_path: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a local file to the VM."""
    contents = ContentsClient(type("State", (), {"url": url, "token": token})())
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
    """Download a remote file from the VM."""
    contents = ContentsClient(type("State", (), {"url": url, "token": token})())
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
    """Delete a remote file."""
    contents = ContentsClient(type("State", (), {"url": url, "token": token})())
    try:
        contents.rm(path)
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(500, str(e))
    return {"message": "Deleted"}

# ----- Automation endpoints -----
@app.post("/sessions/{endpoint}/automation/auth")
def run_auth(endpoint: str, req: AutomationRequest):
    """Run `auth.authenticate_user()` on the VM (Drive OAuth)."""
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
    """Mount Google Drive on the VM."""
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
    """Install Python packages via uv/pip."""
    commands = []
    if req.requirement:
        # Upload requirements file to the VM
        contents = ContentsClient(type("State", (), {"url": req.url, "token": req.token})())
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
    """Get a browser URL to attach to the existing session."""
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
    """Return tokeninfo for the given credentials (debug)."""
    creds = Credentials.from_authorized_user_info(credentials.dict())
    if not creds.valid:
        creds.refresh(Request())
    import urllib.request
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={creds.token}"
    try:
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

# ---------------------------------------------------------------------
# Run with uvicorn (if executed directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
