import os
import json
import uuid
import tempfile
import logging
import time
from typing import Optional, List, Tuple
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel

# Patch jupyter_kernel_client
import jupyter_kernel_client
if not hasattr(jupyter_kernel_client, 'KernelClient'):
    try:
        from jupyter_kernel_client.client import KernelClient
        jupyter_kernel_client.KernelClient = KernelClient
    except ImportError:
        try:
            from jupyter_kernel_client.kernelclient import KernelClient
            jupyter_kernel_client.KernelClient = KernelClient
        except ImportError:
            pass

from colab_cli.client import Client, Prod, ColabRequestError, PostAssignmentResponse
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient
from colab_cli.utils import get_status_code
from colab_cli.auth import PUBLIC_SCOPES

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google_auth_oauthlib.flow import InstalledAppFlow

os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def get_auth_url(code_challenge: str, code_challenge_method: str = "S256") -> str:
    flow = InstalledAppFlow.from_client_config(
        CLIENT_CONFIG,
        PUBLIC_SCOPES,
        redirect_uri=REMOTE_REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        token_usage="remote",
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )
    return auth_url

def exchange_code(code: str, code_verifier: str) -> Credentials:
    flow = InstalledAppFlow.from_client_config(
        CLIENT_CONFIG,
        PUBLIC_SCOPES,
        redirect_uri=REMOTE_REDIRECT_URI,
    )
    flow.fetch_token(code=code, code_verifier=code_verifier)
    return flow.credentials

app = FastAPI(title="Colab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        logger.info("Request: %s %s", request.method, request.url.path)
        if request.query_params:
            logger.info("Query params: %s", dict(request.query_params))
        if request.method in ("POST", "PUT"):
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" not in content_type:
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

class AutomationRequest(SessionContext):
    timeout: Optional[float] = 600

class InstallRequest(SessionContext):
    packages: Optional[List[str]] = None
    requirement: Optional[str] = None
    timeout: Optional[float] = 600

# ---- Helpers with token refresh ----
def creds_from_model(model: CredentialsModel) -> Credentials:
    return Credentials.from_authorized_user_info(model.model_dump())

def refresh_credentials_if_needed(
    creds_model: CredentialsModel,
    force_refresh: bool = False
) -> Tuple[Credentials, Optional[CredentialsModel]]:
    """
    Check if the access token is expired or will expire in the next 5 minutes.
    If so, refresh it using the refresh_token.
    Returns (credentials, updated_model_or_None).
    """
    creds = creds_from_model(creds_model)
    updated_model = None
    need_refresh = force_refresh

    if not need_refresh and creds.expiry:
        # Check if expiry is within 5 minutes from now
        if isinstance(creds.expiry, datetime):
            # If expiry is timezone-naive, assume UTC
            if creds.expiry.tzinfo is None:
                expiry_utc = creds.expiry.replace(tzinfo=timezone.utc)
            else:
                expiry_utc = creds.expiry.astimezone(timezone.utc)
            now_utc = datetime.now(timezone.utc)
            if expiry_utc - now_utc < timedelta(minutes=5):
                need_refresh = True

    if need_refresh:
        try:
            logger.info("Refreshing access token...")
            creds.refresh(Request())
            # Build updated model
            updated_model = CredentialsModel(
                token=creds.token,
                refresh_token=creds.refresh_token,
                expiry=creds.expiry.isoformat() if creds.expiry else None,
                scopes=creds.scopes if creds.scopes else [],
                token_uri=creds.token_uri,
                client_id=creds.client_id,
                client_secret=creds.client_secret,
            )
            logger.info("Token refreshed successfully.")
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            # Continue with old credentials; they might still work.

    return creds, updated_model

def get_authorized_session_and_updated(
    creds_model: CredentialsModel,
    force_refresh: bool = False
) -> Tuple[AuthorizedSession, Optional[CredentialsModel]]:
    creds, updated = refresh_credentials_if_needed(creds_model, force_refresh)
    return AuthorizedSession(creds), updated

def get_authorized_session(creds_model: CredentialsModel) -> AuthorizedSession:
    session, _ = get_authorized_session_and_updated(creds_model)
    return session

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
            logger.error("Propagation GET failed: %d", resp.status_code)
            return False

        text = resp.text
        if text.startswith(")]}'\n"):
            text = text[4:]
        data = json.loads(text)
        token = data.get("token")
        if not token:
            logger.error("No token in propagation response")
            return False

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
    creds = creds_from_model(req.credentials)
    sess = AuthorizedSession(creds)
    colab = Client(Prod(), sess)

    runtime = ColabRuntime(
        req.url,
        req.token,
        kernel_id=req.kernel_id,
        session_id=req.session_id,
    )

    if drive_hook_enabled:
        runtime.colab_request_hook = make_drive_hook(creds, req.endpoint)

    return colab, runtime

# ---- Endpoints ----
@app.get("/auth/url")
def auth_url(code_challenge: str, code_challenge_method: str = "S256"):
    try:
        return {"auth_url": get_auth_url(code_challenge, code_challenge_method)}
    except Exception as e:
        logger.exception("Auth URL generation failed")
        raise HTTPException(500, str(e))

@app.post("/auth/token")
def auth_token(request: dict = Body(...)):
    code = request.get("code")
    code_verifier = request.get("code_verifier")
    if not code or not code_verifier:
        raise HTTPException(400, "Missing code or code_verifier")
    try:
        creds = exchange_code(code, code_verifier)
        return {"credentials": json.loads(creds.to_json())}
    except Exception as e:
        logger.exception("Token exchange failed")
        raise HTTPException(400, str(e))

@app.post("/sessions")
def create_session(credentials: CredentialsModel, gpu: Optional[str] = None, tpu: Optional[str] = None):
    logger.info("Creating session with gpu=%s, tpu=%s", gpu, tpu)
    sess, updated_creds = get_authorized_session_and_updated(credentials)
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
        logger.error("Assignment failed with status %d", status)
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
    else:
        endpoint = res.endpoint
        token = getattr(res, "runtime_proxy_token", None) or getattr(res, "token", None)
        url = getattr(res, "runtime_proxy_info", {}).get("url", "")
        variant_val = variant.value
        accel_val = accelerator.value

    response = {
        "endpoint": endpoint,
        "token": token,
        "url": url,
        "variant": variant_val,
        "accelerator": accel_val,
    }
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/keep-alive")
def keep_alive(endpoint: str, req: SessionContext):
    logger.info("Keep-alive for %s", endpoint)
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)
    client = Client(Prod(), sess)
    try:
        client.keep_alive_assignment(endpoint)
    except Exception as e:
        logger.exception("Keep-alive failed")
        raise HTTPException(500, str(e))
    response = {"status": "ok"}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/execute")
def execute(endpoint: str, req: ExecuteRequest):
    logger.info("Execute request for %s", endpoint)
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)

    # Update the request's credentials with the possibly refreshed ones
    if updated_creds:
        req.credentials = updated_creds

    # Build runtime using the refreshed credentials
    creds = creds_from_model(req.credentials)
    runtime = ColabRuntime(
        req.url,
        req.token,
        kernel_id=req.kernel_id,
        session_id=req.session_id,
    )

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

    response = {"outputs": outputs}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

# ---- File endpoints (with refresh) ----
@app.post("/sessions/{endpoint}/files/list")
def list_files_endpoint(endpoint: str, req: SessionContext, path: str = "content"):
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = req.url
    dummy.token = req.token
    contents = ContentsClient(dummy)
    try:
        data = contents.list_dir(path)
    except Exception as e:
        logger.exception("List files failed")
        raise HTTPException(500, str(e))
    response = {"files": data.get("content", [])}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/files/upload")
async def upload_file(
    endpoint: str,
    credentials: CredentialsModel,
    token: str = Form(...),
    url: str = Form(...),
    remote_path: str = Form(...),
    file: UploadFile = File(...),
):
    sess, updated_creds = get_authorized_session_and_updated(credentials)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)

    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        local_path = tmp.name

    try:
        contents.upload(local_path, remote_path)
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(local_path)
    response = {"message": "Uploaded"}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.get("/sessions/{endpoint}/files/download/{path:path}")
def download_file(endpoint: str, path: str, token: str, url: str):
    # Download doesn't need credentials refresh (uses token/url only)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.url = url
    dummy.token = token
    contents = ContentsClient(dummy)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    local_path = tmp.name
    tmp.close()

    try:
        contents.download(path, local_path)
        return FileResponse(
            local_path,
            filename=os.path.basename(path),
            background=BackgroundTask(os.unlink, local_path)
        )
    except Exception as e:
        if os.path.exists(local_path):
            os.unlink(local_path)
        logger.exception("Download failed")
        raise HTTPException(500, str(e))

@app.delete("/sessions/{endpoint}/files/{path:path}")
def delete_file(endpoint: str, path: str, token: str, url: str):
    # Delete also doesn't use credentials directly
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

# ---- Automation endpoints ----
@app.post("/sessions/{endpoint}/automation/auth")
def run_auth(endpoint: str, req: AutomationRequest):
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)
    # Update req.credentials for runtime
    if updated_creds:
        req.credentials = updated_creds
    creds = creds_from_model(req.credentials)

    runtime = ColabRuntime(
        req.url,
        req.token,
        kernel_id=req.kernel_id,
        session_id=req.session_id,
    )
    runtime.colab_request_hook = make_drive_hook(creds, req.endpoint)

    code = (
        "import os\n"
        "os.environ['USE_AUTH_EPHEM'] = '0'\n"
        "from google.colab import auth\n"
        "auth.authenticate_user()\n"
    )
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

    response = {"outputs": outputs}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/automation/drivemount")
def run_drivemount(endpoint: str, req: AutomationRequest, mount_path: str = "/content/drive"):
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)
    if updated_creds:
        req.credentials = updated_creds
    creds = creds_from_model(req.credentials)

    runtime = ColabRuntime(
        req.url,
        req.token,
        kernel_id=req.kernel_id,
        session_id=req.session_id,
    )
    runtime.colab_request_hook = make_drive_hook(creds, req.endpoint)

    code = f"from google.colab import drive\ndrive.mount('{mount_path}')"
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

    response = {"outputs": outputs}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/automation/install")
def run_install(endpoint: str, req: InstallRequest):
    sess, updated_creds = get_authorized_session_and_updated(req.credentials)
    if updated_creds:
        req.credentials = updated_creds
    _, runtime = get_session_and_runtime(req)

    commands = []
    if req.requirement:
        commands.extend(["-r", req.requirement])
    if req.packages:
        commands.extend(req.packages)
    if not commands:
        raise HTTPException(400, "No packages or requirements specified")

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

    response = {"outputs": outputs}
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.post("/sessions/{endpoint}/url")
def connect_url(endpoint: str, credentials: CredentialsModel, token: str, url: str,
                host: str = "https://colab.research.google.com"):
    # No token refresh needed; just return URL
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
    creds, updated_creds = refresh_credentials_if_needed(credentials, force_refresh=True)
    token = creds.token
    import urllib.request
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={token}"
    try:
        with urllib.request.urlopen(url) as resp:
            info = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(400, str(e))
    response = {
        "email": info.get("email"),
        "scopes": info.get("scope", "").split(),
        "expires_in": info.get("expires_in"),
    }
    if updated_creds:
        response["updated_credentials"] = updated_creds.model_dump()
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
