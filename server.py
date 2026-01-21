from fastmcp import FastMCP
from utils.tool_loader import register_tools_from_dir
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastmcp.server.server import create_sse_app
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

try:
    import jwt  # PyJWT
except Exception:
    jwt = None  # type: ignore


WEB_SEC_HOST = '0.0.0.0'
WEB_SEC_PORT = 8999


load_dotenv()


# ---- JWT Security Configuration ----
JWT_ENABLED = True
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "GuapoMCP"
JWT_AUDIENCE = "mcp"
JWT_TTL_SECONDS = 3600
JWT_SECRET = "dkjknvisxd_otrtpsnvotry3795jdbvctwlgoiyebn"  # random string for encoding


# Shared secret for requesting tokens
TOKEN_API_KEY = "CHANGE_ME_TOKEN_API_KEY"


AUTH_EXEMPT_PATHS = {
    "/auth/token",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

def _create_token(subject: str) -> str:
    if jwt is None:
        raise RuntimeError("PyJWT not installed")

    now = datetime.now(timezone.utc)
    payload = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=JWT_TTL_SECONDS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_token(token: str) -> dict:
    if jwt is None:
        raise RuntimeError("PyJWT not installed")

    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
    )


class JWTAuthASGIMiddleware:
    """
    SSE-safe JWT middleware. Avoids Starlette BaseHTTPMiddleware which can break streaming.
    """

    def __init__(self, fast_api_app):
        self.app = fast_api_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not JWT_ENABLED:
            return await self.app(scope, receive, send)

        path = scope.get("path") or ""

        # allow unauth'd routes
        if path in AUTH_EXEMPT_PATHS:
            return await self.app(scope, receive, send)

        # only protect SSE/MCP surface
        if not path.startswith("/sse"):
            return await self.app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")

        if not auth.lower().startswith("bearer "):
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail":"Missing bearer token"}'
            })
            return

        token = auth.split(" ", 1)[1].strip()
        try:
            _verify_token(token)
        except Exception as e:
            msg = str(e).replace('"', '\\"').encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail":"' + msg + b'"}'
            })
            return

        return await self.app(scope, receive, send)


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


class TokenRequest(BaseModel):
    api_key: str
    sub: Optional[str] = None


@app.post("/auth/token")
def issue_token(req: TokenRequest):
    if req.api_key != TOKEN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid api_key")

    subject = req.sub or "api_user"
    token = _create_token(subject)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_TTL_SECONDS,
    }


mcp_server = FastMCP(
    name="GuapoMCP"
)

mcp_app = create_sse_app(
    server=mcp_server,
    sse_path='/sse',
    message_path=f'/sse/msg',
)

app.mount('', mcp_app)

tool_path = Path(__file__).resolve().parent / "tools"

register_tools_from_dir(
    server=mcp_server,
    tools_dir=tool_path,
)

# define routes on fastapi_app (health/auth/token)
# mount mcp on fastapi_app
app = JWTAuthASGIMiddleware(app)

if __name__ == "__main__":
    uvicorn.run(app, host=WEB_SEC_HOST, port=WEB_SEC_PORT)
