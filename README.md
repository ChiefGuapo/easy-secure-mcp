# mcpserver

A lightweight MCP server module with JWT protection designed to work safely with long-lived SSE (Server-Sent Events) connections.

This module exposes:
- a health endpoint
- a token minting endpoint (API key → JWT)
- an MCP SSE endpoint (protected by JWT)

It also includes a CLI test harness that verifies connectivity, token issuance, tool listing (via the FastMCP client wrapper), and negative auth behavior.

---

## What’s in here

### Main server
- `server.py`

Responsibilities:
- Creates and runs the FastAPI app
- Adds `JWTAuthASGIMiddleware` to protect MCP SSE traffic without breaking SSE
- Exposes routes:
  - `GET /health`
  - `POST /auth/token`
  - `/sse` (MCP transport)

### Test harness
- `test_mcp.py`

Responsibilities:
- Tests that the server is reachable
- Mints a JWT via `/auth/token`
- Runs negative auth tests (should fail gracefully)
- Lists MCP tools using an async MCP client wrapper:

### Tools and utils
- `included some example tool files`
- `dynamic tool loader in utils/tool_loader.py`

---

## Requirements

- Python 3.10+ recommended
- Dependencies used by this module typically include:
  - `fastapi`
  - `uvicorn`
  - `pyyaml`
  - `requests`
  - JWT library (commonly `python-jose` or `pyjwt`, depending on your implementation)

Install dependencies using your project’s existing method (pip/poetry/uv).

---

## Running the server

```bash
python server.py
```


## Running the test script

```bash
python test_mcp.py --host 192.168.1.159 --api-key "CHANGE_ME_TOKEN_API_KEY"  --all
```

## Endpoints summary
- GET /health → service health
- POST /auth/token → mint JWT using API key
- /sse → MCP SSE transport (JWT required)

# Security notes
- Treat TOKEN_API_KEY and JWT secrets as sensitive.
- Do not log raw JWTs in production.
- Rotate secrets periodically.
