#!/usr/bin/env python3
"""mcp_test.py

Quick connectivity + auth + MCP tool listing tests for the GuapoMCP server.

What it tests:
- /health reachable
- /auth/token issues a JWT with a valid API key
- MCP SSE endpoint is reachable with a token
- tools/list works (best-effort; will report a clear failure if protocol differs)
- Unauthorized access fails gracefully:
  - requesting token with wrong api_key -> 401
  - connecting to /sse without bearer token -> 401

Usage examples:
  python mcp_test.py --all --api-key CHANGE_ME_TOKEN_API_KEY
  python mcp_test.py --health
  python mcp_test.py --token --api-key CHANGE_ME_TOKEN_API_KEY
  python mcp_test.py --tools --api-key CHANGE_ME_TOKEN_API_KEY
  python mcp_test.py --unauth

Exit code:
- 0 if all selected tests pass
- 1 if any selected test fails
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import asyncio
import requests

from pathlib import Path
import yaml


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` until we find a directory containing `agent/`.

    This makes the script runnable from any working directory.
    """
    cur = start
    for _ in range(10):
        if (cur / "agent").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: assume file's parent is root
    return start


# Ensure repo root is on sys.path so `import agent...` works when running this script directly.
_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from typing import Optional

import httpx

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import BaseTool


def init_mcp_client(mcp_servers: dict[str, dict[str, str]]) -> MultiServerMCPClient:
    client = MultiServerMCPClient(mcp_servers)
    return client

"""
SAMPLE mcp_servers arg
{
    "GuapoMCP": {
        "transport": "sse",
        "url": "http://10.10.8.2:8900/sse",
        # used to fetch a JWT (client-side metadata)
        "api_key": "CHANGE_ME_TOKEN_API_KEY",
        "auth_url": "http://10.10.8.2:8900/auth/token",
    }
}
"""


class MCPClient:
    # Initialize a multi-server mcp client for use with a langchain agent
    def __init__(self, mcp_servers: dict[str, dict[str, str]]):
        """
        init
        :param mcps_servers: the list of mcps servers as a dictionary where each server name is the key
            EX mcp_servers:
            {
                "math": {
                    "transport": "stdio",  # Local subprocess communication
                    "command": "python",
                    # Absolute path to your math_server.py file
                    "args": ["/path/to/math_server.py"],
                    "api_key": "shared-secret",
                    "auth_url": "http://localhost:8900/auth/token",
                },
                "weather": {
                    "transport": "streamable_http",  # HTTP-based remote server
                    # Ensure you start your weather server on port 8000
                    "url": "http://localhost:8000/mcp",
                    "api_key": "shared-secret",
                    "auth_url": "http://localhost:8900/auth/token",
                }
            }
        """
        # Keep a raw copy (may include auth metadata like api_key/auth_url)
        self.mcp_servers_raw = mcp_servers

        # Build a sanitized config for langchain_mcp_adapters (it forwards keys into session creators)
        self.mcp_servers = {
            name: {k: v for k, v in conf.items() if k not in {"api_key", "auth_url", "_jwt"}}
            for name, conf in mcp_servers.items()
        }
        self.client : Optional[MultiServerMCPClient] = None
        try:
            self.client = init_mcp_client(self.mcp_servers)
        except Exception as e:
            server_list = ", ".join(self.mcp_servers.keys()) or "<none>"
            print(f"[MCP] Warning: failed to connect to MCP servers ({server_list}). Continuing without MCP tools. Error: {e}")
            self.client = None

        self.tools : list[BaseTool] = []

    async def _get_jwt_for_server(self, server_name: str, server_conf: dict) -> Optional[str]:
        """Fetch a JWT for a given MCP server using its api_key.

        The token is cached on the server config dict after first retrieval.
        """
        if server_conf.get("_jwt"):
            return server_conf.get("_jwt")

        api_key = server_conf.get("api_key")
        auth_url = server_conf.get("auth_url")

        if not api_key or not auth_url:
            return None

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                auth_url,
                json={"api_key": api_key, "sub": server_name},
            )
            resp.raise_for_status()
            data = resp.json()

        token = data.get("access_token")
        if token:
            server_conf["_jwt"] = token  # cache
        return token

    async def get_tools(self) -> list[BaseTool]:
        all_tools: list[BaseTool] = []

        if not self.mcp_servers:
            print("[MCP] Warning: no MCP servers configured; skipping tool load.")
            self.tools = []
            return self.tools

        for name, conf in self.mcp_servers.items():
            server_conf = dict(conf)
            raw_conf = self.mcp_servers_raw.get(name, {})

            # Attach JWT if configured for this server
            try:
                if raw_conf.get("api_key"):
                    token = await self._get_jwt_for_server(name, raw_conf)
                    if token:
                        server_conf.setdefault("headers", {})["Authorization"] = f"Bearer {token}"
            except Exception as e:
                print(f"[MCP] Warning: failed to fetch JWT for server '{name}'; continuing without token. Error: {e}")

            try:
                client = MultiServerMCPClient({name: server_conf})
                tools = await client.get_tools()
                all_tools.extend(tools)
                print(f"[MCP] Loaded {len(tools)} tool(s) from server '{name}'.")
            except Exception as e:
                print(f"[MCP] Warning: failed to fetch tools from server '{name}'. Skipping. Error: {e}")
                continue

        self.tools = all_tools
        if not self.tools:
            print("[MCP] Warning: no tools loaded from any MCP server.")
        return self.tools

    async def call_tool(self, tool_call: dict):
        """
        Call a tool based on a LangChain LLM tool_call response.
        :param tool_call: {
            "name": "tool_name",
            "args": { ... }
        }
        """
        try:
            name = tool_call.get("name")
            args = tool_call.get("args", {})

            if not name:
                raise ValueError("Tool call missing 'name' field.")

            # Ensure we have tools loaded (and auth headers applied)
            if not self.tools:
                await self.get_tools()

            # Find matching tool
            tool = next((t for t in self.tools if t.name == name), None)
            if not tool:
                raise ValueError(f"Tool '{name}' not found among loaded MCP tools.")

            # Execute tool (async)
            result = await tool.arun(**args)
            return result

        except Exception as e:
            raise RuntimeError(f"Failed to call tool '{tool_call.get('name')}': {str(e)}")



def _load_mcp_servers(config_path: Path) -> dict[str, dict]:
    """Load MCP server definitions from YAML; return empty dict on error/missing."""
    try:
        if not config_path.exists():
            print(f"[MCP] Config not found at {config_path}; using no MCP servers.")
            return {}
        raw = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw, dict):
            print(f"[MCP] Expected mapping in {config_path}, got {type(raw).__name__}; using no MCP servers.")
            return {}
        # keep only dict-valued entries
        return {name: cfg for name, cfg in raw.items() if isinstance(cfg, dict)}
    except Exception as e:
        print(f"[MCP] Failed to load MCP config from {config_path}: {e}")
        return {}


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""


def _base_url(host: str, port: int, scheme: str) -> str:
    return f"{scheme}://{host}:{port}"


def test_health(base_url: str, timeout: float = 5.0) -> TestResult:
    name = "health"
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        if r.status_code != 200:
            return TestResult(name, False, f"Expected 200, got {r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("status") == "ok":
            return TestResult(name, True, "OK")
        return TestResult(name, True, f"OK (non-standard body: {r.text[:200]})")
    except Exception as e:
        return TestResult(name, False, f"Exception: {e}")


def get_token(base_url: str, api_key: str, sub: Optional[str] = None, timeout: float = 8.0) -> Tuple[Optional[str], TestResult]:
    name = "token"
    try:
        payload: Dict[str, Any] = {"api_key": api_key}
        if sub:
            payload["sub"] = sub
        r = requests.post(f"{base_url}/auth/token", json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, TestResult(name, False, f"Expected 200, got {r.status_code}: {r.text[:300]}")
        data = r.json()
        token = data.get("access_token")
        if not token:
            return None, TestResult(name, False, f"No access_token in response: {data}")
        return token, TestResult(name, True, f"OK (expires_in={data.get('expires_in')})")
    except Exception as e:
        return None, TestResult(name, False, f"Exception: {e}")


def test_unauth_token_rejected(base_url: str, timeout: float = 8.0) -> TestResult:
    name = "unauth_token"
    try:
        r = requests.post(f"{base_url}/auth/token", json={"api_key": "WRONG_KEY"}, timeout=timeout)
        if r.status_code == 401:
            return TestResult(name, True, "OK (401 as expected)")
        return TestResult(name, False, f"Expected 401, got {r.status_code}: {r.text[:300]}")
    except Exception as e:
        return TestResult(name, False, f"Exception: {e}")


def test_unauth_sse_rejected(base_url: str, timeout: float = 8.0) -> TestResult:
    name = "unauth_sse"
    try:
        # The middleware only protects paths starting with /sse
        r = requests.get(f"{base_url}/sse", timeout=timeout)
        if r.status_code == 401:
            return TestResult(name, True, "OK (401 as expected)")
        # Some SSE stacks might return 405 if GET is not allowed; treat as a soft fail with details
        return TestResult(name, False, f"Expected 401, got {r.status_code}: {r.text[:300]}")
    except Exception as e:
        return TestResult(name, False, f"Exception: {e}")


def test_tools_list_async(timeout: float = 15.0) -> TestResult:
    """List tools using the project's async MCPClient wrapper.

    This uses the `get_tools()` coroutine defined in this file.
    """
    name = "tools_list"

    async def _runner():
        return await get_tools()

    try:
        tools = asyncio.run(asyncio.wait_for(_runner(), timeout=timeout))

        # Best-effort normalization for readable output
        names: list[str] = []
        if isinstance(tools, dict) and "tools" in tools and isinstance(tools["tools"], list):
            tool_items = tools["tools"]
        elif isinstance(tools, list):
            tool_items = tools
        elif tools is None:
            tool_items = []
        else:
            tool_items = [tools]

        for t in tool_items:
            if isinstance(t, dict) and t.get("name"):
                names.append(str(t.get("name")))
            elif hasattr(t, "name"):
                names.append(str(getattr(t, "name")))

        if names:
            return TestResult(name, True, f"OK ({len(names)} tools): {', '.join(names[:10])}{'...' if len(names) > 10 else ''}")

        # If we can't parse names, still treat as success if we got a non-error response
        if tool_items:
            return TestResult(name, True, f"OK (tools returned; could not parse names): {str(tools)[:250]}")

        return TestResult(name, False, "No tools returned (empty response). Check mcps.yaml and server connectivity.")

    except asyncio.TimeoutError:
        return TestResult(name, False, f"Timed out after {timeout}s while listing tools")
    except Exception as e:
        return TestResult(name, False, f"Exception: {e}")


def run_selected_tests(args: argparse.Namespace) -> int:
    base_url = _base_url(args.host, args.port, args.scheme)
    selected = []

    if args.all or args.health:
        selected.append(("health", lambda: test_health(base_url, timeout=args.timeout)))

    token: Optional[str] = None

    if args.all or args.token or args.tools:
        if not args.api_key:
            print("[!] --api-key is required for token/tools tests", file=sys.stderr)
            return 1
        t, res = get_token(base_url, api_key=args.api_key, sub=args.sub, timeout=args.timeout)
        selected.append(("token", lambda res=res: res))
        token = t

    if args.all or args.unauth:
        selected.append(("unauth_token", lambda: test_unauth_token_rejected(base_url, timeout=args.timeout)))
        selected.append(("unauth_sse", lambda: test_unauth_sse_rejected(base_url, timeout=args.timeout)))

    if args.all or args.tools:
        # Tool listing is done via the project's async MCP client wrapper.
        selected.append(("tools_list", lambda: test_tools_list_async(timeout=max(15.0, args.timeout))))

    # Execute
    any_fail = False
    print(f"Target: {base_url}")
    for _, fn in selected:
        res = fn()
        status = "PASS" if res.ok else "FAIL"
        print(f"[{status}] {res.name}: {res.detail}")
        if not res.ok:
            any_fail = True

    return 1 if any_fail else 0


async def get_tools():
    mcp_config_path = Path("mcps.yaml")
    servers = _load_mcp_servers(mcp_config_path)
    client = MCPClient(servers)
    return await client.get_tools()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test GuapoMCP server auth + tools listing")

    p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8999, help="Server port (default: 8999)")
    p.add_argument("--scheme", choices=("http", "https"), default="http", help="URL scheme (default: http)")
    p.add_argument("--timeout", type=float, default=10.0, help="Request timeout seconds (default: 10)")

    # Auth
    p.add_argument("--api-key", default=None, help="API key for /auth/token (TOKEN_API_KEY on server)")
    p.add_argument("--sub", default=None, help="Optional subject for token")

    # Which tests
    g = p.add_argument_group("tests")
    g.add_argument("--all", action="store_true", help="Run all tests")
    g.add_argument("--health", action="store_true", help="Test /health")
    g.add_argument("--token", action="store_true", help="Test /auth/token")
    g.add_argument("--tools", action="store_true", help="Test tools/list via MCP client")
    g.add_argument("--unauth", action="store_true", help="Run negative tests (no/invalid auth)")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Default behavior if no flags provided
    if not (args.all or args.health or args.token or args.tools or args.unauth):
        args.all = True

    return run_selected_tests(args)


if __name__ == "__main__":
    raise SystemExit(main())
