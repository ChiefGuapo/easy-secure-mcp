# ---- Tool auto-registration helpers ----

from pathlib import Path
import importlib.util
import inspect
import sys
from types import ModuleType
from typing import Iterable, Set, Union
from fastmcp import FastMCP

def register_tools_from_dir(server: FastMCP, tools_dir: Union[str, Path], *, recursive: bool = True) -> None:
    """Discover and register all functions decorated with @tool_label from a tools directory.

    This will:
      - Walk the directory for .py files
      - Import each module by file path (no sys.path hacks required)
      - Find callables with the `_tool_label` marker added by @tool_label
      - Register them with `server.add_tool(func)`

    Args:
        server: An already-initialized MCP server instance.
        tools_dir: Path to the directory containing tool modules.
        recursive: Whether to recurse into subdirectories.
    """
    tools_path = Path(tools_dir).resolve()
    if not tools_path.exists() or not tools_path.is_dir():
        raise FileNotFoundError(f"Tools directory not found or not a directory: {tools_path}")

    py_files: Iterable[Path]
    py_files = tools_path.rglob("*.py") if recursive else tools_path.glob("*.py")

    registered: Set[str] = set()

    for py_file in py_files:
        # Skip package markers / private modules
        if py_file.name.startswith("__"):
            continue

        module = _import_module_from_path(py_file)

        # Inspect module members; we only register functions marked by @tool_label
        for _, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and getattr(obj, "_tool_label", False):
                tool_name = getattr(obj, "__tool_name__", obj.__name__)

                # Avoid double-registering by name
                if tool_name in registered:
                    continue

                # MCP server registration
                t = server.tool(obj)
                server.add_tool(t)
                registered.add(tool_name)


def _import_module_from_path(path: Path) -> ModuleType:
    """Import a python module from an absolute file path.

    We avoid modifying sys.path by using importlib's file-location loading.
    Each loaded module gets a unique name to prevent collisions.
    """
    unique_name = f"mcp_tools_{path.stem}_{abs(hash(str(path)))}"

    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module
