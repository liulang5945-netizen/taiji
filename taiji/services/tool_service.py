"""Tool service — single source of truth for tool listing.

Aggregates tools from:
1. Built-in tool registry (taiji.agent_ext.tool_registry)
2. MCP servers (taiji.agent_ext.mcp_manager)
3. Dynamic plugins (plugins/ directory)

Used by both runtime status and agent routes.
"""
import os
import logging
from typing import Any

from taiji.core.utils import get_external_path

logger = logging.getLogger("Taiji.Services.Tool")


def list_tools() -> dict:
    """Return all available tools with metadata.

    Returns:
        {
            "status": "ok" | "partial" | "error",
            "tools": [ToolInfo, ...],
            "count": int,
            "error": str,
        }
    """
    from taiji.agent_ext.tool_registry import registry

    tools = []
    seen = set()
    mcp_error = ""

    # 1. Built-in registry tools
    for tool in registry.list_tools(enabled_only=True):
        seen.add(tool.name)
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "source": tool.source,
            "source_id": tool.source_id,
            "category": tool.category,
            "enabled": tool.enabled,
        })

    # 2. MCP tools (may trigger server connections on first call)
    try:
        from taiji.agent_ext.mcp_manager import mcp_manager

        for tool in mcp_manager.get_all_mcp_tools():
            name = tool.get("name") or tool.get("function", {}).get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": tool.get("description") or tool.get("function", {}).get("description", ""),
                "parameters": tool.get("parameters") or tool.get("function", {}).get("parameters", {}),
                "source": "mcp",
                "source_id": tool.get("server_id") or tool.get("source_id", ""),
                "category": "MCP",
                "enabled": True,
            })
    except Exception as exc:
        mcp_error = f"MCP tools are unavailable: {exc}"

    # 3. Dynamic plugins
    plugins_dir = get_external_path("plugins")
    if os.path.exists(plugins_dir):
        for filename in os.listdir(plugins_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            name = filename[:-3]
            if name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": "Dynamic hot-loaded plugin",
                "parameters": {},
                "source": "plugin",
                "source_id": name,
                "category": "plugin",
                "enabled": True,
            })

    return {
        "status": "partial" if mcp_error else "ok",
        "tools": tools,
        "count": len(tools),
        "error": mcp_error,
    }


def get_registry_schemas() -> list[dict]:
    """Get tool schemas in JSON Schema format from the registry."""
    from taiji.agent_ext.tool_registry import registry

    schemas = registry.get_tool_schemas()
    result = []
    for s in schemas:
        func = s.get("function", {})
        result.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return result


def execute_tool(tool_name: str, tool_args: dict) -> Any:
    """Execute a registered tool by name."""
    from taiji.agent_ext.tool_registry import registry
    return registry.execute(tool_name, tool_args)
