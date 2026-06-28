"""
MCP 服务器管理器
管理 MCP 服务器的生命周期（安装/启动/停止/卸载）、工具桥接、Cline 配置导入。
市场逻辑已提取至 mcp_marketplace.py。
"""
import json
import logging
import os
import shutil
import threading
from typing import Dict, List, Optional

from taiji.agent_ext.mcp_client import MCPClient, MCPServerConfig
from taiji.agent_ext.mcp_marketplace import MCPMarketplace
from taiji.agent_ext.tool_registry import registry, ToolDef

logger = logging.getLogger("MCPManager")


class MCPManager:
    """MCP 服务器生命周期管理器"""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._installed_servers: Dict[str, dict] = {}
        self._lock = threading.Lock()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._config_path = os.path.join(base_dir, "mcp_servers_config.json")
        self.marketplace = MCPMarketplace(config_base_dir=base_dir)

        self._load_config()
        self._auto_start_core()

    # ======================== 配置持久化 ========================

    def _load_config(self):
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self._installed_servers = config.get("servers", {})
                logger.info(f"已加载 MCP 配置: {len(self._installed_servers)} 个已安装服务器")
        except Exception as e:
            logger.error(f"加载 MCP 配置失败: {e}")
            self._installed_servers = {}

    def _save_config(self):
        try:
            config = {"version": "1.0.0", "servers": self._installed_servers}
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 MCP 配置失败: {e}")

    # ======================== 自动启动核心 MCP ========================

    def _auto_start_core(self):
        """后台自动安装并启动核心 MCP 服务器"""
        if not shutil.which("npx"):
            logger.info("npx 不可用，跳过 MCP 自动启动")
            return

        def _do_start():
            for server_id in self.marketplace.CORE_MCP_SERVERS:
                try:
                    if server_id not in self._installed_servers:
                        result = self.install(server_id)
                        if result.get("status") not in ("ok", "already_installed"):
                            logger.warning(f"自动安装 MCP '{server_id}' 失败: {result.get('message')}")
                            continue
                        logger.info(f"已自动安装 MCP: {server_id}")
                    result = self.start(server_id)
                    if result.get("status") in ("ok", "already_running"):
                        logger.info(f"已自动启动 MCP: {server_id} ({result.get('tools_count', 0)} 个工具)")
                    else:
                        logger.warning(f"自动启动 MCP '{server_id}' 失败: {result.get('message')}")
                except Exception as e:
                    logger.warning(f"MCP '{server_id}' 自动启动异常: {e}")

        t = threading.Thread(target=_do_start, daemon=True, name="mcp-auto-start")
        t.start()

    # ======================== 安装/卸载 ========================

    def install(self, server_id: str, custom_config: dict = None) -> dict:
        if server_id in self._installed_servers:
            return {"status": "already_installed", "message": f"服务器 '{server_id}' 已安装"}

        config = custom_config or {}
        if not config:
            entry = self.marketplace.find_by_id(server_id)
            if not entry:
                return {"status": "error", "message": f"未在市场中找到服务器 '{server_id}'"}
            config = entry

        record = {
            "id": server_id,
            "name": config.get("name", server_id),
            "command": config.get("command", "cmd"),
            "args": config.get("args", []),
            "args_template": config.get("args_template", []),
            "env": config.get("env", {}),
            "npm_package": config.get("npm_package", ""),
            "description": config.get("description", ""),
            "icon": config.get("icon", "🧩"),
            "category": config.get("category", "通用"),
            "enabled": True,
            "auto_start": False,
        }

        self._installed_servers[server_id] = record
        self._save_config()
        logger.info(f"已安装 MCP: {server_id}")
        return {"status": "ok", "message": f"服务器 '{config.get('name', server_id)}' 安装成功", "config": record}

    def uninstall(self, server_id: str) -> dict:
        if server_id not in self._installed_servers:
            return {"status": "error", "message": f"服务器 '{server_id}' 未安装"}
        if server_id in self._clients:
            self.stop(server_id)
        del self._installed_servers[server_id]
        self._save_config()
        logger.info(f"已卸载 MCP: {server_id}")
        return {"status": "ok", "message": f"服务器 '{server_id}' 已卸载"}

    def add_custom(self, server_id: str, name: str, command: str, args: list,
                   npm_package: str = "", description: str = "", env: dict = None) -> dict:
        return self.install(server_id, {
            "command": command, "args": args, "name": name,
            "npm_package": npm_package, "description": description,
            "icon": "➕", "category": "自定义", "env": env or {},
        })

    # ======================== 启动/停止 ========================

    def start(self, server_id: str, workspace_path: str = "") -> dict:
        if server_id not in self._installed_servers:
            return {"status": "error", "message": f"服务器 '{server_id}' 未安装"}
        if server_id in self._clients and self._clients[server_id].is_running():
            return {"status": "already_running", "message": f"服务器 '{server_id}' 已在运行"}

        cfg = self._installed_servers[server_id]
        args = list(cfg.get("args", []))
        for tmpl in cfg.get("args_template", []):
            args.append(workspace_path or os.getcwd() if tmpl == "{workspace_path}" else tmpl)

        config = MCPServerConfig(
            id=server_id, name=cfg.get("name", server_id),
            command=cfg.get("command", "cmd"), args=args,
            env=cfg.get("env", {}), npm_package=cfg.get("npm_package", ""),
            description=cfg.get("description", ""), icon=cfg.get("icon", "🧩"),
            category=cfg.get("category", "通用"),
        )

        client = MCPClient(config)
        with self._lock:
            if server_id in self._clients:
                self._clients[server_id].stop()
            self._clients[server_id] = client

        if client.start():
            self._register_tools(server_id, client)
            return {"status": "ok", "message": f"服务器 '{cfg.get('name', server_id)}' 启动成功",
                    "tools": client.get_tool_names(), "tools_count": len(client.get_tools())}
        else:
            with self._lock:
                self._clients.pop(server_id, None)
            return {"status": "error", "message": f"服务器 '{server_id}' 启动失败"}

    def stop(self, server_id: str) -> dict:
        if server_id not in self._clients:
            return {"status": "error", "message": f"服务器 '{server_id}' 未在运行"}
        self._clients[server_id].stop()
        registry.unregister_by_source(server_id)
        with self._lock:
            self._clients.pop(server_id, None)
        logger.info(f"已停止 MCP: {server_id}")
        return {"status": "ok", "message": f"服务器 '{server_id}' 已停止"}

    def restart(self, server_id: str, workspace_path: str = "") -> dict:
        if server_id in self._clients:
            self.stop(server_id)
        return self.start(server_id, workspace_path)

    def shutdown_all(self):
        for sid in list(self._clients.keys()):
            try:
                self.stop(sid)
            except Exception as e:
                logger.error(f"停止服务器 {sid} 失败: {e}")
        logger.info("所有 MCP 服务器已停止")

    # ======================== 工具桥接 ========================

    def _register_tools(self, server_id: str, client: MCPClient):
        registered = 0
        for tool in client.get_tools():
            def _make_proxy(c, t_name):
                def proxy(*args, **kwargs):
                    if kwargs:
                        arguments = kwargs
                    elif args:
                        if len(args) == 1 and isinstance(args[0], dict):
                            arguments = args[0]
                        elif len(args) == 1:
                            arguments = {"input": str(args[0])}
                        else:
                            arguments = {"input": " ".join(str(a) for a in args)}
                    else:
                        arguments = {}
                    return c.call_tool(t_name, arguments)
                proxy.__name__ = f"mcp_{t_name}"
                proxy.__doc__ = f"MCP 工具: {t_name}"
                return proxy

            params = tool.input_schema or {"type": "object", "properties": {}}
            tool_def = ToolDef(
                name=f"mcp_{server_id}_{tool.name}",
                description=f"[MCP:{server_id}] {tool.description}",
                parameters=params,
                func=_make_proxy(client, tool.name),
                source="mcp",
                source_id=server_id,
                enabled=True,
                category=client.config.category,
            )
            registry.register(tool_def)
            registered += 1
        logger.info(f"已注册 {registered} 个 MCP 工具 (来源: {server_id})")

    # ======================== Cline 导入 ========================

    def import_from_cline(self, cline_path: str = "") -> dict:
        if not cline_path:
            cline_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Code",
                                      "User", "globalStorage", "saoudrizwan.claude-dev",
                                      "settings", "cline_mcp_settings.json")
        if not os.path.exists(cline_path):
            return {"status": "error", "message": f"Cline 配置文件不存在: {cline_path}"}

        try:
            with open(cline_path, "r", encoding="utf-8") as f:
                cline = json.load(f)
            imported = 0
            for sid, sc in cline.get("mcpServers", {}).items():
                if sc.get("disabled") or sid in self._installed_servers:
                    continue
                npm_pkg = self._extract_npm(sc.get("args", []))
                entry = self.marketplace.find_by_npm(npm_pkg) if npm_pkg else None
                self._installed_servers[sid] = {
                    "id": sid,
                    "name": entry["name"] if entry else sid.split("/")[-1],
                    "command": sc.get("command", "cmd"),
                    "args": sc.get("args", []),
                    "args_template": [],
                    "env": sc.get("env", {}),
                    "npm_package": npm_pkg,
                    "description": entry["description"] if entry else f"从 Cline 导入: {sid}",
                    "icon": entry["icon"] if entry else "🧩",
                    "category": entry["category"] if entry else "导入",
                    "enabled": True,
                    "auto_start": False,
                }
                imported += 1
            self._save_config()
            return {"status": "ok", "message": f"从 Cline 导入 {imported} 个 MCP 服务器",
                    "imported": imported, "total": len(cline.get("mcpServers", {}))}
        except Exception as e:
            return {"status": "error", "message": f"导入失败: {e}"}

    @staticmethod
    def _extract_npm(args: list) -> str:
        for arg in args:
            if arg.startswith("@") or (not arg.startswith("-") and not arg.startswith("/") and arg != "npx"):
                return arg
        for i, arg in enumerate(args):
            if arg == "-y" and i + 1 < len(args):
                return args[i + 1]
        return ""

    # ======================== 查询 ========================

    def get_marketplace(self, category: str = "", keyword: str = "") -> dict:
        servers = self.marketplace.query(category, keyword)
        for s in servers["servers"]:
            s["installed"] = s["id"] in self._installed_servers
            s["running"] = s["id"] in self._clients and self._clients[s["id"]].is_running()
            if s["id"] in self._installed_servers:
                s["enabled"] = self._installed_servers[s["id"]].get("enabled", True)
        return servers

    def get_server_detail(self, server_id: str) -> Optional[dict]:
        detail = self.marketplace.find_by_id(server_id)
        if detail:
            detail = dict(detail)
            detail["installed"] = server_id in self._installed_servers
            detail["running"] = server_id in self._clients and self._clients[server_id].is_running()
            if server_id in self._clients:
                detail["runtime_info"] = self._clients[server_id].get_server_info()
        return detail

    def list_installed(self) -> list:
        result = []
        for sid, cfg in self._installed_servers.items():
            info = dict(cfg)
            info["installed"] = True
            info["running"] = sid in self._clients and self._clients[sid].is_running()
            if sid in self._clients:
                info["runtime_info"] = self._clients[sid].get_server_info()
            result.append(info)
        return result

    def list_running(self) -> list:
        return [c.get_server_info() for c in self._clients.values() if c.is_running()]

    def list_all_tools(self) -> list:
        return [t.to_info() for t in registry.list_tools(source="mcp")]

    def get_status(self) -> dict:
        return {
            "marketplace_count": self.marketplace.server_count,
            "installed_count": len(self._installed_servers),
            "running_count": sum(1 for c in self._clients.values() if c.is_running()),
            "mcp_tools_count": len(registry.list_tools(source="mcp")),
            "installed_servers": self.list_installed(),
        }

    # ======================== 插 件 市 场（简化委托给 marketplace 数据文件） ========================

    def get_plugin_marketplace(self, category: str = "", keyword: str = "") -> dict:
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin_marketplace.json")
        local_data = {"plugins": [], "categories": []}
        try:
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
        except Exception:
            pass

        remote_path = os.path.join(os.path.dirname(self.marketplace._marketplace_path), "plugin_remote_cache.json")
        remote_data = {"plugins": []}
        try:
            if os.path.exists(remote_path):
                with open(remote_path, "r", encoding="utf-8") as f:
                    remote_data = json.load(f)
        except Exception:
            pass

        local_ids = {p["id"] for p in local_data.get("plugins", [])}
        merged = list(local_data.get("plugins", []))
        for p in remote_data.get("plugins", []):
            if p.get("id") not in local_ids:
                merged.append(p)

        if category:
            merged = [p for p in merged if p.get("category") == category]
        if keyword:
            kw = keyword.lower()
            merged = [p for p in merged if kw in p.get("name", "").lower() or kw in p.get("description", "").lower()]

        return {"plugins": merged, "categories": local_data.get("categories", []),
                "total": len(merged), "last_remote_update": remote_data.get("fetch_time", "")}

    def refresh_plugin_marketplace(self) -> dict:
        import datetime
        import urllib.request

        def _do_refresh():
            try:
                sources = [("https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md", "awesome_mcp")]
                plugins = []
                for url, src in sources:
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "Taiji-Agent/1.0"})
                        with urllib.request.urlopen(req, timeout=15) as resp:
                            import re
                            repos = re.findall(r'https://github\.com/([\w-]+/[\w-]+)', resp.read().decode())
                            for repo in repos[:30]:
                                if any(k in repo.lower() for k in ["plugin", "tool", "agent", "mcp"]):
                                    plugins.append({
                                        "id": repo.replace("/", "-"),
                                        "name": repo.split("/")[-1].replace("-", " ").title(),
                                        "icon": "🔗", "category": "开发工具",
                                        "description": f"来自 GitHub: {repo}",
                                        "github": f"https://github.com/{repo}",
                                        "author": repo.split("/")[0], "source": src,
                                    })
                    except Exception:
                        pass
                if plugins:
                    cache = os.path.join(os.path.dirname(self.marketplace._marketplace_path), "plugin_remote_cache.json")
                    with open(cache, "w", encoding="utf-8") as f:
                        json.dump({"plugins": plugins, "fetch_time": datetime.datetime.now().isoformat(), "count": len(plugins)}, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"刷新插件市场失败: {e}")

        t = threading.Thread(target=_do_refresh, daemon=True, name="plugin-market-refresh")
        t.start()
        return {"status": "refreshing", "message": "正在从远程源刷新插件市场..."}

    # ======================== 向后兼容别名 ========================
    install_server = install
    uninstall_server = uninstall
    start_server = start
    stop_server = stop
    restart_server = restart
    add_custom_server = add_custom
    get_installed_servers = list_installed
    get_running_servers = list_running
    get_all_mcp_tools = list_all_tools

    def refresh_marketplace(self) -> dict:
        """刷新 MCP 市场（委托给 marketplace 模块）"""
        return self.marketplace.refresh()

    def __del__(self):
        try:
            self.shutdown_all()
        except Exception:
            pass


# 惰性单例 — 首次访问时才初始化，避免 import 时副作用
_mcp_instance: Optional[MCPManager] = None
_mcp_lock = threading.Lock()


def get_mcp_manager() -> MCPManager:
    """获取 MCPManager 单例（惰性初始化，首次调用才触发线程/远程拉取）。"""
    global _mcp_instance
    if _mcp_instance is None:
        with _mcp_lock:
            if _mcp_instance is None:
                _mcp_instance = MCPManager()
    return _mcp_instance


def __getattr__(name: str):
    """向后兼容：模块级 mcp_manager 属性 → 惰性获取单例"""
    if name == "mcp_manager":
        return get_mcp_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
