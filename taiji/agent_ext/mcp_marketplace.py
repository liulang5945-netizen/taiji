"""
MCP 市场模块
提供 MCP 服务器和插件的市场浏览、npm/GitHub 远程拉取、分类等功能。
从 mcp_manager.py 提取，减少主管理器的职责范围。
"""
import json
import logging
import os
import re
import threading
import urllib.request
from typing import Optional

logger = logging.getLogger("MCPMarketplace")


class MCPMarketplace:
    """MCP 服务器 & 插件市场管理器"""

    # 核心 MCP 服务器的硬编码配置（兜底用）
    CORE_SERVER_DEFAULTS = {
        "playwright": {
            "id": "playwright",
            "name": "Playwright Browser Automation",
            "npm_package": "@playwright/mcp",
            "command": "cmd",
            "args": ["/c", "npx", "-y", "@playwright/mcp"],
            "description": "浏览器自动化（Playwright MCP）",
            "icon": "🌐",
            "category": "浏览器",
        },
        "fetch": {
            "id": "fetch",
            "name": "Web Fetch",
            "npm_package": "mcp-server-fetch",
            "command": "python",
            "args": ["-m", "mcp_server_fetch"],
            "description": "网页内容抓取（HTTP fetch）",
            "icon": "📥",
            "category": "网络",
        },
    }

    # 辅助 MCP 自动启动
    CORE_MCP_SERVERS = ["playwright", "fetch"]

    def __init__(self, config_base_dir: str = ""):
        base = config_base_dir or os.path.dirname(os.path.abspath(__file__))
        self._marketplace_path = os.path.join(base, "mcp_marketplace.json")
        self._data: dict = {"version": "1.0.0", "servers": [], "categories": []}
        self._load()

    # ======================== 初始化 ========================

    def _load(self):
        """加载市场数据（本地 + 远程缓存合并，本地优先）"""
        local_data = {"version": "1.0.0", "servers": [], "categories": []}
        try:
            if os.path.exists(self._marketplace_path):
                with open(self._marketplace_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
        except Exception as e:
            logger.error(f"加载本地市场数据失败: {e}")

        remote_path = os.path.join(os.path.dirname(self._marketplace_path), "mcp_remote_cache.json")
        remote_data = {"servers": []}
        try:
            if os.path.exists(remote_path):
                with open(remote_path, "r", encoding="utf-8") as f:
                    remote_data = json.load(f)
        except Exception:
            pass

        local_ids = {s["id"] for s in local_data.get("servers", [])}
        merged = list(local_data.get("servers", []))
        for s in remote_data.get("servers", []):
            if s.get("id") not in local_ids:
                merged.append(s)
                local_ids.add(s["id"])

        self._data = {
            "version": local_data.get("version", "1.0.0"),
            "categories": local_data.get("categories", ["核心", "开发", "网络", "浏览器", "AI推理", "媒体", "系统", "数据库", "云服务", "效率"]),
            "servers": merged,
            "last_remote_update": remote_data.get("fetch_time", ""),
        }
        logger.info(f"已加载 MCP 市场数据: {len(merged)} 个服务器")

    # ======================== 远程刷新 ========================

    def refresh(self) -> dict:
        """从远程源刷新 MCP 市场数据（后台线程）"""
        def _do_refresh():
            try:
                servers = self._fetch_remote()
                if servers:
                    cache_path = os.path.join(os.path.dirname(self._marketplace_path), "mcp_remote_cache.json")
                    import datetime
                    cache = {"servers": servers, "fetch_time": datetime.datetime.now().isoformat(), "count": len(servers)}
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2, ensure_ascii=False)
                    self._load()
                    logger.info(f"远程 MCP 市场刷新完成: {len(servers)} 个服务器")
                else:
                    logger.warning("远程 MCP 市场未返回数据")
            except Exception as e:
                logger.error(f"刷新远程 MCP 市场失败: {e}")

        t = threading.Thread(target=_do_refresh, daemon=True, name="mcp-market-refresh")
        t.start()
        return {"status": "refreshing", "message": "正在从远程源刷新 MCP 市场..."}

    def _fetch_remote(self) -> list:
        """从 npm + GitHub 获取 MCP 服务器列表"""
        all_servers = []
        seen = set()

        # npm 搜索
        for url, size in [("text=mcp+server+model+context+protocol", 50), ("text=mcp-server", 100)]:
            try:
                req = urllib.request.Request(f"https://registry.npmjs.org/-/v1/search?{url}&size={size}",
                                             headers={"User-Agent": "Taiji-MCP/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                for obj in data.get("objects", []):
                    pkg = obj.get("package", {})
                    name = pkg.get("name", "")
                    if not name or name in seen:
                        continue
                    known_prefixes = ["@modelcontextprotocol/server-", "@anthropic/", "@executeautomation/",
                                      "@upstash/", "@agentdeskai/", "@benborla29/", "mcp-"]
                    is_mcp = any(name.startswith(p) for p in known_prefixes) or "mcp-server" in name or "mcp_server" in name or "mcp" in name.lower()
                    if not is_mcp:
                        continue
                    seen.add(name)
                    desc = pkg.get("description", "")
                    links = pkg.get("links", {})
                    all_servers.append({
                        "id": name.replace("/", "-").replace("@", ""),
                        "name": desc[:40] if desc else name,
                        "icon": "📦",
                        "category": self._guess_category(name, desc),
                        "npm_package": name,
                        "description": desc,
                        "command": "cmd",
                        "args": ["/c", "npx", "-y", name],
                        "args_template": [],
                        "tools_count": 0,
                        "rating": 3,
                        "author": pkg.get("publisher", {}).get("username", ""),
                        "version": pkg.get("version", ""),
                        "npm_url": links.get("npm", f"https://www.npmjs.com/package/{name}"),
                        "github_url": links.get("repository", ""),
                        "source": "npm_search",
                        "tags": self._extract_tags(name, desc),
                    })
                logger.info(f"npm 搜索获取 {len(all_servers)} 个 MCP 包")
            except Exception as e:
                logger.warning(f"npm 搜索失败: {e}")

        # GitHub awesome-mcp-servers
        try:
            url = "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
            req = urllib.request.Request(url, headers={"User-Agent": "Taiji-MCP/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
                npm_links = re.findall(r'https?://(?:www\.)?npmjs\.com/package/([@\w/-]+)', content)
                for pkg_name in npm_links[:50]:
                    if pkg_name not in seen and ("mcp" in pkg_name.lower() or "server" in pkg_name.lower()):
                        seen.add(pkg_name)
                        all_servers.append({
                            "id": pkg_name.replace("/", "-").replace("@", ""),
                            "name": pkg_name.split("/")[-1].replace("server-", "").replace("mcp-", "").replace("-", " ").title(),
                            "icon": "⭐",
                            "category": self._guess_category(pkg_name, ""),
                            "npm_package": pkg_name,
                            "description": "来自 awesome-mcp-servers 社区推荐",
                            "command": "cmd",
                            "args": ["/c", "npx", "-y", pkg_name],
                            "args_template": [],
                            "tools_count": 0,
                            "rating": 4,
                            "author": "community",
                            "source": "awesome_list",
                            "tags": ["社区推荐"],
                        })
                logger.info(f"awesome-mcp-servers 获取 {len(npm_links)} 个包链接")
        except Exception as e:
            logger.warning(f"GitHub awesome-mcp-servers 获取失败: {e}")

        return all_servers

    # ======================== 分类 & 标签 ========================

    @staticmethod
    def _guess_category(name: str, desc: str) -> str:
        text = (name + " " + desc).lower()
        cats = {
            "数据库": ["database", "sql", "postgres", "mysql", "redis", "mongo", "sqlite"],
            "浏览器": ["browser", "playwright", "puppeteer", "web", "fetch", "scrape"],
            "网络": ["search", "brave", "google", "wikipedia"],
            "开发": ["git", "github", "gitlab", "ci", "deploy", "docker", "kubernetes"],
            "AI推理": ["memory", "think", "reason", "agent", "chain"],
            "核心": ["file", "fs", "directory", "path"],
            "效率": ["slack", "discord", "email", "calendar", "notion"],
            "媒体": ["image", "video", "audio", "media", "tts"],
            "系统": ["system", "process", "monitor", "log"],
            "云服务": ["aws", "azure", "gcloud", "cloud", "s3"],
        }
        for cat, keywords in cats.items():
            if any(k in text for k in keywords):
                return cat
        return "开发"

    @staticmethod
    def _extract_tags(name: str, desc: str) -> list:
        tags = set()
        text = (name + " " + desc).lower()
        tag_kw = {"数据库": ["database", "sql", "db"], "搜索": ["search", "query"],
                   "文件": ["file", "filesystem"], "浏览器": ["browser", "web"],
                   "Git": ["git", "version"], "API": ["api", "rest", "http"],
                   "AI": ["ai", "llm", "agent"], "云": ["cloud", "aws", "azure"]}
        for tag, keywords in tag_kw.items():
            if any(k in text for k in keywords):
                tags.add(tag)
        return list(tags)[:5]

    # ======================== 查询 ========================

    def query(self, category: str = "", keyword: str = "") -> dict:
        """浏览/搜索市场"""
        servers = self._data.get("servers", [])
        if category:
            servers = [s for s in servers if s.get("category") == category]
        if keyword:
            kw = keyword.lower()
            servers = [s for s in servers if
                       kw in s.get("name", "").lower() or kw in s.get("description", "").lower() or
                       kw in s.get("npm_package", "").lower() or any(kw in tag for tag in s.get("tags", []))]
        return {"servers": servers, "categories": self._data.get("categories", []), "total": len(servers)}

    def find_by_id(self, server_id: str) -> Optional[dict]:
        """查找服务器，核心服务器有硬编码兜底"""
        for s in self._data.get("servers", []):
            if s["id"] == server_id:
                return s
        if server_id in self.CORE_SERVER_DEFAULTS:
            return self.CORE_SERVER_DEFAULTS[server_id]
        return None

    def find_by_npm(self, npm_package: str) -> Optional[dict]:
        if not npm_package:
            return None
        for s in self._data.get("servers", []):
            if s.get("npm_package") == npm_package:
                return s
        return None

    @property
    def server_count(self) -> int:
        return len(self._data.get("servers", []))

    def get_market_data(self) -> dict:
        return dict(self._data)
