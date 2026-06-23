"""
工具注册表 (Tool Registry)
==========================
统一管理所有 Agent 工具的注册、查找、执行。
支持本地工具和 MCP 远程工具。

使用方式:
    from taiji.agent_ext.tool_registry import registry
    
    # 注册本地工具
    registry.register(ToolDef(
        name="my_tool",
        description="工具描述",
        parameters={"type": "object", "properties": {...}},
        func=my_function,
    ))
    
    # 执行工具
    result = registry.execute("my_tool", {"arg1": "value1"})
    
    # 获取所有工具的 JSON Schema（用于 LLM function calling）
    schemas = registry.get_tool_schemas()
"""
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ToolRegistry")


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    func: Optional[Callable] = None
    source: str = "local"          # "local" | "mcp" | "plugin"
    source_id: str = ""            # 来源标识，如 MCP 服务器名
    enabled: bool = True
    category: str = "通用"

    def to_schema(self) -> dict:
        """转换为 OpenAI function calling 格式的 JSON Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_info(self) -> dict:
        """转换为前端展示用的信息字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "source_id": self.source_id,
            "enabled": self.enabled,
            "category": self.category,
        }


class ToolRegistry:
    """工具注册表：管理所有工具的注册、查找、执行"""

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}

    # ======================== 注册 ========================

    def register(self, tool: ToolDef):
        """注册一个工具"""
        if not tool.name:
            logger.warning("尝试注册无名工具，已忽略")
            return
        if tool.name in self._tools:
            logger.info(f"工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool
        logger.debug(f"已注册工具: {tool.name} (来源: {tool.source})")

    def register_many(self, tools: List[ToolDef]):
        """批量注册工具"""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> bool:
        """注销一个工具"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"已注销工具: {name}")
            return True
        return False

    def unregister_by_source(self, source_id: str):
        """注销来自特定来源的所有工具"""
        to_remove = [name for name, t in self._tools.items() if t.source_id == source_id]
        for name in to_remove:
            del self._tools[name]
        logger.info(f"已注销来源 '{source_id}' 的 {len(to_remove)} 个工具")

    # ======================== 查询 ========================

    def get(self, name: str) -> Optional[ToolDef]:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self, source: str = None, enabled_only: bool = True) -> List[ToolDef]:
        """列出工具"""
        tools = list(self._tools.values())
        if source:
            tools = [t for t in tools if t.source == source]
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    def list_names(self, enabled_only: bool = True) -> List[str]:
        """列出所有工具名"""
        return [t.name for t in self.list_tools(enabled_only=enabled_only)]

    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    def count(self) -> int:
        """工具总数"""
        return len(self._tools)

    # ======================== 启用/禁用 ========================

    def enable(self, name: str):
        """启用工具"""
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str):
        """禁用工具"""
        if name in self._tools:
            self._tools[name].enabled = False

    def enable_source(self, source_id: str):
        """启用来自特定来源的所有工具"""
        for t in self._tools.values():
            if t.source_id == source_id:
                t.enabled = True

    def disable_source(self, source_id: str):
        """禁用来自特定来源的所有工具"""
        for t in self._tools.values():
            if t.source_id == source_id:
                t.enabled = False

    # ======================== 执行 ========================

    def execute(self, name: str, args: dict) -> str:
        """执行指定工具"""
        tool = self._tools.get(name)
        if not tool:
            return f"❌ 工具 '{name}' 不存在。可用工具: {', '.join(self.list_names())}"
        if not tool.enabled:
            return f"❌ 工具 '{name}' 已禁用"
        if not tool.func:
            return f"❌ 工具 '{name}' 没有可执行的函数"

        try:
            # 支持两种参数传递方式：
            # 1. 如果工具参数只有一个 "input" 属性，直接传字符串
            # 2. 否则传整个字典
            params = tool.parameters.get("properties", {})
            if len(params) == 1 and "input" in params:
                # 单参数 "input" 模式
                if isinstance(args, dict):
                    value = args.get("input", args.get("value", json.dumps(args, ensure_ascii=False)))
                else:
                    value = str(args)
                result = tool.func(str(value))
            elif len(params) == 1:
                # 单参数模式（参数名不是 "input"）
                param_name = list(params.keys())[0]
                if isinstance(args, dict):
                    value = args.get(param_name, json.dumps(args, ensure_ascii=False))
                else:
                    value = str(args)
                result = tool.func(str(value))
            else:
                # 多参数模式
                if isinstance(args, dict):
                    result = tool.func(**args)
                else:
                    result = tool.func(str(args))

            return str(result) if result is not None else "✅ 工具执行完成（无返回值）"
        except TypeError as e:
            # 参数类型不匹配，尝试用 input 字符串重试
            try:
                if isinstance(args, dict):
                    result = tool.func(json.dumps(args, ensure_ascii=False))
                else:
                    result = tool.func(str(args))
                return str(result) if result is not None else "✅ 工具执行完成"
            except Exception as retry_e:
                return f"❌ 工具 '{name}' 参数错误: {retry_e}"
        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {traceback.format_exc()}")
            return f"❌ 工具 '{name}' 执行失败: {e}"

    # ======================== Schema 导出 ========================

    def get_tool_schemas(self, enabled_only: bool = True) -> list:
        """获取所有工具的 JSON Schema（用于 LLM function calling）"""
        tools = self.list_tools(enabled_only=enabled_only)
        return [t.to_schema() for t in tools]

    def get_tool_descriptions(self, enabled_only: bool = True) -> str:
        """获取所有工具的文本描述（用于 prompt）"""
        tools = self.list_tools(enabled_only=enabled_only)
        if not tools:
            return "暂无可用工具。"

        lines = ["可用工具:"]
        for t in tools:
            params = t.parameters.get("properties", {})
            param_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
            if param_desc:
                lines.append(f"- **{t.name}**({param_desc}): {t.description}")
            else:
                lines.append(f"- **{t.name}**: {t.description}")
        return "\n".join(lines)

    def get_all_info(self, enabled_only: bool = True) -> list:
        """获取所有工具的详细信息（用于前端展示）"""
        return [t.to_info() for t in self.list_tools(enabled_only=enabled_only)]

    def clear(self):
        """清空所有工具"""
        self._tools.clear()

    def __repr__(self):
        return f"ToolRegistry(tools={len(self._tools)})"


# ======================== 全局单例 ========================

registry = ToolRegistry()


# ── 自修改工具注册（让态极能自主发现和安装新工具） ──
def _register_self_modification_tools():
    """注册自修改工具到全局注册表"""
    try:
        from taiji.agent_ext.self_modification import get_self_modification_engine
        _sm_engine = get_self_modification_engine()

        def _discover_tools(input_str: str) -> str:
            """搜索可用工具。输入: 能力描述关键词"""
            keyword = input_str.strip()
            if not keyword:
                return "请输入要搜索的能力关键词，如：浏览器、翻译、数据库"
            matches = _sm_engine._discovery.find_matching_tools(keyword, registry)
            if not matches:
                return f"未找到与 '{keyword}' 匹配的工具"
            lines = [f"找到 {len(matches)} 个匹配工具:"]
            for m in matches:
                lines.append(f"  [{m['source']}] {m['name']} - {m['description'][:60]} (匹配度: {m['match_score']:.1f})")
            return "\n".join(lines)

        def _install_tool(input_str: str) -> str:
            """安装一个新工具。输入: 工具ID（MCP服务器ID或插件ID）"""
            tool_id = input_str.strip()
            if not tool_id:
                return "请输入要安装的工具ID"
            result = _sm_engine._try_install({"source": "mcp", "id": tool_id}, registry)
            if result.get("success"):
                return f"✅ {result['message']}"
            result = _sm_engine._try_install({"source": "plugin", "id": tool_id}, registry)
            if result.get("success"):
                return f"✅ {result['message']}"
            return f"❌ 安装失败: {result.get('message', '未知错误')}"

        def _my_capabilities(input_str: str) -> str:
            """查看当前已具备的能力"""
            tools = registry.list_tools(enabled_only=True)
            lines = [f"当前已注册 {len(tools)} 个工具:"]
            for t in tools:
                lines.append(f"  - {t.name}: {t.description[:50]}")
            return "\n".join(lines)

        registry.register(ToolDef(
            name="discover_tools",
            description="搜索可用的新工具，根据能力关键词在MCP市场和插件目录中查找",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "能力关键词，如：浏览器、翻译、数据库"}}},
            func=_discover_tools,
            source="self_modification",
            category="自修改",
        ))
        registry.register(ToolDef(
            name="install_tool",
            description="安装一个新的MCP服务器或插件工具",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "工具ID"}}},
            func=_install_tool,
            source="self_modification",
            category="自修改",
        ))
        def _evolve(input_str: str) -> str:
            """自主进化：遇到不会的能力时，自动搜索或编写工具补齐。输入: 缺失的能力描述"""
            ability = input_str.strip()
            if not ability:
                return "请描述你需要但不具备的能力"
            result = _sm_engine.evolve(ability, registry)
            if result.get("evolved"):
                return f"✅ 进化成功! {result.get('message', '')}"
            return f"❌ 进化失败: {result.get('message', '未知原因')}"

        registry.register(ToolDef(
            name="my_capabilities",
            description="查看当前已具备的所有工具能力列表",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "留空即可"}}},
            func=_my_capabilities,
            source="self_modification",
            category="自修改",
        ))
        registry.register(ToolDef(
            name="evolve",
            description="自主进化：遇到不具备的能力时，自动搜索市场或自己编写代码生成工具补齐能力",
            parameters={"type": "object", "properties": {"input": {"type": "string", "description": "缺失的能力描述，如：翻译、数据库操作、图表生成"}}},
            func=_evolve,
            source="self_modification",
            category="自修改",
        ))
        logger.info("自修改工具已注册: discover_tools, install_tool, my_capabilities, evolve")
    except Exception as e:
        logger.debug(f"自修改工具注册失败: {e}")


def register_local_tools():
    """注册所有本地内置工具到注册表"""
    from taiji.agent_ext.agent_tools import (
        read_local_file, write_file, edit_file, delete_file,
        list_directory, create_directory, create_project,
        install_dependency, analyze_code,
    )
    from taiji.agent_ext.agent_planner import (
        create_plan, update_plan, get_plan, list_plans,
        save_context, load_context,
    )

    local_tools = [
        ToolDef(
            name="read_local_file",
            description="读取工作台文件内容，支持分页。输入文件路径（可选逗号后跟页码）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "文件路径，如 data.txt, 2 表示第2页"}
            }, "required": ["input"]},
            func=read_local_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="write_file",
            description="在工作台中创建或覆盖写入文件。输入格式: 文件路径 | 文件内容",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 文件路径 | 文件内容"}
            }, "required": ["input"]},
            func=write_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="edit_file",
            description="精确编辑工作台中的文件内容。输入格式: 文件路径 | 旧文本 | 新文本",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 文件路径 | 旧文本 | 新文本"}
            }, "required": ["input"]},
            func=edit_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="delete_file",
            description="删除工作台中的文件或空目录。输入文件路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "要删除的文件或目录路径"}
            }, "required": ["input"]},
            func=delete_file,
            source="local", category="文件",
        ),
        ToolDef(
            name="list_directory",
            description="列出工作台目录内容。输入目录路径（留空则列出根目录）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "目录路径（可选）"}
            }},
            func=list_directory,
            source="local", category="文件",
        ),
        ToolDef(
            name="create_directory",
            description="在工作台中创建目录。输入目录路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "要创建的目录路径"}
            }, "required": ["input"]},
            func=create_directory,
            source="local", category="文件",
        ),
        ToolDef(
            name="create_project",
            description="创建完整项目脚手架。输入格式: 项目类型 | 项目名。支持: python-script, web-app, vue-app 等。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 项目类型 | 项目名"}
            }, "required": ["input"]},
            func=create_project,
            source="local", category="开发",
        ),
        ToolDef(
            name="install_dependency",
            description="安装 Python 依赖包。输入包名。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "包名，如 requests"}
            }, "required": ["input"]},
            func=install_dependency,
            source="local", category="开发",
        ),
        ToolDef(
            name="analyze_code",
            description="分析代码文件语法。支持 .py, .js, .json。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "文件路径"}
            }, "required": ["input"]},
            func=analyze_code,
            source="local", category="开发",
        ),
        ToolDef(
            name="create_plan",
            description="为复杂开发任务创建执行计划。输入任务描述。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "任务描述"}
            }, "required": ["input"]},
            func=create_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="update_plan",
            description="更新计划步骤状态。输入格式: 计划ID | 步骤序号 | 状态(done/failed/skip)",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: 计划ID | 步骤序号 | 状态"}
            }, "required": ["input"]},
            func=update_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="get_plan",
            description="获取任务计划进度。输入计划ID或'all'查看全部。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "计划ID或'all'"}
            }, "required": ["input"]},
            func=get_plan,
            source="local", category="规划",
        ),
        ToolDef(
            name="save_context",
            description="保存开发上下文信息。输入格式: key | value",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "格式: key | value"}
            }, "required": ["input"]},
            func=save_context,
            source="local", category="规划",
        ),
        ToolDef(
            name="load_context",
            description="读取已保存的上下文信息。输入key。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "上下文key"}
            }, "required": ["input"]},
            func=load_context,
            source="local", category="规划",
        ),
    ]

    # Python 代码执行
    try:
        from taiji.agent_ext.sandbox_executor import execute_python_code_safe
        local_tools.append(ToolDef(
            name="execute_python",
            description="在安全沙箱中执行 Python 代码。输入必须是纯 Python 代码。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "Python 代码"}
            }, "required": ["input"]},
            func=execute_python_code_safe,
            source="local", category="开发",
        ))
    except ImportError:
        pass

    # 搜索引擎
    try:
        from taiji.agent_ext.agent import _create_robust_search
        import json as _json
        # 读取用户搜索配置
        _search_engine = "智能多核"
        _search_key = ""
        _ui_settings = {}
        try:
            from taiji.services.settings_service import load_settings

            _ui_settings = load_settings()
            _search_engine = _ui_settings.get("search_engine", "\u667a\u80fd\u591a\u6838")
            _search_key = _ui_settings.get("search_key", "")
        except Exception:
            pass
        # 优先使用统一 web 模块，回退到旧版搜索
        try:
            from taiji.tools.web import web_search
            search_func = web_search
        except Exception:
            search_func = _create_robust_search(_search_engine, _ui_settings, _search_key, "")

        local_tools.append(ToolDef(
            name="search",
            description="在互联网上搜索最新信息。输入简短的搜索关键词。支持 DuckDuckGo、Bing、百度多引擎。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "搜索关键词"}
            }, "required": ["input"]},
            func=search_func,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 网页阅读（使用统一 web 模块）
    try:
        from taiji.tools.web import web_fetch
        local_tools.append(ToolDef(
            name="read_webpage",
            description="深入阅读指定网址的网页正文，自动转为 Markdown。输入完整 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=web_fetch,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 高级浏览器访问（优先 Playwright MCP，降级到 read_webpage）
    try:
        def _browse_web(url: str) -> str:
            """浏览器访问网页，支持 JS 渲染。优先用 Playwright MCP，降级到 requests。"""
            # 优先尝试 Playwright MCP
            pw_tools = [n for n in registry.list_names() if "playwright" in n and ("get_content" in n or "navigate" in n)]
            if pw_tools:
                try:
                    # 先导航
                    nav_tool = [n for n in pw_tools if "navigate" in n]
                    if nav_tool:
                        registry.execute(nav_tool[0], {"url": url})
                    # 再获取内容
                    content_tool = [n for n in pw_tools if "get_content" in n or "get_text" in n or "page_content" in n]
                    if content_tool:
                        result = registry.execute(content_tool[0], {})
                        if result and len(str(result).strip()) > 50:
                            return str(result)[:8000]
                except Exception as e:
                    logger.debug(f"Playwright MCP 浏览失败，降级: {e}")

            # 降级到 read_webpage
            try:
                from taiji.agent_ext.agent import read_webpage
                return read_webpage(url)
            except Exception:
                return f"无法访问: {url}"

        local_tools.append(ToolDef(
            name="browse_web",
            description="用浏览器访问网页（支持 JavaScript 渲染的动态页面）。输入完整 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=_browse_web,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 智能网页抓取（优先 MCP fetch，降级到 requests）
    try:
        def _smart_fetch(url: str) -> str:
            """智能抓取网页，返回 Markdown 格式正文。优先用 MCP fetch，降级到 requests。"""
            # 优先尝试 MCP fetch
            fetch_tools = [n for n in registry.list_names() if "fetch" in n and "markdown" in n]
            if fetch_tools:
                try:
                    result = registry.execute(fetch_tools[0], {"url": url})
                    if result and len(str(result).strip()) > 50:
                        return str(result)[:10000]
                except Exception as e:
                    logger.debug(f"MCP fetch 失败，降级: {e}")

            # 降级到 urllib + regex（纯 stdlib）
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode('utf-8', errors='ignore')
                # 移除无用标签并提取文本
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<aside[^>]*>.*?</aside>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<(?:br|p|div|h[1-6]|li)[^>]*/?>', '\n', text, flags=re.IGNORECASE)
                text = re.sub(r'<[^>]+>', '', text)
                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&nbsp;', ' ')
                text = re.sub(r'\n{3,}', '\n\n', text)
                return text.strip()[:10000]
            except Exception as e:
                return f"抓取失败: {e}"

        local_tools.append(ToolDef(
            name="smart_fetch",
            description="智能抓取网页正文，返回 Markdown 格式。适合文章、文档、博客等。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL 地址"}
            }, "required": ["input"]},
            func=_smart_fetch,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # 命令行执行
    try:
        from taiji.agent_ext.agent import run_command
        local_tools.append(ToolDef(
            name="run_command",
            description="运行本地命令行命令（安全白名单）。输入命令字符串。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "命令字符串"}
            }, "required": ["input"]},
            func=run_command,
            source="local", category="系统",
        ))
    except Exception:
        pass

    # B 站字幕
    try:
        from taiji.tools.bilibili_subtitle import read_bilibili_subtitle
        local_tools.append(ToolDef(
            name="read_bilibili_subtitle",
            description="读取 B 站视频官方 CC 字幕。输入 B 站视频 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "B 站视频 URL"}
            }, "required": ["input"]},
            func=read_bilibili_subtitle,
            source="local", category="媒体",
        ))
    except ImportError:
        pass

    # 通用知识自学习工具
    try:
        from taiji.agent_ext.knowledge_learner import get_knowledge_learner
        _learner = get_knowledge_learner()

        def _learn_knowledge(input_str: str) -> str:
            """启动某领域的知识学习。输入格式: 领域名 | 来源URL1,URL2,... | 深度(shallow/medium/deep)"""
            try:
                parts = [p.strip() for p in input_str.split("|")]
                domain = parts[0]
                sources = None
                depth = "medium"
                if len(parts) > 1 and parts[1]:
                    sources = [s.strip() for s in parts[1].split(",") if s.strip()]
                if len(parts) > 2 and parts[2]:
                    depth = parts[2].strip()
                session = _learner.start_learning(domain, sources=sources, depth=depth)
                return (
                    f"✅ 学习完成 [{domain}]\n"
                    f"状态: {session.status}\n"
                    f"采集: {session.entries_collected} 条\n"
                    f"新增: {session.entries_new} | 更新: {session.entries_updated} | 跳过: {session.entries_skipped}\n"
                    f"验证得分: {session.verify_score:.0%}\n"
                    + "\n".join(session.log[-5:])
                )
            except Exception as e:
                return f"❌ 学习失败: {e}"

        def _query_knowledge(input_str: str) -> str:
            """查询已学习的知识。输入: 查询问题（可选 | 领域名 限定范围）"""
            try:
                parts = [p.strip() for p in input_str.split("|")]
                question = parts[0]
                domain = parts[1] if len(parts) > 1 else ""
                return _learner.query(question, domain=domain)
            except Exception as e:
                return f"❌ 查询失败: {e}"

        def _learning_report(input_str: str) -> str:
            """查看学习进度报告。输入: 领域名（留空查看全部）"""
            try:
                return _learner.get_learning_report(domain=input_str.strip())
            except Exception as e:
                return f"❌ 获取报告失败: {e}"

        local_tools.extend([
            ToolDef(
                name="learn_knowledge",
                description="启动某领域的知识自学习（自动采集、结构化、存储、验证）。输入格式: 领域名 | 来源URL(逗号分隔,可选) | 深度(shallow/medium/deep)",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "格式: 领域名 | 来源URL | 深度"}
                }, "required": ["input"]},
                func=_learn_knowledge,
                source="local", category="学习",
            ),
            ToolDef(
                name="query_knowledge",
                description="查询已学习的知识库。输入问题（可选 | 领域名）。",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "查询问题（可选|领域名）"}
                }, "required": ["input"]},
                func=_query_knowledge,
                source="local", category="学习",
            ),
            ToolDef(
                name="learning_report",
                description="查看知识学习进度报告。输入领域名（留空查看全部）。",
                parameters={"type": "object", "properties": {
                    "input": {"type": "string", "description": "领域名（可选）"}
                }},
                func=_learning_report,
                source="local", category="学习",
            ),
        ])
    except Exception as e:
        logger.warning(f"知识学习工具注册失败: {e}")

    # ═══════════════════════════════════════════════
    # 补充工具：完整能力，不做限制
    # ═══════════════════════════════════════════════

    # 时间日期
    def _datetime(input_str: str = "") -> str:
        import datetime, time
        now = datetime.datetime.now()
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        return (f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"星期: {weekdays[now.weekday()]}\n"
                f"时间戳: {time.time():.0f}")

    local_tools.append(ToolDef(
        name="datetime",
        description="获取当前日期、时间、星期几。输入留空即可。",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "留空即可"}
        }},
        func=_datetime,
        source="local", category="工具",
    ))

    # 计算器 — 完整数学能力
    def _calculator(input_str: str) -> str:
        """数学计算，支持所有 math 模块函数"""
        import math, cmath, ast, operator, statistics
        allowed_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.Mod: operator.mod,
            ast.USub: operator.neg, ast.FloorDiv: operator.floordiv,
            ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
            ast.BitXor: operator.xor, ast.Invert: operator.invert,
            ast.LShift: operator.lshift, ast.RShift: operator.rshift,
        }
        # 完整 math 函数集
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        allowed_names.update({
            'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf,
            'sqrt': math.sqrt, 'cbrt': lambda x: x ** (1/3),
            'abs': abs, 'round': round, 'int': int, 'float': float,
            'min': min, 'max': max, 'sum': sum, 'len': len,
            'pow': pow, 'divmod': divmod,
            'mean': statistics.mean, 'median': statistics.median,
            'stdev': statistics.stdev, 'variance': statistics.variance,
            'factorial': math.factorial, 'comb': math.comb, 'perm': math.perm,
            'gcd': math.gcd, 'lcm': math.lcm,
        })
        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in allowed_names:
                    return allowed_names[fname](*[_eval(a) for a in node.args])
            elif isinstance(node, ast.Name) and node.id in allowed_names:
                return allowed_names[node.id]
            elif isinstance(node, ast.List):
                return [_eval(e) for e in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(_eval(e) for e in node.elts)
            raise ValueError(f"不支持: {ast.dump(node)}")
        try:
            expr = ast.parse(input_str.strip(), mode='eval')
            result = _eval(expr)
            return f"计算结果: {input_str.strip()} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    local_tools.append(ToolDef(
        name="calculator",
        description="数学计算。支持所有 math 函数（sin/cos/tan/log/exp/sqrt/factorial/comb/perm/gcd/lcm 等）、统计函数（mean/median/stdev）、位运算。输入数学表达式。",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "数学表达式，如 2**10, log2(1024), factorial(10), mean([1,2,3,4,5])"}
        }, "required": ["input"]},
        func=_calculator,
        source="local", category="工具",
    ))

    # 文本摘要
    def _text_summarize(input_str: str) -> str:
        """提取文本摘要"""
        import re
        text = input_str.strip()
        if len(text) < 200:
            return f"文本较短（{len(text)} 字），无需摘要。\n\n{text}"
        sentences = re.split(r'[。！？.!?\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return text[:500]
        word_freq = {}
        for s in sentences:
            for word in s:
                if len(word) > 1:
                    word_freq[word] = word_freq.get(word, 0) + 1
        scored = []
        for i, s in enumerate(sentences):
            score = sum(word_freq.get(w, 0) for w in s if len(w) > 1)
            score *= (1 - i * 0.02)
            scored.append((score, i, s))
        scored.sort(reverse=True)
        top_count = max(3, len(sentences) // 5)
        top = sorted(scored[:top_count], key=lambda x: x[1])
        summary = '。'.join(s[2] for s in top) + '。'
        return f"【摘要】({len(sentences)}句 → {len(top)}句)\n{summary}"

    local_tools.append(ToolDef(
        name="text_summarize",
        description="对长文本提取摘要。输入需要摘要的文本。",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "需要摘要的长文本"}
        }, "required": ["input"]},
        func=_text_summarize,
        source="local", category="工具",
    ))

    # 文件对比
    def _diff_text(input_str: str) -> str:
        """对比两段文本的差异"""
        import difflib
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "格式错误: 请用 | 分隔两段文本，如 '文本A | 文本B'"
        text_a = parts[0].strip()
        text_b = parts[1].strip()
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        diff = list(difflib.unified_diff(lines_a, lines_b, lineterm='', n=3))
        if not diff:
            return "两段文本完全相同。"
        return "\n".join(diff[:100])

    local_tools.append(ToolDef(
        name="diff_text",
        description="对比两段文本的差异。输入格式: 文本A | 文本B",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "格式: 文本A | 文本B（用 | 分隔）"}
        }, "required": ["input"]},
        func=_diff_text,
        source="local", category="工具",
    ))

    # 数据处理 — 完整查询能力
    def _data_query(input_str: str) -> str:
        """查询/分析 CSV、JSON、TXT 数据文件"""
        import os, json, csv, re
        parts = input_str.split("|", 1)
        file_path = parts[0].strip()
        query = parts[1].strip() if len(parts) > 1 else "head"

        # 尝试多个路径
        candidates = [
            os.path.join("agent_workspace", file_path),
            os.path.join("taiji_data", file_path),
            file_path,
        ]
        safe_path = None
        for p in candidates:
            if os.path.exists(p):
                safe_path = p
                break
        if not safe_path:
            return f"文件不存在: {file_path}"

        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.json':
                with open(safe_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    result = f"JSON 数组，共 {len(data)} 条记录。"
                    if query == "head":
                        result += f"\n前10条:\n{json.dumps(data[:10], ensure_ascii=False, indent=2)}"
                    elif query == "tail":
                        result += f"\n后10条:\n{json.dumps(data[-10:], ensure_ascii=False, indent=2)}"
                    elif query == "count":
                        result += f"\n总记录数: {len(data)}"
                    elif query == "keys" and data:
                        result += f"\n字段: {list(data[0].keys()) if isinstance(data[0], dict) else '非对象数组'}"
                    elif query.startswith("filter"):
                        # filter: key=value
                        match = re.match(r'filter:\s*(\w+)\s*=\s*(.+)', query)
                        if match and isinstance(data[0], dict):
                            k, v = match.group(1), match.group(2)
                            filtered = [d for d in data if str(d.get(k, '')) == v]
                            result += f"\n过滤 {k}={v}: {len(filtered)} 条\n{json.dumps(filtered[:5], ensure_ascii=False, indent=2)}"
                        else:
                            result += "\n过滤格式: filter: key=value"
                    elif query.startswith("select"):
                        # select: key1,key2
                        match = re.match(r'select:\s*(.+)', query)
                        if match and isinstance(data[0], dict):
                            keys = [k.strip() for k in match.group(1).split(',')]
                            selected = [{k: d.get(k) for k in keys} for d in data[:10]]
                            result += f"\n选取 {keys}:\n{json.dumps(selected, ensure_ascii=False, indent=2)}"
                    else:
                        result += f"\n示例:\n{json.dumps(data[:5], ensure_ascii=False, indent=2)}"
                elif isinstance(data, dict):
                    keys = list(data.keys())
                    result = f"JSON 对象，{len(keys)} 个键: {keys[:20]}"
                    if query == "keys":
                        result = f"所有键:\n{json.dumps(keys, ensure_ascii=False)}"
                    elif query.startswith("get"):
                        match = re.match(r'get:\s*(.+)', query)
                        if match:
                            k = match.group(1).strip()
                            val = data.get(k, f"键 '{k}' 不存在")
                            result += f"\n{ k}: {json.dumps(val, ensure_ascii=False, indent=2)}"
                return result

            elif ext == '.csv':
                with open(safe_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames
                    rows = list(reader)
                if not rows:
                    return "CSV 文件为空。"
                result = f"CSV: {len(headers)} 列, {len(rows)} 行\n列名: {', '.join(headers)}"

                if query == "head":
                    result += f"\n前5行:\n"
                    for row in rows[:5]:
                        result += " | ".join(str(row.get(h, ''))[:20] for h in headers[:6]) + "\n"
                elif query == "tail":
                    result += f"\n后5行:\n"
                    for row in rows[-5:]:
                        result += " | ".join(str(row.get(h, ''))[:20] for h in headers[:6]) + "\n"
                elif query == "count":
                    result += f"\n总行数: {len(rows)}"
                elif query == "describe":
                    # 基本统计
                    for h in headers[:5]:
                        vals = [row.get(h, '') for row in rows]
                        numeric = []
                        for v in vals:
                            try:
                                numeric.append(float(v))
                            except:
                                pass
                        if numeric:
                            result += f"\n{h}: min={min(numeric):.2f}, max={max(numeric):.2f}, avg={sum(numeric)/len(numeric):.2f}"
                        else:
                            unique = len(set(vals))
                            result += f"\n{h}: {unique} 个唯一值"
                elif query.startswith("filter"):
                    match = re.match(r'filter:\s*(\w+)\s*=\s*(.+)', query)
                    if match:
                        k, v = match.group(1), match.group(2)
                        filtered = [r for r in rows if str(r.get(k, '')) == v]
                        result += f"\n过滤 {k}={v}: {len(filtered)} 行"
                        for r in filtered[:5]:
                            result += "\n" + " | ".join(str(r.get(h, ''))[:20] for h in headers[:6])
                elif query.startswith("select"):
                    match = re.match(r'select:\s*(.+)', query)
                    if match:
                        keys = [k.strip() for k in match.group(1).split(',')]
                        result += f"\n选取列 {keys}:\n"
                        for r in rows[:10]:
                            result += " | ".join(str(r.get(k, ''))[:20] for k in keys) + "\n"
                return result

            elif ext in ('.txt', '.md', '.log', '.py', '.js', '.ts'):
                with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                lines = content.splitlines()
                result = f"文本文件: {len(lines)} 行, {len(content)} 字符"
                if query == "head":
                    result += f"\n前20行:\n" + "\n".join(lines[:20])
                elif query == "tail":
                    result += f"\n后20行:\n" + "\n".join(lines[-20:])
                elif query == "count":
                    result += f"\n行数: {len(lines)}, 字符数: {len(content)}"
                elif query.startswith("grep"):
                    match = re.match(r'grep:\s*(.+)', query)
                    if match:
                        pattern = match.group(1)
                        matched = [(i+1, l) for i, l in enumerate(lines) if pattern in l]
                        result += f"\n匹配 '{pattern}': {len(matched)} 行"
                        for num, line in matched[:20]:
                            result += f"\n  L{num}: {line[:100]}"
                return result

            else:
                # 尝试作为 JSON 读取
                with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)
                return f"文件内容（前5000字符）:\n{content}"

        except Exception as e:
            return f"读取失败: {e}"

    local_tools.append(ToolDef(
        name="data_query",
        description="查询/分析数据文件（CSV/JSON/TXT）。支持: head/tail/count/keys/describe/filter:key=value/select:key1,key2/grep:关键词。输入格式: 文件路径 | 操作",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "格式: 文件路径 | 操作(head/tail/count/keys/describe/filter:key=val/select:key1,key2/grep:词)"}
        }, "required": ["input"]},
        func=_data_query,
        source="local", category="工具",
    ))

    # URL 检查 — 完整抓取能力
    def _url_check(input_str: str) -> str:
        """检查 URL 并抓取内容"""
        import urllib.request, urllib.error
        parts = input_str.split("|", 1)
        url = parts[0].strip()
        mode = parts[1].strip() if len(parts) > 1 else "check"
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TaijiBot/1.0'}
        try:
            if mode == "fetch":
                # 完整抓取
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode('utf-8', errors='ignore')
                    ct = resp.headers.get('Content-Type', '')
                    return f"URL: {url}\n状态码: {resp.status}\n内容类型: {ct}\n大小: {len(content)} 字节\n\n内容（前3000字）:\n{content[:3000]}"
            else:
                # HEAD 检查
                req = urllib.request.Request(url, method='HEAD', headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return f"URL 可访问\n状态码: {resp.status}\n内容类型: {resp.headers.get('Content-Type', '未知')}\n大小: {resp.headers.get('Content-Length', '未知')} 字节"
        except urllib.error.HTTPError as e:
            return f"HTTP 错误: {e.code} {e.reason}"
        except Exception as e:
            return f"无法访问: {e}"

    local_tools.append(ToolDef(
        name="url_check",
        description="检查 URL 可访问性或抓取内容。输入 URL | 模式(check/fetch)。check=仅检查，fetch=抓取内容。",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "URL | 模式(check/fetch)"}
        }, "required": ["input"]},
        func=_url_check,
        source="local", category="工具",
    ))

    # 正则匹配
    def _regex_match(input_str: str) -> str:
        """正则表达式匹配"""
        import re
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "格式: 正则表达式 | 待匹配文本"
        pattern = parts[0].strip()
        text = parts[1].strip()
        try:
            matches = re.findall(pattern, text)
            if not matches:
                return f"没有匹配到任何内容。\n正则: {pattern}"
            result = f"匹配到 {len(matches)} 项:\n"
            for m in matches[:20]:
                result += f"  - {m}\n"
            return result
        except re.error as e:
            return f"正则表达式错误: {e}"

    local_tools.append(ToolDef(
        name="regex_match",
        description="正则表达式匹配。输入格式: 正则表达式 | 待匹配文本",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "格式: 正则 | 文本"}
        }, "required": ["input"]},
        func=_regex_match,
        source="local", category="工具",
    ))

    # 任务调度
    # 已调度任务存储
    _scheduled_tasks = {}

    def _schedule_task(input_str: str) -> str:
        """调度一个延迟任务（实际执行）"""
        import threading, time, json, os
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "格式: 延迟秒数 | 任务描述"
        try:
            delay = int(parts[0].strip())
            task_desc = parts[1].strip()
            if delay < 0 or delay > 86400:
                return "延迟时间范围: 0-86400 秒（24小时）"

            task_id = f"task_{int(time.time())}_{hash(task_desc) % 10000}"

            def _execute_task():
                """任务执行：将任务描述写入待执行队列"""
                try:
                    queue_path = os.path.join("taiji_data", "task_queue.json")
                    os.makedirs(os.path.dirname(queue_path), exist_ok=True)
                    tasks = []
                    if os.path.exists(queue_path):
                        with open(queue_path, 'r', encoding='utf-8') as f:
                            tasks = json.load(f)
                    tasks.append({
                        "task_id": task_id,
                        "description": task_desc,
                        "scheduled_at": time.time(),
                        "executed_at": time.time(),
                        "status": "pending",
                    })
                    with open(queue_path, 'w', encoding='utf-8') as f:
                        json.dump(tasks, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    pass

            timer = threading.Timer(delay, _execute_task)
            timer.daemon = True
            timer.start()
            _scheduled_tasks[task_id] = {"desc": task_desc, "delay": delay, "timer": timer}

            return f"任务已调度\n任务ID: {task_id}\n描述: {task_desc}\n将在 {delay} 秒后执行\n当前队列: {len(_scheduled_tasks)} 个任务"
        except ValueError:
            return "延迟秒数必须是整数"

    def _list_tasks(input_str: str = "") -> str:
        """查看已调度的任务"""
        if not _scheduled_tasks:
            return "当前没有已调度的任务。"
        result = f"已调度 {len(_scheduled_tasks)} 个任务:\n"
        for tid, info in _scheduled_tasks.items():
            result += f"  [{tid}] {info['desc']} (延迟 {info['delay']}s)\n"
        return result

    local_tools.append(ToolDef(
        name="schedule_task",
        description="调度一个延迟任务（实际执行）。输入格式: 延迟秒数 | 任务描述",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "格式: 秒数 | 任务描述"}
        }, "required": ["input"]},
        func=_schedule_task,
        source="local", category="工具",
    ))

    local_tools.append(ToolDef(
        name="list_tasks",
        description="查看已调度的任务列表。输入留空即可。",
        parameters={"type": "object", "properties": {
            "input": {"type": "string", "description": "留空即可"}
        }},
        func=_list_tasks,
        source="local", category="工具",
    ))

    # ═══════════════════════════════════════════════
    # 桌面自动化（打开任何程序）
    # ═══════════════════════════════════════════════
    try:
        from taiji.tools.desktop import (
            desktop_run_command, desktop_run_program, desktop_system_info,
            desktop_processes, desktop_file_op, open_file_manager, open_terminal, open_editor,
        )

        local_tools.append(ToolDef(
            name="run_program",
            description="启动任意程序。输入格式: 程序名 | 参数(可选)。如: notepad | file.txt",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "程序名 | 参数"}
            }, "required": ["input"]},
            func=desktop_run_program,
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="system_info",
            description="获取系统信息（CPU、内存、磁盘、操作系统）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "留空即可"}
            }},
            func=lambda input="": desktop_system_info(),
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="list_processes",
            description="列出系统进程。输入进程名过滤（可选）。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "进程名（可选）"}
            }},
            func=lambda input="": desktop_processes(input),
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="file_operation",
            description="文件操作：复制/移动/删除/压缩。输入格式: 操作(copy/move/delete/zip) | 源路径 | 目标路径",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "操作 | 源 | 目标"}
            }, "required": ["input"]},
            func=desktop_file_op,
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="open_folder",
            description="打开文件管理器。输入目录路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "目录路径"}
            }, "required": ["input"]},
            func=open_file_manager,
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="open_terminal",
            description="打开终端。输入目录路径（可选）和命令（可选）。格式: 路径 | 命令",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "路径 | 命令（可选）"}
            }},
            func=lambda input="": open_terminal(input.split("|")[0].strip() or ".", input.split("|")[1].strip() if "|" in input else ""),
            source="local", category="桌面",
        ))

        local_tools.append(ToolDef(
            name="open_editor",
            description="打开代码编辑器。输入文件路径。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "文件路径"}
            }, "required": ["input"]},
            func=open_editor,
            source="local", category="桌面",
        ))
    except Exception:
        pass

    # ═══════════════════════════════════════════════
    # SearXNG 搜索（聚合 70+ 搜索引擎）
    # ═══════════════════════════════════════════════
    try:
        from taiji.tools.searxng import searxng_search
        local_tools.append(ToolDef(
            name="searxng_search",
            description="SearXNG 智能搜索：聚合 70+ 搜索引擎，自动选择搜索分类（新闻/学术/技术/图片/视频/音乐/通用）。输入搜索关键词。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "搜索关键词"}
            }, "required": ["input"]},
            func=searxng_search,
            source="local", category="网络",
        ))
    except Exception:
        pass

    # ═══════════════════════════════════════════════
    # 浏览器自动化（Playwright）
    # ═══════════════════════════════════════════════
    try:
        from taiji.tools.browser import browse_open, browse_read, browse_click, browse_search, browse_screenshot

        local_tools.append(ToolDef(
            name="browser_open",
            description="用浏览器打开网页（支持 JavaScript 渲染）。输入 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL"}
            }, "required": ["input"]},
            func=browse_open,
            source="local", category="浏览器",
        ))

        local_tools.append(ToolDef(
            name="browser_read",
            description="用浏览器读取网页内容（JS 渲染后）。输入 URL。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL"}
            }, "required": ["input"]},
            func=browse_read,
            source="local", category="浏览器",
        ))

        local_tools.append(ToolDef(
            name="browser_click",
            description="用浏览器打开网页并点击元素。输入格式: URL | CSS选择器",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL | CSS选择器"}
            }, "required": ["input"]},
            func=lambda input: browse_click(input.split("|")[0].strip(), input.split("|")[1].strip()) if "|" in input else "格式: URL | CSS选择器",
            source="local", category="浏览器",
        ))

        local_tools.append(ToolDef(
            name="browser_search",
            description="用浏览器搜索（自动打开 Google 搜索）。输入搜索关键词。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "搜索关键词"}
            }, "required": ["input"]},
            func=browse_search,
            source="local", category="浏览器",
        ))

        local_tools.append(ToolDef(
            name="browser_screenshot",
            description="截取网页截图。输入 URL。截图保存到 agent_workspace/screenshots/。",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "URL"}
            }, "required": ["input"]},
            func=browse_screenshot,
            source="local", category="浏览器",
        ))
    except Exception:
        pass

    registry.register_many(local_tools)
    logger.info(f"已注册 {len(local_tools)} 个本地工具")


# 模块加载时自动注册本地工具
try:
    register_local_tools()
except Exception as e:
    logger.warning(f"本地工具注册失败（可能尚未初始化）: {e}")

# 注册自修改工具（在本地工具之后，确保 registry 已有工具列表）
try:
    _register_self_modification_tools()
except Exception as e:
    logger.debug(f"自修改工具注册失败: {e}")
