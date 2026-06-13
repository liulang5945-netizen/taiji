"""
Agent 核心功能 API 路由（精简版）
保留：工具列表、ReAct 推理引擎、工具注册表、多 Agent 协作

已拆分到独立文件的功能：
- routes_agent_workspace.py → 工作台文件操作、代码执行、项目脚手架、插件上传
- routes_agent_mcp.py      → MCP 服务器市场、安装/卸载、启动/停止
- routes_agent_memory.py   → 记忆系统（短期/工作/长期记忆）
"""
import json
import logging
import os

from fastapi import APIRouter, HTTPException

from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Agent")
router = APIRouter()


# ======================== Agent 工具列表 ========================

@router.get("/api/agent/tools")
def list_agent_tools():
    """列出已加载的 Agent 工具"""
    try:
        from taiji.agent_ext.tool_registry import registry

        tools = []
        seen = set()
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
        except Exception as e:
            logger.debug(f"MCP 工具同步失败: {e}")

        plugins_dir = get_external_path("plugins")
        if os.path.exists(plugins_dir):
            for f in os.listdir(plugins_dir):
                if f.endswith('.py') and not f.startswith('_'):
                    name = f.replace('.py', '')
                    if name not in seen:
                        tools.append({
                            "name": name,
                            "description": "动态热挂载插件",
                            "source": "plugin",
                            "source_id": name,
                            "category": "插件",
                            "enabled": True,
                        })
        return {"status": "ok", "tools": tools, "count": len(tools)}
    except Exception as e:
        logger.error(f"获取 Agent 工具失败: {e}")
        return {"status": "error", "message": str(e), "tools": []}


# ======================== ReAct 推理引擎 ========================

@router.post("/api/agent/react")
async def react_task(req: dict):
    """使用 ReAct 引擎执行自主推理任务"""
    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    max_steps = req.get("max_steps", 15)

    try:
        from taiji.agent_ext.react_engine import AgentController
        from taiji.agent_ext.memory_manager import memory

        memory.add_message("user", task)

        controller = AgentController(max_steps=max_steps)
        app_state._active_agent_engine = controller
        try:
            result = controller.run_task(task)
        finally:
            app_state._active_agent_engine = None

        if result.final_answer:
            memory.add_message("assistant", result.final_answer)

        if result.status == "completed" and result.final_answer:
            memory.remember(f"任务: {task}\n结果: {result.final_answer[:200]}", category="tasks")

        return {
            "status": result.status,
            "final_answer": result.final_answer,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_args": s.action_args,
                    "observation": s.observation,
                    "is_final": s.is_final,
                    "error": s.error,
                    "duration_ms": round(s.duration_ms, 1),
                }
                for s in result.steps
            ],
            "total_duration_ms": round(result.total_duration_ms, 1),
            "total_steps": len(result.steps),
        }
    except Exception as e:
        logger.error(f"ReAct 任务执行失败: {e}")
        return {"status": "error", "message": "Agent 任务执行失败，请查看服务器日志获取详情"}


@router.post("/api/agent/react/stream")
async def react_task_stream(req: dict):
    """ReAct 流式推理（SSE）"""
    from fastapi.responses import StreamingResponse
    import asyncio

    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    max_steps = req.get("max_steps", 15)

    async def event_generator():
        try:
            from taiji.agent_ext.react_engine import ReActEngine
            from taiji.agent_ext.memory_manager import memory

            memory.add_message("user", task)
            engine = ReActEngine(max_steps=max_steps)
            app_state._active_agent_engine = engine

            for event in engine.run_stream(task):
                event_type = event.get("type", "unknown")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"

                if event_type == "final":
                    answer = event.get("data", {}).get("answer", "")
                    if answer:
                        memory.add_message("assistant", answer)
                        memory.remember(f"任务: {task}\n结果: {answer[:200]}", category="tasks")

                await asyncio.sleep(0.01)

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/agent/cancel")
async def cancel_agent():
    """取消当前正在执行的 ReAct 任务"""
    try:
        # 尝试取消存储在 app_state 中的活跃引擎
        engine = getattr(app_state, "_active_agent_engine", None)
        if engine and hasattr(engine, "cancel"):
            engine.cancel()
            return {"status": "ok", "message": "已发送取消信号"}
        # 回退：尝试从 trainer 取消
        trainer = getattr(app_state, "_trainer_ref", None)
        if trainer and hasattr(trainer, "cancel"):
            trainer.cancel()
            return {"status": "ok", "message": "已发送取消信号"}
        return {"status": "ok", "message": "没有正在运行的任务"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================== 工具注册表 ========================

@router.get("/api/agent/tools/registry")
def list_tool_registry():
    """列出工具注册表中所有已注册的工具（JSON Schema 格式）"""
    try:
        from taiji.agent_ext.tool_registry import registry
        schemas = registry.get_tool_schemas()
        tools_info = []
        for s in schemas:
            func = s.get("function", {})
            tools_info.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
        return {"status": "ok", "tools": tools_info, "count": len(tools_info)}
    except Exception as e:
        logger.error(f"获取工具注册表失败: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/api/agent/tools/execute")
async def execute_tool(req: dict):
    """直接执行一个注册的工具"""
    tool_name = req.get("tool", "").strip()
    tool_args = req.get("args", {})
    if not tool_name:
        raise HTTPException(status_code=400, detail="工具名不能为空")

    try:
        from taiji.agent_ext.tool_registry import registry
        result = registry.execute(tool_name, tool_args)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================== 多 Agent 协作 ========================

@router.get("/api/agent/roles")
def list_roles():
    """列出所有可用的 Agent 角色"""
    try:
        from taiji.agent_ext.multi_agent import orchestrator
        roles = []
        for name, role in orchestrator.roles.items():
            roles.append({
                "name": name,
                "display_name": role.name,
                "description": role.description,
                "allowed_tools": role.allowed_tools,
                "max_steps": role.max_steps,
            })
        return {"status": "ok", "roles": roles, "count": len(roles)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/agent/collaborate")
async def collaborate(req: dict):
    """多 Agent 协作执行任务"""
    task = req.get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="任务不能为空")

    try:
        from taiji.agent_ext.multi_agent import orchestrator

        collab_task = orchestrator.decompose_task(task)
        result = orchestrator.execute_task(collab_task.id)

        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"多 Agent 协作失败: {e}")
        return {"status": "error", "message": "多 Agent 协作失败，请查看服务器日志获取详情"}


@router.get("/api/agent/collaborate/tasks")
def list_collab_tasks():
    """列出所有协作任务"""
    try:
        from taiji.agent_ext.multi_agent import orchestrator
        tasks = orchestrator.list_tasks()
        return {"status": "ok", "tasks": tasks}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/api/agent/collaborate/messages")
def collab_messages(topic: str = "", limit: int = 50):
    """获取 Agent 间通信消息"""
    try:
        from taiji.agent_ext.multi_agent import message_bus
        messages = message_bus.get_messages(topic=topic or None, limit=limit)
        return {"status": "ok", "messages": messages, "count": len(messages)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
