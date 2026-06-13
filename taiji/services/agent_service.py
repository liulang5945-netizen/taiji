"""Agent service — ReAct engine and tool execution.

Wraps agent_ext operations so routes don't directly depend on experimental modules.
This is the last service to extract because it touches memory, tools, workspace, MCP, ReAct.
"""
import logging
from typing import Any

from taiji.core.app_state import app_state

logger = logging.getLogger("Taiji.Services.Agent")


def run_react_task(task: str, max_steps: int = 15) -> dict:
    """Run a ReAct task synchronously. Returns structured result."""
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


def run_react_stream(task: str, max_steps: int = 15):
    """Run a ReAct task as a stream of events. Yields dicts."""
    from taiji.agent_ext.react_engine import ReActEngine
    from taiji.agent_ext.memory_manager import memory

    memory.add_message("user", task)
    engine = ReActEngine(max_steps=max_steps)
    app_state._active_agent_engine = engine

    for event in engine.run_stream(task):
        event_type = event.get("type", "unknown")

        if event_type == "final":
            answer = event.get("data", {}).get("answer", "")
            if answer:
                memory.add_message("assistant", answer)
                memory.remember(f"任务: {task}\n结果: {answer[:200]}", category="tasks")

        yield event


def cancel_active_task() -> str:
    """Cancel the currently active ReAct task. Returns status message."""
    engine = getattr(app_state, "_active_agent_engine", None)
    if engine and hasattr(engine, "cancel"):
        engine.cancel()
        return "已发送取消信号"
    trainer = getattr(app_state, "_trainer_ref", None)
    if trainer and hasattr(trainer, "cancel"):
        trainer.cancel()
        return "已发送取消信号"
    return "没有正在运行的任务"


def list_roles() -> list[dict]:
    """List all available Agent roles."""
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
    return roles


def collaborate(task: str) -> dict:
    """Run multi-Agent collaboration."""
    from taiji.agent_ext.multi_agent import orchestrator

    collab_task = orchestrator.decompose_task(task)
    return orchestrator.execute_task(collab_task.id)


def list_collab_tasks() -> list:
    """List all collaboration tasks."""
    from taiji.agent_ext.multi_agent import orchestrator
    return orchestrator.list_tasks()


def get_collab_messages(topic: str = "", limit: int = 50) -> list:
    """Get Agent communication messages."""
    from taiji.agent_ext.multi_agent import message_bus
    return message_bus.get_messages(topic=topic or None, limit=limit)
