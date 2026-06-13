"""Taiji model service — multimodal operations and native training.

Wraps taiji-specific operations (multimodal, feed, sleep, play, training)
so routes_taiji.py can be a thin API layer.
"""
import logging
import os
import time
from typing import Any

from taiji.core.app_state import app_state

logger = logging.getLogger("Taiji.Services.TaijiModel")


def is_available() -> bool:
    """Check if taiji-specific features are available."""
    return app_state.is_taiji() and app_state.get_taiji_engine() is not None


def _get_engine():
    """Get taiji engine, raise if unavailable."""
    if not is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    return app_state.get_taiji_engine()


def get_status() -> dict:
    """Get taiji engine status."""
    engine = app_state.get_taiji_engine()
    try:
        status = engine.get_status()
        status["available"] = True
        status.pop("name", None)
        return status
    except Exception:
        return {"available": False}


def get_tools() -> list:
    """Get available multimodal tools."""
    engine = app_state.get_taiji_engine()
    return engine.MULTIMODAL_TOOLS


def execute_multimodal(tool: str, params: dict) -> Any:
    """Execute a multimodal tool."""
    engine = app_state.get_taiji_engine()
    return engine._execute_multimodal_tool(tool, params)


def run_agent(task: str, tools: list = None) -> dict:
    """Run a multimodal agent task."""
    from taiji.agent_ext.tool_registry import registry

    engine = app_state.get_taiji_engine()
    tool_reg = tools if tools else registry
    result = engine.run(task, tool_reg)

    return {
        "type": "result",
        "status": result.status,
        "final_answer": result.final_answer,
        "steps": len(result.steps),
        "duration_ms": result.total_duration_ms,
    }


def cancel_agent() -> None:
    """Cancel current multimodal task."""
    engine = app_state.get_taiji_engine()
    engine.cancel()


# ===== Life activities =====

def get_feed_status() -> dict:
    from taiji.life.feed_engine import get_feed_engine
    engine = get_feed_engine()
    return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}


def feed() -> dict:
    from taiji.life.feed_engine import get_feed_engine
    engine = get_feed_engine()
    report = engine.feed(reason="manual")
    return {
        "status": "ok",
        "items_fed": report.items_fed,
        "items_rejected": report.items_rejected,
        "samples_generated": report.samples_generated,
        "avg_quality": report.avg_quality,
        "duration_seconds": report.duration_seconds,
        "recommendations": report.recommendations,
    }


def feed_text(text: str, source: str = "manual", category: str = "knowledge") -> dict:
    from taiji.life.feed_engine import get_feed_engine
    engine = get_feed_engine()
    item = engine.feed_text(text=text, source=source, category=category)
    if item:
        return {
            "status": "ok",
            "quality_score": item.quality_score,
            "sample_count": item.sample_count,
            "item_status": item.status,
        }
    return {"status": "ok", "message": "内容已跳过（重复或质量不达标）"}


def get_sleep_status() -> dict:
    from taiji.life.sleep_engine import get_sleep_engine
    engine = get_sleep_engine()
    return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}


def sleep() -> dict:
    from taiji.life.sleep_engine import get_sleep_engine
    engine = get_sleep_engine()
    report = engine.sleep(reason="manual")
    return {
        "status": "ok",
        "phases_completed": report.phases_completed,
        "training_samples_used": report.training_samples_used,
        "training_loss": report.training_loss,
        "health_status": report.health_status,
        "duration_seconds": report.duration_seconds,
    }


def get_play_status() -> dict:
    from taiji.life.play_engine import get_play_engine
    engine = get_play_engine()
    return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}


def play() -> dict:
    from taiji.life.play_engine import get_play_engine
    engine = get_play_engine()
    report = engine.play(reason="manual")
    activities = []
    for a in report.activities:
        activities.append({
            "type": a.activity_type,
            "topic": a.topic,
            "content": a.content,
            "quality": round(a.quality_score, 2),
            "kept": a.kept,
        })
    return {
        "status": "ok",
        "activities": activities,
        "mood": report.mood,
        "traits_discovered": report.personality_traits_discovered,
        "duration_seconds": report.duration_seconds,
    }


def get_personality() -> dict:
    from taiji.life.play_engine import get_play_engine
    engine = get_play_engine()
    return engine.get_personality()


# ===== Life scheduler =====

def get_life_status() -> dict:
    from taiji.life.life_scheduler import get_life_scheduler
    scheduler = get_life_scheduler()
    return {"status": "ok", "data": scheduler.get_status(), "summary": scheduler.get_summary()}


def start_life() -> None:
    from taiji.life.life_scheduler import get_life_scheduler
    get_life_scheduler().start()


def stop_life() -> None:
    from taiji.life.life_scheduler import get_life_scheduler
    get_life_scheduler().stop()


def record_interaction(success: bool = True, topic: str = "") -> dict:
    from taiji.life.life_scheduler import get_life_scheduler
    scheduler = get_life_scheduler()
    scheduler.record_interaction(success=success, topic=topic)
    return scheduler.needs.to_dict()


def force_life_action(action: str) -> dict:
    from taiji.life.life_scheduler import get_life_scheduler
    scheduler = get_life_scheduler()
    result = scheduler.force_action(action)
    return {"status": "ok", "result": result, "needs": scheduler.needs.to_dict()}


def get_life_timeline(hours: int = 24) -> list:
    from taiji.life.life_scheduler import get_life_scheduler
    return get_life_scheduler().get_timeline(hours=hours)
