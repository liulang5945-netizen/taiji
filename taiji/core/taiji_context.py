"""
TaijiContext — 态极器官容器
将所有生命系统引用统一保存在一个轻量 dataclass 中，
便于测试隔离和依赖注入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaijiContext:
    """态极生命体的完整器官集合。

    每个字段代表态极的一个子系统。构造函数不执行任何初始化逻辑，
    所有器官由 Builder 预先创建后注入。
    """

    # 骨骼
    body: Any = None
    model: Any = None
    tokenizer: Any = None

    # 循环系统
    events: Any = None

    # 免疫系统
    safety: Any = None

    # 大脑皮层
    cortex: Any = None

    # 记忆系统
    context_manager: Any = None

    # 生命引擎
    feed: Any = None
    sleep: Any = None
    play: Any = None
    evolution: Any = None
    explore: Any = None
    science: Any = None
    improver: Any = None
    life_scheduler: Any = None

    # 推理
    inference_engine: Any = None

    # 可选扩展
    action_provider: Any = None
    data_collector: Any = None

    def __repr__(self) -> str:
        loaded = "已加载" if self.model is not None else "未加载"
        running = "运行中" if getattr(self.life_scheduler, 'is_running', lambda: False)() else "暂停"
        return f"TaijiContext(model={loaded}, life={running})"
