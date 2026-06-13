"""
态极大脑系统 (Brain System)
============================

态极的大脑 — 天生的意识中枢。

- backbone.py:  Transformer 骨架（ModelSelf）— 从零训练的大脑
- cortex.py:     大脑皮层 — 感知→思考→行动→记忆的意识流
- inference.py:  推理引擎 — 生成文本
- heads/:        多头模块（语言、工具、感知、记忆、规划）
"""

from taiji.brain.cortex import Cortex

__all__ = ["Cortex"]