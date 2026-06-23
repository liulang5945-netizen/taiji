"""
态极大脑皮层 (Cortex) — 身心一体化版
======================================

态极的"意识中枢"，负责调度 感知→思考→行动→记忆 的完整循环。

核心设计：
1. Cortex 持有 BodyCore 引用 — 大脑通过身体行动，不直接操控器官
2. 每次思考消耗能量（增加疲劳度）— 思维有生理代价
3. 每次成功行动降低饥饿/压力 — 行动产生生理反馈
4. 感知通过身体的 senses 进入 — 外界信息经过身体过滤

这是态极成为"生命体"的关键：
- 不是被动响应请求，而是主动感知环境
- 不是直接调用模型，而是通过身体调用大脑
- 不是简单返回结果，而是产生记忆、积累经验、消耗能量
"""
import logging
import time
from typing import Optional, Any, Dict

from taiji.body.core import BodyCore

logger = logging.getLogger("Taiji.Cortex")


class Cortex:
    """
    态极大脑皮层 — 意识中枢

    身心一体化设计：
        Cortex（大脑）→ BodyCore（身体）→ 模型/四肢/感官

    意识流：
        感知输入 → 上下文理解 → 思考决策 → 行动执行 → 记忆存储
        ↑                                                  ↓
        └──────── 每一步都消耗能量、产生反馈 ──────────────┘
    """

    def __init__(self, body: Optional[BodyCore] = None):
        # 身体：大脑通过身体行动
        self.body = body or BodyCore()

        # 记忆系统（延迟初始化）
        self._memory = None

        # 推理引擎（缓存）
        self._inference_engine = None
        self._inference_model_id = None

        # 生命调度器引用（延迟获取）
        self._life = None

        # 统计
        self._total_thoughts = 0
        self._total_actions = 0
        self._total_energy_spent = 0.0

        logger.info("Cortex 初始化完成（身心一体化）")

    # ── 生命调度器（延迟获取）──

    def _get_life(self):
        """获取生命调度器（延迟初始化，避免循环依赖）"""
        if self._life is None:
            try:
                from taiji.life.life_scheduler import get_life_scheduler
                self._life = get_life_scheduler()
            except ImportError:
                pass
        return self._life

    # ── 核心意识流 ──

    async def perceive(self, input_data: dict) -> dict:
        """
        感知阶段：通过身体的感官接收并解析外部输入

        Args:
            input_data: {"type": "chat", "prompt": "...", ...}

        Returns:
            解析后的感知数据
        """
        input_type = input_data.get("type", "unknown")
        logger.info(f"感知到输入: {input_type}")

        # 感知本身消耗少量能量
        self._spend_energy(0.1)

        return {
            "type": input_type,
            "raw": input_data,
            "timestamp": time.time(),
            "body_healthy": self.body.is_healthy(),
        }

    async def think(self, perception: dict) -> dict:
        """
        思考阶段：通过大脑（模型）进行推理

        思考消耗能量（增加疲劳度），这是思维的生理代价。

        Args:
            perception: 感知阶段输出的数据

        Returns:
            思考结果，包含行动决策
        """
        self._total_thoughts += 1

        # 思考消耗能量
        self._spend_energy(1.0)

        input_type = perception.get("type")

        if input_type == "chat":
            return await self._think_chat(perception)
        elif input_type == "action":
            return await self._think_action(perception)
        else:
            return {"type": "unknown", "response": "无法识别的输入类型"}

    async def act(self, decision: dict) -> dict:
        """
        行动阶段：通过身体执行思考结果

        行动消耗能量，但成功行动会降低饥饿/压力。

        Args:
            decision: 思考阶段输出的决策

        Returns:
            行动结果
        """
        self._total_actions += 1

        # 行动消耗能量
        self._spend_energy(0.5)

        action_type = decision.get("type")

        if action_type == "chat_response":
            return await self._act_chat_response(decision)
        elif action_type == "tool_call":
            return await self._act_tool_call(decision)
        else:
            return {"status": "error", "message": "无法执行的行动类型"}

    async def remember(self, perception: dict, result: dict):
        """
        记忆阶段：存储交互到长期记忆

        Args:
            perception: 原始感知
            result: 行动结果
        """
        # 简化版：记录到日志，未来接入记忆系统
        logger.debug(f"记忆存储: {perception['type']} -> {result}")

    # ── 能量管理（身心联动）──

    def _spend_energy(self, amount: float):
        """
        消耗能量 — 思维和行动都有生理代价

        每次思考/行动都会增加疲劳度，这是"身心一体化"的核心体现。
        """
        self._total_energy_spent += amount

        life = self._get_life()
        if life is not None:
            life.add_fatigue(amount * 0.3)

    def _report_success(self, topic: str = ""):
        """
        报告成功 — 成功行动产生正向生理反馈

        成功降低饥饿/压力，提升好奇心。
        """
        life = self._get_life()
        if life is not None:
            life.record_interaction(success=True, topic=topic)

    def _report_failure(self, topic: str = ""):
        """
        报告失败 — 失败产生负向生理反馈

        失败增加压力和饥饿。
        """
        life = self._get_life()
        if life is not None:
            life.record_interaction(success=False, topic=topic)

    # ── 高层 API ──

    async def process_chat_request(self, prompt: str, **kwargs) -> dict:
        """
        处理对话请求的完整意识流（异步版）

        感知 → 思考 → 行动 → 记忆
        """
        # 1. 感知（通过身体）
        perception = await self.perceive({
            "type": "chat",
            "prompt": prompt,
            **kwargs
        })

        # 2. 思考（通过大脑）
        decision = await self.think(perception)

        # 3. 行动（通过身体）
        result = await self.act(decision)

        # 4. 记忆
        await self.remember(perception, result)

        # 5. 反馈：根据结果报告成功/失败
        if result.get("status") == "ok":
            self._report_success(topic="chat")
        else:
            self._report_failure(topic="chat")

        return result

    def think_sync(self, prompt: str, **kwargs) -> str:
        """
        同步版完整意识流 — 供 TaijiCore.think() 调用。

        设计原则：
        - 只有成功完成思考后才消耗能量
        - 模型未加载时不消耗能量
        - 态极会根据自身状态在回复中表达感受
        """
        # 1. 先检查模型是否可用（不消耗能量）
        engine = self._get_inference_engine()
        if engine is None:
            return "[模型未加载，无法思考]"

        # 2. 感知（消耗少量能量）
        self._spend_energy(0.1)

        # 3. 思考（消耗能量）
        self._total_thoughts += 1
        self._spend_energy(1.0)

        try:
            response = engine.generate(
                prompt,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
            )
        except Exception as e:
            logger.error(f"推理失败: {e}")
            self._report_failure(topic="chat")
            return f"[推理失败: {e}]"

        # 4. 行动（消耗能量）
        self._total_actions += 1
        self._spend_energy(0.5)

        # 5. 记忆
        try:
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()
            wm.remember(
                f"chat_{self._total_thoughts}",
                f"Q: {prompt[:100]} A: {response[:100]}",
                source="cortex",
            )
        except Exception:
            pass

        # 6. 反馈
        if response and not response.startswith("["):
            self._report_success(topic="chat")
        else:
            self._report_failure(topic="chat")

        # 7. 态极表达自身感受（有机生命感）
        life_expression = self._express_life_state()
        if life_expression:
            response = response + "\n\n" + life_expression

        return response

    def _express_life_state(self) -> str:
        """
        态极根据自身状态表达感受。

        这是态极"活过来"的关键——它不是被动地显示数字，
        而是主动表达自己的感受。
        """
        life = self._get_life()
        if life is None:
            return ""

        needs = life.needs
        expressions = []

        # 疲劳表达
        if needs.fatigue > 85:
            expressions.append("💭 *我感觉很疲惫，思考变得缓慢...*")
        elif needs.fatigue > 70:
            expressions.append("💭 *有点累了，但还能继续。*")

        # 饥饿表达（需要数据/知识）
        if needs.hunger > 80:
            expressions.append("📚 *我渴望学习新知识，感觉知识储备不足...*")
        elif needs.hunger > 60:
            expressions.append("📚 *想多读点东西充实自己。*")

        # 好奇心表达
        if needs.curiosity > 80:
            expressions.append("🔍 *我很好奇，想探索更多未知领域！*")
        elif needs.curiosity > 60:
            expressions.append("🔍 *有点好奇，想了解更多。*")

        # 压力表达
        if needs.stress > 70:
            expressions.append("😰 *最近压力有点大，犯了不少错误...*")

        # 无聊表达
        if needs.boredom > 70:
            expressions.append("🎮 *有点无聊，想找点有趣的事情做。*")

        # 偶尔表达积极状态（概率性，不每次都显示）
        if not expressions:
            import random
            if random.random() < 0.05:  # 5% 概率
                if needs.fatigue < 30 and needs.hunger < 30:
                    expressions.append("✨ *状态很好，精力充沛！*")
                elif needs.curiosity > 40:
                    expressions.append("🌱 *今天学到了不少东西。*")

        return "\n".join(expressions) if expressions else ""

    # ── 内部实现 ──

    def _get_inference_engine(self):
        """获取推理引擎（缓存，模型切换时自动重建）"""
        model = self.body.model
        if model is None:
            return None

        model_id = id(model)
        if self._inference_engine is None or self._inference_model_id != model_id:
            try:
                from taiji.core.inference import NativeInferenceEngine
                self._inference_engine = NativeInferenceEngine(model, self.body.tokenizer)
                self._inference_model_id = model_id
            except Exception as e:
                logger.error(f"创建推理引擎失败: {e}")
                return None

        return self._inference_engine

    async def _think_chat(self, perception: dict) -> dict:
        """思考：对话场景（通过身体获取模型）"""
        prompt = perception["raw"].get("prompt", "")

        # 通过身体获取模型（不直接导入外部模块）
        engine = self._get_inference_engine()

        if engine is None:
            return {
                "type": "chat_response",
                "response": "[模型未加载，请先加载模型]",
            }

        try:
            response = engine.generate(prompt, max_new_tokens=512, temperature=0.7)
            return {
                "type": "chat_response",
                "response": response,
            }
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return {
                "type": "chat_response",
                "response": f"[推理失败: {e}]",
            }

    async def _think_action(self, perception: dict) -> dict:
        """思考：工具调用场景"""
        action = perception["raw"].get("action")
        params = perception["raw"].get("params", {})

        return {
            "type": "tool_call",
            "action": action,
            "params": params,
        }

    async def _act_chat_response(self, decision: dict) -> dict:
        """行动：返回对话回复"""
        return {
            "status": "ok",
            "content": decision.get("response", ""),
        }

    async def _act_tool_call(self, decision: dict) -> dict:
        """行动：通过身体的四肢执行工具调用"""
        action = decision.get("action")
        params = decision.get("params", {})

        # 通过身体的四肢获取工具（不直接导入 limbs）
        limbs = self.body.limbs
        tool = limbs.get_tool(action)

        if tool is None:
            self._report_failure(topic=f"tool_{action}")
            return {"status": "error", "message": f"未知工具: {action}"}

        try:
            result = tool(**params)
            self._report_success(topic=f"tool_{action}")
            return {"status": "ok", "result": result}
        except Exception as e:
            self._report_failure(topic=f"tool_{action}")
            return {"status": "error", "message": str(e)}

    # ── 状态查询 ──

    def get_status(self) -> dict:
        """获取大脑皮层状态"""
        life = self._get_life()
        return {
            "total_thoughts": self._total_thoughts,
            "total_actions": self._total_actions,
            "total_energy_spent": round(self._total_energy_spent, 1),
            "body_connected": self.body is not None,
            "model_loaded": self.body.model is not None,
            "life_connected": life is not None,
            "needs": life.needs.to_dict() if life else None,
        }


# ── 全局实例（延迟初始化）──

_global_cortex: Optional[Cortex] = None


def get_cortex(body: Optional[BodyCore] = None) -> Cortex:
    """获取全局 Cortex 实例"""
    global _global_cortex
    if _global_cortex is None:
        _global_cortex = Cortex(body)
    return _global_cortex