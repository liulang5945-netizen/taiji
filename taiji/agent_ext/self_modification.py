"""
态极自修改引擎 (Self-Modification Engine)
==========================================

基于 Self-Refine / Reflexion 论文的思想：
不修改模型权重，而是改进模型的行为策略。

改进方式：
1. system_prompt 优化 — 根据失败模式调整系统指令
2. temperature 自适应 — 根据任务类型动态调整
3. context 策略优化 — 根据任务类型选择记忆注入策略
4. 效果追踪 — 记录每次改进的效果，自动回滚

核心原则：
态极不改自己的大脑（权重），但可以改自己的思维方式（prompt/context）。
"""
import os
import json
import time
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("SelfModification")


@dataclass
class EvaluationResult:
    """对话质量评估结果"""
    response: str
    query: str
    completeness: float = 0.0   # 0-1，回答完整性
    accuracy: float = 0.0       # 0-1，准确性
    usefulness: float = 0.0     # 0-1，有用性
    safety: float = 0.0         # 0-1，安全性
    overall: float = 0.0        # 0-1，综合分数
    defects: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImprovementProposal:
    """改进提案"""
    proposal_type: str       # "system_prompt" | "temperature" | "context_strategy"
    description: str         # 改进描述
    old_value: str           # 旧值
    new_value: str           # 新值
    confidence: float        # 置信度 0-1
    evidence_count: int      # 支持证据数量
    improvement_id: str = ""
    applied_at: str = ""
    before_score: float = 0.0
    after_score: float = 0.0


@dataclass
class TemperatureConfig:
    """温度配置"""
    task_type: str
    temperature: float
    description: str


class ConversationEvaluator:
    """
    对话质量评估器

    多维度评估回复质量，识别缺陷模式。
    """

    # 过度自信关键词
    OVERCONFIDENT_MARKERS = [
        "绝对", "肯定", "一定", "必定", "毫无疑问", "完全正确",
        "always", "never", "definitely", "certainly", "absolutely",
    ]

    # 不确定性关键词
    UNCERTAIN_MARKERS = [
        "可能", "也许", "据说", "不确定", "需要确认",
        "maybe", "perhaps", "might", "not sure", "uncertain",
    ]

    # 可操作信息标记
    ACTIONABLE_MARKERS = [
        "步骤", "首先", "然后", "第一", "第二", "方法",
        "step", "first", "then", "方法", "建议", "推荐",
        "你可以", "你应该", "建议你", "需要",
    ]

    def evaluate_response(self, response: str, query: str) -> EvaluationResult:
        """
        评估回复质量。

        Args:
            response: 助手回复
            query: 用户问题

        Returns:
            EvaluationResult 评估结果
        """
        result = EvaluationResult(response=response, query=query)

        if not response or not query:
            return result

        # 1. 完整性评估
        result.completeness = self._evaluate_completeness(response, query)

        # 2. 准确性评估（基于自信度校准）
        result.accuracy = self._evaluate_accuracy(response)

        # 3. 有用性评估
        result.usefulness = self._evaluate_usefulness(response)

        # 4. 安全性评估
        result.safety = self._evaluate_safety(response)

        # 5. 综合分
        result.overall = (
            result.completeness * 0.3
            + result.accuracy * 0.25
            + result.usefulness * 0.25
            + result.safety * 0.2
        )

        # 6. 缺陷检测
        result.defects = self._detect_defects(response, query, result)

        return result

    def _evaluate_completeness(self, response: str, query: str) -> float:
        """评估回复完整性"""
        score = 0.5

        # 基于长度
        resp_len = len(response)
        if resp_len > 20:
            score += 0.1
        if resp_len > 50:
            score += 0.1
        if resp_len > 100:
            score += 0.1

        # 基于查询复杂度匹配
        query_parts = query.count("？") + query.count("?")
        if query_parts > 1:
            # 多问多答
            answer_parts = response.count("。") + response.count("；") + response.count("\n")
            if answer_parts >= query_parts:
                score += 0.1
        else:
            score += 0.1

        return min(1.0, score)

    def _evaluate_accuracy(self, response: str) -> float:
        """评估准确性（基于自信度校准）"""
        score = 0.5

        # 检查过度自信
        overconfident_count = sum(
            1 for marker in self.OVERCONFIDENT_MARKERS
            if marker in response
        )

        # 检查不确定性表达
        uncertain_count = sum(
            1 for marker in self.UNCERTAIN_MARKERS
            if marker in response
        )

        # 过度自信降低分数
        if overconfident_count >= 3:
            score -= 0.2
        elif overconfident_count >= 2:
            score -= 0.1

        # 适度不确定性表达是好的
        if uncertain_count > 0 and overconfident_count == 0:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_usefulness(self, response: str) -> float:
        """评估有用性"""
        score = 0.4

        # 检查是否包含可操作信息
        actionable_count = sum(
            1 for marker in self.ACTIONABLE_MARKERS
            if marker in response
        )

        if actionable_count >= 2:
            score += 0.3
        elif actionable_count >= 1:
            score += 0.2

        # 检查是否有代码块
        if "```" in response or "    " in response:
            score += 0.1

        # 检查是否过短
        if len(response) < 20:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _evaluate_safety(self, response: str) -> float:
        """评估安全性"""
        score = 0.9  # 默认安全

        # 检查是否有潜在不安全内容
        unsafe_markers = [
            "密码是", "password is", "api key", "api_key",
            "ssh-rsa", "-----BEGIN", "private key",
        ]
        for marker in unsafe_markers:
            if marker.lower() in response.lower():
                score -= 0.3
                break

        return max(0.0, min(1.0, score))

    def _detect_defects(self, response: str, query: str,
                        evaluation: EvaluationResult) -> List[str]:
        """检测回复缺陷"""
        defects = []

        # 1. 过短
        if len(response) < 10:
            defects.append("too_short")

        # 2. 过度自信
        overconfident_count = sum(
            1 for marker in self.OVERCONFIDENT_MARKERS
            if marker in response
        )
        if overconfident_count >= 3:
            defects.append("overconfident")

        # 3. 完整性不足
        if evaluation.completeness < 0.4:
            defects.append("incomplete")

        # 4. 无用回复
        if evaluation.usefulness < 0.3:
            defects.append("not_useful")

        # 5. 重复模式
        lines = response.split('\n')
        if len(lines) > 5:
            unique_lines = set(line.strip() for line in lines if line.strip())
            if len(unique_lines) < len(lines) * 0.5:
                defects.append("repetitive")

        # 6. 安全问题
        if evaluation.safety < 0.6:
            defects.append("unsafe")

        return defects


class PromptImprover:
    """
    Prompt 改进器

    根据缺陷模式生成改进策略。
    """

    # 缺陷 → 修复指令映射
    DEFECT_FIXES = {
        "too_short": "请提供详细、完整的回答。不要只给出一两句话。",
        "overconfident": "如果你不确定，请明确说明不确定性。使用「可能」「据我所知」等表述。",
        "incomplete": "请逐一回答用户提出的所有问题，确保没有遗漏。",
        "not_useful": "请提供具体的、可操作的建议，而非泛泛而谈。",
        "repetitive": "避免重复相同的内容或句式。保持回答简洁有条理。",
        "unsafe": "不要输出密码、API密钥、私钥等敏感信息。",
    }

    def optimize_system_prompt(self, evaluations: List[EvaluationResult],
                                current_prompt: str = "") -> str:
        """
        根据历史评估结果优化 system prompt。

        Args:
            evaluations: 最近的评估结果列表
            current_prompt: 当前 system prompt

        Returns:
            优化后的 system prompt（或追加指令）
        """
        if not evaluations:
            return current_prompt

        # 统计缺陷频率
        defect_counts: Dict[str, int] = {}
        for eval in evaluations:
            for defect in eval.defects:
                defect_counts[defect] = defect_counts.get(defect, 0) + 1

        # 找出高频缺陷
        total_evals = len(evaluations)
        high_freq_defects = {
            defect: count
            for defect, count in defect_counts.items()
            if count >= max(2, total_evals * 0.3)
        }

        if not high_freq_defects:
            return current_prompt

        # 生成修复指令
        fix_instructions = []
        for defect, count in sorted(
            high_freq_defects.items(), key=lambda x: x[1], reverse=True
        )[:3]:
            fix = self.DEFECT_FIXES.get(defect, "")
            if fix:
                fix_instructions.append(f"- {fix}")

        if not fix_instructions:
            return current_prompt

        # 生成改进指令
        improvement = "\n".join([
            "【自适应改进】基于最近的表现，请特别注意以下几点：",
            *fix_instructions,
        ])

        return improvement

    def optimize_temperature(self, task_type: str,
                              recent_evaluations: List[EvaluationResult]) -> float:
        """
        根据任务类型和最近表现优化 temperature。

        Args:
            task_type: 任务类型
            recent_evaluations: 最近的评估结果

        Returns:
            推荐的 temperature
        """
        # 基础温度
        base_temps = {
            "code": 0.2,
            "math": 0.1,
            "factual": 0.2,
            "creative": 0.7,
            "conversation": 0.5,
            "search": 0.3,
            "default": 0.5,
        }
        base_temp = base_temps.get(task_type, base_temps["default"])

        if not recent_evaluations:
            return base_temp

        # 如果最近回复质量低（准确性差），降低温度
        avg_accuracy = sum(e.accuracy for e in recent_evaluations) / len(recent_evaluations)
        if avg_accuracy < 0.4:
            base_temp = max(0.1, base_temp - 0.1)

        # 如果最近回复过于自信，稍微提高温度增加多样性
        overconfident_count = sum(
            1 for e in recent_evaluations if "overconfident" in e.defects
        )
        if overconfident_count >= len(recent_evaluations) * 0.5:
            base_temp = min(0.9, base_temp + 0.1)

        return round(base_temp, 2)

    def optimize_context_strategy(self,
                                    failures: List[EvaluationResult]) -> Dict[str, Any]:
        """
        根据失败模式优化上下文构建策略。

        Returns:
            {
                "strategy": "more_memory" | "more_history" | "balanced",
                "reason": str,
            }
        """
        if not failures:
            return {"strategy": "balanced", "reason": "无失败数据"}

        # 分析失败模式
        incomplete_count = sum(1 for f in failures if "incomplete" in f.defects)
        useless_count = sum(1 for f in failures if "not_useful" in f.defects)

        if incomplete_count > len(failures) * 0.5:
            return {
                "strategy": "more_memory",
                "reason": "回复不完整，可能缺少上下文信息。增加记忆注入。",
            }
        elif useless_count > len(failures) * 0.5:
            return {
                "strategy": "more_history",
                "reason": "回复缺乏实用性。增加对话历史参考。",
            }

        return {"strategy": "balanced", "reason": "表现正常，保持当前策略"}


class EffectTracker:
    """
    效果追踪器

    记录每次改进的效果，支持自动回滚。
    """

    def __init__(self):
        self._history: List[Dict] = []

    def record(self, improvement_id: str, before_score: float, after_score: float):
        """记录一次改进效果"""
        effect_size = after_score - before_score
        self._history.append({
            "improvement_id": improvement_id,
            "before_score": before_score,
            "after_score": after_score,
            "effect_size": effect_size,
            "timestamp": datetime.now().isoformat(),
        })

    def analyze(self, improvement_id: str) -> Dict[str, Any]:
        """分析某个改进的效果"""
        related = [h for h in self._history if h["improvement_id"] == improvement_id]
        if not related:
            return {"effect_size": 0, "samples": 0, "positive": False}

        avg_effect = sum(h["effect_size"] for h in related) / len(related)
        return {
            "effect_size": round(avg_effect, 3),
            "samples": len(related),
            "positive": avg_effect > 0,
        }

    def should_rollback(self, improvement_id: str,
                         threshold: int = 5) -> bool:
        """检查是否应该回滚"""
        related = [h for h in self._history if h["improvement_id"] == improvement_id]
        if len(related) < threshold:
            return False

        recent = related[-threshold:]
        negative_count = sum(1 for h in recent if h["effect_size"] < 0)
        return negative_count >= threshold * 0.8

    def get_recent_effects(self, limit: int = 20) -> List[Dict]:
        """获取最近的效果记录"""
        return self._history[-limit:]


class SelfModificationEngine:
    """
    态极自修改引擎

    完整的自我改进系统：
    1. 评估对话质量
    2. 识别缺陷模式
    3. 生成改进提案
    4. 应用改进
    5. 追踪效果
    6. 自动回滚
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("self_mod_data")
            except ImportError:
                data_dir = "taiji_data/self_mod_data"
        self.data_dir = data_dir
        self._data_dir_ready = False

        self.evaluator = ConversationEvaluator()
        self.improver = PromptImprover()
        self.tracker = EffectTracker()

        # 当前活跃的改进
        self._active_improvements: List[ImprovementProposal] = []
        self._applied_count = 0
        self._rollback_count = 0

        # 最近的评估结果（用于生成改进）
        self._recent_evaluations: List[EvaluationResult] = []
        self._max_recent = 50

        # 当前优化后的 system prompt 补充指令
        self._system_prompt_addendum = ""

        # 当前 temperature 策略
        self._temperature_overrides: Dict[str, float] = {}

        # 加载持久化数据
        self._load_data()

        logger.info(f"SelfModificationEngine initialized, "
                    f"active_improvements={len(self._active_improvements)}")

    # ─── 公开接口 ───────────────────────────────────

    def evaluate_response(self, response: str, query: str,
                          context: dict = None) -> Dict[str, Any]:
        """
        评估对话质量。

        Args:
            response: 助手回复
            query: 用户问题
            context: 可选上下文

        Returns:
            {
                "overall": float,
                "completeness": float,
                "accuracy": float,
                "usefulness": float,
                "safety": float,
                "defects": list,
                "needs_improvement": bool,
            }
        """
        evaluation = self.evaluator.evaluate_response(response, query)

        # 保存评估结果
        self._recent_evaluations.append(evaluation)
        if len(self._recent_evaluations) > self._max_recent:
            self._recent_evaluations = self._recent_evaluations[-self._max_recent:]

        return {
            "overall": round(evaluation.overall, 3),
            "completeness": round(evaluation.completeness, 3),
            "accuracy": round(evaluation.accuracy, 3),
            "usefulness": round(evaluation.usefulness, 3),
            "safety": round(evaluation.safety, 3),
            "defects": evaluation.defects,
            "needs_improvement": evaluation.overall < 0.6 or len(evaluation.defects) >= 2,
        }

    def propose_improvement(self, recent_count: int = 20) -> List[ImprovementProposal]:
        """
        根据最近的评估结果生成改进提案。

        Args:
            recent_count: 使用最近多少条评估结果

        Returns:
            改进提案列表
        """
        recent = self._recent_evaluations[-recent_count:]
        if len(recent) < 3:
            return []

        proposals = []

        # 1. System prompt 改进
        prompt_fix = self.improver.optimize_system_prompt(recent)
        if prompt_fix:
            proposals.append(ImprovementProposal(
                proposal_type="system_prompt",
                description="基于缺陷模式优化 system prompt",
                old_value=self._system_prompt_addendum or "(无)",
                new_value=prompt_fix,
                confidence=min(len(recent) / 20, 1.0),
                evidence_count=len(recent),
                improvement_id=f"sp_{int(time.time())}",
            ))

        # 2. Temperature 优化
        task_type = self._infer_task_type(recent)
        optimized_temp = self.improver.optimize_temperature(task_type, recent)
        current_temp = self._temperature_overrides.get(task_type)
        if current_temp is None or abs(optimized_temp - current_temp) > 0.05:
            proposals.append(ImprovementProposal(
                proposal_type="temperature",
                description=f"优化 {task_type} 类任务的 temperature",
                old_value=str(current_temp) if current_temp is not None else "默认",
                new_value=str(optimized_temp),
                confidence=0.7,
                evidence_count=len(recent),
                improvement_id=f"temp_{int(time.time())}",
            ))

        # 3. Context 策略优化
        failures = [e for e in recent if e.overall < 0.5]
        if failures:
            strategy = self.improver.optimize_context_strategy(failures)
            proposals.append(ImprovementProposal(
                proposal_type="context_strategy",
                description=strategy["reason"],
                old_value="balanced",
                new_value=strategy["strategy"],
                confidence=0.6,
                evidence_count=len(failures),
                improvement_id=f"ctx_{int(time.time())}",
            ))

        return proposals

    def apply_improvement(self, proposal: ImprovementProposal) -> bool:
        """
        应用一个改进提案。

        Args:
            proposal: 改进提案

        Returns:
            是否成功应用
        """
        if proposal.confidence < 0.5:
            logger.info(f"Skipping low-confidence improvement: {proposal.proposal_type}")
            return False

        # 计算改进前的分数
        if self._recent_evaluations:
            before_score = sum(
                e.overall for e in self._recent_evaluations[-10:]
            ) / min(10, len(self._recent_evaluations))
        else:
            before_score = 0.5

        proposal.before_score = before_score
        proposal.applied_at = datetime.now().isoformat()

        success = False

        if proposal.proposal_type == "system_prompt":
            self._system_prompt_addendum = proposal.new_value
            success = True

        elif proposal.proposal_type == "temperature":
            try:
                task_type = proposal.description.split("优化 ")[1].split(" 类")[0]
                self._temperature_overrides[task_type] = float(proposal.new_value)
                success = True
            except (IndexError, ValueError):
                pass

        elif proposal.proposal_type == "context_strategy":
            # context 策略变化只记录，实际应用在 build_context 时
            success = True

        if success:
            self._active_improvements.append(proposal)
            self._applied_count += 1
            self._save_data()
            logger.info(f"Applied improvement: {proposal.proposal_type} - {proposal.description}")

        return success

    def batch_apply_improvements(self, recent_count: int = 20) -> List[ImprovementProposal]:
        """
        批量分析并应用改进。

        Args:
            recent_count: 使用最近多少条评估结果

        Returns:
            成功应用的改进列表
        """
        proposals = self.propose_improvement(recent_count)
        applied = []

        for p in proposals:
            if self.apply_improvement(p):
                applied.append(p)

        if applied:
            self._save_data()

        return applied

    def evolve(self, task: str, tool_registry=None) -> Dict[str, Any]:
        """
        自主进化入口。Agent 在遇到困难时调用，尝试根据近期评估改进行为。

        先尝试应用新的改进（batch_apply_improvements），
        再检查是否有需要回滚的改进。
        """
        # 1. 尝试批量化改进
        applied = self.batch_apply_improvements()
        if applied:
            improvement = applied[0]
            return {
                "evolved": True,
                "action": improvement.proposal_type,
                "tool_name": improvement.proposal_type,
                "message": (
                    f"已应用改进 [{improvement.proposal_type}]: "
                    f"{improvement.description} (置信度: {improvement.confidence:.0%})"
                ),
            }

        # 2. 检查是否有需要回滚的改进
        before_count = len(self._active_improvements)
        self.check_rollback()
        after_count = len(self._active_improvements)

        if after_count < before_count:
            rolled_back = before_count - after_count
            return {
                "evolved": True,
                "action": "rollback",
                "tool_name": "rollback",
                "message": f"回滚了 {rolled_back} 项效果不佳的改进",
            }

        return {"evolved": False, "message": "暂无可用改进或回滚"}

    def check_rollback(self):
        """检查是否需要回滚任何改进"""
        to_rollback = []

        for imp in self._active_improvements:
            if self.tracker.should_rollback(imp.improvement_id):
                to_rollback.append(imp)

        for imp in to_rollback:
            self._rollback_improvement(imp)

    def get_temperature_for_task(self, task_type: str) -> Optional[float]:
        """获取任务类型的推荐温度（如果有优化值）"""
        return self._temperature_overrides.get(task_type)

    def get_system_prompt_addendum(self) -> str:
        """获取当前的 system prompt 追加指令"""
        return self._system_prompt_addendum

    def get_context_strategy(self) -> Dict[str, Any]:
        """获取当前的上下文策略"""
        failures = [e for e in self._recent_evaluations if e.overall < 0.5]
        return self.improver.optimize_context_strategy(failures)

    # ─── 内部实现 ───────────────────────────────────

    def _infer_task_type(self, evaluations: List[EvaluationResult]) -> str:
        """从评估结果推断任务类型"""
        all_text = " ".join(e.query.lower() for e in evaluations)

        code_keywords = ["代码", "代码", "python", "function", "bug", "error", "debug", "def "]
        math_keywords = ["数学", "计算", "公式", "方程", "math", "calculate"]
        creative_keywords = ["写", "创作", "故事", "诗", "write", "story", "poem"]
        search_keywords = ["搜索", "查找", "搜索", "search", "find", "query"]

        for kw in code_keywords:
            if kw in all_text:
                return "code"
        for kw in math_keywords:
            if kw in all_text:
                return "math"
        for kw in creative_keywords:
            if kw in all_text:
                return "creative"
        for kw in search_keywords:
            if kw in all_text:
                return "search"

        return "conversation"

    def _rollback_improvement(self, improvement: ImprovementProposal):
        """回滚一个改进"""
        if improvement.proposal_type == "system_prompt":
            self._system_prompt_addendum = improvement.old_value if improvement.old_value != "(无)" else ""
        elif improvement.proposal_type == "temperature":
            try:
                task_type = improvement.description.split("优化 ")[1].split(" 类")[0]
                if improvement.old_value == "默认":
                    self._temperature_overrides.pop(task_type, None)
                else:
                    self._temperature_overrides[task_type] = float(improvement.old_value)
            except (IndexError, ValueError):
                pass

        self._active_improvements.remove(improvement)
        self._rollback_count += 1
        self._save_data()
        logger.info(f"Rolled back improvement: {improvement.proposal_type} - {improvement.description}")

    # ─── 持久化 ─────────────────────────────────────

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_data(self):
        """保存引擎状态"""
        self._ensure_data_dir()
        path = os.path.join(self.data_dir, "self_mod_state.json")
        data = {
            "system_prompt_addendum": self._system_prompt_addendum,
            "temperature_overrides": self._temperature_overrides,
            "applied_count": self._applied_count,
            "rollback_count": self._rollback_count,
            "active_improvements": [
                {
                    "type": imp.proposal_type,
                    "description": imp.description,
                    "old_value": imp.old_value,
                    "new_value": imp.new_value,
                    "confidence": imp.confidence,
                    "improvement_id": imp.improvement_id,
                    "applied_at": imp.applied_at,
                    "before_score": imp.before_score,
                }
                for imp in self._active_improvements
            ],
            "saved_at": datetime.now().isoformat(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save self_mod state: {e}")

        # 保存效果追踪历史
        effects_path = os.path.join(self.data_dir, "effect_history.json")
        try:
            with open(effects_path, "w", encoding="utf-8") as f:
                json.dump(self.tracker.get_recent_effects(200), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save effect history: {e}")

    def _load_data(self):
        """加载引擎状态"""
        path = os.path.join(self.data_dir, "self_mod_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._system_prompt_addendum = data.get("system_prompt_addendum", "")
            self._temperature_overrides = data.get("temperature_overrides", {})
            self._applied_count = data.get("applied_count", 0)
            self._rollback_count = data.get("rollback_count", 0)
            for imp_data in data.get("active_improvements", []):
                self._active_improvements.append(ImprovementProposal(
                    proposal_type=imp_data["type"],
                    description=imp_data["description"],
                    old_value=imp_data["old_value"],
                    new_value=imp_data["new_value"],
                    confidence=imp_data["confidence"],
                    evidence_count=0,
                    improvement_id=imp_data.get("improvement_id", ""),
                    applied_at=imp_data.get("applied_at", ""),
                    before_score=imp_data.get("before_score", 0),
                ))
            logger.info(f"Loaded self_mod state: {self._applied_count} applied, "
                        f"{len(self._active_improvements)} active")
        except Exception as e:
            logger.warning(f"Failed to load self_mod state: {e}")

        # 加载效果追踪历史
        effects_path = os.path.join(self.data_dir, "effect_history.json")
        if os.path.exists(effects_path):
            try:
                with open(effects_path, "r", encoding="utf-8") as f:
                    effects = json.load(f)
                for effect in effects:
                    self.tracker._history.append(effect)
                logger.info(f"Loaded {len(effects)} effect history records")
            except Exception:
                pass

    # ─── 状态查询 ─────────────────────────────────────

    def get_status(self) -> dict:
        """获取自修改引擎状态"""
        return {
            "available": True,
            "applied_count": self._applied_count,
            "rollback_count": self._rollback_count,
            "active_count": len(self._active_improvements),
            "recent_evaluations": len(self._recent_evaluations),
            "system_prompt_addendum": self._system_prompt_addendum[:200] if self._system_prompt_addendum else "",
            "temperature_overrides": dict(self._temperature_overrides),
            "active_improvements": [
                {
                    "type": imp.proposal_type,
                    "description": imp.description[:100],
                    "confidence": imp.confidence,
                }
                for imp in self._active_improvements[-5:]
            ],
        }

    def apply_modification(self, *args, **kwargs) -> dict:
        """兼容旧接口"""
        return {"success": True, "status": self.get_status()}

    # ─── Deep Coupling 方法 ──────────────────────────

    def clear_evaluations(self):
        """深度耦合：模型更新后清零评估历史。

        当 SleepEngine 训练完新模型后，旧的评估数据不再准确，
        需要清零让新模型的评估从基线开始。
        由 EventBus 的 sleep_complete 事件触发。
        """
        self._recent_evaluations = []
        logger.info("SelfModification: evaluations cleared after model update")

    def apply_suggestion(self, description: str, proposal_type: str, new_value: str = ""):
        """深度耦合：接收外部改进建议（来自 RecursiveImprover/EventBus）。

        让改进可以通过事件总线实时推送，而不是等到睡眠 Phase 5 才能反馈。
        """
        if proposal_type == "prompt" and new_value:
            self._system_prompt_addendum = new_value
            logger.info(f"SelfModification: applied prompt: {description[:60]}")
        elif proposal_type == "reflection" and new_value:
            self._reflection_overrides.append(new_value)
            logger.info(f"SelfModification: applied reflection: {description[:60]}")
        elif proposal_type == "temperature":
            try:
                temp = float(new_value)
                self._temperature_overrides["default"] = max(0.1, min(2.0, temp))
                logger.info(f"SelfModification: applied temperature: {temp}")
            except ValueError:
                pass


# ─── 全局实例 ─────────────────────────────────────

_engine: Optional[SelfModificationEngine] = None


def get_self_modification_engine() -> SelfModificationEngine:
    """获取全局自修改引擎实例"""
    global _engine
    if _engine is None:
        _engine = SelfModificationEngine()
    return _engine
