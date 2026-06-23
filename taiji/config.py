"""
Taiji Model Configuration

态极递归蒸馏进化 (Taiji Recursive Distillation) 配置。

进化路线：
  Qwen2.5-0.5B (启蒙老师) → 态极火种 → 蒸馏 → 态极1B → 蒸馏 → 态极3B → ...

基底模型: Qwen2.5 系列（提供语言能力）
生命系统: 饥饿、好奇、疲劳、进化（提供态极性）
核心哲学: 小态极把自己的结构投射到更大的计算载体中
"""
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Taiji 特殊 token 系统
# ═══════════════════════════════════════════════════════════════
# 特殊 token 以「内容」定义，不以「ID」定义。
# ID 在运行时从 tokenizer 动态查询，兼容任何词表大小。
#
# 两种模式：
#   1. 原生态极 (32K SentencePiece): ID 固定从 32000 开始
#   2. 转化模型 (151K+ BPE): ID 从 base_vocab_size 开始
# ═══════════════════════════════════════════════════════════════

# 特殊 token 内容定义（name → content）
SPECIAL_TOKEN_CONTENT = {
    # === 思考与输出 ===
    "think_start":      "<think>",
    "think_end":        "</think>",
    "tool_call":        "<tool_call>",
    "tool_result":      "<tool_result>",
    "answer":           "<final_answer>",

    # === 感知系统 ===
    "observe":          "<observe>",
    "observe_end":      "</observe>",
    "env_tree":         "<tree>",
    "env_state":        "<state>",
    "env_result":       "<result>",

    # === 记忆系统 ===
    "mem_read":         "<mem_read>",
    "mem_write":        "<mem_write>",
    # mem_slot_base: 20 个短期记忆槽 (mem_slot_0 ~ mem_slot_19)
    # mem_long_base: 10 个长期记忆槽 (mem_long_0 ~ mem_long_9)

    # === 规划系统 ===
    "plan_start":       "<plan>",
    "plan_end":         "</plan>",
    "plan_step":        "<step>",
    "plan_step_end":    "</step>",
    "plan_done":        "<plan_done>",
    "plan_replan":      "<replan>",

    # === 反思系统 ===
    "reflect_start":    "<reflect>",
    "reflect_end":      "</reflect>",
    "reflect_detect":   "<detect>",
    "reflect_cause":    "<cause>",
    "reflect_correct":  "<correct>",
    "reflect_confirm":  "<confirm>",
}

# 默认 ID 映射（原生态极 32K SentencePiece 模式）
# 转化模型会在运行时通过 SpecialTokenResolver 重新映射
SPECIAL_TOKENS = {
    "think_start":      32000,
    "think_end":        32001,
    "tool_call":        32002,
    "tool_result":      32003,
    "answer":           32004,
    "observe":          32005,
    "observe_end":      32006,
    "env_tree":         32007,
    "env_state":        32008,
    "env_result":       32009,
    "mem_read":         32100,
    "mem_write":        32101,
    "mem_slot_base":    32102,
    "mem_long_base":    32122,
    "plan_start":       32132,
    "plan_end":         32133,
    "plan_step":        32134,
    "plan_step_end":    32135,
    "plan_done":        32136,
    "plan_replan":      32137,
    "reflect_start":    32138,
    "reflect_end":      32139,
    "reflect_detect":   32140,
    "reflect_cause":    32141,
    "reflect_correct":  32142,
    "reflect_confirm":  32143,
    "tool_name_base":   32150,
}


# ═══════════════════════════════════════════════════════════════
# 多模态离散 token 区域定义
# ═══════════════════════════════════════════════════════════════
# 策略: 在词表末尾划分固定区域，每个模态有独立的 token 范围
# 新模态只需在末尾追加，不影响已有 token
# ═══════════════════════════════════════════════════════════════

MULTIMODAL_TOKENS = {
    # 图像 token: VQ-VAE codebook 映射到词表
    "image_token_base": 152936,       # 图像 token 起始 ID
    "image_codebook_size": 8192,      # VQ-VAE codebook 大小
    # 范围: 152936 ~ 161127 (8192 个)

    # 音频 token: EnCodec codebook 映射到词表
    "audio_token_base": 161128,       # 音频 token 起始 ID
    "audio_codebook_size": 4096,      # 音频 codebook 大小
    # 范围: 161128 ~ 165223 (4096 个)

    # 多模态控制 token
    "mm_control_base": 165224,        # 控制 token 起始 ID
    "mm_control_size": 100,           # 控制 token 数量
    # 范围: 165224 ~ 165323 (100 个)
}

# 多模态控制 token 内容（用于标记模态边界）
MM_CONTROL_TOKENS = {
    "image_start":    165224,  # <image> — 图像序列开始
    "image_end":      165225,  # </image> — 图像序列结束
    "audio_start":    165226,  # <audio> — 音频序列开始
    "audio_end":      165227,  # </audio> — 音频序列结束
    "img_row":        165228,  # <row> — 图像行分隔（16×16 网格）
    "gen_image":      165229,  # <gen_image> — 触发图像生成
    "gen_audio":      165230,  # <gen_audio> — 触发音频生成
}

# 计算总词表大小
MULTIMODAL_VOCAB_SIZE = (
    MULTIMODAL_TOKENS["image_token_base"]        # 152936 (文本 + 特殊 token)
    + MULTIMODAL_TOKENS["image_codebook_size"]    # + 8192 (图像)
    + MULTIMODAL_TOKENS["audio_codebook_size"]    # + 4096 (音频)
    + MULTIMODAL_TOKENS["mm_control_size"]        # + 100 (控制)
)
# = 165,324


class SpecialTokenResolver:
    """
    动态特殊 token ID 解析器。

    从 tokenizer 中查询特殊 token 的实际 ID，
    兼容原生态极 (32K SentencePiece) 和转化模型 (151K+ BPE)。

    用法:
        resolver = SpecialTokenResolver(tokenizer)
        tool_call_id = resolver["tool_call"]  # 动态获取 ID
    """

    def __init__(self, tokenizer):
        self._cache = {}
        self._resolve(tokenizer)

    def _resolve(self, tokenizer):
        """从 tokenizer 中解析所有特殊 token 的 ID"""
        for name, content in SPECIAL_TOKEN_CONTENT.items():
            token_id = self._find_token_id(tokenizer, content)
            if token_id is not None:
                self._cache[name] = token_id
            else:
                # 回退到默认 ID
                self._cache[name] = SPECIAL_TOKENS.get(name, 32000)

        # 工具名基址：找到 <tool_sep> 或使用默认
        tool_sep_id = self._find_token_id(tokenizer, "<tool_sep>")
        self._cache["tool_name_base"] = tool_sep_id if tool_sep_id else SPECIAL_TOKENS["tool_name_base"]

    def _find_token_id(self, tokenizer, content: str) -> int:
        """在 tokenizer 中查找 token 内容对应的 ID"""
        # 方法 1: convert_tokens_to_ids
        try:
            tid = tokenizer.convert_tokens_to_ids(content)
            if tid is not None and tid != tokenizer.unk_token_id:
                return tid
        except Exception:
            pass

        # 方法 2: 直接 encode
        try:
            ids = tokenizer.encode(content, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        except Exception:
            pass

        # 方法 3: added_tokens_decoder 中查找
        try:
            added = getattr(tokenizer, 'added_tokens_decoder', {})
            for tid_str, info in added.items():
                if hasattr(info, 'content') and info.content == content:
                    return int(tid_str)
                elif isinstance(info, dict) and info.get('content') == content:
                    return int(tid_str)
        except Exception:
            pass

        return None

    def __getitem__(self, name: str) -> int:
        return self._cache.get(name, SPECIAL_TOKENS.get(name, 32000))

    def get(self, name: str, default=None) -> int:
        return self._cache.get(name, default)

    def resolve_all(self) -> dict:
        """返回完整的 ID 映射字典"""
        return dict(self._cache)


def get_taiji_data_path(subdir: str) -> str:
    """
    获取态极持久化数据路径。优先存到当前模型目录下，确保数据和模型绑定。

    优先级：
      1. 当前加载的模型目录/{subdir}（如 taiji_350M/sleep_data/）
      2. 外部目录 taiji_data/{subdir}（兜底）

    这样打包不影响数据，且模型目录就是态极的"身份"。
    """
    import os

    # 优先：存到当前模型目录下
    try:
        from core.app_state import app_state
        model_path = getattr(app_state, "_loaded_model_name", "") or ""
        if model_path and os.path.isdir(model_path):
            # model_path 指向 taiji_350M/ 或 best/，用其父目录
            model_dir = model_path if os.path.exists(os.path.join(model_path, "config.json")) else os.path.dirname(model_path)
            data_path = os.path.join(model_dir, subdir)
            os.makedirs(data_path, exist_ok=True)
            return data_path
    except (ImportError, Exception):
        pass

    # 兜底：外部目录
    try:
        from core.config import get_external_path
        base = get_external_path(os.path.join("taiji_data", subdir))
    except ImportError:
        base = os.path.join(os.getcwd(), "taiji_data", subdir)
    os.makedirs(base, exist_ok=True)
    return base


@dataclass
class ModelConfig:
    """模型架构配置"""

    # === 基础参数 ===
    vocab_size: int = 242612          # 新训练的大词表 (原32000基础 + Qwen2.5扩展)
    hidden_size: int = 768            # 隐藏层维度
    intermediate_size: int = 2048     # FFN 中间层维度
    num_hidden_layers: int = 12       # Transformer 层数
    num_attention_heads: int = 12     # 注意力头数
    num_key_value_heads: int = 12     # KV 头数 (GQA)

    # === 活跃头配置 ===
    # 控制哪些多头分支实际创建和训练。未激活的头不消耗参数。
    # 可选: "language", "tool", "perception", "memory", "plan"
    active_heads: list = None         # None = 全部激活（向后兼容）

    # === 序列参数 ===
    max_position_embeddings: int = 4096

    # === 注意力 bias ===
    attention_bias: bool = False         # Q/K/V 投影是否使用 bias（Qwen2 用 True，原生态极用 False）

    # === 归一化 ===
    rms_norm_eps: float = 1e-5

    # === RoPE ===
    rope_theta: float = 500000.0

    # === 词表参数 ===
    base_vocab_size: int = 32000      # 基础 SentencePiece 词表大小
    num_special_tokens: int = 1000    # 特殊 token 数量 (扩展后，含 750 个工具位)

    # === 记忆系统参数 ===
    num_short_term_slots: int = 20    # 短期记忆槽数
    num_long_term_slots: int = 10     # 长期记忆槽数
    memory_dim: int = 64              # 记忆向量维度

    @classmethod
    def size_125m(cls) -> "ModelConfig":
        """125M 参数 — 轻量 Agent，适合 CPU 推理"""
        return cls(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
        )

    @classmethod
    def size_350m(cls) -> "ModelConfig":
        """350M 参数 — 标准 Agent"""
        return cls(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
        )

    @classmethod
    def size_1b(cls) -> "ModelConfig":
        """1B 参数 — 复杂推理 + Agent"""
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_3b(cls) -> "ModelConfig":
        """3B 参数 — 高级推理 + 多任务 Agent"""
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def size_7b(cls) -> "ModelConfig":
        """7B 参数 — 复杂多步推理 + 专业级 Agent"""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=4,
        )

    @classmethod
    def from_qwen(cls, model_name: str = "Qwen/Qwen2.5-0.5B") -> "ModelConfig":
        """
        从 Qwen 模型配置创建 ModelConfig。
        用于态极递归蒸馏的第一阶段（启蒙老师）。

        Qwen2.5-0.5B: hidden=896, layers=24, heads=14, kv=2, ffn=4864
        Qwen2.5-1.5B: hidden=1536, layers=28, heads=12, kv=2, ffn=8960
        Qwen2.5-3B:   hidden=2048, layers=36, heads=16, kv=2, ffn=11008
        """
        qwen_configs = {
            "Qwen/Qwen2.5-0.5B": cls(
                vocab_size=151936, hidden_size=896, intermediate_size=4864,
                num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
            "Qwen/Qwen2.5-1.5B": cls(
                vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
            "Qwen/Qwen2.5-3B": cls(
                vocab_size=151936, hidden_size=2048, intermediate_size=11008,
                num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
                max_position_embeddings=32768,
            ),
        }
        if model_name in qwen_configs:
            return qwen_configs[model_name]
        # 默认返回 0.5B 配置
        return qwen_configs["Qwen/Qwen2.5-0.5B"]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def count_parameters(self) -> int:
        """估算参数量"""
        embed = self.vocab_size * self.hidden_size
        # 每层: attention(Q+K+V+O) + FFN(W1+W_gate+W2) + 2*norm
        head_dim = self.head_dim
        attn = (
            self.hidden_size * self.num_attention_heads * head_dim +  # Q
            self.hidden_size * self.num_key_value_heads * head_dim +  # K
            self.hidden_size * self.num_key_value_heads * head_dim +  # V
            self.num_attention_heads * head_dim * self.hidden_size    # O
        )
        ffn = (
            self.hidden_size * self.intermediate_size +   # W1
            self.hidden_size * self.intermediate_size +   # W_gate
            self.intermediate_size * self.hidden_size     # W2
        )
        norm = self.hidden_size * 2  # attention_norm + ffn_norm
        per_layer = attn + ffn + norm
        return embed + per_layer * self.num_hidden_layers

    @property
    def size_label(self) -> str:
        """从配置自动推断模型规模标签（不依赖硬编码）"""
        params = self.count_parameters()
        if params >= 3e9:
            return "7B"
        elif params >= 1e9:
            return "3B"
        elif params >= 500e6:
            return "1B"
        elif params >= 200e6:
            return "350M"
        elif params >= 80e6:
            return "125M"
        else:
            return f"{params/1e6:.0f}M"

    def describe(self) -> str:
        """返回人类可读的配置描述"""
        params = self.count_parameters()
        if params >= 1e9:
            size_str = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            size_str = f"{params/1e6:.0f}M"
        else:
            size_str = f"{params/1e3:.0f}K"
        return (
            f"ModelSelf-{size_str} | "
            f"hidden={self.hidden_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads} kv={self.num_key_value_heads} "
            f"ffn={self.intermediate_size} vocab={self.vocab_size}"
        )
