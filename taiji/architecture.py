"""Clean Taiji architecture implementation for the native-v2 stack."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock

logger = logging.getLogger("ModelSelf")


@dataclass
class ModelOutput:
    """Standard model forward output."""

    logits: torch.Tensor
    tool_logits: Optional[torch.Tensor] = None
    perception_logits: Optional[torch.Tensor] = None
    memory_logits: Optional[torch.Tensor] = None
    plan_logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    kv_cache: Optional[list] = None


class ModelSelfBackbone(nn.Module):
    """Shared Transformer backbone."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    bias=getattr(config, "attention_bias", False),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _build_causal_mask(
        self,
        seq_len: int,
        total_len: int,
        start_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        col_indices = torch.arange(total_len, device=device).unsqueeze(0)
        allowed = col_indices <= (start_pos + row_indices)
        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward_embeddings(
        self,
        hidden: torch.Tensor,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        seq_len = hidden.shape[1]
        start_pos = kv_cache[0][0].shape[1] if kv_cache is not None else 0
        total_len = start_pos + seq_len
        mask = self._build_causal_mask(seq_len, total_len, start_pos, hidden.device, hidden.dtype)

        new_kv_cache = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            layer_kv = kv_cache[layer_index] if kv_cache is not None else None
            hidden, layer_new_kv = layer(hidden, mask, layer_kv, use_cache)
            if use_cache:
                new_kv_cache.append(layer_new_kv)

        hidden = self.norm(hidden)
        return hidden, new_kv_cache

    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        hidden = self.embedding(tokens)
        return self.forward_embeddings(hidden, kv_cache=kv_cache, use_cache=use_cache)


class MultimodalProjector(nn.Module):
    """Project external encoder features into the Taiji hidden space."""

    def __init__(self, encoder_dim: int, hidden_size: int, num_tokens: int = 1):
        super().__init__()
        self.num_tokens = num_tokens
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)
        batch_size = encoder_output.shape[0]
        pooled = encoder_output.mean(dim=1)
        projected = self.projector(pooled)
        projected = projected.view(batch_size, self.num_tokens, -1)
        return self.norm(projected)


class PerceptionHead(nn.Module):
    """Environment perception head."""

    def __init__(self, hidden_size: int, num_env_tokens: int = 200):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_env_tokens),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class MemoryHead(nn.Module):
    """Differentiable memory read/write head."""

    def __init__(self, hidden_size: int, num_slots: int = 30, memory_dim: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.memory_keys = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_size, memory_dim, bias=False)
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_slots),
        )
        self.value_proj = nn.Linear(hidden_size, memory_dim, bias=False)
        self.output_proj = nn.Linear(memory_dim, hidden_size, bias=False)

    def read(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(hidden)
        scores = torch.matmul(query, self.memory_keys.t()) / (self.memory_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        read_raw = torch.matmul(attn_weights, self.memory_values)
        gate = self.read_gate(torch.cat([hidden, read_raw], dim=-1))
        read_out = self.output_proj(read_raw * gate)
        return read_out, attn_weights

    def get_write_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.write_gate(hidden)

    def get_write_value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_proj(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        _, attn_weights = self.read(hidden)
        return attn_weights


class PlanHead(nn.Module):
    """Task planning head."""

    MAX_SUBGOALS = 5

    def __init__(self, hidden_size: int, num_actions: int = 8):
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_actions),
        )
        self.steps_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus(),
        )
        self.subgoal_proj = nn.Linear(hidden_size, hidden_size // 4, bias=False)
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        return_subgoals: bool = False,
    ) -> Dict[str, torch.Tensor]:
        result = {
            "plan_logits": self.action_head(hidden),
            "predicted_steps": self.steps_predictor(hidden),
            "difficulty": self.difficulty_head(hidden),
        }
        if return_subgoals:
            subgoal_base = self.subgoal_proj(hidden).unsqueeze(1)
            result["subgoals"] = subgoal_base.expand(-1, self.MAX_SUBGOALS, -1)
        return result


class ToolHead(nn.Module):
    """Tool classification head with lightweight structured argument prediction."""

    ARG_TYPES = ["string", "number", "file_path", "boolean", "json"]
    NUM_ARG_SLOTS = 4

    def __init__(self, hidden_size: int, num_tools: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_tools),
        )
        self.schema_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.arg_presence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, self.NUM_ARG_SLOTS),
        )
        self.arg_type_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, len(self.ARG_TYPES)),
                )
                for _ in range(self.NUM_ARG_SLOTS)
            ]
        )
        self.arg_value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, hidden_size),
                )
                for _ in range(self.NUM_ARG_SLOTS)
            ]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        schema_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if schema_embeds is not None:
            schema_proj = self.schema_proj(schema_embeds)
            cond = torch.cat([hidden, schema_proj], dim=-1)
        else:
            cond = torch.cat([hidden, hidden], dim=-1)

        arg_types = [head(cond) for head in self.arg_type_heads]
        arg_values = [head(cond) for head in self.arg_value_heads]
        return {
            "tool_logits": self.mlp(hidden),
            "arg_presence": self.arg_presence(hidden),
            "arg_types": arg_types,
            "arg_values": arg_values,
        }


class ModelSelf(nn.Module):
    """Taiji native model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        active = config.active_heads
        if active is None or len(active) == 0:
            active_heads = {"language", "tool", "perception", "memory", "plan"}
        else:
            active_heads = set(active)
        self._active_heads = active_heads

        self.backbone = ModelSelfBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._max_tools = 750
        self._num_tools = 0
        self.tool_head = ToolHead(config.hidden_size, self._max_tools) if "tool" in active_heads else None
        self.perception_head = PerceptionHead(config.hidden_size, num_env_tokens=200) if "perception" in active_heads else None
        self.memory_head = MemoryHead(config.hidden_size, num_slots=30) if "memory" in active_heads else None
        self.plan_head = PlanHead(config.hidden_size, num_actions=8) if "plan" in active_heads else None

        self.vision_projector = MultimodalProjector(encoder_dim=512, hidden_size=config.hidden_size, num_tokens=4)
        self.audio_projector = MultimodalProjector(encoder_dim=384, hidden_size=config.hidden_size, num_tokens=4)

        self.apply(self._init_weights)
        self.lm_head.weight = self.backbone.embedding.weight

        logger.info("ModelSelf active heads: %s", sorted(active_heads))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_num_tools(self, num_tools: int) -> None:
        """Update the number of registered tools."""

        if self.tool_head is None:
            logger.warning("Tool head is not active; skipping tool registration update")
            return
        if num_tools > self._max_tools:
            logger.warning(
                "Tool count %s exceeds maximum supported tools %s; extra tools will be ignored",
                num_tools,
                self._max_tools,
            )
        self._num_tools = min(num_tools, self._max_tools)

    @property
    def num_tools(self) -> int:
        return self._num_tools

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        tool_targets: Optional[torch.Tensor] = None,
        tool_head_active: bool = False,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
        force_tool_head: bool = False,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if vision_features is not None or audio_features is not None:
            hidden, new_kv_cache = self._forward_with_multimodal(
                tokens,
                vision_features=vision_features,
                audio_features=audio_features,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
        else:
            hidden, new_kv_cache = self.backbone(tokens, kv_cache=kv_cache, use_cache=use_cache)

        logits = self.lm_head(hidden)

        tool_logits = None
        if (tool_head_active or force_tool_head) and self.tool_head is not None and self._num_tools > 0:
            last_hidden = hidden[:, -1, :]
            tool_outputs = self.tool_head(last_hidden)
            tool_logits = tool_outputs["tool_logits"][:, : self._num_tools]

        loss = None
        if targets is not None or tool_targets is not None:
            loss = self._compute_loss(
                logits=logits,
                targets=targets,
                tool_logits=tool_logits,
                tool_targets=tool_targets,
                tool_head_active=tool_head_active or force_tool_head,
            )

        return ModelOutput(
            logits=logits,
            tool_logits=tool_logits,
            loss=loss,
            kv_cache=new_kv_cache,
        )

    def get_agent_outputs(self, hidden: torch.Tensor) -> Dict[str, Any]:
        if hidden.dim() == 3:
            last_hidden = hidden[:, -1, :]
        else:
            last_hidden = hidden

        result: Dict[str, Any] = {
            "perception": None,
            "memory_read": None,
            "memory_attn": None,
            "memory_write_logits": None,
            "plan": None,
        }

        if self.perception_head is not None:
            result["perception"] = self.perception_head(last_hidden)
        if self.memory_head is not None:
            mem_out, mem_attn = self.memory_head.read(last_hidden)
            result["memory_read"] = mem_out
            result["memory_attn"] = mem_attn
            result["memory_write_logits"] = self.memory_head.get_write_logits(last_hidden)
        if self.plan_head is not None:
            result["plan"] = self.plan_head(last_hidden)
        return result

    def get_tool_arg_outputs(
        self,
        hidden: torch.Tensor,
        schema_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.tool_head is None:
            return {}
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]
        return self.tool_head(hidden, schema_embeds)

    def _forward_with_multimodal(
        self,
        tokens: torch.Tensor,
        vision_features: Optional[torch.Tensor],
        audio_features: Optional[torch.Tensor],
        kv_cache: Optional[list],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        text_embeds = self.backbone.embedding(tokens)
        segments = []
        if vision_features is not None:
            segments.append(self.vision_projector(vision_features))
        if audio_features is not None:
            segments.append(self.audio_projector(audio_features))
        prefix_len = sum(segment.shape[1] for segment in segments)
        segments.append(text_embeds)

        combined = torch.cat(segments, dim=1)
        hidden, new_kv_cache = self.backbone.forward_embeddings(
            combined,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden = hidden[:, prefix_len:, :]
        return hidden, new_kv_cache

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        tool_logits: Optional[torch.Tensor],
        tool_targets: Optional[torch.Tensor],
        tool_head_active: bool,
    ) -> torch.Tensor:
        losses = []
        if targets is not None:
            losses.append(
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
            )

        if tool_head_active and tool_logits is not None and tool_targets is not None:
            valid_mask = tool_targets != -100
            if valid_mask.any():
                losses.append(F.cross_entropy(tool_logits[valid_mask], tool_targets[valid_mask]))

        if not losses:
            return torch.zeros((), device=logits.device, requires_grad=True)
        return sum(losses) / len(losses)

    def get_num_parameters(self) -> Dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        tool_head_params = sum(parameter.numel() for parameter in self.tool_head.parameters()) if self.tool_head is not None else 0
        other_head_params = 0
        for head_name in ("perception_head", "memory_head", "plan_head"):
            head = getattr(self, head_name, None)
            if head is not None:
                other_head_params += sum(parameter.numel() for parameter in head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": total - tool_head_params - other_head_params,
            "tool_head": tool_head_params,
            "other_heads": other_head_params,
            "active_heads": sorted(self._active_heads),
        }

    def print_model_info(self) -> None:
        params = self.get_num_parameters()
        print("\n" + "=" * 60)
        print("ModelSelf Architecture")
        print("=" * 60)
        print(self.config.describe())
        print(f"Active Heads:         {', '.join(params['active_heads'])}")
        print(f"Total Parameters:     {params['total']:,} ({params['total'] / 1e6:.1f}M)")
        print(f"  Backbone:           {params['backbone']:,}")
        print(f"  Tool Head:          {params['tool_head']:,}")
        print(f"  Other Heads:        {params['other_heads']:,}")
        print(f"  Tools registered:   {self._num_tools}")
        print("=" * 60 + "\n")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self(generated)
                next_token_logits = output.logits[:, -1, :] / max(temperature, 1e-6)

                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                    probs = torch.softmax(sorted_logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, sampled)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break

        return generated
