"""Clean native inference engine for the Taiji native-v2 stack."""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn.functional as F

from taiji.architecture import ModelSelf
from taiji.config import SpecialTokenResolver

logger = logging.getLogger("ModelSelf")


class NativeInferenceEngine:
    """Native autoregressive inference engine."""

    def __init__(
        self,
        model: ModelSelf,
        tokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._compiled = False
        self._tokens = SpecialTokenResolver(tokenizer)

        logger.info(
            "Resolved structural tokens: tool_call=%s answer=%s",
            self._tokens["tool_call"],
            self._tokens["answer"],
        )

        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.half()
            logger.info("Enabled FP16 inference on CUDA")
        elif device == "cpu":
            try:
                from taiji.core.quantization import TaijiQuantizer

                self.model = TaijiQuantizer.quantize_dynamic(self.model)
                logger.info("Enabled INT8 dynamic quantization on CPU")
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Skipping CPU dynamic quantization: %s", exc)

        self.model.eval()
        self.try_compile()

    def try_compile(self) -> None:
        if self._compiled:
            return
        if sys.platform == "win32":
            logger.info("Skipping torch.compile on Windows")
            return
        try:
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                logger.info("Enabled torch.compile for native inference")
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("torch.compile unavailable, falling back to eager mode: %s", exc)

    def generate(
        self,
        prompt: str,
        tokenizer=None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vision_features=None,
        audio_features=None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self._generate_tokens(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                vision_features=vision_features,
                audio_features=audio_features,
            )

        new_ids = output_ids[0][prompt_len:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return self._lightweight_critique(result, prompt)

    def _lightweight_critique(self, response: str, prompt: str) -> str:
        if not response or len(response) < 10:
            return response
        try:
            from taiji.safety.constitutional_ai import get_constitutional_ai

            critic = get_constitutional_ai()
            critique = critic.critique(response, {"task": prompt[:200]})
            if not critique.passed and critique.revised_response:
                return critique.revised_response
        except Exception:  # pragma: no cover - best effort
            pass
        return response

    def generate_stream(
        self,
        prompt: str,
        tokenizer=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        kv_cache = None
        current_ids = input_ids
        token_buffer = []
        decode_batch = 8
        repeat_limit = 20
        last_token_id = None
        repeat_count = 0

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set():
                    break

                output = self.model(current_ids, kv_cache=kv_cache, use_cache=True)
                kv_cache = output.kv_cache

                next_logits = output.logits[:, -1, :] / max(temperature, 1e-6)
                next_token = self._sample_token(next_logits, top_p)
                token_id = int(next_token[0].item())

                if token_id == last_token_id:
                    repeat_count += 1
                    if repeat_count >= repeat_limit:
                        logger.debug("Detected repeated token loop, stopping early")
                        break
                else:
                    repeat_count = 0
                last_token_id = token_id

                token_buffer.append(next_token[0])

                if len(token_buffer) >= decode_batch:
                    text = self.tokenizer.decode(torch.stack(token_buffer), skip_special_tokens=True)
                    if text:
                        yield text
                    token_buffer = []

                if token_id == self.tokenizer.eos_token_id:
                    break

                current_ids = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

        if token_buffer:
            text = self.tokenizer.decode(torch.stack(token_buffer), skip_special_tokens=True)
            if text:
                yield text

    def generate_react_step(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            output_ids = self._generate_tokens(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )

        new_ids = output_ids[0][input_ids.shape[1] :]
        result = self._parse_react_output(new_ids)
        if "final_answer" in result:
            result = self._self_critique(result, prompt)
        return result

    def _self_critique(self, result: dict, prompt: str) -> dict:
        try:
            from taiji.safety.constitutional_ai import get_constitutional_ai

            critic = get_constitutional_ai()
            critique = critic.critique(
                result["final_answer"],
                {"task": prompt[:200], "observation": result.get("observation", "")},
            )
            if not critique.passed and critique.revised_response:
                result["final_answer"] = critique.revised_response
                result["_critique_applied"] = True
                result["_violations"] = [item["principle"] for item in critique.violations]
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Skipping self-critique: %s", exc)
        return result

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_on: Optional[list] = None,
        vision_features=None,
        audio_features=None,
    ) -> torch.Tensor:
        kv_cache = None
        current_ids = input_ids
        generated = []
        multimodal_injected = False

        for _ in range(max_new_tokens):
            if multimodal_injected:
                output = self.model(current_ids, kv_cache=kv_cache, use_cache=True)
            else:
                output = self.model(
                    current_ids,
                    kv_cache=kv_cache,
                    use_cache=True,
                    vision_features=vision_features,
                    audio_features=audio_features,
                )
                multimodal_injected = True

            kv_cache = output.kv_cache
            next_logits = output.logits[:, -1, :] / max(temperature, 1e-6)

            if generated:
                previous_ids = torch.stack(
                    [item[0] for item in generated if item.dim() == 1 or item.shape[0] == 1]
                ).unique()
                next_logits[0, previous_ids] /= 1.2

            next_token = self._sample_token(next_logits, top_p)
            token_id = int(next_token[0].item())

            if token_id == self.tokenizer.eos_token_id:
                break
            if stop_on and token_id in stop_on:
                break

            generated.append(next_token)
            current_ids = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

        if not generated:
            return input_ids

        generated_tensors = [item.unsqueeze(0) if item.dim() == 1 else item for item in generated]
        return torch.cat([input_ids] + generated_tensors, dim=1)

    def _sample_token(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        with torch.no_grad():
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
            probs = F.softmax(filtered_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _parse_react_output(self, token_ids: torch.Tensor) -> Dict[str, Any]:
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        thought_parts = []
        actions = []
        final_answer = None

        index = 0
        while index < len(ids):
            token_id = ids[index]

            if token_id == self._tokens["tool_call"]:
                if index + 1 >= len(ids):
                    break

                tool_id = ids[index + 1]
                tool_name = self.tokenizer.get_tool_name(tool_id) if hasattr(self.tokenizer, "get_tool_name") else None
                if not tool_name:
                    logger.warning("Unknown tool token ID: %s", tool_id)
                    index += 1
                    continue

                index += 2
                arg_token_ids = []
                while index < len(ids) and ids[index] not in (
                    self._tokens["tool_call_end"],
                    self._tokens["tool_result"],
                    self._tokens["answer"],
                    self._tokens["tool_call"],
                ):
                    arg_token_ids.append(ids[index])
                    index += 1

                if index < len(ids) and ids[index] == self._tokens["tool_call_end"]:
                    index += 1

                arg_text = self.tokenizer.decode(arg_token_ids, skip_special_tokens=True).strip()
                try:
                    action_args = json.loads(arg_text) if arg_text else {}
                except json.JSONDecodeError:
                    action_args = {"input": arg_text}

                actions.append({"action": tool_name, "action_args": action_args})
                continue

            if token_id == self._tokens["tool_result"]:
                index += 1
                while index < len(ids) and ids[index] != self._tokens["tool_result_end"]:
                    index += 1
                if index < len(ids) and ids[index] == self._tokens["tool_result_end"]:
                    index += 1
                continue

            if token_id == self._tokens["answer"]:
                final_answer = self.tokenizer.decode(ids[index + 1 :], skip_special_tokens=True).strip()
                break

            thought_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=True))
            index += 1

        thought_text = "".join(thought_parts).strip()

        if actions:
            return {
                "thought": thought_text,
                "action": actions[0]["action"],
                "action_args": actions[0]["action_args"],
                "extra_actions": actions[1:] if len(actions) > 1 else [],
            }

        if final_answer:
            return {"thought": thought_text, "final_answer": final_answer}

        if thought_text:
            logger.info("No structural ReAct markers detected; falling back to plain answer")
            return {"thought": thought_text, "final_answer": thought_text}

        return {"thought": "", "final_answer": "(模型未生成有效输出)"}

    def register_tools(self, tool_names: list) -> None:
        for name in tool_names:
            self.tokenizer.register_tool(name)
        self.model.set_num_tools(len(self.tokenizer._tool_name_to_id))
