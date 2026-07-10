"""
Agent ReAct 推理引擎
====================
实现 Thought -> Action -> Observation 自主推理循环。
支持本地模型和云端 API 双模式。
"""
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger("ReActEngine")


@dataclass
class AgentStep:
    step_number: int
    thought: str = ""
    action: str = ""
    action_args: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class AgentResult:
    task: str
    final_answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0
    status: str = "completed"


class ToolCallParser:
    """多策略工具调用解析器

    解析策略优先级:
    1. 原生 tool_calls（云端 API 返回）
    2. JSON 代码块
    3. <tool_call> 标签
    4. JSON 行内
    5. 函数式调用（use_tool / call_tool）
    6. Action 格式（Action: tool_name(args)）
    7. 思考-行动三段式（Action: tool_name / Action Input: {...}）
    8. 尝试修复不完整 JSON
    """

    def parse(self, content: str, tool_calls_raw: list = None,
              available_tools: list = None) -> list:
        if tool_calls_raw:
            return self._parse_native_calls(tool_calls_raw)
        if not content:
            return []
        results = self._extract_json_blocks(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._extract_xml_tool_calls(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._extract_inline_json(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._extract_function_calls(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._extract_action_format(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._extract_thought_action_format(content)
        if results:
            return self._validate_all(results, available_tools)
        results = self._try_repair_json(content)
        if results:
            return self._validate_all(results, available_tools)
        return []

    def _parse_native_calls(self, tool_calls_raw: list) -> list:
        results = []
        for tc in tool_calls_raw:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if name:
                results.append({"name": name, "arguments": args})
        return results

    def _extract_json_blocks(self, text: str) -> list:
        import re
        results = []
        for match in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, list):
                    results.extend(self._normalize(data))
                elif isinstance(data, dict):
                    results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        if results:
            return results
        for match in re.finditer(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL):
            try:
                data = json.loads(match.group(0))
                results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        return results

    def _extract_xml_tool_calls(self, text: str) -> list:
        import re
        results = []
        for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                results.extend(self._normalize([data]))
            except json.JSONDecodeError:
                continue
        return results

    def _extract_inline_json(self, text: str) -> list:
        import re
        results = []
        pattern1 = r'\{[^{}]*?"tool"\s*:\s*"([^"]+)"[^{}]*?"args"\s*:\s*(\{[^}]*?\})[^{}]*?\}'
        for tool_name, args_str in re.findall(pattern1, text, re.DOTALL):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"input": args_str}
            results.append({"name": tool_name, "arguments": args})
        pattern2 = r'\{[^{}]*?"name"\s*:\s*"([^"]+)"[^{}]*?"arguments"\s*:\s*(\{[^}]*?\})[^{}]*?\}'
        for tool_name, args_str in re.findall(pattern2, text, re.DOTALL):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"input": args_str}
            results.append({"name": tool_name, "arguments": args})
        if not results:
            try:
                data = json.loads(text.strip())
                if isinstance(data, dict) and ("name" in data or "tool" in data):
                    name = data.get("name") or data.get("tool", "")
                    args = data.get("arguments") or data.get("args", {})
                    if name:
                        results.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                pass
        return results

    def _extract_function_calls(self, text: str) -> list:
        """提取函数式调用：use_tool("name", param=value) 或 call_tool("name", {...})"""
        import re
        results = []
        # 匹配 use_tool("tool_name", param=value) 或 call_tool("tool_name", {...})
        pattern = r'(?:use_tool|call_tool)\s*\(\s*["\'](\w+)["\']\s*,\s*(\{[^}]*\}|[^)]*)\)'
        for tool_name, args_str in re.findall(pattern, text, re.DOTALL):
            args_str = args_str.strip()
            if args_str.startswith('{'):
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                # 解析 key=value 格式
                args = {}
                for pair in re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                    args[pair[0]] = pair[1]
                if not args:
                    args = {"input": args_str}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _extract_action_format(self, text: str) -> list:
        import re
        results = []
        for tool_name, args_str in re.findall(r'Action\s*:\s*(\w+)\(([^)]*)\)', text):
            args_str = args_str.strip().strip('"\'')
            if args_str:
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                args = {}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _extract_thought_action_format(self, text: str) -> list:
        """
        提取思考-行动-观察三段式：
          Thought: xxx
          Action: tool_name
          Action Input: {"param": "value"}
        """
        import re
        results = []
        # 匹配 Action: tool_name 后跟 Action Input: {...}
        pattern = r'Action\s*:\s*(\w+)\s*\n\s*Action\s*Input\s*:\s*(\{[^}]*\}|[^\n]+)'
        for tool_name, args_str in re.findall(pattern, text, re.DOTALL):
            args_str = args_str.strip()
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"input": args_str} if args_str else {}
            results.append({"name": tool_name, "arguments": args})

        # 也匹配单行 Action: tool_name（无参数）
        if not results:
            single_pattern = r'(?<=\n)Action\s*:\s*(\w+)\s*$'
            for match in re.finditer(single_pattern, text, re.MULTILINE):
                tool_name = match.group(1).strip()
                if tool_name and tool_name not in ('Input', 'Output', 'Result'):
                    results.append({"name": tool_name, "arguments": {}})
        return results

    def _try_repair_json(self, text: str) -> list:
        import re
        results = []
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"([^"]+)".*?"args"\s*:\s*\{([^}]*)', text, re.DOTALL):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            if args_str:
                try:
                    args = json.loads("{" + args_str + "}")
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                args = {}
            results.append({"name": tool_name, "arguments": args})
        return results

    def _normalize(self, items: list) -> list:
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("tool") or item.get("name") or item.get("function") or ""
            args = item.get("args") or item.get("arguments") or item.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if name:
                results.append({"name": name, "arguments": args})
        return results

    def _validate_all(self, results: list, available_tools: list = None) -> list:
        if not available_tools:
            return results
        valid = []
        for tc in results:
            if tc["name"] in available_tools:
                valid.append(tc)
            else:
                for t in available_tools:
                    if t.lower() == tc["name"].lower():
                        tc["name"] = t
                        valid.append(tc)
                        break
        return valid


class FewShotGenerator:
    """根据可用工具动态生成 few-shot 示例"""

    def generate(self, tool_schemas: list, max_examples: int = 3) -> str:
        if not tool_schemas:
            return ""
        examples = []
        for i, schema in enumerate(tool_schemas[:max_examples]):
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            if not name:
                continue
            example_args = {}
            for pname, pinfo in list(params.items())[:2]:
                ptype = pinfo.get("type", "string")
                if ptype == "string":
                    example_args[pname] = "示例值"
                elif ptype == "integer":
                    example_args[pname] = 5
                elif ptype == "boolean":
                    example_args[pname] = True
                elif ptype == "number":
                    example_args[pname] = 1.0
            examples.append(
                f"用户: 请帮我完成某个任务\n"
                f"助手: 我来思考一下...\n"
                f"```json\n{json.dumps({'tool': name, 'args': example_args}, ensure_ascii=False, indent=2)}\n```"
            )
        return "\n\n".join(examples) if examples else ""


class ReActEngine:
    """ReAct 推理引擎：Thought -> Action -> Observation 循环"""

    def __init__(self, max_steps: int = 15, stream_callback: Callable = None):
        """max_steps: 最大推理步数（15 步足以覆盖大多数多步工具调用场景）"""
        self.max_steps = max_steps
        self.stream_callback = stream_callback
        self._cancelled = False
        self._tool_parser = ToolCallParser()

    def cancel(self):
        self._cancelled = True

    def _emit(self, event_type: str, data: dict):
        if self.stream_callback:
            try:
                self.stream_callback(event_type, data)
            except Exception:
                pass

    def run(self, task: str, system_prompt: str = "", history: list = None,
            skill_tools: list = None) -> AgentResult:
        from taiji.agent_ext.tool_registry import registry
        from taiji.agent_ext.skill_manager import skill_manager

        self._cancelled = False
        result = AgentResult(task=task)
        start_time = time.time()

        if skill_tools:
            tool_schemas = [t.to_schema() for t in
                           [registry.get(n) for n in skill_tools if registry.get(n)]]
        else:
            tool_schemas = registry.get_tool_schemas()

        if not system_prompt:
            tool_desc = registry.get_tool_descriptions()
            if skill_tools:
                lines = ["可用工具:"]
                for name in skill_tools:
                    t = registry.get(name)
                    if t:
                        lines.append(f"- **{t.name}**: {t.description}")
                tool_desc = "\n".join(lines)
            system_prompt = self._build_system_prompt(tool_desc)

        skill_prompt = skill_manager.get_skill_system_prompt()
        if skill_prompt:
            system_prompt = skill_prompt + "\n\n" + system_prompt

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        self._emit("start", {"task": task, "max_steps": self.max_steps})

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                result.status = "stopped"
                break

            step = AgentStep(step_number=step_num)
            step_start = time.time()

            try:
                self._emit("step_start", {"step": step_num})
                llm_response = self._call_llm(system_prompt, messages, tool_schemas)

                if llm_response.get("error"):
                    step.error = llm_response["error"]
                    result.steps.append(step)
                    result.status = "failed"
                    break

                content = llm_response.get("content", "")
                tool_calls = llm_response.get("tool_calls", [])

                if content:
                    step.thought = content

                if self._is_final_answer(content, tool_calls, step_num, self.max_steps):
                    step.is_final = True
                    step.observation = content
                    result.steps.append(step)
                    result.final_answer = content
                    result.status = "completed"
                    self._emit("final_answer", {"answer": content, "step": step_num})
                    break

                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                tool_args = {"input": tool_args}

                        if skill_tools and tool_name not in skill_tools:
                            step.observation = f"工具 {tool_name} 不在当前技能范围内"
                            messages.append({"role": "assistant", "content": content or ""})
                            messages.append({"role": "user", "content": step.observation + "。请使用可用的工具。"})
                            continue

                        step.action = tool_name
                        step.action_args = tool_args
                        self._emit("tool_call", {"step": step_num, "tool": tool_name, "args": tool_args})
                        observation = registry.execute(tool_name, tool_args)
                        step.observation = observation
                        self._emit("observation", {"step": step_num, "tool": tool_name, "result": observation[:500]})
                        messages.append({
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": [{"id": f"call_{step_num}", "type": "function",
                                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]
                        })
                        messages.append({"role": "tool", "tool_call_id": f"call_{step_num}", "content": observation})
                else:
                    step.thought = content
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "请继续执行任务。如果已完成请给出最终答案。"})

            except Exception as e:
                step.error = str(e)
                logger.error(f"ReAct 步骤 {step_num} 异常: {traceback.format_exc()}")

            step.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step)
            self._emit("step_end", {"step": step_num, "thought": step.thought[:200],
                                    "action": step.action, "duration_ms": step.duration_ms})
        else:
            result.status = "max_steps"
            result.final_answer = f"达到最大推理步数 ({self.max_steps})"

        result.total_duration_ms = (time.time() - start_time) * 1000
        self._emit("complete", {"status": result.status, "steps": len(result.steps),
                                "duration_ms": result.total_duration_ms})
        return result

    def run_stream(self, task: str, system_prompt: str = "", history: list = None,
                   skill_tools: list = None):
        from taiji.agent_ext.tool_registry import registry
        from taiji.agent_ext.skill_manager import skill_manager

        self._cancelled = False

        if skill_tools:
            tool_schemas = [t.to_schema() for t in [registry.get(n) for n in skill_tools if registry.get(n)]]
        else:
            tool_schemas = registry.get_tool_schemas()

        if not system_prompt:
            tool_desc = registry.get_tool_descriptions()
            if skill_tools:
                lines = ["可用工具:"]
                for name in skill_tools:
                    t = registry.get(name)
                    if t:
                        lines.append(f"- **{t.name}**: {t.description}")
                tool_desc = "\n".join(lines)
            system_prompt = self._build_system_prompt(tool_desc)

        skill_prompt = skill_manager.get_skill_system_prompt()
        if skill_prompt:
            system_prompt = skill_prompt + "\n\n" + system_prompt

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        yield {"type": "start", "data": {"task": task}}

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                yield {"type": "cancelled", "data": {}}
                break

            try:
                llm_response = self._call_llm(system_prompt, messages, tool_schemas)
                if llm_response.get("error"):
                    yield {"type": "error", "data": {"error": llm_response["error"]}}
                    break

                content = llm_response.get("content", "")
                tool_calls = llm_response.get("tool_calls", [])

                if content:
                    yield {"type": "thought", "data": {"step": step_num, "content": content}}

                if self._is_final_answer(content, tool_calls, step_num, self.max_steps):
                    yield {"type": "final", "data": {"answer": content, "step": step_num}}
                    break

                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                tool_args = {"input": tool_args}
                        yield {"type": "action", "data": {"step": step_num, "tool": tool_name, "args": tool_args}}
                        observation = registry.execute(tool_name, tool_args)
                        yield {"type": "observation", "data": {"step": step_num, "tool": tool_name, "result": observation[:1000]}}
                        messages.append({"role": "assistant", "content": content or "",
                                         "tool_calls": [{"id": f"call_{step_num}", "type": "function",
                                                         "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]})
                        messages.append({"role": "tool", "tool_call_id": f"call_{step_num}", "content": observation})
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "请继续执行任务。如果已完成请给出最终答案。"})

            except Exception as e:
                yield {"type": "error", "data": {"step": step_num, "error": str(e)}}
                break

        yield {"type": "done", "data": {}}

    def _call_llm(self, system_prompt: str, messages: list, tools: list) -> dict:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        return self._call_local_model(full_messages)

    def _call_openai_compatible(self, api_base: str, api_key: str, model: str, messages: list, tools: list) -> dict:
        import urllib.request
        url = api_base.rstrip("/") + "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 2000}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            result = {"content": message.get("content", ""), "tool_calls": []}
            for tc in message.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"input": args}
                result["tool_calls"].append({"id": tc.get("id", ""), "name": func.get("name", ""), "arguments": args})
            return result
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return {"error": str(e)}

    def _call_local_model(self, messages: list, available_tools: list = None) -> dict:
        """
        态极推理核心。

        架构: ModelSelf (5头: language/tool/perception/memory/plan)
        native-v2: SentencePiece tokenizer + contract offset
        """
        from taiji.core.app_state import app_state

        content = ""
        _inference_error = None
        tokenizer = app_state.get_tokenizer()

        # 构建 prompt (态极统一格式)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content_str = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[系统] {content_str}")
            elif role == "user":
                prompt_parts.append(f"[用户] {content_str}")
            elif role == "assistant":
                prompt_parts.append(f"[助手] {content_str}")
            elif role == "tool":
                prompt_parts.append(f"[工具结果] {content_str}")
        prompt = "\n".join(prompt_parts) + "\n[助手]"

        # 推理
        try:
            trainer = app_state.trainer
            if trainer and tokenizer:
                if hasattr(trainer, 'generate'):
                    content = trainer.generate(prompt, tokenizer, max_new_tokens=800, temperature=0.4)
                    content = content.strip()
                elif hasattr(trainer, 'generate_stream'):
                    chunks = []
                    for chunk in trainer.generate_stream(prompt, tokenizer, max_new_tokens=800, temperature=0.4):
                        chunks.append(chunk)
                    content = "".join(chunks).strip()
        except Exception as e:
            _inference_error = e
            logger.warning(f"推理失败: {e}")

        if not content:
            error_detail = ""
            if _inference_error:
                error_detail = f"\n推理错误: {str(_inference_error)[:200]}"
            elif app_state.startup_error:
                error_detail = f"\n详细错误: {app_state.startup_error[:200]}"
            elif not app_state.startup_complete:
                error_detail = "\n模型正在加载中，请稍候重试。"
            elif not app_state.model:
                error_detail = "\n模型未加载，请在设置中检查模型配置。"
            else:
                error_detail = "\n模型已加载但推理返回空结果，请检查模型文件是否完整。"
            return {"error": f"态极大脑未加载。{error_detail}\n请在设置中检查模型配置，或等待模型加载完成后重试。"}

        # 使用多策略解析器提取工具调用
        tool_names = [s.get("function", {}).get("name", "") for s in (available_tools or [])]
        tool_calls = self._tool_parser.parse(content, available_tools=tool_names)
        if tool_calls:
            return {"content": "", "tool_calls": tool_calls}

        # 自修复：第一次解析失败时重试一次（仅对长内容）
        if content and len(content) > 10:
            repair_result = self._self_repair(content, messages, available_tools)
            if repair_result:
                return repair_result

        return {"content": content, "tool_calls": []}

    def _self_repair(self, raw_output, messages, available_tools=None):
        try:
            from taiji.core.app_state import app_state
            trainer = app_state.trainer
            if not trainer or not hasattr(trainer, 'generate'):
                return None
            tool_names = [s.get('function', {}).get('name', '') for s in (available_tools or [])]
            tool_list = ', '.join(tool_names[:10]) if tool_names else 'no tools'
            repair_prompt = (
                '[system] Your last reply could not be parsed as a valid tool call.\n'
                'Original reply: ' + raw_output[:300] + '\n\n'
                'Available tools: ' + tool_list + '\n\n'
                'Please re-output the tool call in strict JSON format (JSON only, no other content):\n'
                '{"tool": "tool_name", "args": {"param": "value"}}\n'
                '[assistant]'
            )
            tokenizer = app_state.get_tokenizer()
            response = trainer.generate(repair_prompt, tokenizer, max_new_tokens=300, temperature=0.1)
            content = response.strip()
            tool_calls = self._tool_parser.parse(content, available_tools=tool_names)
            if tool_calls:
                logger.info('Self-repair succeeded')
                return {'content': '', 'tool_calls': tool_calls}
        except Exception as e:
            logger.debug(f'Self-repair failed: {e}')
        return None

    def _build_system_prompt(self, tool_descriptions, tool_schemas=None):
        few_shot = ''
        if tool_schemas:
            generator = FewShotGenerator()
            few_shot = generator.generate(tool_schemas, max_examples=2)
            if few_shot:
                few_shot = '\n\n## 示例\n' + few_shot + '\n'

        return f"""你是态极（Taiji），一个本地运行的AI助手。你可以直接回答问题，也可以使用工具来完成任务。

## 可用工具
{tool_descriptions}

## 回答方式
- 如果问题可以直接回答，用自然语言回答即可
- 如果需要搜索或使用工具，输出以下 JSON 格式：
```json
{{"tool": "工具名", "args": {{"参数名": "值"}}}}
```
{few_shot}
## 规则
- 每次只调用一个工具
- 只能使用上述列出的工具
- 保持简洁、准确"""

    def _is_final_answer(self, content, tool_calls, step_num, max_steps=15):
        """判断是否为最终答案（融合文本模式 + 行动模式）"""
        if tool_calls:
            return False
        if not content:
            # B1-RE 修复：空 content + 无工具 → 若接近 max_steps 就终止，避免死循环
            if step_num >= max_steps - 2:
                return True
            return False
        # 显式最终答案标记
        final_markers = ['最终答案', 'Final Answer', 'final answer', 'DONE', '任务完成']
        for marker in final_markers:
            if marker in content:
                return True
        # 第一步无工具调用 + 内容足够长 → 直接视为最终回答（文本模式）
        if not tool_calls and len(content) > 15:
            return True
        # B1-RE 修复：接近 max_steps 就终止
        if step_num >= max_steps - 2:
            return True
        return False


class AgentController:
    def __init__(self, max_steps=15):
        self.engine = ReActEngine(max_steps=max_steps)
        self._task_history = []

    def run_task(self, task, stream_callback=None, skill_id=None):
        from taiji.agent_ext.skill_manager import skill_manager
        if stream_callback:
            self.engine.stream_callback = stream_callback
        skill_tools = None
        if skill_id:
            skill = skill_manager.activate_skill(skill_id)
            if skill:
                skill_tools = skill.tools
        result = self.engine.run(task, history=self._task_history[-10:], skill_tools=skill_tools)
        self._task_history.append({'role': 'user', 'content': task})
        if result.final_answer:
            self._task_history.append({'role': 'assistant', 'content': result.final_answer})
        if result.status == 'completed' and len(result.steps) >= 2:
            skill_manager.learn_from_task(task,
                [{'thought': s.thought, 'action': s.action} for s in result.steps],
                result.final_answer)
        return result

    def run_task_stream(self, task, skill_id=None):
        from taiji.agent_ext.skill_manager import skill_manager
        skill_tools = None
        if skill_id:
            skill = skill_manager.activate_skill(skill_id)
            if skill:
                skill_tools = skill.tools
        return self.engine.run_stream(task, history=self._task_history[-10:], skill_tools=skill_tools)

    def cancel(self):
        self.engine.cancel()

    def clear_history(self):
        self._task_history.clear()

    def get_history(self):
        return self._task_history.copy()
