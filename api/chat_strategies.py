"""
态极推理策略
============

态极是独立生命体，用自己的大脑推理。
不依赖外部云端 API，不套壳。

两种模式：
- 态极思维：直接文本生成（快速对话）
- 态极行动：ReAct 推理 + 工具调用（复杂任务）
"""
import asyncio
import json
import logging
import threading

logger = logging.getLogger("ApiServer.Chat.Strategies")


def _apply_rag(prompt, app_state):
    """从知识库检索相关上下文，注入 prompt"""
    if app_state.rag_kb and app_state.rag_kb.chunks:
        context = app_state.rag_kb.search_with_fallback(prompt)
        if context:
            context_str = "\n---\n".join(context)
            return (f"基于以下参考资料回答问题。\n\n"
                    f"【参考资料】\n{context_str}\n\n【问题】\n{prompt}")
    return prompt


def _get_life_state():
    """读取态极生命状态"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        life = get_life_scheduler()
        life.record_interaction(success=True, topic="chat")
        return life.needs.to_dict()
    except Exception:
        return {}


def _get_memory_context():
    """读取近期记忆，注入推理上下文（使用统一上下文管理器）"""
    try:
        from taiji.agent.context_manager import get_context_manager
        ctx = get_context_manager()
        wm_context = ctx._get_working_memory_context(500)
        if wm_context:
            return f"【近期记忆】\n{wm_context}\n\n"
    except Exception:
        pass

    # 回退到旧方式
    try:
        from taiji.agent.working_memory import get_working_memory
        wm = get_working_memory()
        all_memories = wm.export_all()
        if all_memories:
            lines = [f"- {str(v)[:100]}" for k, v in list(all_memories.items())[:5]]
            if lines:
                return "【近期记忆】\n" + "\n".join(lines) + "\n\n"
    except Exception:
        pass
    return ""


def _build_history(request):
    """构建对话历史"""
    history = []
    if request.history:
        for u, a in request.history:
            if u:
                history.append({"role": "user", "content": u})
            if a:
                history.append({"role": "assistant", "content": a})
    return history


def _record_evolution(prompt, result_text, success):
    """记录到进化引擎"""
    try:
        from taiji.life.evolution_engine import get_evolution_engine
        evo = get_evolution_engine()
        if success:
            evo.record_task_success(
                task=prompt[:200],
                steps=[{"action": "chat", "tool": "react"}],
                final_answer=result_text[:200],
            )
        else:
            evo.record_task_failure(
                task=prompt[:200],
                error=result_text[:200] if result_text else "empty",
            )
    except Exception:
        pass


async def _stream_react(request, prompt, app_state, collector):
    """
    态极行动模式 — ReAct 推理 + 工具调用。

    产出结构化 SSE 事件：
      {"type":"life","data":{"needs":{...}}}
      {"type":"thought","data":{"step":1,"content":"..."}}
      {"type":"action","data":{"step":1,"tool":"search","args":{...}}}
      {"type":"observation","data":{"step":1,"tool":"search","result":"..."}}
      {"type":"final","data":{"answer":"...","step":2}}
    """
    from taiji.agent_ext.react_engine import ReActEngine

    life_needs = _get_life_state()
    fatigue = life_needs.get("fatigue", 0)
    curiosity = life_needs.get("curiosity", 50)

    # 生命状态影响推理深度
    max_steps = 8
    if fatigue > 80:
        max_steps = 4
    elif curiosity > 80:
        max_steps = 12

    engine = ReActEngine(max_steps=max_steps)

    # 使用统一上下文管理器构建上下文
    try:
        from taiji.agent.context_manager import get_context_manager
        ctx = get_context_manager()

        # 注入对话历史到上下文管理器
        if request.history:
            for u, a in request.history:
                if u:
                    ctx.add_message("user", u)
                if a:
                    ctx.add_message("assistant", a)

        # 构建带记忆的 prompt
        enriched_prompt = ctx.build_context(
            task=prompt,
            system_prompt=request.system_prompt or "",
            include_history=True,
            include_memory=True,
        )

        # 构建消息列表（用于 ReAct 引擎）
        history = ctx._get_recent_history_messages()
    except Exception:
        # 回退到旧方式
        memory_context = _get_memory_context()
        enriched_prompt = (memory_context + prompt) if memory_context else prompt
        history = _build_history(request)

    # 发送生命状态
    if life_needs:
        yield f'data: {json.dumps({"type": "life", "data": {"needs": life_needs}}, ensure_ascii=False)}\n\n'
    full_text = ""

    try:
        for event in engine.run_stream(
            task=enriched_prompt,
            system_prompt=request.system_prompt or "",
            history=history,
        ):
            event_type = event.get("type", "")
            event_data = event.get("data", {})

            if event_type == "final":
                full_text = event_data.get("answer", "")
            elif event_type == "thought":
                full_text += event_data.get("content", "")

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"ReAct engine error: {e}")
        yield f'data: {json.dumps({"type": "error", "data": {"error": str(e)}}, ensure_ascii=False)}\n\n'

    success = bool(full_text and not full_text.startswith("["))
    _record_evolution(request.prompt, full_text, success)

    try:
        if collector and full_text:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"


async def _stream_think(request, prompt, app_state, stop_event, collector):
    """
    态极思维模式 — 直接文本生成（快速对话）。

    产出纯文本 SSE 流。使用统一上下文管理器。
    """
    # 使用统一上下文管理器构建上下文
    try:
        from taiji.agent.context_manager import get_context_manager
        ctx = get_context_manager()

        # 注入对话历史
        if request.history:
            for u, a in request.history:
                if u:
                    ctx.add_message("user", u)
                if a:
                    ctx.add_message("assistant", a)

        formatted = ctx.build_context(
            task=prompt,
            system_prompt=request.system_prompt or "",
            include_history=True,
            include_memory=True,
        )
    except Exception:
        # 回退到旧方式
        from taiji.agent_ext.token_optimizer import compress_history
        compressed = compress_history(request.history, max_rounds=3, max_chars_per_round=300)
        context_str = ""
        if compressed:
            context_str = "【上下文】\n" + "\n".join(f"用户: {u}\n助手: {a}" for u, a in compressed) + "\n\n"
        formatted = (f"{request.system_prompt or ''}\n\n{context_str}"
                     f"### Instruction:\n{prompt}\n### Response:\n")

    # 注入当前系统时间，让模型知道"现在是什么时候"
    try:
        from datetime import datetime
        import pytz
        cst = pytz.timezone('Asia/Shanghai')
        now_str = datetime.now(cst).strftime('%Y年%m月%d日 %H:%M')
        weekday_map = ['一', '二', '三', '四', '五', '六', '日']
        weekday = weekday_map[datetime.now(cst).weekday()]
        time_hint = f"【系统】当前时间：{now_str}（周{weekday}），时区 Asia/Shanghai\n\n"
        formatted = time_hint + formatted
    except Exception:
        # pytz 不可用时用简单方式
        try:
            from datetime import datetime
            from datetime import timezone, timedelta
            cst_offset = timezone(timedelta(hours=8))
            now_str = datetime.now(cst_offset).strftime('%Y年%m月%d日 %H:%M')
            formatted = f"【系统】当前时间：{now_str}，时区 Asia/Shanghai\n\n" + formatted
        except Exception:
            pass

    # 读取生命状态
    life_needs = _get_life_state()
    if life_needs:
        yield f'data: {json.dumps({"type": "life", "data": {"needs": life_needs}}, ensure_ascii=False)}\n\n'

    taiji = app_state.get_taiji_engine()
    tokenizer = app_state.get_tokenizer()
    full_text = ""

    if taiji and tokenizer:
        # 态极原生推理
        try:
            for chunk in taiji.generate_stream(formatted, tokenizer, max_new_tokens=512, stop_event=stop_event):
                if stop_event.is_set():
                    logger.info("推理被用户停止")
                    break
                full_text += chunk
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.warning(f"态极推理失败: {e}")
            yield f"data: {json.dumps(f'[推理失败: {e}]', ensure_ascii=False)}\n\n"
    else:
        # 回退到 trainer
        try:
            trainer = app_state.get_trainer()
            if trainer and hasattr(trainer, 'generate_stream'):
                for chunk in trainer.generate_stream(formatted, tokenizer, max_new_tokens=512, stop_event=stop_event):
                    if stop_event.is_set():
                        logger.info("推理被用户停止")
                        break
                    full_text += chunk
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
            elif trainer and hasattr(trainer, 'generate'):
                full_text = trainer.generate(formatted, tokenizer, max_new_tokens=512)
                if "### Response:\n" in full_text:
                    full_text = full_text.split("### Response:\n", 1)[-1].strip()
                yield f"data: {json.dumps(full_text, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps('[模型未加载]', ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[推理失败: {e}]', ensure_ascii=False)}\n\n"

    success = bool(full_text and not full_text.startswith("["))
    _record_evolution(request.prompt, full_text, success)

    try:
        if collector and full_text:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"


def create_event_generator(request, app_state, collector_factory):
    """
    统一推理入口。

    态极只有两种模式：
    - agent/行动：ReAct 推理 + 工具调用（结构化事件）
    - 其他/思维：直接文本生成（纯文本流）
    """
    async def event_generator():
        stop_event = threading.Event()
        try:
            collector = None
            try:
                collector = collector_factory()
            except Exception:
                pass

            prompt = _apply_rag(request.prompt, app_state)

            # 行动模式：ReAct 推理 + 工具调用
            if "agent" in request.engine or "react" in request.engine:
                async for chunk in _stream_react(request, prompt, app_state, collector):
                    yield chunk
                return

            # 思维模式：直接文本生成
            async for chunk in _stream_think(request, prompt, app_state, stop_event, collector):
                yield chunk

        except (GeneratorExit, RuntimeError, asyncio.CancelledError):
            logger.info("推理客户端已断开连接，正在停止生成...")
            stop_event.set()
        except Exception as e:
            logger.error(f"推理生成出错: {e}")
            yield f"data: {json.dumps(f'生成出错: {e}', ensure_ascii=False)}\n\n"
        finally:
            stop_event.set()

    return event_generator
