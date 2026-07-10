"""
事件总线订阅注册 (Event Subscriptions)
======================================

所有引擎在启动时通过此模块注册彼此的事件订阅，
取代原来通过 JSON 文件系统通联的松耦合方式。

这是深度耦合的核心——引擎间实时通知，不再靠文件轮询。
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("Taiji.EventSubscriptions")


def register_all_subscriptions():
    """
    在应用启动时调用一次，注册所有引擎间的事件订阅。

    这个函数把原来通过 JSON 文件传递的松耦合，
    升级为 EventBus 发布-订阅的实时通信。
    """

    # ─── 1. FeedEngine 喂食完成 → 自动触发消化 ───
    def _on_feed_complete(event_data: Dict[str, Any]):
        """食物摄取完，自动通知睡眠引擎准备消化"""
        try:
            from taiji.life.sleep_engine import get_sleep_engine
            engine = get_sleep_engine()
            samples = event_data.get("samples", 0)
            logger.info(f"EventBus: feed_complete ({samples} samples) → 通知 SleepEngine")
            # 如果有足够的新数据，建议来一次小睡
            if samples >= 5 and not engine.is_sleeping():
                engine.nap(duration_minutes=2)  # 短睡快速消化
        except Exception as e:
            logger.debug(f"EventBus feed_complete handler: {e}")

    # ─── 2. 探索完成 → 通知进化引擎 ───
    def _on_explore_complete(event_data: Dict[str, Any]):
        """探索到新知识，通知进化引擎记录"""
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            engine = get_evolution_engine()
            topic = event_data.get("topic", "")
            pages = event_data.get("pages_read", 0)
            if topic:
                engine.metrics.knowledge_domains[topic] = \
                    engine.metrics.knowledge_domains.get(topic, 0.0) + pages
                logger.debug(f"EventBus: explore_complete ({topic}) → evolution_engine")
        except Exception as e:
            logger.debug(f"EventBus explore_complete handler: {e}")

    # ─── 3. 睡眠训练完成 → 通知 SelfModificationEngine 刷新策略 ───
    def _on_sleep_complete(event_data: Dict[str, Any]):
        """睡眠结束，模型可能变了，通知自修改引擎清零评估历史"""
        try:
            from taiji.agent_ext.self_modification import get_self_modification_engine
            engine = get_self_modification_engine()
            # 模型更新后，旧的评估数据可能不准确
            engine.clear_evaluations()
            logger.info("EventBus: sleep_complete → cleared self_modification evaluations")
        except Exception as e:
            logger.debug(f"EventBus sleep_complete handler: {e}")

    # ─── 4. 递归改进产生建议 → 推送给 SelfModificationEngine ───
    def _on_improvement_proposal(event_data: Dict[str, Any]):
        """RecursiveImprover 产生了改进建议，即时推给自修改引擎"""
        try:
            proposal = event_data.get("proposal", {})
            if proposal.get("confidence", 0) >= 0.7:
                from taiji.agent_ext.self_modification import get_self_modification_engine
                engine = get_self_modification_engine()
                engine.apply_suggestion(proposal.get("description", ""),
                                        proposal.get("proposal_type", ""),
                                        proposal.get("new_value", ""))
                logger.info(f"EventBus: improvement_proposal → self_modification: {proposal.get('description', '')[:60]}")
        except Exception as e:
            logger.debug(f"EventBus improvement_proposal handler: {e}")

    # ─── 5. 智能爬取完成 → 触发本地索引搜索能力刷新 ───
    def _on_crawl_complete(event_data: Dict[str, Any]):
        """爬取完成，更新索引统计"""
        try:
            indexed = event_data.get("indexed", 0)
            if indexed > 0:
                logger.info(f"EventBus: crawl_complete ({indexed} pages) → index ready for search")
                # 索引已经在 Pipeline 中实时更新了，这里只是通知
        except Exception as e:
            logger.debug(f"EventBus crawl_complete handler: {e}")

    # ─── 6. 科学研究完成 → 发现入知识库 ───
    def _on_research_complete(event_data: Dict[str, Any]):
        """科学研究完成，把发现存入知识库"""
        try:
            domain = event_data.get("domain", "")
            hypothesis = event_data.get("hypothesis", "")
            if domain and hypothesis:
                from taiji.agent_ext.knowledge_learner import get_knowledge_learner
                learner = get_knowledge_learner()
                learner.record_research_finding(domain, hypothesis)
                logger.info(f"EventBus: research_complete ({domain}) → knowledge_learner")
        except Exception as e:
            logger.debug(f"EventBus research_complete handler: {e}")

    # ─── 7. 用户交互成功 → 更新需求值（压力下降） ───
    def _on_interaction(event_data: Dict[str, Any]):
        """用户交互完成，调整生命需求"""
        try:
            from taiji.life.life_scheduler import get_life_scheduler
            scheduler = get_life_scheduler()
            success = event_data.get("success", True)
            if success:
                scheduler.needs.stress = max(0, scheduler.needs.stress - 5)
                scheduler.needs.boredom = max(0, scheduler.needs.boredom - 3)
        except Exception as e:
            logger.debug(f"EventBus interaction handler: {e}")

    # ─── 8. 模型错误 → 触发紧急自我评估 ───
    def _on_model_error(event_data: Dict[str, Any]):
        """模型出错，触发即时评估"""
        try:
            from taiji.infra.self_evaluator import get_self_evaluator
            evaluator = get_self_evaluator()
            error_type = event_data.get("error_type", "unknown")
            evaluator.record_incident(error_type)
            logger.debug(f"EventBus: model_error ({error_type}) → self_evaluator")
        except Exception as e:
            logger.debug(f"EventBus model_error handler: {e}")

    # ─── 注册所有订阅 ───
    from taiji.infra.events import get_event_bus, EventType
    bus = get_event_bus()

    # 已有事件类型
    bus.subscribe(EventType.FEED_COMPLETE, _on_feed_complete)
    bus.subscribe(EventType.EXPLORE_COMPLETE, _on_explore_complete)
    bus.subscribe(EventType.SLEEP_COMPLETE, _on_sleep_complete)
    bus.subscribe(EventType.RESEARCH_COMPLETE, _on_research_complete)
    bus.subscribe(EventType.INTERACTION_SUCCESS, lambda d: _on_interaction({"success": True}))
    bus.subscribe(EventType.INTERACTION_FAILURE, lambda d: _on_interaction({"success": False}))

    # 新增事件类型
    bus.subscribe("crawl_complete", _on_crawl_complete)
    bus.subscribe("improvement_proposal", _on_improvement_proposal)
    bus.subscribe("model_error", _on_model_error)

    logger.info(f"EventBus subscriptions registered ({bus.get_subscriber_count()})")
