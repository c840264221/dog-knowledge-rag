"""
RootAgent Observability 工具模块。

功能：
    为 RootAgent（根智能体 / 根调度器）提供可观测能力。

    当前模块主要负责：
    1. 构建 RootAgent 路由调试元数据。
    2. 尝试把 RootAgent 路由决策写入 Runtime Timeline。
    3. 保证可观测能力失败时，不影响主路由流程。

设计原则：
    1. RootAgent 的核心职责是路由。
    2. Observability（可观测性）是增强能力。
    3. 可观测写入失败不能导致用户请求失败。
    4. 输出数据必须是 dict，方便 LangGraph checkpoint 序列化。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.root_agent.schemas import RootRouteDecision
from src.logger import logger
from src.runtime.context import runtime_ctx


def build_root_route_observability_payload(
        question: str,
        decision: RootRouteDecision,
        current_agent: str = "root_agent",
) -> dict[str, Any]:
    """
    构建 RootAgent 路由可观测数据。

    功能：
        将 RootRouteDecision 转换成可写入 state、timeline、
        debug report 的普通 dict 数据。

        该函数只负责构建数据，不负责写入 timeline。

    参数：
        question:
            用户原始问题。

        decision:
            RootRouteDecision，RootAgent 生成的新版路由决策对象。

        current_agent:
            当前 Agent 名称。
            默认是 root_agent。

    返回值：
        dict[str, Any]:
            RootAgent 路由可观测数据。
            该 dict 可以安全写入 LangGraph state。
    """

    return {
        "component": "root_agent",
        "event_type": "route",
        "event_name": "root_route_decision",
        "question": question,
        "route": decision.route,
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "requires_rag": decision.requires_rag,
        "requires_tool": decision.requires_tool,
        "requires_memory": decision.requires_memory,
        "source": decision.source,
        "hints": decision.hints,
        "current_agent": current_agent,
        "next_agent": decision.route,
        "created_at": datetime.now(
            timezone.utc,
        ).isoformat(),
    }


def record_root_route_timeline(
        payload: dict[str, Any],
) -> bool:
    """
    尝试将 RootAgent 路由数据写入 Runtime Timeline。

    功能：
        从 runtime_ctx 中获取当前 Runtime Context，
        然后尝试写入 timeline event。

        如果当前没有 runtime context，或者 timeline 写入失败，
        本函数不会抛出异常，而是返回 False。

    参数：
        payload:
            RootAgent 路由可观测数据。
            通常由 build_root_route_observability_payload 生成。

    返回值：
        bool:
            True 表示成功写入 Runtime Timeline。
            False 表示没有写入成功，但不会影响主流程。
    """

    try:
        runtime_context = runtime_ctx.get()
    except Exception as exc:
        logger.debug(
            f"RootAgent 获取 runtime context 失败，跳过 timeline 写入: {exc}"
        )
        return False

    if runtime_context is None:
        logger.debug(
            "RootAgent 当前没有 runtime context，跳过 timeline 写入"
        )
        return False

    timeline = _safe_get_timeline(
        runtime_context=runtime_context,
    )

    if timeline is None:
        logger.debug(
            "RootAgent 当前 runtime context 中没有 timeline，跳过写入"
        )
        return False

    return _safe_add_timeline_event(
        timeline=timeline,
        payload=payload,
    )


def _safe_get_timeline(
        runtime_context: Any,
) -> Any | None:
    """
    安全获取 Runtime Timeline。

    功能：
        兼容两种 timeline 获取方式：

        1. runtime_context.timeline()
        2. runtime_context.timeline

        这样可以降低不同 Runtime Context 实现之间的耦合。

    参数：
        runtime_context:
            当前运行时上下文对象。

    返回值：
        Any | None:
            如果能获取 timeline，则返回 timeline 对象。
            如果无法获取，则返回 None。
    """

    timeline_attr = getattr(
        runtime_context,
        "timeline",
        None,
    )

    if timeline_attr is None:
        return None

    if callable(
            timeline_attr,
    ):
        try:
            return timeline_attr()
        except Exception as exc:
            logger.debug(
                f"RootAgent 调用 runtime_context.timeline() 失败: {exc}"
            )
            return None

    return timeline_attr


def _safe_add_timeline_event(
        timeline: Any,
        payload: dict[str, Any],
) -> bool:
    """
    安全写入 timeline event。

    功能：
        兼容多种可能的 timeline.add_event 调用形式。

        优先尝试：
            add_event(event_type=..., name=..., metadata=...)

        如果不支持，再尝试：
            add_event(event_type=..., name=..., payload=...)

        如果仍不支持，再尝试：
            add_event(payload)

        这样可以减少 RootAgent 对具体 Timeline 实现的依赖。

    参数：
        timeline:
            Runtime Timeline 对象。

        payload:
            需要写入的事件数据。

    返回值：
        bool:
            True 表示写入成功。
            False 表示写入失败。
    """

    add_event = getattr(
        timeline,
        "add_event",
        None,
    )

    if not callable(
            add_event,
    ):
        logger.debug(
            "RootAgent timeline 对象没有 add_event 方法"
        )
        return False

    try:
        add_event(
            event_type=payload.get(
                "event_type",
                "route",
            ),
            name=payload.get(
                "event_name",
                "root_route_decision",
            ),
            metadata=payload,
        )
        return True
    except TypeError:
        pass
    except Exception as exc:
        logger.debug(
            f"RootAgent 使用 metadata 写入 timeline 失败: {exc}"
        )
        return False

    try:
        add_event(
            event_type=payload.get(
                "event_type",
                "route",
            ),
            name=payload.get(
                "event_name",
                "root_route_decision",
            ),
            payload=payload,
        )
        return True
    except TypeError:
        pass
    except Exception as exc:
        logger.debug(
            f"RootAgent 使用 payload 写入 timeline 失败: {exc}"
        )
        return False

    try:
        add_event(
            payload,
        )
        return True
    except Exception as exc:
        logger.debug(
            f"RootAgent 使用单参数写入 timeline 失败: {exc}"
        )
        return False