"""
RootAgent Debug Report 字段构建模块。

功能：
    将 RootAgent 的可观测数据整理成 Debug Report 可以直接使用的字段。

    当前模块不负责：
    1. 执行主图路由。
    2. 写入 Runtime Timeline。
    3. 调用 LLM。
    4. 生成完整 Markdown 文件。

设计原则：
    1. RootAgent 负责主路由。
    2. observability.py 负责运行时可观测数据。
    3. debug_report.py 负责报告展示字段。
    4. 输出必须是普通 dict，方便 LangGraph checkpoint 和 JSON 序列化。
"""

from __future__ import annotations

from typing import Any


def build_root_debug_report_fields(
        root_observability: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    构建 RootAgent Debug Report 字段。

    功能：
        将 root_observability 转换成更适合报告展示的结构。

        该函数只做数据整理，不写文件、不写 timeline、
        不修改原始 root_observability。

    参数：
        root_observability:
            RootAgent 可观测数据。
            通常来自 state["root_observability"]。

    返回值：
        dict[str, Any]:
            RootAgent Debug Report 字段。
            可以写入 state["root_debug_report"]。
    """

    if not root_observability:
        return build_empty_root_debug_report_fields()

    route = str(
        root_observability.get(
            "route",
            "",
        )
    )

    query_type = str(
        root_observability.get(
            "query_type",
            "",
        )
    )

    confidence = root_observability.get(
        "confidence",
        0.0,
    )

    reason = str(
        root_observability.get(
            "reason",
            "",
        )
    )

    return {
        "section": "root_agent",
        "section_title": "RootAgent 路由决策",
        "status": "available",
        "summary": build_root_route_summary(
            route=route,
            query_type=query_type,
            confidence=confidence,
        ),
        "question": root_observability.get(
            "question",
            "",
        ),
        "route": route,
        "query_type": query_type,
        "confidence": confidence,
        "reason": reason,
        "requires": {
            "rag": bool(
                root_observability.get(
                    "requires_rag",
                    False,
                )
            ),
            "tool": bool(
                root_observability.get(
                    "requires_tool",
                    False,
                )
            ),
            "memory": bool(
                root_observability.get(
                    "requires_memory",
                    False,
                )
            ),
        },
        "agent_flow": {
            "current_agent": root_observability.get(
                "current_agent",
                "root_agent",
            ),
            "next_agent": root_observability.get(
                "next_agent",
                route,
            ),
        },
        "timeline": {
            "recorded": bool(
                root_observability.get(
                    "timeline_recorded",
                    False,
                )
            ),
            "event_type": root_observability.get(
                "event_type",
                "route",
            ),
            "event_name": root_observability.get(
                "event_name",
                "root_route_decision",
            ),
        },
        "source": root_observability.get(
            "source",
            "",
        ),
        "hints": root_observability.get(
            "hints",
            {},
        ),
        "created_at": root_observability.get(
            "created_at",
            "",
        ),
    }


def build_empty_root_debug_report_fields() -> dict[str, Any]:
    """
    构建空的 RootAgent Debug Report 字段。

    功能：
        当 state 中没有 root_observability 时，
        返回一个稳定的空报告结构，避免 Debug Report 生成阶段报错。

    参数：
        无。

    返回值：
        dict[str, Any]:
            空 RootAgent Debug Report 字段。
    """

    return {
        "section": "root_agent",
        "section_title": "RootAgent 路由决策",
        "status": "not_available",
        "summary": "RootAgent 路由可观测数据不可用。",
        "question": "",
        "route": "",
        "query_type": "",
        "confidence": 0.0,
        "reason": "",
        "requires": {
            "rag": False,
            "tool": False,
            "memory": False,
        },
        "agent_flow": {
            "current_agent": "root_agent",
            "next_agent": "",
        },
        "timeline": {
            "recorded": False,
            "event_type": "route",
            "event_name": "root_route_decision",
        },
        "source": "",
        "hints": {},
        "created_at": "",
    }


def build_root_route_summary(
        route: str,
        query_type: str,
        confidence: Any,
) -> str:
    """
    构建 RootAgent 路由摘要。

    功能：
        将 route、query_type、confidence 转换成一段适合 Debug Report 展示的中文摘要。

    参数：
        route:
            RootAgent 判断出的目标路由。

        query_type:
            用户问题类型。

        confidence:
            路由置信度。
            可能来自模型或规则，可以是 float，也可能是其他可转字符串的值。

    返回值：
        str:
            中文路由摘要。
    """

    confidence_text = format_confidence(
        confidence,
    )

    route_label = get_route_label(
        route=route,
    )

    query_type_label = get_query_type_label(
        query_type=query_type,
    )

    return (
        f"RootAgent 将该问题路由到 {route_label}，"
        f"问题类型为 {query_type_label}，"
        f"置信度为 {confidence_text}。"
    )


def format_confidence(
        confidence: Any,
) -> str:
    """
    格式化置信度。

    功能：
        将 confidence 转换成报告中更易读的百分比文本。

    参数：
        confidence:
            原始置信度。
            通常是 0 到 1 之间的 float。

    返回值：
        str:
            格式化后的置信度文本。
    """

    try:
        confidence_number = float(
            confidence,
        )
    except (
            TypeError,
            ValueError,
    ):
        return str(
            confidence,
        )

    percentage = confidence_number * 100

    return f"{percentage:.1f}%"


def get_route_label(
        route: str,
) -> str:
    """
    获取 route 的中文展示名称。

    功能：
        将内部 route key 转换成更适合 Debug Report 阅读的中文名称。

    参数：
        route:
            内部路由 key，例如 dog_knowledge_agent。

    返回值：
        str:
            中文展示名称。
    """

    labels = {
        "dog_knowledge_agent": "DogKnowledgeAgent（狗狗知识智能体）",
        "general_agent": "GeneralAgent（通用问答智能体）",
        "tool_agent": "ToolAgent（工具智能体）",
        "FINISH": "FINISH（结束节点）",
    }

    return labels.get(
        route,
        f"{route}（未知路由）",
    )


def get_query_type_label(
        query_type: str,
) -> str:
    """
    获取 query_type 的中文展示名称。

    功能：
        将 RootAgent 内部问题类型转换成中文说明。

    参数：
        query_type:
            问题类型 key，例如 dog_recommendation。

    返回值：
        str:
            中文问题类型说明。
    """

    labels = {
        "dog_knowledge": "dog_knowledge（狗狗知识问答）",
        "dog_recommendation": "dog_recommendation（狗狗推荐）",
        "tool_request": "tool_request（工具请求）",
        "general_chat": "general_chat（普通聊天）",
        "finish": "finish（结束请求）",
    }

    return labels.get(
        query_type,
        f"{query_type}（未知问题类型）",
    )


def render_root_debug_report_markdown(
        root_debug_report: dict[str, Any],
) -> str:
    """
    渲染 RootAgent Debug Report Markdown 片段。

    功能：
        将 root_debug_report 字段渲染成 Markdown 文本片段。
        后续如果要接入完整 Debug Report 文件，可以直接复用该函数。

    参数：
        root_debug_report:
            RootAgent Debug Report 字段。
            通常来自 state["root_debug_report"]。

    返回值：
        str:
            Markdown 格式的 RootAgent Debug Report 片段。
    """

    if not root_debug_report:
        root_debug_report = build_empty_root_debug_report_fields()

    requires = root_debug_report.get(
        "requires",
        {},
    )

    agent_flow = root_debug_report.get(
        "agent_flow",
        {},
    )

    timeline = root_debug_report.get(
        "timeline",
        {},
    )

    lines = [
        f"## {root_debug_report.get('section_title', 'RootAgent 路由决策')}",
        "",
        f"- 状态: {root_debug_report.get('status', '')}",
        f"- 摘要: {root_debug_report.get('summary', '')}",
        f"- 用户问题: {root_debug_report.get('question', '')}",
        f"- 路由目标: {root_debug_report.get('route', '')}",
        f"- 问题类型: {root_debug_report.get('query_type', '')}",
        f"- 路由置信度: {format_confidence(root_debug_report.get('confidence', 0.0))}",
        f"- 路由原因: {root_debug_report.get('reason', '')}",
        f"- 是否需要 RAG: {requires.get('rag', False)}",
        f"- 是否需要工具: {requires.get('tool', False)}",
        f"- 是否需要记忆: {requires.get('memory', False)}",
        f"- 当前 Agent: {agent_flow.get('current_agent', '')}",
        f"- 下一个 Agent: {agent_flow.get('next_agent', '')}",
        f"- Timeline 是否记录成功: {timeline.get('recorded', False)}",
        f"- 决策来源: {root_debug_report.get('source', '')}",
        f"- 创建时间: {root_debug_report.get('created_at', '')}",
    ]

    return "\n".join(
        lines,
    )