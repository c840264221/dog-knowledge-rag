"""
RootAgent Debug Report 单元测试。

功能：
    测试 RootAgent Debug Report 字段构建与 Markdown 渲染能力。

测试目标：
    1. root_observability 可以转换成 root_debug_report。
    2. 缺失 root_observability 时可以返回稳定的空报告结构。
    3. root_supervisor_node 会返回 root_debug_report。
    4. Markdown 渲染函数可以生成可读报告片段。
"""

from __future__ import annotations

import pytest

from src.agents.root_agent.debug_report import (
    build_empty_root_debug_report_fields,
    build_root_debug_report_fields,
    format_confidence,
    render_root_debug_report_markdown,
)
from src.agents.root_agent.supervisor import (
    root_supervisor_node,
)


def test_build_root_debug_report_fields() -> None:
    """
    测试构建 RootAgent Debug Report 字段。

    功能：
        验证 root_observability 可以被转换成报告字段。

    参数：
        无。

    返回值：
        None。
    """

    root_observability = {
        "component": "root_agent",
        "event_type": "route",
        "event_name": "root_route_decision",
        "question": "推荐几种适合公寓养的狗",
        "route": "dog_knowledge_agent",
        "query_type": "dog_recommendation",
        "confidence": 0.9,
        "reason": "命中狗狗推荐类关键词。",
        "requires_rag": True,
        "requires_tool": False,
        "requires_memory": True,
        "source": "root_supervisor_rule_v1",
        "hints": {
            "matched_keywords": [
                "推荐",
            ]
        },
        "current_agent": "root_agent",
        "next_agent": "dog_knowledge_agent",
        "created_at": "2026-07-01T00:00:00+00:00",
        "timeline_recorded": True,
    }

    report = build_root_debug_report_fields(
        root_observability=root_observability,
    )

    assert report[
        "section"
    ] == "root_agent"

    assert report[
        "section_title"
    ] == "RootAgent 路由决策"

    assert report[
        "status"
    ] == "available"

    assert report[
        "route"
    ] == "dog_knowledge_agent"

    assert report[
        "query_type"
    ] == "dog_recommendation"

    assert report[
        "requires"
    ][
        "rag"
    ] is True

    assert report[
        "requires"
    ][
        "tool"
    ] is False

    assert report[
        "timeline"
    ][
        "recorded"
    ] is True

    assert "DogKnowledgeAgent" in report[
        "summary"
    ]


def test_build_empty_root_debug_report_fields() -> None:
    """
    测试空 RootAgent Debug Report 字段。

    功能：
        当 root_observability 不存在时，
        报告构建函数应该返回稳定的空结构。

    参数：
        无。

    返回值：
        None。
    """

    report = build_empty_root_debug_report_fields()

    assert report[
        "section"
    ] == "root_agent"

    assert report[
        "status"
    ] == "not_available"

    assert report[
        "route"
    ] == ""

    assert report[
        "requires"
    ][
        "rag"
    ] is False

    assert report[
        "timeline"
    ][
        "recorded"
    ] is False


def test_build_root_debug_report_fields_with_none() -> None:
    """
    测试传入 None 时返回空报告结构。

    功能：
        验证 build_root_debug_report_fields 对缺失数据的容错能力。

    参数：
        无。

    返回值：
        None。
    """

    report = build_root_debug_report_fields(
        root_observability=None,
    )

    assert report[
        "status"
    ] == "not_available"

    assert report[
        "summary"
    ] == "RootAgent 路由可观测数据不可用。"


@pytest.mark.parametrize(
    "confidence, expected_text",
    [
        (
            0.9,
            "90.0%",
        ),
        (
            1,
            "100.0%",
        ),
        (
            0,
            "0.0%",
        ),
        (
            "unknown",
            "unknown",
        ),
    ],
)
def test_format_confidence(
        confidence: object,
        expected_text: str,
) -> None:
    """
    测试置信度格式化。

    功能：
        验证不同类型的 confidence 可以被稳定格式化。

    参数：
        confidence:
            原始置信度。

        expected_text:
            预期格式化结果。

    返回值：
        None。
    """

    assert format_confidence(
        confidence,
    ) == expected_text


def test_render_root_debug_report_markdown() -> None:
    """
    测试 RootAgent Debug Report Markdown 渲染。

    功能：
        验证 root_debug_report 可以被渲染成 Markdown 文本片段。

    参数：
        无。

    返回值：
        None。
    """

    report = build_root_debug_report_fields(
        root_observability={
            "question": "金毛寿命多久？",
            "route": "dog_knowledge_agent",
            "query_type": "dog_knowledge",
            "confidence": 0.88,
            "reason": "命中狗狗知识类关键词。",
            "requires_rag": True,
            "requires_tool": False,
            "requires_memory": True,
            "current_agent": "root_agent",
            "next_agent": "dog_knowledge_agent",
            "timeline_recorded": False,
            "source": "root_supervisor_rule_v1",
            "created_at": "2026-07-01T00:00:00+00:00",
        },
    )

    markdown = render_root_debug_report_markdown(
        root_debug_report=report,
    )

    assert "## RootAgent 路由决策" in markdown
    assert "金毛寿命多久" in markdown
    assert "dog_knowledge_agent" in markdown
    assert "88.0%" in markdown


@pytest.mark.asyncio
async def test_root_supervisor_returns_root_debug_report() -> None:
    """
    测试 root_supervisor_node 返回 root_debug_report。

    功能：
        验证 RootAgent 路由节点会返回 Debug Report 字段。

    参数：
        无。

    返回值：
        None。
    """

    result = await root_supervisor_node(
        {
            "question": "推荐几种适合公寓养的狗",
            "user_id": "test_user",
            "session_id": "test_session",
            "trace_id": "test_trace",
        }
    )

    assert "root_debug_report" in result

    report = result[
        "root_debug_report"
    ]

    assert report[
        "section"
    ] == "root_agent"

    assert report[
        "status"
    ] == "available"

    assert report[
        "route"
    ] == "dog_knowledge_agent"

    assert report[
        "query_type"
    ] == "dog_recommendation"

    assert report[
        "requires"
    ][
        "rag"
    ] is True