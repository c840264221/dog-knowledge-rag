"""
RootAgent Observability 单元测试。

功能：
    测试 RootAgent 路由可观测数据构建和 timeline 写入能力。

测试目标：
    1. 可正确构建 root_observability 字典。
    2. 没有 runtime context 时不会报错。
    3. 有 fake timeline 时可以成功写入事件。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.root_agent import observability as root_observability
from src.agents.root_agent.observability import (
    build_root_route_observability_payload,
    record_root_route_timeline,
)
from src.agents.root_agent.schemas import RootRouteDecision
from src.agents.root_agent.supervisor import root_supervisor_node


class FakeTimeline:
    """
    FakeTimeline 测试用时间线对象。

    功能：
        模拟真实 Runtime Timeline 的 add_event 方法，
        用于验证 RootAgent 是否正确写入 timeline。

    参数：
        无。

    返回值：
        无。
    """

    def __init__(
            self,
    ) -> None:
        """
        初始化 FakeTimeline。

        功能：
            创建 events 列表，用于保存测试中写入的事件。

        参数：
            无。

        返回值：
            None。
        """

        self.events: list[dict[str, Any]] = []

    def add_event(
            self,
            event_type: str,
            name: str,
            metadata: dict[str, Any],
    ) -> None:
        """
        添加测试事件。

        功能：
            模拟 timeline.add_event 方法，把事件写入 self.events。

        参数：
            event_type:
                事件类型，例如 route。

            name:
                事件名称，例如 root_route_decision。

            metadata:
                事件元数据。

        返回值：
            None。
        """

        self.events.append(
            {
                "event_type": event_type,
                "name": name,
                "metadata": metadata,
            }
        )


class FakeRuntimeContext:
    """
    FakeRuntimeContext 测试用运行时上下文。

    功能：
        模拟真实 Runtime Context，只提供 timeline 方法。
    """

    def __init__(
            self,
            timeline: FakeTimeline,
    ) -> None:
        """
        初始化 FakeRuntimeContext。

        参数：
            timeline:
                测试用 FakeTimeline 对象。

        返回值：
            None。
        """

        self._timeline = timeline

    def timeline(
            self,
    ) -> FakeTimeline:
        """
        返回测试用 timeline。

        功能：
            模拟 runtime_context.timeline() 调用。

        参数：
            无。

        返回值：
            FakeTimeline:
                测试用时间线对象。
        """

        return self._timeline


class FakeRuntimeCtxManager:
    """
    FakeRuntimeCtxManager 测试用 runtime_ctx 管理器。

    功能：
        模拟 src.runtime.context.runtime_ctx，
        提供 get 方法返回 fake runtime context。
    """

    def __init__(
            self,
            runtime_context: FakeRuntimeContext | None,
    ) -> None:
        """
        初始化 FakeRuntimeCtxManager。

        参数：
            runtime_context:
                测试用运行时上下文。
                可以为 None，用于测试没有 runtime context 的场景。

        返回值：
            None。
        """

        self._runtime_context = runtime_context

    def get(
            self,
    ) -> FakeRuntimeContext | None:
        """
        获取测试用 runtime context。

        功能：
            模拟 runtime_ctx.get()。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None:
                测试用运行时上下文。
        """

        return self._runtime_context


def test_build_root_route_observability_payload() -> None:
    """
    测试构建 RootAgent 可观测数据。

    功能：
        验证 build_root_route_observability_payload
        可以根据 RootRouteDecision 构建标准 dict。

    参数：
        无。

    返回值：
        None。
    """

    decision = RootRouteDecision(
        route="dog_knowledge_agent",
        query_type="dog_recommendation",
        confidence=0.9,
        reason="测试路由原因",
        requires_rag=True,
        requires_tool=False,
        requires_memory=True,
        source="test",
        hints={
            "matched_keywords": [
                "推荐",
            ]
        },
    )

    payload = build_root_route_observability_payload(
        question="推荐几种适合公寓养的狗",
        decision=decision,
    )

    assert payload[
        "component"
    ] == "root_agent"

    assert payload[
        "event_type"
    ] == "route"

    assert payload[
        "event_name"
    ] == "root_route_decision"

    assert payload[
        "route"
    ] == "dog_knowledge_agent"

    assert payload[
        "query_type"
    ] == "dog_recommendation"

    assert payload[
        "requires_rag"
    ] is True

    assert payload[
        "requires_tool"
    ] is False

    assert payload[
        "next_agent"
    ] == "dog_knowledge_agent"

    assert "created_at" in payload


def test_record_root_route_timeline_without_runtime_context(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试没有 runtime context 时不报错。

    功能：
        当 runtime_ctx.get() 返回 None 时，
        record_root_route_timeline 应该返回 False，
        但不应该抛异常。

    参数：
        monkeypatch:
            pytest 提供的 monkeypatch 工具，
            用于替换 root_observability.runtime_ctx。

    返回值：
        None。
    """

    monkeypatch.setattr(
        root_observability,
        "runtime_ctx",
        FakeRuntimeCtxManager(
            runtime_context=None,
        ),
    )

    result = record_root_route_timeline(
        payload={
            "event_type": "route",
            "event_name": "root_route_decision",
        },
    )

    assert result is False


def test_record_root_route_timeline_with_fake_timeline(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试成功写入 fake timeline。

    功能：
        使用 FakeTimeline 模拟真实 timeline，
        验证 record_root_route_timeline 可以成功写入事件。

    参数：
        monkeypatch:
            pytest 提供的 monkeypatch 工具。

    返回值：
        None。
    """

    fake_timeline = FakeTimeline()

    fake_runtime_context = FakeRuntimeContext(
        timeline=fake_timeline,
    )

    monkeypatch.setattr(
        root_observability,
        "runtime_ctx",
        FakeRuntimeCtxManager(
            runtime_context=fake_runtime_context,
        ),
    )

    payload = {
        "event_type": "route",
        "event_name": "root_route_decision",
        "route": "dog_knowledge_agent",
    }

    result = record_root_route_timeline(
        payload=payload,
    )

    assert result is True

    assert len(
        fake_timeline.events,
    ) == 1

    event = fake_timeline.events[
        0
    ]

    assert event[
        "event_type"
    ] == "route"

    assert event[
        "name"
    ] == "root_route_decision"

    assert event[
        "metadata"
    ][
        "route"
    ] == "dog_knowledge_agent"


@pytest.mark.asyncio
async def test_root_supervisor_returns_root_observability() -> None:
    """
    测试 root_supervisor_node 返回 root_observability。

    功能：
        验证 RootAgent 路由节点除了返回 route_decision，
        也会返回 root_observability。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "推荐几种适合公寓养的狗",
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = await root_supervisor_node(
        state,
    )

    assert "route_decision" in result

    assert "root_observability" in result

    root_metadata = result[
        "root_observability"
    ]

    assert root_metadata[
        "component"
    ] == "root_agent"

    assert root_metadata[
        "route"
    ] == "dog_knowledge_agent"

    assert root_metadata[
        "query_type"
    ] == "dog_recommendation"

    assert "timeline_recorded" in root_metadata