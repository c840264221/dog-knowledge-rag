"""
ToolAgent tool_answer_node 测试。

功能：
    测试工具答案节点能否把 tool_results 转换成 final_answer。

测试重点：
    1. weather 工具结果可以格式化成天气回答。
    2. date 工具结果可以格式化成日期回答。
    3. 失败工具结果可以格式化成失败提示。
    4. 没有 tool_results 时不会生成虚假回答。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.state_adapter import TOOL_AGENT_RESPONSE_STATE_KEY
from src.agents.tool_agent.nodes.tool_answer_node import (
    build_tool_agent_tool_answer_node,
    format_date_tool_result,
    format_failed_tool_result,
    format_generic_tool_result,
    format_weather_tool_result,
)
from src.graph.tools.schemas.tool_result_schema import ToolResult


class FakeCheckpointManager:
    """
    测试用 CheckpointManager。

    功能：
        记录 save_checkpoint 调用次数。

    参数：
        无。

    返回值：
        FakeCheckpointManager:
            测试用检查点管理器。
    """

    def __init__(self) -> None:
        self.save_count = 0

    def save_checkpoint(self) -> None:
        """
        模拟保存 checkpoint。

        功能：
            每次调用时 save_count 加一。

        参数：
            无。

        返回值：
            None。
        """

        self.save_count += 1


class FakeStateScope:
    """
    测试用 StateScope。

    功能：
        记录当前节点名称。

    参数：
        无。

    返回值：
        FakeStateScope:
            测试用状态作用域。
    """

    def __init__(self) -> None:
        self.current_node: str | None = None

    def set_node(
        self,
        node_name: str,
    ) -> None:
        """
        设置当前节点名称。

        功能：
            模拟 RuntimeContext.state().set_node。

        参数：
            node_name:
                节点名称。

        返回值：
            None。
        """

        self.current_node = node_name


class FakeTimelineScope:
    """
    测试用 TimelineScope。

    功能：
        记录 timeline 事件。

    参数：
        无。

    返回值：
        FakeTimelineScope:
            测试用时间线作用域。
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def add_event(
        self,
        event_type: str,
        name: str,
        metadata: dict | None = None,
    ) -> None:
        """
        添加 timeline 事件。

        功能：
            模拟 RuntimeContext.timeline().add_event。

        参数：
            event_type:
                事件类型。

            name:
                事件名称。

            metadata:
                附加元数据。

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
    测试用 RuntimeContext。

    功能：
        提供 state 和 timeline 作用域。

    参数：
        无。

    返回值：
        FakeRuntimeContext:
            测试用运行时上下文。
    """

    def __init__(self) -> None:
        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self) -> FakeStateScope:
        """
        获取状态作用域。

        功能：
            返回 FakeStateScope。

        参数：
            无。

        返回值：
            FakeStateScope:
                测试用状态作用域。
        """

        return self.state_scope

    def timeline(self) -> FakeTimelineScope:
        """
        获取时间线作用域。

        功能：
            返回 FakeTimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope:
                测试用时间线作用域。
        """

        return self.timeline_scope


def build_test_node():
    """
    构建测试用答案节点。

    功能：
        注入 fake checkpoint_manager 和 fake runtime_context。

    参数：
        无。

    返回值：
        tuple:
            node, fake_ctx, checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()
    checkpoint_manager = FakeCheckpointManager()

    def runtime_context_getter():
        """
        获取测试 RuntimeContext。

        功能：
            返回 fake_ctx。

        参数：
            无。

        返回值：
            FakeRuntimeContext:
                测试用运行时上下文。
        """

        return fake_ctx

    node = build_tool_agent_tool_answer_node(
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        checkpoint_manager,
    )


def test_tool_answer_node_should_format_weather_result() -> None:
    """
    测试格式化天气工具结果。

    功能：
        weather 工具返回 dict content 时，应生成天气回答。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node()

    update = node(
        {
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "weather",
                    "content": {
                        "city": "成都",
                        "weather": "晴",
                        "temperature": "30°C",
                        "wind_speed": "9 km/h",
                    },
                }
            ],
        }
    )

    assert update["final_answer"] == "成都天气，晴，温度约 30°C，风速约 9 km/h。"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["final_answer"] == update["final_answer"]
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_answer_node"
    assert checkpoint_manager.save_count == 1


def test_tool_answer_node_should_format_date_result() -> None:
    """
    测试格式化日期工具结果。

    功能：
        date 工具返回字符串 content 时，应生成日期回答。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        _fake_ctx,
        _checkpoint_manager,
    ) = build_test_node()

    update = node(
        {
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "date",
                    "content": "2026-07-07",
                }
            ],
        }
    )

    assert update["final_answer"] == "今天的日期是 2026-07-07。"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"


def test_tool_answer_node_should_format_failed_result() -> None:
    """
    测试格式化失败工具结果。

    功能：
        success=False 时，应生成失败提示。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        _fake_ctx,
        _checkpoint_manager,
    ) = build_test_node()

    update = node(
        {
            "tool_results": [
                {
                    "success": False,
                    "tool_name": "weather",
                    "error": "网络错误",
                }
            ],
        }
    )

    assert update["final_answer"] == "工具调用失败：网络错误"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "failed"


def test_tool_answer_node_should_keep_empty_answer_without_results() -> None:
    """
    测试没有工具结果时不生成虚假回答。

    功能：
        tool_results 为空时，保留原 final_answer。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node()

    update = node(
        {
            "final_answer": "已有回答",
            "tool_results": [],
        }
    )

    assert update["final_answer"] == "已有回答"
    assert update["tool_agent_answer_source"] == "empty_tool_results"
    assert checkpoint_manager.save_count == 0


def test_format_helpers_should_return_readable_text() -> None:
    """
    测试格式化辅助函数。

    功能：
        确认 weather/date/generic/failed 几类格式化结果稳定。

    参数：
        无。

    返回值：
        None。
    """

    assert format_weather_tool_result(
        ToolResult(
            success=True,
            tool_name="weather",
            content="成都，温度 30°C",
        )
    ) == "天气查询结果：成都，温度 30°C"
    assert format_date_tool_result(
        ToolResult(
            success=True,
            tool_name="date",
            content="2026-07-07",
        )
    ) == "今天的日期是 2026-07-07。"
    assert format_generic_tool_result(
        ToolResult(
            success=True,
            tool_name="demo",
            content="ok",
        )
    ) == "demo 工具返回：ok"
    assert format_failed_tool_result(
        ToolResult(
            success=False,
            tool_name="demo",
            error="失败",
        )
    ) == "工具调用失败：失败"
