"""
ToolAgent tool_confirm_node 测试。

功能：
    测试 ToolAgent 工具确认节点是否能根据 require_confirm 生成确认计划。

测试重点：
    1. 没有 tool_calls 时不需要确认。
    2. date 这类低风险工具可以跳过确认。
    3. weather 设置 require_confirm=True 时生成批量确认提示。
    4. 多个需要确认的工具只生成一次 batch confirmation。
    5. 用户确认或拒绝后返回结构化权限状态。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.state_adapter import TOOL_AGENT_RESPONSE_STATE_KEY
from src.agents.tool_agent.nodes.tool_confirm_node import (
    build_batch_confirmation_prompt,
    build_tool_agent_tool_confirm_node,
    format_tool_args,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_metadata import ToolMetadata


class FakeTool:
    """
    测试用工具对象。

    功能：
        模拟真实工具，只提供 metadata 字段。

    参数：
        metadata:
            工具元数据。

    返回值：
        FakeTool:
            测试用工具对象。
    """

    def __init__(
        self,
        metadata: ToolMetadata,
    ) -> None:
        self.metadata = metadata


class FakeRegistry:
    """
    测试用工具注册表。

    功能：
        模拟 ToolRegistry，提供 get_tool 方法。

    参数：
        tools:
            工具名称到工具对象的映射。

    返回值：
        FakeRegistry:
            测试用工具注册表。
    """

    def __init__(
        self,
        tools: dict[str, FakeTool],
    ) -> None:
        self.tools = tools

    def get_tool(
        self,
        name: str,
    ) -> FakeTool | None:
        """
        根据名称获取工具。

        功能：
            模拟 ToolRegistry.get_tool。

        参数：
            name:
                工具名称。

        返回值：
            FakeTool | None:
                找到返回工具，找不到返回 None。
        """

        return self.tools.get(
            name
        )


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
                当前节点名称。

        返回值：
            None。
        """

        self.current_node = node_name


class FakeTimelineScope:
    """
    测试用 TimelineScope。

    功能：
        记录节点事件。

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
        添加时间线事件。

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


class FakeCheckpointManager:
    """
    测试用 CheckpointManager。

    功能：
        记录保存 checkpoint 的次数。

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
            记录保存次数。

        参数：
            无。

        返回值：
            None。
        """

        self.save_count += 1


def build_fake_registry() -> FakeRegistry:
    """
    构建测试用工具注册表。

    功能：
        date 不需要确认，weather 需要确认。

    参数：
        无。

    返回值：
        FakeRegistry:
            测试用工具注册表。
    """

    return FakeRegistry(
        tools={
            "date": FakeTool(
                ToolMetadata(
                    name="date",
                    description="获取当前日期",
                    require_confirm=False,
                )
            ),
            "weather": FakeTool(
                ToolMetadata(
                    name="weather",
                    description="查询天气",
                    require_confirm=True,
                )
            ),
        }
    )


def build_test_node(
    interrupt_func=None,
):
    """
    构建测试用确认节点。

    功能：
        注入 fake registry、fake checkpoint_manager、fake runtime_context
        和可选 fake interrupt。

    参数：
        interrupt_func:
            测试用中断函数。为 None 时保持 pending 行为。

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

    node = build_tool_agent_tool_confirm_node(
        tool_registry=build_fake_registry(),
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
        interrupt_func=interrupt_func,
    )

    return (
        node,
        fake_ctx,
        checkpoint_manager,
    )


def test_tool_confirm_node_should_not_require_confirmation_without_tool_calls() -> None:
    """
    测试没有 tool_calls 时不需要确认。

    功能：
        确认节点返回 not_required 状态。

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
            "question": "你好",
        }
    )

    assert update["tool_confirmed"] == "not_required"
    assert update["tool_confirmation_required"] is False
    assert update["tool_agent_permission"]["status"] == "not_required"
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_confirm_node"
    assert checkpoint_manager.save_count == 1


def test_tool_confirm_node_should_skip_confirmation_when_tools_do_not_require_it() -> None:
    """
    测试低风险工具跳过确认。

    功能：
        date 的 require_confirm=False，因此不需要用户确认。

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
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }
    )

    assert update["tool_confirmed"] == "not_required"
    assert update["tool_confirmation_required"] is False
    assert update["tool_agent_permission"]["status"] == "not_required"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "not_required"
    assert checkpoint_manager.save_count == 1


def test_tool_confirm_node_should_create_batch_confirmation_for_weather() -> None:
    """
    测试 weather 需要确认。

    功能：
        weather 的 require_confirm=True，因此节点生成 pending 批量确认提示。

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
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )

    assert update["tool_confirmed"] == "pending"
    assert update["tool_confirmation_required"] is True
    assert update["tool_confirmation_mode"] == "batch"
    assert update["tool_agent_permission"]["status"] == "pending"
    assert "查询天气" in update["tool_confirmation_prompt"]
    assert "city=成都" in update["tool_confirmation_prompt"]
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "pending"
    assert checkpoint_manager.save_count == 1


def test_tool_confirm_node_should_confirm_when_interrupt_returns_yes() -> None:
    """
    测试 interrupt 返回 y 时确认工具调用。

    功能：
        模拟 LangGraph interrupt 恢复后的用户输入。
        当用户输入 y 时，节点应返回 confirmed，而不是 pending。

    参数：
        无。

    返回值：
        None。
    """

    captured_prompts: list[str] = []

    def fake_interrupt(
        prompt: str,
    ) -> str:
        """
        测试用 fake interrupt。

        功能：
            记录确认提示，并模拟用户输入 y。

        参数：
            prompt:
                确认节点生成的提示文本。

        返回值：
            str:
                模拟用户确认输入。
        """

        captured_prompts.append(
            prompt
        )
        return "y"

    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        interrupt_func=fake_interrupt,
    )

    update = node(
        {
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )

    assert captured_prompts
    assert "查询天气" in captured_prompts[0]
    assert update["tool_confirmed"] == "confirmed"
    assert update["tool_confirmation_required"] is False
    assert update["tool_agent_permission"]["status"] == "confirmed"
    assert checkpoint_manager.save_count == 2


def test_tool_confirm_node_should_reject_when_interrupt_returns_no() -> None:
    """
    测试 interrupt 返回 n 时拒绝工具调用。

    功能：
        模拟 LangGraph interrupt 恢复后的用户拒绝输入。
        当用户输入 n 时，节点应清空 tool_calls 并写入取消结果。

    参数：
        无。

    返回值：
        None。
    """

    def fake_interrupt(
        prompt: str,
    ) -> str:
        """
        测试用 fake interrupt。

        功能：
            忽略提示文本，并模拟用户输入 n。

        参数：
            prompt:
                确认节点生成的提示文本。

        返回值：
            str:
                模拟用户拒绝输入。
        """

        return "n"

    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        interrupt_func=fake_interrupt,
    )

    update = node(
        {
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update["tool_agent_permission"]["status"] == "rejected"
    assert update["tool_results"] == [
        "用户取消了工具调用。"
    ]
    assert checkpoint_manager.save_count == 2


def test_tool_confirm_node_should_confirm_when_user_says_yes() -> None:
    """
    测试用户确认工具调用。

    功能：
        tool_confirmed=yes 时，节点返回 confirmed 权限状态。

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
            "tool_confirmed": "yes",
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )

    assert update["tool_confirmed"] == "confirmed"
    assert update["tool_confirmation_required"] is False
    assert update["tool_agent_permission"]["status"] == "confirmed"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "confirmed"


def test_tool_confirm_node_should_reject_when_user_says_no() -> None:
    """
    测试用户拒绝工具调用。

    功能：
        tool_confirmed=no 时，节点清空 tool_calls 并写入取消结果。

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
            "tool_confirmed": "no",
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update["tool_results"] == [
        "用户取消了工具调用。"
    ]
    assert update["tool_agent_permission"]["status"] == "rejected"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "rejected"


def test_build_batch_confirmation_prompt_should_format_multiple_tools() -> None:
    """
    测试批量确认提示格式。

    功能：
        确认多个工具可以合并成一次用户可读提示。

    参数：
        无。

    返回值：
        None。
    """

    prompt = build_batch_confirmation_prompt(
        tool_calls=[
            ToolCall(
                name="date",
                args={},
            ),
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            ),
        ],
        tool_registry=build_fake_registry(),
    )

    assert "1. 获取当前日期" in prompt
    assert "2. 查询天气" in prompt
    assert "参数：无" in prompt
    assert "city=成都" in prompt


def test_format_tool_args_should_return_readable_text() -> None:
    """
    测试工具参数格式化。

    功能：
        确认空参数显示为“无”，普通参数显示为 key=value。

    参数：
        无。

    返回值：
        None。
    """

    assert format_tool_args({}) == "无"
    assert format_tool_args(
        {
            "city": "成都",
        }
    ) == "city=成都"
