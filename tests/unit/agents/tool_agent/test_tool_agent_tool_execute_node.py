"""
ToolAgent tool_execute_node 测试。

功能：
    测试 ToolAgent 工具执行节点是否会根据权限状态决定是否执行工具。

测试重点：
    1. pending 状态不执行工具。
    2. rejected 状态不执行工具。
    3. confirmed 状态执行工具。
    4. not_required 状态执行工具。
    5. 没有 tool_calls 时安全跳过。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.runtime_adapter import (
    TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY,
)
from src.agents.tool_agent.adapters.state_adapter import TOOL_AGENT_RESPONSE_STATE_KEY
from src.agents.tool_agent.nodes.tool_execute_node import (
    build_tool_agent_tool_execute_node,
    get_permission_status,
)
from src.graph.tools.schemas.tool_result_schema import ToolResult


class FakeExecutor:
    """
    测试用工具执行器。

    功能：
        模拟 ToolExecutor.execute，记录调用并返回 ToolResult。

    参数：
        无。

    返回值：
        FakeExecutor:
            测试用工具执行器。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(
        self,
        tool_name: str,
        args: dict,
    ) -> ToolResult:
        """
        模拟执行工具。

        功能：
            记录工具名和参数，并返回成功 ToolResult。

        参数：
            tool_name:
                工具名称。

            args:
                工具参数。

        返回值：
            ToolResult:
                模拟工具执行结果。
        """

        self.calls.append(
            {
                "tool_name": tool_name,
                "args": args,
            }
        )
        return ToolResult(
            success=True,
            tool_name=tool_name,
            content={
                "args": args,
            },
            metadata={
                "source": "fake_executor",
            },
        )


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
            每调用一次，save_count 加一。

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
    构建测试用执行节点。

    功能：
        注入 fake executor、fake checkpoint_manager 和 fake runtime_context。

    参数：
        无。

    返回值：
        tuple:
            node, fake_executor, fake_ctx, checkpoint_manager。
    """

    fake_executor = FakeExecutor()
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

    node = build_tool_agent_tool_execute_node(
        executor=fake_executor,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_executor,
        fake_ctx,
        checkpoint_manager,
    )


@pytest.mark.asyncio
async def test_tool_execute_node_should_skip_when_permission_pending() -> None:
    """
    测试 pending 权限不执行工具。

    功能：
        工具等待用户确认时，执行节点只刷新响应契约，不调用 executor。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_executor,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node()

    update = await node(
        {
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
            "tool_agent_permission": {
                "status": "pending",
            },
        }
    )

    assert fake_executor.calls == []
    assert update["tool_agent_execute_skipped"] is True
    assert "pending" in update["tool_agent_execute_skip_reason"]
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "pending"
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_execute_node_should_skip_when_permission_rejected() -> None:
    """
    测试 rejected 权限不执行工具。

    功能：
        用户拒绝工具调用后，执行节点不应调用 executor。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_executor,
        _fake_ctx,
        _checkpoint_manager,
    ) = build_test_node()

    update = await node(
        {
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
            "tool_agent_permission": {
                "status": "rejected",
            },
        }
    )

    assert fake_executor.calls == []
    assert update["tool_agent_execute_skipped"] is True
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_tool_execute_node_should_execute_when_permission_confirmed() -> None:
    """
    测试 confirmed 权限执行工具。

    功能：
        用户确认后，执行节点调用 executor 并写入 tool_results。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_executor,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node()

    update = await node(
        {
            "tool_round": 2,
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
            "tool_agent_permission": {
                "status": "confirmed",
            },
        }
    )

    assert fake_executor.calls == [
        {
            "tool_name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert update["tool_calls"] == []
    assert update["need_tool"] is False
    assert update["tool_round"] == 3
    assert update["tool_results"][0]["success"] is True
    assert TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY in update
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_execute_node"
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_execute_node_should_execute_when_permission_not_required() -> None:
    """
    测试 not_required 权限直接执行工具。

    功能：
        低风险工具不需要确认时，执行节点可以直接执行。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_executor,
        _fake_ctx,
        _checkpoint_manager,
    ) = build_test_node()

    update = await node(
        {
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
            "tool_agent_permission": {
                "status": "not_required",
            },
        }
    )

    assert fake_executor.calls[0]["tool_name"] == "date"
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_execute_node_should_skip_without_tool_calls() -> None:
    """
    测试没有 tool_calls 时安全跳过。

    功能：
        没有工具调用时，不调用 executor，只返回跳过原因。

    参数：
        无。

    返回值：
        None。
    """

    (
        node,
        fake_executor,
        _fake_ctx,
        _checkpoint_manager,
    ) = build_test_node()

    update = await node({})

    assert fake_executor.calls == []
    assert update["tool_agent_execute_skipped"] is True
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"


def test_get_permission_status_should_fallback_to_tool_confirmed() -> None:
    """
    测试权限状态读取兼容 tool_confirmed。

    功能：
        当 tool_agent_permission 不存在时，回退读取旧字段 tool_confirmed。

    参数：
        无。

    返回值：
        None。
    """

    assert get_permission_status(
        {
            "tool_confirmed": "confirmed",
        }
    ) == "confirmed"
