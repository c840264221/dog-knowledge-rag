"""
ToolAgent 统一入口测试。

功能：
    测试 build_tool_agent 是否能把 parse -> confirm -> response_adapter
    三个节点按顺序串起来。

测试重点：
    1. 天气工具会生成待确认状态。
    2. 普通问题会生成 no_tool 响应。
    3. 已有 tool_calls 时不会重复调用 parser。
    4. state 合并函数不会原地修改输入 state。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.agent import (
    build_tool_agent,
    merge_state_update,
)
from src.agents.tool_agent.adapters.state_adapter import TOOL_AGENT_RESPONSE_STATE_KEY
from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.schemas.tool_result_schema import ToolResult


class FakeAinvokeParser:
    """
    测试用异步解析器。

    功能：
        模拟带 ainvoke 方法的 parser，并记录调用输入。

    参数：
        result:
            parser 需要返回的解析结果。

    返回值：
        FakeAinvokeParser:
            测试用 parser。
    """

    def __init__(
        self,
        result: dict,
    ) -> None:
        self.result = result
        self.inputs: list[dict] = []

    async def ainvoke(
        self,
        parser_input: dict,
    ) -> dict:
        """
        模拟异步解析调用。

        功能：
            记录 parser_input，并返回预设结果。

        参数：
            parser_input:
                工具解析输入，包含 question 和 state。

        返回值：
            dict:
                预设工具解析结果。
        """

        self.inputs.append(
            parser_input
        )
        return self.result


class FakeTool:
    """
    测试用工具对象。

    功能：
        模拟真实工具对象，只暴露 metadata。

    参数：
        metadata:
            工具元数据。

    返回值：
        FakeTool:
            测试用工具。
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
        模拟 ToolRegistry.get_tool，用于确认节点查询工具元数据。

    参数：
        tools:
            工具名到工具对象的映射。

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
        根据工具名获取工具。

        功能：
            模拟真实 ToolRegistry 的 get_tool 方法。

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


class FakeExecutor:
    """
    测试用工具执行器。

    功能：
        模拟 ToolExecutor.execute，避免单元测试调用真实外部工具。

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
        模拟工具执行。

        功能：
            记录工具调用，并返回成功结果。

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
            content="2026-07-07"
            if tool_name == "date"
            else "执行成功",
            metadata={
                "source": "fake_executor",
            },
        )


def build_fake_registry() -> FakeRegistry:
    """
    构建测试用工具注册表。

    功能：
        weather 需要确认，date 不需要确认。

    参数：
        无。

    返回值：
        FakeRegistry:
            测试用工具注册表。
    """

    return FakeRegistry(
        tools={
            "weather": FakeTool(
                metadata=ToolMetadata(
                    name="weather",
                    description="查询天气",
                    require_confirm=True,
                )
            ),
            "date": FakeTool(
                metadata=ToolMetadata(
                    name="date",
                    description="获取当前日期",
                    require_confirm=False,
                )
            ),
        }
    )


@pytest.mark.asyncio
async def test_tool_agent_should_build_pending_confirmation_for_weather() -> None:
    """
    测试天气工具进入待确认状态。

    功能：
        parser 输出 weather 工具调用后，ToolAgent 统一入口应生成确认提示和 pending 权限。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": True,
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
    executor = FakeExecutor()
    node = build_tool_agent(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
    )

    # 调用 ToolAgent 统一入口，内部会依次执行解析、确认、执行、响应适配。
    result = await node(
        {
            "question": "今天成都天气怎么样？",
        }
    )

    assert result["need_tool"] is True
    assert TOOL_AGENT_TOOL_CATALOG_STATE_KEY in result
    assert result["tool_call_validation_ok"] is True
    assert result["tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert result["tool_confirmed"] == "pending"
    assert result["tool_confirmation_required"] is True
    assert "查询天气" in result["tool_confirmation_prompt"]
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "pending"
    assert result["final_answer"] == ""
    assert executor.calls == []
    assert len(
        parser.inputs
    ) == 1


@pytest.mark.asyncio
async def test_tool_agent_should_return_no_tool_for_general_question() -> None:
    """
    测试普通问题不需要工具。

    功能：
        parser 输出 need_tool=False 时，ToolAgent 统一入口应返回 no_tool 响应状态。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": False,
            "tool_calls": [],
            "response": "普通回答",
        }
    )
    node = build_tool_agent(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=FakeExecutor(),
        runtime_context_getter=lambda: None,
    )

    # 普通问题走完整 ToolAgent 链路后，不应该产生待确认工具。
    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["need_tool"] is False
    assert result["tool_calls"] == []
    assert result["tool_call_validation_ok"] is True
    assert result["tool_confirmed"] == "not_required"
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"


@pytest.mark.asyncio
async def test_tool_agent_should_not_reparse_existing_tool_calls() -> None:
    """
    测试已有 tool_calls 时不重复解析。

    功能：
        当上游 state 已经有工具调用计划时，parse 节点返回空 update，
        ToolAgent 继续进入确认节点处理已有 tool_calls。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": False,
            "tool_calls": [],
        }
    )
    node = build_tool_agent(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=FakeExecutor(),
        runtime_context_getter=lambda: None,
    )

    # 已有 tool_calls 时，解析节点会跳过，确认节点直接处理原有调用。
    result = await node(
        {
            "question": "今天成都天气怎么样？",
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

    assert parser.inputs == []
    assert result["tool_call_validation_ok"] is True
    assert result["tool_confirmed"] == "pending"
    assert result["tool_confirmation_required"] is True
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "pending"


@pytest.mark.asyncio
async def test_tool_agent_should_execute_tool_when_confirmation_not_required() -> None:
    """
    测试不需要确认的工具会被执行。

    功能：
        date 工具 require_confirm=False，ToolAgent 统一入口应在确认节点后进入执行节点。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }
    )
    executor = FakeExecutor()
    node = build_tool_agent(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
    )

    # date 不需要用户确认，所以执行节点会调用 fake executor。
    result = await node(
        {
            "question": "今天几号？",
        }
    )

    assert executor.calls == [
        {
            "tool_name": "date",
            "args": {},
        }
    ]
    assert result["tool_results"][0]["success"] is True
    assert result["tool_calls"] == []
    assert result["tool_call_validation_ok"] is True
    assert result["final_answer"] == "今天的日期是 2026-07-07。"
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_agent_should_filter_invalid_tool_call_before_confirmation() -> None:
    """
    测试非法工具调用在确认前被过滤。

    功能：
        parser 输出未知工具时，ToolAgent 应先由 validate 节点记录错误并清空 tool_calls，
        确认节点不应为未知工具生成确认提示，执行节点也不应调用 executor。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "unknown_tool",
                    "args": {},
                }
            ],
        }
    )
    executor = FakeExecutor()
    node = build_tool_agent(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
    )

    result = await node(
        {
            "question": "调用不存在的工具",
        }
    )

    assert result["need_tool"] is False
    assert result["tool_call_validation_ok"] is False
    assert result["tool_calls"] == []
    assert result["tool_call_validation_errors"][0]["code"] == "unknown_tool"
    assert result["tool_confirmed"] == "not_required"
    assert result["tool_confirmation_required"] is False
    assert executor.calls == []


def test_merge_state_update_should_not_mutate_original_state() -> None:
    """
    测试 state 合并函数不修改原始 state。

    功能：
        确认 merge_state_update 返回新 dict，避免调用方 state 被原地污染。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "你好",
    }

    # 合并 update 时应创建新 dict，而不是原地改传入的 state。
    merged = merge_state_update(
        state=state,
        update={
            "need_tool": False,
        },
    )

    assert state == {
        "question": "你好",
    }
    assert merged == {
        "question": "你好",
        "need_tool": False,
    }
    assert merged is not state
