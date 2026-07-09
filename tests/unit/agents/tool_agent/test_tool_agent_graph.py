"""
ToolAgent graph.py 测试。

功能：
    测试 ToolAgent LangGraph 子图是否能按条件边正确执行。

测试重点：
    1. route_after_tool_confirm 能根据权限状态返回路由。
    2. weather 需要确认时不会执行工具。
    3. date 不需要确认时会执行工具并生成 final_answer。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.state_adapter import TOOL_AGENT_RESPONSE_STATE_KEY
from src.agents.tool_agent.graph import (
    build_tool_agent_graph,
    route_after_tool_confirm,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.schemas.tool_result_schema import ToolResult


class FakeAinvokeParser:
    """
    测试用异步解析器。

    功能：
        模拟带 ainvoke 方法的 parser。

    参数：
        result:
            parser 需要返回的解析结果。

    返回值：
        FakeAinvokeParser:
            测试用解析器。
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
            记录 parser_input，并返回预设解析结果。

        参数：
            parser_input:
                工具解析输入。

        返回值：
            dict:
                预设解析结果。
        """

        self.inputs.append(
            parser_input
        )
        return self.result


class FakeTool:
    """
    测试用工具对象。

    功能：
        模拟真实工具对象，只提供 metadata。

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
        模拟 ToolRegistry.get_tool。

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
            模拟工具注册表查询。

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
        模拟 ToolExecutor.execute，避免调用真实外部工具。

    参数：
        无。

    返回值：
        FakeExecutor:
            测试用执行器。
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
            记录调用并返回成功 ToolResult。

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
            content="2026-07-08"
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


def test_route_after_tool_confirm_should_return_expected_route() -> None:
    """
    测试确认后路由函数。

    功能：
        根据权限状态返回 pending_confirmation、rejected 或 allowed。

    参数：
        无。

    返回值：
        None。
    """

    assert route_after_tool_confirm(
        {
            "tool_agent_permission": {
                "status": "pending",
            },
        }
    ) == "pending_confirmation"
    assert route_after_tool_confirm(
        {
            "tool_agent_permission": {
                "status": "rejected",
            },
        }
    ) == "rejected"
    assert route_after_tool_confirm(
        {
            "tool_agent_permission": {
                "status": "confirmed",
            },
        }
    ) == "allowed"
    assert route_after_tool_confirm(
        {
            "tool_agent_permission": {
                "status": "not_required",
            },
        }
    ) == "allowed"


@pytest.mark.asyncio
async def test_tool_agent_graph_should_stop_before_execute_when_pending() -> None:
    """
    测试 pending 时不执行工具。

    功能：
        weather 需要确认，子图应在 response_adapter 后结束，不调用 executor。

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
    graph = build_tool_agent_graph(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
    )

    # 调用编译后的 LangGraph 子图，验证 pending 分支不会进入执行节点。
    result = await graph.ainvoke(
        {
            "question": "今天成都天气怎么样？",
        }
    )

    assert executor.calls == []
    assert result["tool_confirmed"] == "pending"
    assert result["tool_confirmation_required"] is True
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["permission"]["status"] == "pending"


@pytest.mark.asyncio
async def test_tool_agent_graph_should_execute_weather_when_interrupt_confirms() -> None:
    """
    测试 interrupt 确认后执行 weather 工具。

    功能：
        weather 默认需要确认。
        当 ToolAgent graph 注入 fake interrupt 且返回 y 时，
        子图应从 tool_confirm 继续进入 tool_execute。

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
    captured_prompts: list[str] = []

    def fake_interrupt(
        prompt: str,
    ) -> str:
        """
        测试用 fake interrupt。

        功能：
            记录确认提示，并模拟用户确认。

        参数：
            prompt:
                工具确认节点生成的提示文本。

        返回值：
            str:
                模拟用户输入 y。
        """

        captured_prompts.append(
            prompt
        )
        return "y"

    graph = build_tool_agent_graph(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
        interrupt_func=fake_interrupt,
    )

    # fake interrupt 返回 y 后，条件边会进入 tool_execute。
    result = await graph.ainvoke(
        {
            "question": "今天成都天气怎么样？",
        }
    )

    assert captured_prompts
    assert "查询天气" in captured_prompts[0]
    assert executor.calls == [
        {
            "tool_name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert result["tool_confirmed"] == "confirmed"
    assert result["tool_calls"] == []
    assert result["tool_results"][0]["success"] is True
    assert result["final_answer"] == "天气查询结果：执行成功"
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_agent_graph_should_execute_date_when_confirmation_not_required() -> None:
    """
    测试不需要确认时执行工具。

    功能：
        date 工具不需要确认，子图应进入 execute 和 answer 节点。

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
    graph = build_tool_agent_graph(
        parser=parser,
        tool_registry=build_fake_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
    )

    # date 权限为 not_required，所以条件边会路由到 tool_execute。
    result = await graph.ainvoke(
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
    assert result["tool_calls"] == []
    assert result["tool_results"][0]["success"] is True
    assert result["final_answer"] == "今天的日期是 2026-07-08。"
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "completed"
