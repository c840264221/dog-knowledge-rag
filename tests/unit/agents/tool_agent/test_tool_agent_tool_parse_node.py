"""
ToolAgent tool_parse_node 测试。

功能：
    测试新版 ToolAgent 工具解析节点是否能通过注入 parser 生成工具调用 state。

测试重点：
    1. parser 返回 dict 时可以生成 tool_calls。
    2. parser 返回 ToolParseResult 时可以正常处理。
    3. 已有 tool_calls 时跳过重复解析。
    4. parser 抛异常时返回安全 fallback。
    5. 输出必须是普通 dict，不能把 Pydantic 对象写入 state。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
)
from src.agents.tool_agent.nodes.tool_parse_node import (
    build_tool_agent_tool_parse_node,
    call_tool_parser,
    normalize_tool_parse_result,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall, ToolParseResult


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
        提供 state 和 timeline 两个作用域。

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
        获取测试状态作用域。

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
        获取测试时间线作用域。

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


class FakeAinvokeParser:
    """
    测试用异步解析器。

    功能：
        模拟带 ainvoke 方法的 parser。

    参数：
        result:
            parser 返回值。

        error:
            parser 需要抛出的异常。

    返回值：
        FakeAinvokeParser:
            测试用解析器。
    """

    def __init__(
        self,
        result=None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.inputs: list[dict] = []

    async def ainvoke(
        self,
        parser_input: dict,
    ):
        """
        模拟异步 parser 调用。

        功能：
            记录输入，按配置返回结果或抛异常。

        参数：
            parser_input:
                parser 输入。

        返回值：
            object:
                配置的解析结果。
        """

        self.inputs.append(
            parser_input
        )

        if self.error is not None:
            raise self.error

        return self.result


class FakeLLMProvider:
    """
    测试用 LLM Provider。

    功能：
        模拟项目中的 llm_provider，提供 backup_llm 和 safe_ainvoke。

    参数：
        response_text:
            safe_ainvoke 返回的 LLM 文本。

        error:
            safe_ainvoke 需要抛出的异常。

    返回值：
        FakeLLMProvider:
            测试用 LLM Provider。
    """

    def __init__(
        self,
        response_text: str,
        error: Exception | None = None,
    ) -> None:
        self.backup_llm = object()
        self.response_text = response_text
        self.error = error
        self.calls: list[dict] = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response: str,
    ) -> str:
        """
        模拟安全调用 LLM。

        功能：
            记录调用参数，并返回预设文本。

        参数：
            llm:
                被调用的大语言模型。

            prompt:
                渲染后的 prompt。

            fallback_response:
                LLM 调用失败时的兜底文本。

        返回值：
            str:
                模拟 LLM 返回文本。
        """

        self.calls.append(
            {
                "llm": llm,
                "prompt": prompt,
                "fallback_response": fallback_response,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response_text


def build_test_node(
    parser,
    with_checkpoint: bool = True,
    with_runtime_context: bool = True,
):
    """
    构建测试用 ToolAgent 工具解析节点。

    功能：
        注入 fake parser、fake checkpoint_manager 和 fake runtime_context。

    参数：
        parser:
            测试用 parser。

        with_checkpoint:
            是否提供 checkpoint_manager。

        with_runtime_context:
            是否提供 runtime context。

    返回值：
        tuple:
            node, fake_ctx, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()
    checkpoint_manager = (
        FakeCheckpointManager()
        if with_checkpoint
        else None
    )

    def runtime_context_getter():
        """
        获取测试用 RuntimeContext。

        功能：
            根据 with_runtime_context 决定是否返回 fake_ctx。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None:
                测试用运行时上下文或 None。
        """

        if not with_runtime_context:
            return None

        return fake_ctx

    node = build_tool_agent_tool_parse_node(
        parser=parser,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        checkpoint_manager,
    )


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_support_llm_provider() -> None:
    """
    测试节点支持注入 llm_provider。

    功能：
        不传 parser，只传 fake llm_provider，模拟旧 tool_parse_node 的 LLM JSON 输出。

    参数：
        无。

    返回值：
        None。
    """

    llm_provider = FakeLLMProvider(
        response_text="""
        {
          "need_tool": true,
          "tool_calls": [
            {
              "name": "weather",
              "args": {
                "city": "成都"
              }
            }
          ],
          "response": ""
        }
        """
    )
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

    node = build_tool_agent_tool_parse_node(
        llm_provider=llm_provider,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    update = await node(
        {
            "question": "今天成都天气怎么样？",
        }
    )

    assert update["need_tool"] is True
    assert update["tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "pending_confirmation"
    assert len(
        llm_provider.calls
    ) == 1
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_parse_dict_result() -> None:
    """
    测试 parser 返回 dict 时生成工具调用。

    功能：
        确认节点输出 need_tool、tool_calls、tool_results、tool_round 和 tool_agent_response。

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
    (
        node,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天成都天气怎么样？",
            "tool_round": 2,
        }
    )

    assert update["need_tool"] is True
    assert update["tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert update["tool_results"] == []
    assert update["tool_round"] == 3
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "pending_confirmation"
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_parse_node"
    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "tool_agent_tool_parse_node",
            "metadata": None,
        }
    ]
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_parse_tool_parse_result() -> None:
    """
    测试 parser 返回 ToolParseResult。

    功能：
        确认节点能处理底层工具 schema 中的 ToolParseResult。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result=ToolParseResult(
            need_tool=False,
            tool_calls=[],
            response="不需要工具。",
        )
    )
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "你好",
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update["tool_round"] == 1
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_skip_when_tool_calls_exist() -> None:
    """
    测试已有 tool_calls 时跳过解析。

    功能：
        避免重复解析覆盖上游已经生成的工具调用。

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
    (
        node,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天几号？",
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }
    )

    assert update == {}
    assert parser.inputs == []
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_parse_node"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_fallback_when_parser_failed() -> None:
    """
    测试 parser 抛异常时返回安全 fallback。

    功能：
        确认解析失败不会打断 ToolAgent 链路。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        error=RuntimeError(
            "parser failed"
        )
    )
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天几号？",
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_fallback_when_question_missing() -> None:
    """
    测试缺少 question 时返回安全 fallback。

    功能：
        确认没有用户问题时不会调用 parser。

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
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node({})

    assert update["need_tool"] is False
    assert parser.inputs == []
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_call_tool_parser_should_support_callable_parser() -> None:
    """
    测试普通 callable parser。

    功能：
        确认 call_tool_parser 可以调用普通函数解析器。

    参数：
        无。

    返回值：
        None。
    """

    def parser(
        parser_input: dict,
    ) -> dict:
        """
        测试用普通解析函数。

        功能：
            返回 date 工具调用。

        参数：
            parser_input:
                工具解析输入。

        返回值：
            dict:
                工具解析结果。
        """

        assert parser_input["question"] == "今天几号？"
        return {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }

    result = await call_tool_parser(
        parser=parser,
        question="今天几号？",
        state={},
    )

    assert result["tool_calls"][0]["name"] == "date"


def test_normalize_tool_parse_result_should_skip_invalid_tool_calls() -> None:
    """
    测试归一化时跳过非法 tool_calls。

    功能：
        确认坏数据不会直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    result = normalize_tool_parse_result(
        {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                },
                {
                    "args": {},
                },
                "bad_call",
                ToolCall(
                    name="date",
                    args={},
                ),
            ],
        }
    )

    assert [
        tool_call.name
        for tool_call in result.tool_calls
    ] == [
        "weather",
        "date",
    ]
