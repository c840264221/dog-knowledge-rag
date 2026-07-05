from typing import Any

import pytest

from src.graph import graph_run
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
)


class FakeRuntimeContext:
    """
    测试用 RuntimeContext。

    功能：
        模拟 graph_run.py 需要写入的 trace_id、user_id 和 session_id 字段。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContext:
            一个最小可用的运行时上下文对象。
    """

    def __init__(self) -> None:
        self.trace_id = None
        self.user_id = None
        self.session_id = None


class FakeInterrupt:
    """
    测试用 interrupt 对象。

    功能：
        模拟 LangGraph interrupt 对象的 value 字段。

    参数含义：
        value:
            interrupt 提示文本。

    返回值含义：
        FakeInterrupt:
            一个包含 value 的测试对象。
    """

    def __init__(
        self,
        value: str,
    ) -> None:
        self.value = value


class FakeTask:
    """
    测试用 task 对象。

    功能：
        模拟 LangGraph current_state.tasks 中保存 interrupts 的结构。

    参数含义：
        prompt:
            interrupt 提示文本。

    返回值含义：
        FakeTask:
            一个包含 interrupts 列表的测试对象。
    """

    def __init__(
        self,
        prompt: str,
    ) -> None:
        self.interrupts = [
            FakeInterrupt(prompt),
        ]


class FakeCurrentState:
    """
    测试用 current state。

    功能：
        模拟 app.aget_state(config) 返回的 LangGraph 当前状态对象。

    参数含义：
        values:
            当前 state values。
        next_nodes:
            current_state.next，非空表示图处于中断状态。
        prompt:
            interrupt 提示文本。

    返回值含义：
        FakeCurrentState:
            一个最小 current state 对象。
    """

    def __init__(
        self,
        values: dict[str, Any] | None = None,
        next_nodes: tuple[str, ...] = (),
        prompt: str = "是否继续？",
    ) -> None:
        self.values = values or {}
        self.next = next_nodes
        self.tasks = [
            FakeTask(prompt),
        ] if next_nodes else []


class FakeGraphApp:
    """
    测试用 graph app。

    功能：
        模拟 LangGraph compiled graph 的 aget_state 和 astream 方法。

    参数含义：
        current_state:
            aget_state 返回的状态。
        resume_events:
            astream 恢复执行时产出的事件列表。

    返回值含义：
        FakeGraphApp:
            一个支持异步调用的测试 graph app。
    """

    def __init__(
        self,
        current_state: FakeCurrentState,
        resume_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.current_state = current_state
        self.resume_events = resume_events or []
        self.received_configs: list[dict[str, Any]] = []

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> FakeCurrentState:
        """
        模拟读取 LangGraph 当前状态。

        功能：
            记录传入 config，并返回预设 current_state。

        参数含义：
            config:
                LangGraph config。

        返回值含义：
            FakeCurrentState:
                预设状态对象。
        """

        self.received_configs.append(config)
        return self.current_state

    async def astream(
        self,
        *_args: Any,
        **_kwargs: Any,
    ):
        """
        模拟恢复执行的异步事件流。

        功能：
            按顺序 yield 预设 resume_events。

        参数含义：
            *_args:
                兼容 LangGraph astream 位置参数。
            **_kwargs:
                兼容 LangGraph astream 关键字参数。

        返回值含义：
            AsyncIterator[dict[str, Any]]:
                异步事件流。
        """

        for event in self.resume_events:
            yield event


async def fake_stream_runner(
    **_kwargs: Any,
):
    """
    测试用主图流式执行函数。

    功能：
        模拟 safe_stream_graph 产出一个状态更新事件。

    参数含义：
        **_kwargs:
            兼容 safe_stream_graph 的 graph、state、config、stream_mode 参数。

    返回值含义：
        AsyncIterator[dict[str, Any]]:
            异步状态事件。
    """

    yield {
        "memory_extract": {},
    }


@pytest.fixture()
def runtime_context() -> FakeRuntimeContext:
    """
    创建测试 runtime context。

    功能：
        为每个测试提供独立的 FakeRuntimeContext。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContext:
            测试运行时上下文。
    """

    return FakeRuntimeContext()


@pytest.fixture(autouse=True)
def patch_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    固定测试 user_id。

    功能：
        避免 create_initial_state 依赖真实用户管理模块。

    参数含义：
        monkeypatch:
            pytest 提供的动态替换工具。

    返回值含义：
        None。
    """

    monkeypatch.setattr(
        graph_run,
        "get_user_id",
        lambda: "test_user",
    )
    monkeypatch.setattr(
        graph_run,
        "write_rag_debug_report_if_enabled",
        lambda **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_return_final_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试新问题正常完成时返回 GraphFinalResult。

    功能：
        使用 mock graph app 模拟完整主图运行完成，并验证结构化结果字段。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "answer": "金毛通常很友好。",
            },
        )
    )

    result = await graph_run.run_main_graph_with_result(
        question="金毛性格怎么样？",
        thread_id="thread_001",
        trace_id="trace_001",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "金毛通常很友好。"
    assert result.thread_id == "thread_001"
    assert result.trace_id == "trace_001"
    assert runtime_context.user_id == "test_user"
    assert runtime_context.session_id == "thread_001"
    assert app.received_configs[0]["configurable"] == {
        "thread_id": "thread_001",
    }


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_return_interrupt_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试新问题触发中断时返回 GraphInterruptResult。

    功能：
        使用 mock current_state.next 模拟图停在 interrupt 节点。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            next_nodes=("ask_confirm",),
            prompt="是否允许调用天气工具？",
        )
    )

    result = await graph_run.run_main_graph_with_result(
        question="今天成都天气怎么样？",
        thread_id="thread_002",
        trace_id="trace_002",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "是否允许调用天气工具？"
    assert result.thread_id == "thread_002"
    assert result.trace_id == "trace_002"


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_resume_to_final_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试恢复执行后直接返回最终结果。

    功能：
        使用 mock astream 模拟 Command resume 后产出 answer 事件。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(),
        resume_events=[
            {
                "answer": "已完成工具调用。",
            },
        ],
    )
    result = await graph_run.run_main_graph_with_result(
        question="y",
        thread_id="thread_003",
        trace_id="trace_003",
        resume_value="y",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "已完成工具调用。"
    assert result.metadata["source"] == "resume_stream_event"


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_resume_to_interrupt_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试恢复执行后再次中断。

    功能：
        使用 mock astream 返回空事件，再通过 aget_state 模拟图再次停在 interrupt。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            next_nodes=("ask_confirm",),
            prompt="还需要二次确认吗？",
        ),
        resume_events=[],
    )
    result = await graph_run.run_main_graph_with_result(
        question="y",
        thread_id="thread_004",
        trace_id="trace_004",
        resume_value="y",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "还需要二次确认吗？"
    assert result.thread_id == "thread_004"
