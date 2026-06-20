"""
answer_gen_node 单元测试。

answer_gen_node（最终回答生成节点）：
用于把用户问题、历史消息、长期记忆、工具结果组装成 Prompt，
然后调用 LLM 生成最终回答。

build_answer_gen_node（最终回答节点构建函数）：
通过 Dependency Injection（依赖注入）方式，把 llm_provider、
checkpoint_manager、runtime_context_getter 注入到 node 中。

Tool Results（工具结果）：
工具执行后的结果集合。这里必须按 list 结构处理，
不能再按单个字符串处理。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from src.graph.nodes.answer_gen_node import (
    build_answer_gen_node,
    format_history,
    normalize_tool_results,
    format_tool_results,
    build_answer_prompt,
)


class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 Graph 正在执行的 node。
    """

    def __init__(self):
        """
        初始化假 StateScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.current_node = None

    def set_node(self, node_name):
        """
        设置当前 node 名称。

        参数：
            node_name：
                当前执行的节点名称。

        返回值：
            None：无业务返回值。
        """

        self.current_node = node_name


class FakeTimelineScope:
    """
    测试用假 TimelineScope。

    TimelineScope（时间线作用域）：
    用于记录 node 执行事件。
    """

    def __init__(self):
        """
        初始化假 TimelineScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.events = []

    def add_event(self, event_type, name, metadata=None):
        """
        添加时间线事件。

        参数：
            event_type：
                事件类型，例如 node。

            name：
                事件名称，例如 answer_gen_node。

            metadata：
                事件附加信息，默认为 None。

        返回值：
            None：无业务返回值。
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
    测试用假 RuntimeContext。

    RuntimeContext（运行时上下文）：
    表示一次请求执行过程中的运行时环境。
    """

    def __init__(self):
        """
        初始化假 RuntimeContext。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self):
        """
        获取假 StateScope。

        参数：
            无。

        返回值：
            FakeStateScope：
                测试用状态作用域。
        """

        return self.state_scope

    def timeline(self):
        """
        获取假 TimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope：
                测试用时间线作用域。
        """

        return self.timeline_scope


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于保存运行时 checkpoint。
    """

    def __init__(self, error=None):
        """
        初始化假 CheckpointManager。

        参数：
            error：
                调用 save_checkpoint 时需要抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.error = error
        self.save_count = 0

    def save_checkpoint(self):
        """
        模拟保存 checkpoint。

        参数：
            无。

        返回值：
            None：无业务返回值。
        """

        self.save_count += 1

        if self.error:
            raise self.error


class FakeLLMResponse:
    """
    测试用假 LLM 响应。

    LLM Response（大语言模型响应）：
    真实模型返回值通常包含 content 字段。
    """

    def __init__(self, content):
        """
        初始化假 LLM 响应。

        参数：
            content：
                模型响应内容。

        返回值：
            None：构造函数无返回值。
        """

        self.content = content


class FakeLLMProvider:
    """
    测试用假 LLMProvider。

    LLMProvider（大语言模型提供者）：
    用于提供 main_llm，并封装 safe_ainvoke 调用。
    """

    def __init__(self, response=None, error=None):
        """
        初始化假 LLMProvider。

        参数：
            response：
                safe_ainvoke 成功时返回的响应。

            error：
                safe_ainvoke 需要主动抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.main_llm = "fake_main_llm"
        self.response = response
        self.error = error
        self.calls = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response,
    ):
        """
        模拟异步调用 LLM。

        参数：
            llm：
                当前使用的大语言模型对象。

            prompt：
                传给 LLM 的提示词。

            fallback_response：
                模型失败时的兜底回答。

        返回值：
            object：
                模拟 LLM 响应。
        """

        self.calls.append(
            {
                "llm": llm,
                "prompt": prompt,
                "fallback_response": fallback_response,
            }
        )

        if self.error:
            raise self.error

        return self.response


def build_test_node(
    llm_response=None,
    llm_error=None,
    with_checkpoint=True,
    checkpoint_error=None,
    with_runtime_context=True,
):
    """
    构建测试用 answer_gen_node。

    功能：
        创建 fake llm_provider、fake checkpoint_manager、fake runtime_context。
        然后通过 build_answer_gen_node 注入依赖，得到真正可执行的 node。

    参数：
        llm_response：
            假 LLMProvider 成功返回的响应。

        llm_error：
            假 LLMProvider 需要抛出的异常。

        with_checkpoint：
            是否传入 checkpoint_manager。

        checkpoint_error：
            checkpoint_manager.save_checkpoint 需要抛出的异常。

        with_runtime_context：
            是否提供 fake runtime context。

    返回值：
        tuple：
            node, fake_ctx, fake_llm_provider, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()

    fake_llm_provider = FakeLLMProvider(
        response=llm_response,
        error=llm_error,
    )

    fake_checkpoint_manager = (
        FakeCheckpointManager(
            error=checkpoint_error,
        )
        if with_checkpoint
        else None
    )

    def runtime_context_getter():
        """
        获取测试用 RuntimeContext。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None：
                根据 with_runtime_context 决定是否返回 fake_ctx。
        """

        if not with_runtime_context:
            return None

        return fake_ctx

    node = build_answer_gen_node(
        llm_provider=fake_llm_provider,
        checkpoint_manager=fake_checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    )


def test_normalize_tool_results_should_return_empty_list_when_none():
    """
    测试 tool_results 为 None 时，是否归一化为空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert normalize_tool_results(
        None,
    ) == []


def test_normalize_tool_results_should_keep_list():
    """
    测试 tool_results 已经是 list 时，是否保持 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_results = [
        "result1",
        "result2",
    ]

    result = normalize_tool_results(
        tool_results,
    )

    assert result == tool_results
    assert result is tool_results


def test_normalize_tool_results_should_convert_string_to_list():
    """
    测试历史遗留字符串 tool_results 是否转换成 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = normalize_tool_results(
        "历史字符串工具结果",
    )

    assert result == [
        "历史字符串工具结果",
    ]


def test_normalize_tool_results_should_wrap_dict_to_list():
    """
    测试 dict 类型 tool_results 是否包装成 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    raw_result = {
        "tool_name": "weather",
        "result": "北京天气：晴",
    }

    result = normalize_tool_results(
        raw_result,
    )

    assert result == [
        raw_result,
    ]


def test_format_history_should_format_human_and_assistant_messages():
    """
    测试历史消息是否正确格式化。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    messages = [
        HumanMessage(
            content="你好",
        ),
        SimpleNamespace(
            content="你好，我是助手",
        ),
    ]

    result = format_history(
        messages,
    )

    assert result == (
        "用户: 你好\n"
        "助手: 你好，我是助手"
    )


def test_format_history_should_return_empty_string_when_no_messages():
    """
    测试没有历史消息时，是否返回空字符串。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_history(
        [],
    )

    assert result == ""


def test_format_tool_results_should_return_empty_string_when_empty():
    """
    测试没有工具结果时，是否返回空字符串。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_tool_results(
        [],
    )

    assert result == ""


def test_format_tool_results_should_format_list_of_strings():
    """
    测试 list[str] 工具结果是否正确格式化。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_tool_results(
        [
            "日期是 2026-06-20",
            "天气是晴天",
        ]
    )

    assert result == (
        "1. 日期是 2026-06-20\n\n"
        "2. 天气是晴天"
    )


def test_format_tool_results_should_format_list_of_dicts():
    """
    测试 list[dict] 工具结果是否正确格式化。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_tool_results(
        [
            {
                "tool_name": "weather",
                "result": "北京天气：晴",
                "success": True,
            },
            {
                "tool_name": "date",
                "result": "2026-06-20",
                "success": True,
            },
        ]
    )

    assert result == (
        "1. 工具：【weather】\n"
        "是否成功：True\n"
        "结果：北京天气：晴\n\n"
        "2. 工具：【date】\n"
        "是否成功：True\n"
        "结果：2026-06-20"
    )


def test_format_tool_results_should_format_dict_without_success():
    """
    测试工具结果 dict 缺少 success 字段时，是否仍能格式化。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_tool_results(
        [
            {
                "name": "weather",
                "content": "上海天气：多云",
            }
        ]
    )

    assert result == (
        "1. 工具：【weather】\n"
        "结果：上海天气：多云"
    )


def test_format_tool_results_should_compatible_with_legacy_string():
    """
    测试历史遗留字符串 tool_results 是否能格式化。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = format_tool_results(
        "用户取消了工具调用。",
    )

    assert result == "1. 用户取消了工具调用。"


def test_build_answer_prompt_should_include_tool_results_when_exists():
    """
    测试 Prompt 中是否正确注入工具结果。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    prompt = build_answer_prompt(
        question="北京天气怎么样？",
        memory_context="暂无用户记忆",
        history_text="用户: 查询天气",
        tool_results_text=(
            "1. 工具：【weather】\n"
            "是否成功：True\n"
            "结果：北京天气：晴"
        ),
    )

    assert prompt.startswith(
        "【工具结果】"
    )

    assert "北京天气：晴" in prompt
    assert "【用户长期记忆】" in prompt
    assert "【对话历史】" in prompt
    assert "【用户问题】" in prompt


def test_build_answer_prompt_should_not_include_tool_results_when_empty():
    """
    测试没有工具结果时，Prompt 是否不添加工具结果区域。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    prompt = build_answer_prompt(
        question="你好",
        memory_context="暂无用户记忆",
        history_text="",
        tool_results_text="",
    )

    assert not prompt.startswith(
        "【工具结果】"
    )

    assert "【用户问题】" in prompt
    assert "你好" in prompt


@pytest.mark.asyncio
async def test_answer_gen_node_should_generate_answer_with_tool_results():
    """
    测试有工具结果时，answer_gen_node 是否调用 LLM 并返回最终回答。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "北京今天是晴天。",
        )
    )

    state = {
        "question": "北京天气怎么样？",
        "memory_context": "用户喜欢简洁回答",
        "messages": [
            HumanMessage(
                content="帮我查北京天气",
            )
        ],
        "tool_results": [
            {
                "tool_name": "weather",
                "result": "北京天气：晴",
                "success": True,
            }
        ],
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "北京今天是晴天。"
    }

    assert len(
        fake_llm_provider.calls
    ) == 1

    call = fake_llm_provider.calls[0]

    assert call["llm"] == "fake_main_llm"
    assert call["fallback_response"] == "模型暂时不可用"

    assert "【工具结果】" in call["prompt"]
    assert "工具：【weather】" in call["prompt"]
    assert "结果：北京天气：晴" in call["prompt"]
    assert "用户喜欢简洁回答" in call["prompt"]
    assert "用户: 帮我查北京天气" in call["prompt"]
    assert "北京天气怎么样？" in call["prompt"]

    assert fake_ctx.state_scope.current_node == "answer_gen_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "answer_gen_node",
            "metadata": None,
        }
    ]

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_answer_gen_node_should_generate_answer_without_tool_results():
    """
    测试没有工具结果时，Prompt 是否不包含工具结果区域。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "你好，有什么可以帮你？",
        )
    )

    state = {
        "question": "你好",
        "messages": [],
        "tool_results": [],
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "你好，有什么可以帮你？"
    }

    prompt = fake_llm_provider.calls[0]["prompt"]

    assert "【工具结果】" not in prompt
    assert "暂无用户记忆" in prompt
    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_answer_gen_node_should_compatible_with_legacy_string_tool_results():
    """
    测试历史遗留字符串 tool_results 是否会格式化后注入 Prompt。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        _fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "用户取消了工具调用。",
        )
    )

    state = {
        "question": "工具执行了吗？",
        "tool_results": "用户取消了工具调用。",
    }

    result = await node(
        state,
    )

    prompt = fake_llm_provider.calls[0]["prompt"]

    assert result == {
        "answer": "用户取消了工具调用。"
    }

    assert "【工具结果】" in prompt
    assert "1. 用户取消了工具调用。" in prompt


@pytest.mark.asyncio
async def test_answer_gen_node_should_use_fallback_when_llm_raises_error():
    """
    测试 LLM 调用失败时，是否返回兜底回答。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_error=RuntimeError(
            "llm error",
        )
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "模型暂时不可用"
    }

    assert len(
        fake_llm_provider.calls
    ) == 1

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_answer_gen_node_should_not_break_when_checkpoint_save_failed():
    """
    测试 checkpoint 保存失败时，节点是否仍然返回回答。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "最终回答",
        ),
        checkpoint_error=RuntimeError(
            "checkpoint error",
        ),
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "最终回答"
    }

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_answer_gen_node_should_work_without_checkpoint_manager():
    """
    测试未注入 checkpoint_manager 时，节点是否仍可正常工作。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "没有 checkpoint 也能回答",
        ),
        with_checkpoint=False,
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "没有 checkpoint 也能回答"
    }

    assert len(
        fake_llm_provider.calls
    ) == 1

    assert fake_checkpoint_manager is None


@pytest.mark.asyncio
async def test_answer_gen_node_should_work_without_runtime_context():
    """
    测试 runtime_context_getter 返回 None 时，节点是否仍可正常工作。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response=FakeLLMResponse(
            "没有 runtime context 也能回答",
        ),
        with_runtime_context=False,
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result == {
        "answer": "没有 runtime context 也能回答"
    }

    assert len(
        fake_llm_provider.calls
    ) == 1

    assert fake_checkpoint_manager.save_count == 1

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.timeline_scope.events == []