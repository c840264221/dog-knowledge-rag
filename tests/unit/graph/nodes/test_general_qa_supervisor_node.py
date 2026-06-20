"""
general_qa_supervisor_node 单元测试。

Supervisor（监督者节点）：
负责根据当前 state 判断 general_qa_agent 下一步应该交给哪个 worker 执行。

Worker（工作节点）：
Graph 中执行具体任务的节点，例如 tool_parse、ask_confirm、execute_tool、answer_gen。

本测试覆盖：
1. build_state_summary 是否正确提取 state 摘要
2. 合法 worker 是否原样返回
3. finish 是否可以作为终止信号返回
4. 非法 worker 是否兜底到 answer_gen
5. LLM 返回字符串时是否兼容
6. 是否保存 checkpoint
7. 是否写入 runtime context 当前 node
8. 是否记录 timeline 事件
9. runtime_context_getter 返回 None 时是否不报错
10. checkpoint_manager 为 None 时是否不报错
"""

import pytest

from src.agents.general_qa_agent.supervisor import (
    build_general_qa_supervisor_node,
    build_state_summary,
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
                事件名称，例如 general_qa_supervisor_node。

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

    def __init__(self):
        """
        初始化假 CheckpointManager。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

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

    def __init__(self, response):
        """
        初始化假 LLMProvider。

        参数：
            response：
                safe_ainvoke 返回的响应，可以是 FakeLLMResponse，也可以是普通字符串。

        返回值：
            None：构造函数无返回值。
        """

        self.main_llm = "fake_main_llm"
        self.response = response
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
                模型不可用时的兜底响应。

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

        return self.response


def build_test_node(
    decision="answer_gen",
    with_checkpoint=True,
    with_runtime_context=True,
    response_as_string=False,
):
    """
    构建测试用 general_qa_supervisor_node。

    功能：
        创建 fake llm_provider、fake checkpoint_manager、fake runtime_context。
        然后通过 build_general_qa_supervisor_node 注入依赖，得到真正可执行的 node。

    参数：
        decision：
            模拟 LLM 返回的 worker 决策。

        with_checkpoint：
            是否传入 checkpoint_manager。

        with_runtime_context：
            是否提供 fake runtime context。

        response_as_string：
            是否让 fake LLM 直接返回字符串，而不是返回带 content 字段的对象。

    返回值：
        tuple：
            node, fake_ctx, fake_llm_provider, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()

    response = (
        decision
        if response_as_string
        else FakeLLMResponse(
            decision,
        )
    )

    fake_llm_provider = FakeLLMProvider(
        response=response,
    )

    fake_checkpoint_manager = (
        FakeCheckpointManager()
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

    node = build_general_qa_supervisor_node(
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


def test_build_state_summary_should_extract_expected_fields():
    """
    测试 build_state_summary 是否正确提取 supervisor 需要的字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "question": "北京天气怎么样？",
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": "weather",
                "result": "北京天气：晴",
            }
        ],
        "answer": "北京天气晴",
        "tool_confirmed": True,
        "unused_field": "不应该进入 summary",
    }

    result = build_state_summary(
        state,
    )

    assert result == {
        "question": "北京天气怎么样？",
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": "weather",
                "result": "北京天气：晴",
            }
        ],
        "has_answer": True,
        "tool_confirmed": True,
    }


def test_build_state_summary_should_set_has_answer_false_when_answer_missing():
    """
    测试 answer 缺失时 has_answer 是否为 False。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "question": "你好",
    }

    result = build_state_summary(
        state,
    )

    assert result["has_answer"] is False


@pytest.mark.parametrize(
    "decision",
    [
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
        "finish",
    ],
)
@pytest.mark.asyncio
async def test_supervisor_should_return_valid_worker(
    decision,
):
    """
    测试 LLM 返回合法 worker 时，supervisor 是否原样返回。

    参数：
        decision：
            参数化传入的合法 worker。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        decision=decision,
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result["next_worker"] == decision
    assert result["messages"][0].content == f"Supervisor决策: {decision}"

    assert fake_ctx.state_scope.current_node == "general_qa_supervisor_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "general_qa_supervisor_node",
            "metadata": None,
        }
    ]

    assert len(
        fake_llm_provider.calls
    ) == 1

    assert fake_llm_provider.calls[0]["llm"] == "fake_main_llm"
    assert fake_llm_provider.calls[0]["fallback_response"] == "所有模型均不可用！"

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_supervisor_should_strip_and_lowercase_decision():
    """
    测试 LLM 返回带空格和大写的 worker 时，是否会 strip 并 lowercase。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_llm_provider,
        _fake_checkpoint_manager,
    ) = build_test_node(
        decision="  TOOL_PARSE  ",
    )

    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["next_worker"] == "tool_parse"


@pytest.mark.asyncio
async def test_supervisor_should_fallback_to_answer_gen_when_decision_invalid():
    """
    测试 LLM 返回非法 worker 时，是否兜底到 answer_gen。

    fallback（兜底）：
    当模型输出不符合合法 worker 集合时，不让 Graph 因非法路由中断，
    而是返回安全默认节点 answer_gen。

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
        decision="unknown_worker",
    )

    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["next_worker"] == "answer_gen"
    assert result["messages"][0].content == "Supervisor决策: answer_gen"
    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_supervisor_should_support_plain_string_response():
    """
    测试 LLM 直接返回字符串时，supervisor 是否仍能处理。

    场景：
        有些 fake LLM 或 fallback 逻辑可能直接返回 str，
        而不是返回带 content 字段的对象。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_llm_provider,
        _fake_checkpoint_manager,
    ) = build_test_node(
        decision="answer_gen",
        response_as_string=True,
    )

    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["next_worker"] == "answer_gen"


@pytest.mark.asyncio
async def test_supervisor_should_work_without_checkpoint_manager():
    """
    测试 checkpoint_manager 为 None 时，supervisor 是否仍可正常运行。

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
        decision="finish",
        with_checkpoint=False,
    )

    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["next_worker"] == "finish"
    assert len(
        fake_llm_provider.calls
    ) == 1
    assert fake_checkpoint_manager is None


@pytest.mark.asyncio
async def test_supervisor_should_work_without_runtime_context():
    """
    测试 runtime_context_getter 返回 None 时，supervisor 是否仍可正常运行。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        _fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        decision="answer_gen",
        with_runtime_context=False,
    )

    result = await node(
        {
            "question": "你好",
        }
    )

    assert result["next_worker"] == "answer_gen"

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.timeline_scope.events == []

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_supervisor_prompt_should_include_state_summary():
    """
    测试传给 LLM 的 prompt 中是否包含 state_summary 信息。

    注意：
        GENERAL_QA_SUPERVISOR_PROMPT.format_messages 通常返回 LangChain messages。
        因此这里不强行断言完整 prompt 字符串，只检查调用确实发生，
        并且 prompt 中能转换出关键字段内容。

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
        decision="answer_gen",
    )

    state = {
        "question": "北京天气怎么样？",
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                },
            }
        ],
        "tool_results": [],
    }

    await node(
        state,
    )

    prompt = fake_llm_provider.calls[0]["prompt"]

    prompt_text = str(
        prompt
    )

    assert "北京天气怎么样？" in prompt_text
    assert "weather" in prompt_text
    assert "北京" in prompt_text