"""
tool_parse_node 单元测试。

tool_parse_node（工具解析节点）：
用于分析用户问题是否需要调用工具，并将 LLM 输出解析成 tool_calls。

build_tool_parse_node（工具解析节点构建函数）：
用于通过 Dependency Injection（依赖注入）方式，把 llm_provider、
checkpoint_manager、runtime_context_getter 注入到 node 中。

Dependency Injection，DI（依赖注入）：
不在函数内部直接 import container 或创建依赖，而是从外部传入依赖。

Graph Node（图节点）：
LangGraph 中的执行节点，接收 state，返回 dict，用于更新 Graph 状态。

State Merge（状态合并）：
LangGraph 会把 node 返回的 dict 合并回当前 state。
"""

import pytest

from src.graph.nodes.tool_parse_node import (
    build_tool_parse_node,
)


class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 Graph 正在执行的 node、agent、tool 等运行状态。
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
    用于记录当前请求执行过程中的事件。
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
                事件名称，例如 tool_parse_node。

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


class FakeLLMProvider:
    """
    测试用假 LLM Provider。

    LLM Provider（大语言模型提供者）：
    项目中用于统一提供模型和安全调用方法的对象。
    """

    def __init__(
        self,
        response_text=None,
        error=None,
    ):
        """
        初始化假 LLM Provider。

        参数：
            response_text：
                safe_ainvoke 成功时返回的文本内容。

            error：
                safe_ainvoke 需要主动抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.backup_llm = object()
        self.response_text = response_text
        self.error = error
        self.calls = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response,
    ):
        """
        模拟安全异步调用 LLM。

        参数：
            llm：
                被调用的 LLM 对象。

            prompt：
                传入 LLM 的 prompt。

            fallback_response：
                调用失败时的兜底返回文本。

        返回值：
            str：
                模拟 LLM 返回的字符串。
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

        return self.response_text


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于保存、恢复、清理运行时检查点。
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


def build_test_node(
    llm_response_text=None,
    llm_error=None,
    with_checkpoint=True,
    with_runtime_context=True,
):
    """
    构建测试用 tool_parse_node。

    功能：
        创建 fake llm_provider、fake checkpoint_manager、fake runtime_context。
        然后通过 build_tool_parse_node 注入依赖，得到真正可执行的 node。

    参数：
        llm_response_text：
            假 LLM 成功返回的文本。

        llm_error：
            假 LLM 需要抛出的异常。

        with_checkpoint：
            是否传入 checkpoint_manager。
            True 表示传入 fake checkpoint_manager。
            False 表示传入 None。

        with_runtime_context：
            是否提供 fake runtime context。
            True 表示 runtime_context_getter 返回 fake_ctx。
            False 表示 runtime_context_getter 返回 None。

    返回值：
        tuple：
            node, fake_ctx, fake_llm_provider, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()

    fake_llm_provider = FakeLLMProvider(
        response_text=llm_response_text,
        error=llm_error,
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

    node = build_tool_parse_node(
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


@pytest.mark.asyncio
async def test_tool_parse_node_should_return_weather_tool_call_when_llm_requires_weather():
    """
    测试 LLM 返回需要 weather 工具时，tool_parse_node 是否正确返回 tool_calls。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    llm_response_text = """
    {
      "need_tool": true,
      "tool_calls": [
        {
          "name": "weather",
          "args": {
            "city": "北京",
            "date": "2025-03-15"
          }
        }
      ],
      "response": ""
    }
    """

    (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response_text=llm_response_text,
    )

    state = {
        "question": "北京 2025-03-15 天气怎么样？",
        "tool_round": 2,
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                    "date": "2025-03-15",
                },
            }
        ],
        "tool_results": [],
        "tool_round": 3,
    }

    assert fake_ctx.state_scope.current_node == "tool_parse_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "tool_parse_node",
            "metadata": None,
        }
    ]

    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_parse_node_should_return_no_tool_when_llm_says_no_tool():
    """
    测试 LLM 返回不需要工具时，tool_parse_node 是否返回 need_tool=False。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    llm_response_text = """
    {
      "need_tool": false,
      "tool_calls": [],
      "response": "你好，我可以帮你解答问题。"
    }
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response_text=llm_response_text,
    )

    state = {
        "question": "你好",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "tool_round": 1,
    }

    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_parse_node_should_return_empty_dict_when_tool_calls_already_exist():
    """
    测试 state 中已经存在 tool_calls 时，tool_parse_node 是否返回空 dict。

    empty dict（空字典）：
    在 LangGraph node 中通常表示不更新 state。

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
        llm_response_text="""
        {
          "need_tool": true,
          "tool_calls": [
            {
              "name": "date",
              "args": {}
            }
          ],
          "response": ""
        }
        """,
    )

    state = {
        "question": "今天几号？",
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = await node(
        state,
    )

    assert result == {}

    assert fake_ctx.state_scope.current_node == "tool_parse_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "tool_parse_node",
            "metadata": None,
        }
    ]

    assert fake_llm_provider.calls == []
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_parse_node_should_fallback_when_llm_call_failed():
    """
    测试 LLM 调用失败时，tool_parse_node 是否返回 need_tool=False。

    fallback（兜底）：
    当 LLM 调用或解析失败时，返回安全默认结果，避免 Graph 中断。

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
            "llm failed",
        ),
    )

    state = {
        "question": "今天几号？",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
    }

    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_parse_node_should_fallback_when_parser_failed():
    """
    测试 LLM 返回非法 JSON 时，tool_parse_node 是否返回 need_tool=False。

    parser（解析器）：
    当前使用 PydanticOutputParser，将 LLM 文本解析成 ToolParseResult。

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
        llm_response_text="这不是合法 JSON",
    )

    state = {
        "question": "今天几号？",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
    }

    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_parse_node_should_use_default_tool_round_when_missing():
    """
    测试 state 中缺少 tool_round 时，是否默认从 0 递增到 1。

    tool_round（工具轮次）：
    用于记录工具调用轮数，防止工具链路无限循环。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    llm_response_text = """
    {
      "need_tool": true,
      "tool_calls": [
        {
          "name": "date",
          "args": {}
        }
      ],
      "response": ""
    }
    """

    (
        node,
        _fake_ctx,
        _fake_llm_provider,
        _fake_checkpoint_manager,
    ) = build_test_node(
        llm_response_text=llm_response_text,
    )

    state = {
        "question": "今天几号？",
    }

    result = await node(
        state,
    )

    assert result["tool_round"] == 1

    assert result["tool_calls"] == [
        {
            "name": "date",
            "args": {},
        }
    ]


@pytest.mark.asyncio
async def test_tool_parse_node_should_fallback_when_question_missing():
    """
    测试缺少 question 字段时，tool_parse_node 是否安全兜底。

    当前推荐实现：
        question = state.get("question", "")
        如果 question 为空，则返回 {"need_tool": False}

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
        llm_response_text="""
        {
          "need_tool": true,
          "tool_calls": [
            {
              "name": "date",
              "args": {}
            }
          ],
          "response": ""
        }
        """,
    )

    state = {}

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
    }

    assert fake_llm_provider.calls == []
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_parse_node_should_work_without_checkpoint_manager():
    """
    测试未注入 checkpoint_manager 时，tool_parse_node 是否仍可正常工作。

    checkpoint_manager=None：
    表示当前节点不保存 checkpoint，但不应该影响工具解析逻辑。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    llm_response_text = """
    {
      "need_tool": true,
      "tool_calls": [
        {
          "name": "date",
          "args": {}
        }
      ],
      "response": ""
    }
    """

    (
        node,
        _fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response_text=llm_response_text,
        with_checkpoint=False,
    )

    state = {
        "question": "今天几号？",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
        "tool_results": [],
        "tool_round": 1,
    }

    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager is None


@pytest.mark.asyncio
async def test_tool_parse_node_should_work_without_runtime_context():
    """
    测试 runtime_context_getter 返回 None 时，tool_parse_node 是否仍可正常工作。

    runtime_context_getter（运行时上下文获取函数）：
    用于在 node 执行时获取当前 RuntimeContext。
    如果返回 None，node 应该跳过 state/timeline 写入，而不是报错。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    llm_response_text = """
    {
      "need_tool": false,
      "tool_calls": [],
      "response": "不需要工具。"
    }
    """

    (
        node,
        fake_ctx,
        fake_llm_provider,
        fake_checkpoint_manager,
    ) = build_test_node(
        llm_response_text=llm_response_text,
        with_runtime_context=False,
    )

    state = {
        "question": "什么是金毛？",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "tool_round": 1,
    }

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.timeline_scope.events == []
    assert len(fake_llm_provider.calls) == 1
    assert fake_checkpoint_manager.save_count == 1