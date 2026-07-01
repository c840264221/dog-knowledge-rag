"""
Graph route 路由函数单元测试。

Route（路由）：
在 LangGraph 中，route 函数负责根据当前 state（状态）判断下一步应该进入哪个 node（节点）。

State（状态）：
Graph 节点之间传递的数据结构，本项目中主要使用 DogState。

Unit Test（单元测试）：
只测试一个函数的输入和输出，不依赖真实 LLM、Tool、VectorDB、Checkpoint 等外部能力。
"""

import pytest

from src.graph.routes.route_afer_confirm import (
    route_after_confirm,
)

from src.graph.routes. route_after_ask_user import (
    route_after_ask_user,
)

from src.graph.routes. route_after_retrieval import (
    route_after_retrieval,
)

# 重点：这里建议 import 整个模块，而不是只 import 函数
# 因为 route_after_semantic 内部用了 runtime_ctx，测试时需要 monkeypatch 替换它
import src.graph.routes.route_after_semantic as semantic_route_module
import src.agents.root_agent.routes as root_route_module


def test_route_after_confirm_should_call_tool_when_need_tool_and_tool_calls_exist():
    """
    测试 need_tool=True 且 tool_calls 存在时，是否进入 call_tool。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = route_after_confirm(
        state,
    )

    assert result == "call_tool"


def test_route_after_confirm_should_not_call_tool_when_need_tool_false():
    """
    测试 need_tool=False 时，即使 tool_calls 存在，也不调用工具。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "need_tool": False,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = route_after_confirm(
        state,
    )

    assert result == "no_call_tool"


def test_route_after_confirm_should_not_call_tool_when_tool_calls_empty():
    """
    测试 need_tool=True 但 tool_calls 为空时，不调用工具。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "need_tool": True,
        "tool_calls": [],
    }

    result = route_after_confirm(
        state,
    )

    assert result == "no_call_tool"


def test_route_after_confirm_should_not_call_tool_when_fields_missing():
    """
    测试缺少 need_tool 和 tool_calls 字段时，是否安全兜底到 no_call_tool。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {}

    result = route_after_confirm(
        state,
    )

    assert result == "no_call_tool"


def test_route_after_ask_user_should_return_retry_when_feedback_is_1():
    """
    测试用户输入 1 时，是否进入 retry。

    retry（重试）：
    表示重新执行检索流程，通常会回到 retrieve 或 retry node。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "user_feedback": "1",
    }

    result = route_after_ask_user(
        state,
    )

    assert result == "retry"


def test_route_after_ask_user_should_return_modify_filter_when_feedback_is_2():
    """
    测试用户输入 2 时，是否进入 modify_filter。

    modify_filter（修改过滤条件）：
    表示放宽或修改检索条件，以获取更多搜索结果。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "user_feedback": "2",
    }

    result = route_after_ask_user(
        state,
    )

    assert result == "modify_filter"


def test_route_after_ask_user_should_return_generate_when_feedback_is_3():
    """
    测试用户输入 3 时，是否直接进入 generate。

    generate（生成）：
    表示不再继续检索或修改条件，直接生成最终答案。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "user_feedback": "3",
    }

    result = route_after_ask_user(
        state,
    )

    assert result == "generate"


def test_route_after_ask_user_should_strip_feedback_before_routing():
    """
    测试 user_feedback 前后有空格时，是否会先 strip 再判断。

    strip（去除空白）：
    去掉字符串前后的空格、换行符等空白字符。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "user_feedback": " 1 ",
    }

    result = route_after_ask_user(
        state,
    )

    assert result == "retry"


def test_route_after_ask_user_should_fallback_to_generate_when_feedback_unknown():
    """
    测试用户输入未知内容时，是否兜底进入 generate。

    fallback（兜底）：
    当输入不符合预期时，选择一个安全默认分支。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "user_feedback": "a",
    }

    result = route_after_ask_user(
        state,
    )

    assert result == "generate"


def test_route_after_ask_user_should_fallback_to_generate_when_feedback_missing():
    """
    测试缺少 user_feedback 字段时，是否兜底进入 generate。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {}

    result = route_after_ask_user(
        state,
    )

    assert result == "generate"


def test_route_after_retrieval_should_return_good_when_retrieval_ok_true():
    """
    测试 retrieval_ok=True 时，是否返回 good。

    retrieval_ok（检索是否成功）：
    表示当前检索结果是否满足生成答案的要求。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "retrieval_ok": True,
    }

    result = route_after_retrieval(
        state,
    )

    assert result == "good"


def test_route_after_retrieval_should_return_retry_when_retrieval_ok_false():
    """
    测试 retrieval_ok=False 时，是否返回 retry。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "retrieval_ok": False,
    }

    result = route_after_retrieval(
        state,
    )

    assert result == "retry"


def test_route_after_retrieval_should_return_retry_when_retrieval_ok_missing():
    """
    测试缺少 retrieval_ok 字段时，是否安全兜底返回 retry。

    fallback（兜底）：
    当 state 中缺少预期字段时，不让 Graph 路由函数直接报错，
    而是返回一个安全的默认分支。

    retry（重试）：
    表示当前检索状态不明确或检索未通过时，重新进入重试流程。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state = {}

    result = route_after_retrieval(
        state,
    )

    assert result == "retry"

class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 agent、node、tool、phase 等运行状态。
    """

    def __init__(self):
        """
        初始化假状态作用域。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.current_agent = None

    def set_agent(self, agent_name):
        """
        设置当前 agent。

        参数：
            agent_name：
                当前路由选中的 agent 名称。

        返回值：
            None：无业务返回值。
        """

        self.current_agent = agent_name


class FakeTimelineScope:
    """
    测试用假 TimelineScope。

    TimelineScope（时间线作用域）：
    用于记录运行过程中的事件。
    """

    def __init__(self):
        """
        初始化假时间线作用域。

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
                事件类型，例如 route。

            name：
                事件名称，例如 route_after_semantic。

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
    表示一次请求执行过程中的上下文对象。
    """

    def __init__(self):
        """
        初始化假运行时上下文。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self):
        """
        返回假 StateScope。

        参数：
            无。

        返回值：
            FakeStateScope：测试用状态作用域。
        """

        return self.state_scope

    def timeline(self):
        """
        返回假 TimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope：测试用时间线作用域。
        """

        return self.timeline_scope


class FakeRuntimeCtxVar:
    """
    测试用假 runtime_ctx。

    runtime_ctx：
    项目中用于保存当前 RuntimeContext 的上下文变量包装对象。
    """

    def __init__(self, ctx):
        """
        初始化假 runtime_ctx。

        参数：
            ctx：
                假 RuntimeContext 对象。

        返回值：
            None：构造函数无返回值。
        """

        self.ctx = ctx

    def get(self):
        """
        获取当前 RuntimeContext。

        参数：
            无。

        返回值：
            FakeRuntimeContext：当前测试上下文。
        """

        return self.ctx


def test_route_after_semantic_should_return_valid_next_agent(monkeypatch):
    """
    测试 next_agent 合法时，route_after_semantic 是否返回该 agent。

    monkeypatch（猴子补丁）：
    pytest 提供的测试工具，可以在测试期间临时替换模块变量或函数。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {
        "next_agent": "exact_agent",
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "dog_knowledge_agent"
    assert fake_ctx.state_scope.current_agent == "dog_knowledge_agent"
    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "route",
            "name": "route_after_root_supervisor",
            "metadata": None,
        }
    ]


def test_route_after_semantic_should_fallback_to_general_agent_when_next_agent_missing(monkeypatch):
    """
    测试缺少 next_agent 时，是否兜底到 general_agent。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {}

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "general_agent"
    assert fake_ctx.state_scope.current_agent == "general_agent"


def test_route_after_semantic_should_fallback_to_general_agent_when_next_agent_empty(monkeypatch):
    """
    测试 next_agent 为空字符串时，是否兜底到 general_agent。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {
        "next_agent": "",
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "general_agent"
    assert fake_ctx.state_scope.current_agent == "general_agent"


def test_route_after_semantic_should_strip_next_agent(monkeypatch):
    """
    测试 next_agent 前后有空格时，是否会 strip 后再判断。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {
        "next_agent": " exact_agent ",
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "dog_knowledge_agent"
    assert fake_ctx.state_scope.current_agent == "dog_knowledge_agent"


def test_route_after_semantic_should_fallback_to_general_agent_when_next_agent_invalid(monkeypatch):
    """
    测试 next_agent 非法时，是否兜底到 general_agent。

    invalid route（非法路由）：
    LangGraph conditional_edges 中不存在的路由 key。
    如果不兜底，可能会导致 LangGraph 抛出 KeyError。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {
        "next_agent": "unknown_agent",
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "general_agent"
    assert fake_ctx.state_scope.current_agent == "general_agent"


@pytest.mark.parametrize(
    "next_agent, expected_route",
    [
        ("dog_knowledge_agent", "dog_knowledge_agent"),
        ("recommendation_agent", "dog_knowledge_agent"),
        ("exact_agent", "dog_knowledge_agent"),
        ("exact_search_agent", "dog_knowledge_agent"),
        ("general_agent", "general_agent"),
        ("tool_agent", "general_agent"),
        ("FINISH", "FINISH"),
    ],
)
def test_route_after_semantic_should_allow_all_valid_routes(
    monkeypatch,
    next_agent,
    expected_route,
):
    """
    测试 route_after_semantic 是否允许所有合法路由。

    parametrize（参数化测试）：
    pytest 提供的能力，可以用多组输入重复执行同一个测试逻辑。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

        next_agent：
            当前参数化传入的 agent 名称。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    fake_ctx = FakeRuntimeContext()

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        FakeRuntimeCtxVar(
            fake_ctx,
        ),
    )

    state = {
        "next_agent": next_agent,
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == expected_route
    assert fake_ctx.state_scope.current_agent == expected_route


def test_route_after_semantic_should_still_return_route_when_runtime_context_failed(monkeypatch):
    """
    测试 runtime context 写入失败时，route_after_semantic 是否仍然返回路由。

    这个测试对应你代码里的 try / except。
    即使 runtime_ctx.get().state().set_agent(...) 失败，
    route 函数也不能影响 LangGraph 路由结果。

    参数：
        monkeypatch：
            pytest fixture，用于替换 semantic_route_module.runtime_ctx。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    class BrokenRuntimeCtxVar:
        """
        测试用损坏 runtime_ctx。

        用于模拟 runtime_ctx.get() 抛出异常。
        """

        def get(self):
            """
            模拟获取 RuntimeContext 失败。

            参数：
                无。

            返回值：
                None：本方法会抛出 RuntimeError。
            """

            raise RuntimeError(
                "runtime context failed",
            )

    monkeypatch.setattr(
        root_route_module,
        "runtime_ctx",
        BrokenRuntimeCtxVar(),
    )

    state = {
        "next_agent": "exact_agent",
    }

    result = semantic_route_module.route_after_semantic(
        state,
    )

    assert result == "dog_knowledge_agent"
