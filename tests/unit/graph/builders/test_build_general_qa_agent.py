"""
build_general_qa_agent 单元测试。

Graph Builder（图构建器）：
负责创建 LangGraph StateGraph，注册 node，添加 edge，
配置 conditional_edges，并最终 compile 成可运行 graph。

本测试重点：
1. 检查 llm_provider 必填
2. 检查无 memory_provider 时入口是 supervisor
3. 检查有 memory_provider + checkpoint_provider 时入口是 memory_retrieve
4. 检查核心节点是否全部注册
5. 检查 conditional_edges 的 route key 是否完整
6. 检查 graph.compile 是否被调用
"""

import pytest
from langgraph.graph import END

from src.agents.general_qa_agent.agent import (
    build_general_qa_agent,
)


class FakeCompiledGraph:
    """
    测试用假 CompiledGraph。

    CompiledGraph（已编译图）：
    LangGraph compile 后返回的可运行图对象。
    """

    def __init__(self, graph):
        """
        初始化假编译图。

        参数：
            graph：
                FakeStateGraph 实例。

        返回值：
            None：构造函数无返回值。
        """

        self.graph = graph


class FakeStateGraph:
    """
    测试用假 StateGraph。

    StateGraph（状态图）：
    LangGraph 中用于注册节点、边和条件边的图构建对象。

    这里用 fake 版本记录 add_node、add_edge、add_conditional_edges 等调用，
    避免测试依赖 LangGraph 内部实现细节。
    """

    latest_instance = None

    def __init__(self, state_schema):
        """
        初始化假 StateGraph。

        参数：
            state_schema：
                Graph 使用的状态类型，例如 DogState。

        返回值：
            None：构造函数无返回值。
        """

        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.entry_point = None
        self.compile_called = False

        FakeStateGraph.latest_instance = self

    def add_node(self, name, node):
        """
        注册 node。

        参数：
            name：
                节点名称。

            node：
                节点函数或 Runnable。

        返回值：
            None：无业务返回值。
        """

        self.nodes[name] = node

    def add_edge(self, start, end):
        """
        添加普通边。

        参数：
            start：
                起始节点名称。

            end：
                结束节点名称。

        返回值：
            None：无业务返回值。
        """

        self.edges.append(
            {
                "start": start,
                "end": end,
            }
        )

    def add_conditional_edges(
        self,
        source,
        path,
        path_map,
    ):
        """
        添加条件边。

        参数：
            source：
                条件边起始节点名称。

            path：
                route 函数。

            path_map：
                route 返回值到目标节点的映射。

        返回值：
            None：无业务返回值。
        """

        self.conditional_edges.append(
            {
                "source": source,
                "path": path,
                "path_map": path_map,
            }
        )

    def set_entry_point(self, name):
        """
        设置入口节点。

        参数：
            name：
                Graph 入口节点名称。

        返回值：
            None：无业务返回值。
        """

        self.entry_point = name

    def compile(self):
        """
        模拟编译 graph。

        参数：
            无。

        返回值：
            FakeCompiledGraph：
                假编译图对象。
        """

        self.compile_called = True

        return FakeCompiledGraph(
            self,
        )


class FakeLLMProvider:
    """
    测试用假 LLMProvider。

    LLMProvider（大语言模型提供者）：
    用于注入 build_general_qa_agent。
    """

    def __init__(self):
        """
        初始化假 LLMProvider。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.main_llm = "fake_main_llm"
        self.backup_llm = "fake_backup_llm"

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response,
    ):
        """
        模拟 LLMProvider.safe_ainvoke。

        参数：
            llm：
                模型对象。

            prompt：
                提示词。

            fallback_response：
                兜底响应。

        返回值：
            str：
                假响应。
        """

        return "fake response"


class FakeMemoryProvider:
    """
    测试用假 MemoryProvider。

    MemoryProvider（记忆提供者）：
    用于提供 semantic_recall 能力。
    """

    async def semantic_recall(
        self,
        *args,
        **kwargs,
    ):
        """
        模拟语义记忆召回。

        参数：
            *args：
                位置参数。

            **kwargs：
                关键字参数。

        返回值：
            list：
                假记忆召回结果。
        """

        return []


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于保存 checkpoint。
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


class FakeCheckpointProvider:
    """
    测试用假 CheckpointProvider。

    CheckpointProvider（检查点提供者）：
    对外暴露 manager。
    """

    def __init__(self):
        """
        初始化假 CheckpointProvider。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.manager = FakeCheckpointManager()


def patch_state_graph(
    monkeypatch,
):
    """
    替换 agent 模块中的 StateGraph。

    功能：
        build_general_qa_agent 文件里是通过 from langgraph.graph import StateGraph 引入的。
        所以测试时需要 monkeypatch 当前 agent 模块里的 StateGraph 名称，
        而不是 monkeypatch langgraph.graph.StateGraph。

    参数：
        monkeypatch：
            pytest 提供的 monkeypatch fixture（临时替换工具）。

    返回值：
        module：
            被 patch 的 agent 模块。
    """

    import src.agents.general_qa_agent.agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "StateGraph",
        FakeStateGraph,
    )

    return agent_module


def test_build_general_qa_agent_should_raise_error_when_llm_provider_missing():
    """
    测试缺少 llm_provider 时，是否抛出 ValueError。

    参数：
        无。

    返回值：
        None：
            pytest 会根据异常断言判断测试是否通过。
    """

    with pytest.raises(
        ValueError,
        match="缺少 llm_provider",
    ):
        build_general_qa_agent(
            llm_provider=None,
        )


def test_build_general_qa_agent_without_memory_should_use_supervisor_entry(
    monkeypatch,
):
    """
    测试未传 memory_provider 时，入口是否是 supervisor。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        memory_provider=None,
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    assert graph.entry_point == "supervisor"
    assert graph.compile_called is True

    assert "memory_retrieve" not in graph.nodes

    assert set(
        graph.nodes.keys()
    ) == {
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
        "supervisor",
    }


def test_build_general_qa_agent_with_memory_should_use_memory_retrieve_entry(
    monkeypatch,
):
    """
    测试传入 memory_provider 和 checkpoint_provider 时，入口是否是 memory_retrieve。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        memory_provider=FakeMemoryProvider(),
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    assert graph.entry_point == "memory_retrieve"
    assert graph.compile_called is True

    assert set(
        graph.nodes.keys()
    ) == {
        "memory_retrieve",
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
        "supervisor",
    }

    assert {
        "start": "memory_retrieve",
        "end": "supervisor",
    } in graph.edges


def test_build_general_qa_agent_should_register_ask_confirm_conditional_edges(
    monkeypatch,
):
    """
    测试 ask_confirm 的 conditional_edges 是否正确注册。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    ask_confirm_edges = [
        item
        for item in graph.conditional_edges
        if item["source"] == "ask_confirm"
    ]

    assert len(
        ask_confirm_edges
    ) == 1

    assert ask_confirm_edges[0]["path_map"] == {
        "call_tool": "execute_tool",
        "no_call_tool": "answer_gen",
    }


def test_build_general_qa_agent_should_register_execute_tool_conditional_edges(
    monkeypatch,
):
    """
    测试 execute_tool 的 conditional_edges 是否正确注册。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    execute_tool_edges = [
        item
        for item in graph.conditional_edges
        if item["source"] == "execute_tool"
    ]

    assert len(
        execute_tool_edges
    ) == 1

    assert execute_tool_edges[0]["path_map"] == {
        "ask_confirm": "ask_confirm",
        "answer_gen": "answer_gen",
    }


def test_build_general_qa_agent_should_register_supervisor_conditional_edges(
    monkeypatch,
):
    """
    测试 supervisor 的 conditional_edges 是否正确注册。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    supervisor_edges = [
        item
        for item in graph.conditional_edges
        if item["source"] == "supervisor"
    ]

    assert len(
        supervisor_edges
    ) == 1

    assert set(
        supervisor_edges[0]["path_map"].keys()
    ) == {
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
        "finish",
    }


def test_build_general_qa_agent_should_end_after_answer_generation(
    monkeypatch,
):
    """
    测试 answer_gen 是否直接结束，其他普通 worker 是否回到 supervisor。

    功能：
        answer_gen 已经生成最终回答，不应再次进入 LLM Supervisor，否则可能
        被重复路由回 answer_gen；tool_parse 等非终态 worker 仍需返回调度器。

    参数：
        monkeypatch：
            pytest 提供的临时替换工具。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    patch_state_graph(
        monkeypatch,
    )

    compiled_graph = build_general_qa_agent(
        llm_provider=FakeLLMProvider(),
        checkpoint_provider=FakeCheckpointProvider(),
    )

    graph = compiled_graph.graph

    assert {
        "start": "tool_parse",
        "end": "supervisor",
    } in graph.edges

    assert {
        "start": "answer_gen",
        "end": "supervisor",
    } not in graph.edges

    assert {
        "start": "answer_gen",
        "end": END,
    } in graph.edges

    assert {
        "start": "ask_confirm",
        "end": "supervisor",
    } not in graph.edges

    assert {
        "start": "execute_tool",
        "end": "supervisor",
    } not in graph.edges
