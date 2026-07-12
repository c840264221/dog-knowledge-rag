import pytest

from src.agents.dog_knowledge_agent import agent as dog_knowledge_agent_module
from src.agents.dog_knowledge_agent.agent import (
    build_dog_knowledge_agent,
)
from src.graph.nodes.generate_node import (
    resolve_memory_text,
)


class FakeLLMProvider:
    """
    测试用 LLMProvider 假对象。

    功能：
        用于 build_dog_knowledge_agent 的构建级 smoke test。
        当前测试只验证 graph.compile() 是否成功，
        不会真正调用 safe_ainvoke，也不会访问 main_llm。

    参数：
        无。

    返回值：
        FakeLLMProvider 实例。

    专业名词：
        Fake：
            假对象。用于测试时替代真实外部依赖。

        LLMProvider：
            大语言模型提供者。真实项目中负责提供 main_llm、safe_ainvoke 等能力。
    """

    pass


class FakeMemoryProvider:
    """
    测试用 MemoryProvider 假对象。

    功能：
        用于替代真实记忆服务。
        当前测试只验证图构建，不会真实召回用户记忆。

    参数：
        无。

    返回值：
        FakeMemoryProvider 实例。
    """

    semantic_recall = object()


class FakeRetrieverProvider:
    """
    测试用 RetrieverProvider 假对象。

    功能：
        用于替代真实 RAG 检索器提供者。
        当前测试只验证 build_retrieve_node 能否被正常构建，
        不会真实访问 Chroma 或 Retriever。

    参数：
        无。

    返回值：
        FakeRetrieverProvider 实例。

    专业名词：
        Retriever：
            检索器。用于从知识库中召回相关文档或 chunk。

        Provider：
            提供者。用于统一管理和注入服务能力。
    """

    pass


class FakeRerankerProvider:
    """
    测试用 RerankerProvider 假对象。

    功能：
        用于替代真实 reranker provider。
        当前测试只验证 build_rerank_node 能否被正常构建，
        不会真实执行重排序。

    参数：
        无。

    返回值：
        FakeRerankerProvider 实例。

    专业名词：
        Reranker：
            重排序器。用于对 Retriever 召回结果进行二次排序。
    """

    pass


class FakeCompiledGraph:
    """
    测试用假编译图。

    功能：
        保存 FakeStateGraph，方便测试检查节点和边是否注册正确。

    参数：
        graph:
            FakeStateGraph 实例。

    返回值：
        FakeCompiledGraph 实例。
    """

    def __init__(
            self,
            graph,
    ):
        self.graph = graph


class FakeStateGraph:
    """
    测试用假 StateGraph。

    功能：
        记录 DogKnowledgeAgent 构图时注册的节点和边。
        这样测试可以验证图结构，不需要真正运行 LLM、RAG 或 LangGraph。

    参数：
        state_schema:
            图使用的状态结构。

    返回值：
        FakeStateGraph 实例。
    """

    latest_instance = None

    def __init__(
            self,
            state_schema,
    ):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.conditional_entry_point = None
        self.conditional_edges = []
        self.compile_called = False
        FakeStateGraph.latest_instance = self

    def add_node(
            self,
            name,
            node,
    ):
        """
        记录节点注册。

        参数：
            name:
                节点名称。

            node:
                节点函数。

        返回值：
            None。
        """

        self.nodes[name] = node

    def add_edge(
            self,
            start,
            end,
    ):
        """
        记录普通边注册。

        参数：
            start:
                起始节点名称。

            end:
                目标节点名称。

        返回值：
            None。
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
        记录条件边注册。

        参数：
            source:
                条件边起始节点。

            path:
                路由函数。

            path_map:
                路由结果到节点名称的映射。

        返回值：
            None。
        """

        self.conditional_edges.append(
            {
                "source": source,
                "path": path,
                "path_map": path_map,
            }
        )

    def set_conditional_entry_point(
            self,
            path,
            path_map,
    ):
        """
        记录条件入口注册。

        参数：
            path:
                入口路由函数。

            path_map:
                路由结果到入口节点名称的映射。

        返回值：
            None。
        """

        self.conditional_entry_point = {
            "path": path,
            "path_map": path_map,
        }

    def compile(self):
        """
        模拟 LangGraph compile。

        参数：
            无。

        返回值：
            FakeCompiledGraph:
                假编译图对象。
        """

        self.compile_called = True
        return FakeCompiledGraph(self)


def test_build_dog_knowledge_agent_success():
    """
    测试 dog_knowledge_agent 可以正常构建。

    功能：
        验证 build_dog_knowledge_agent 在传入必要 provider 后，
        可以成功完成 LangGraph compile。

        这个测试属于 smoke test，不验证真实业务执行。
        它主要用于发现：
        1. import 路径错误。
        2. 节点注册错误。
        3. edge 连接错误。
        4. 条件路由映射错误。
        5. provider 参数漏传错误。

    参数：
        无。

    返回值：
        无。
    """

    graph = build_dog_knowledge_agent(
        llm_provider=FakeLLMProvider(),
        memory_provider=FakeMemoryProvider(),
        checkpoint_provider=None,
        retriever_provider=FakeRetrieverProvider(),
        reranker_provider=FakeRerankerProvider(),
    )

    assert graph is not None


def test_build_dog_knowledge_agent_should_register_layer_contract_nodes(
        monkeypatch,
) -> None:
    """
    测试 DogKnowledgeAgent 构图时注册 V1.7.4 分层契约节点。

    功能：
        验证 DogKnowledgeAgent 内部已经接入：
        1. query_layer_output
        2. retrieve -> query_layer_output -> evaluate
        3. evaluate 的 rerank 分支先进入 retrieval_layer_output
        4. retrieval_layer_output -> rerank
        5. rerank -> memory_retrieve -> generate
        6. generate -> generation_layer_output
        7. generation_layer_output -> fallback_layer_output
        8. fallback_layer_output 后面的分层收敛链路：
           legacy_state_to_layer_outputs
           aggregate_layer_outputs
           finalize_answer

        其中 fallback_layer_output 后面的分层收敛链路包括：
        1. legacy_state_to_layer_outputs
        2. aggregate_layer_outputs
        3. finalize_answer

    参数：
        monkeypatch:
            pytest 提供的临时替换工具。

    返回值：
        None。
    """

    monkeypatch.setattr(
        dog_knowledge_agent_module,
        "StateGraph",
        FakeStateGraph,
    )

    compiled_graph = build_dog_knowledge_agent(
        llm_provider=FakeLLMProvider(),
        memory_provider=FakeMemoryProvider(),
        checkpoint_provider=None,
        retriever_provider=FakeRetrieverProvider(),
        reranker_provider=FakeRerankerProvider(),
    )

    graph = compiled_graph.graph

    assert graph.compile_called is True
    assert "query_layer_output" in graph.nodes
    assert "retrieval_layer_output" in graph.nodes
    assert "memory_retrieve" in graph.nodes
    assert "generation_layer_output" in graph.nodes
    assert "fallback_layer_output" in graph.nodes
    assert "legacy_state_to_layer_outputs" in graph.nodes
    assert "aggregate_layer_outputs" in graph.nodes
    assert {
        "start": "retrieve",
        "end": "query_layer_output",
    } in graph.edges
    assert {
        "start": "query_layer_output",
        "end": "evaluate",
    } in graph.edges
    evaluate_edges = [
        item
        for item in graph.conditional_edges
        if item["source"] == "evaluate"
    ]
    assert evaluate_edges[0]["path_map"]["rerank"] == "retrieval_layer_output"
    assert evaluate_edges[0]["path_map"]["retry"] == "retry"
    assert evaluate_edges[0]["path_map"]["ask_user"] == "ask_user"
    assert evaluate_edges[0]["path_map"]["generate"] == "memory_retrieve"
    assert {
        "start": "retrieval_layer_output",
        "end": "rerank",
    } in graph.edges
    assert {
        "start": "rerank",
        "end": "memory_retrieve",
    } in graph.edges
    assert {
        "start": "memory_retrieve",
        "end": "generate",
    } in graph.edges
    ask_user_edges = [
        item
        for item in graph.conditional_edges
        if item["source"] == "ask_user"
    ]
    assert ask_user_edges[0]["path_map"]["generate"] == "memory_retrieve"
    assert {
        "start": "generate",
        "end": "generation_layer_output",
    } in graph.edges
    assert {
        "start": "generation_layer_output",
        "end": "fallback_layer_output",
    } in graph.edges
    assert {
        "start": "fallback_layer_output",
        "end": "legacy_state_to_layer_outputs",
    } in graph.edges
    assert {
        "start": "legacy_state_to_layer_outputs",
        "end": "aggregate_layer_outputs",
    } in graph.edges
    assert {
        "start": "aggregate_layer_outputs",
        "end": "finalize_answer",
    } in graph.edges


@pytest.mark.asyncio
async def test_resolve_memory_text_should_prefer_state_memory_context() -> None:
    """
    测试答案生成前优先复用 state 中的记忆上下文。

    功能：
        当 memory_retrieve 节点已经写入 memory_context 时，
        resolve_memory_text 直接返回该内容，不再通过 MemoryProvider
        发起第二次记忆召回。

    参数：
        无。

    返回值：
        None。
    """

    result = await resolve_memory_text(
        user_id="user_001",
        question="金毛的性格怎么样？",
        memory_provider=object(),
        memory_context="用户喜欢金毛寻回犬。",
    )

    assert result == "用户喜欢金毛寻回犬。"
