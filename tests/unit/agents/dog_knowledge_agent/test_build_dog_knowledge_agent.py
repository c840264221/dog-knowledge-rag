from src.agents.dog_knowledge_agent.agent import (
    build_dog_knowledge_agent,
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

    pass


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