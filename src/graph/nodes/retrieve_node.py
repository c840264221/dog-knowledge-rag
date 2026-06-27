import json
from typing import Any

from src.graph.states.dog_state import DogState
from src.logger import logger
from src.rag.adapters import (
    rag_context_to_documents
)
from src.rag.query_builders import (
    build_rag_query_from_state
)
from src.runtime.context import runtime_ctx
from src.runtime.scopes.retrieval_scope import RetrievalScope

# 导入rag召回诊断工具
from src.rag.observation.diagnostics import (
    build_retrieval_diagnostics,
    build_retrieval_quality_log_summary,
)


def build_retrieve_node(
        retriever_provider=None,
        checkpoint_provider=None,
):
    """
    构建新版 RAG retrieve_node。

    功能：
        使用闭包方式注入 RetrieverProvider 和 CheckpointProvider，
        让 exact_search_agent 的 retrieve 节点接入新版 RAG 召回链路。

        v1.5 当前职责：
        1. 从 state 构建 RagQuery。
        2. 调用 MetadataFilterRetriever.async_retrieve 执行召回。
        3. 得到 RagContext。
        4. 将 RagContext 适配成旧版 docs。
        5. 返回 rag_query、rag_context、docs、retrieval_ok。

    技术名词：
        RetrieverProvider：
            召回器提供者，统一管理 DogQueryFilterParser 和 MetadataFilterRetriever。

        RagQuery：
            新版 RAG 查询对象。

        RagContext：
            新版 RAG 检索上下文。

        Backward Compatibility：
            向后兼容。这里表示继续返回 docs 字段，避免旧节点断掉。

    参数：
        retriever_provider:
            RetrieverProvider 实例。
            中文释义：用于获取 parser 和 retriever。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint。

    返回值：
        callable:
            返回一个 async retrieve_node 函数，供 LangGraph 注册使用。
    """

    async def retrieve_node(
            state: DogState
    ) -> dict[str, Any]:
        """
        执行新版 RAG 召回节点。

        功能：
            1. 校验 retriever_provider 是否存在。
            2. 调用 _execute_new_rag_retrieve 执行核心召回逻辑。
            3. 返回需要合并进 DogState 的状态更新。

        参数：
            state:
                当前 DogState。

        返回值：
            dict[str, Any]:
                包含 rag_query、rag_context、docs、retrieval_ok。
        """

        if retriever_provider is None:
            raise RuntimeError(
                "build_retrieve_node 缺少 retriever_provider，"
                "请确认 GraphRuntimeService 已注入 container.get('retriever')。"
            )

        return await execute_new_rag_retrieve(
            state=state,
            retriever_provider=retriever_provider,
            checkpoint_provider=checkpoint_provider,
        )

    return retrieve_node


async def retrieve_node(
        state: DogState
) -> dict[str, Any]:
    """
    旧版兼容 retrieve_node。

    功能：
        保留旧函数名，避免其他旧代码仍然直接导入 retrieve_node 时立刻报错。

        注意：
            v1.5 新代码推荐使用 build_retrieve_node 注入 Provider。
            这个函数只是兼容入口，不建议新代码继续依赖它。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            新版 RAG 召回结果和旧版 docs 兼容字段。
    """

    from src.runtime.container.init import container

    retriever_provider = container.get(
        "retriever"
    )

    checkpoint_provider = container.get(
        "checkpoint"
    )

    return await execute_new_rag_retrieve(
        state=state,
        retriever_provider=retriever_provider,
        checkpoint_provider=checkpoint_provider,
    )


async def execute_new_rag_retrieve(
        state: DogState,
        retriever_provider,
        checkpoint_provider=None,
) -> dict[str, Any]:
    """
    执行新版 RAG 召回核心流程。

    功能：
        这是 retrieve_node 的核心运行逻辑。

        具体步骤：
        1. 设置 runtime 当前节点。
        2. 从 RetrieverProvider 获取 DogQueryFilterParser。
        3. 从 RetrieverProvider 获取 MetadataFilterRetriever。
        4. 使用 build_rag_query_from_state 构建 RagQuery。
        5. 调用 retriever.async_retrieve 执行召回。
        6. 使用 rag_context_to_documents 转换旧 docs。
        7. 写入 RetrievalScope。
        8. 保存 checkpoint。
        9. 返回状态更新。

    参数：
        state:
            当前 DogState。

        retriever_provider:
            RetrieverProvider 实例。

        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        dict[str, Any]:
            包含 rag_query、rag_context、docs、retrieval_ok。
    """

    runtime_context = runtime_ctx.get()

    runtime_context.state().set_node(
        "retrieve_node"
    )

    runtime_context.timeline().add_event(
        event_type="node",
        name="retrieve_node"
    )

    question = str(
        state.get(
            "question",
            ""
        )
        or ""
    ).strip()

    if not question:
        raise ValueError(
            "retrieve_node 缺少 question"
        )

    logger.info(
        f"进入新版 RAG retrieve_node，question={question}"
    )

    parser = retriever_provider.dog_query_filter_parser

    retriever = retriever_provider.metadata_filter_retriever

    rag_query = build_rag_query_from_state(
        state=state,
        parser=parser,
    )

    logger.debug(
        f"新版 RAG RagQuery 构建完成: {json.dumps(rag_query.model_dump(), indent=4, ensure_ascii=False)}"
    )

    rag_context = await retriever.async_retrieve(
        query=rag_query
    )

    docs = rag_context_to_documents(
        rag_context=rag_context
    )

    runtime_context.service(
        RetrievalScope
    ).set_docs(
        docs
    )

    logger.info(
        "新版 RAG retrieve_node 执行完成，"
        f"status={rag_context.status}, "
        f"chunks={len(rag_context.chunks)}, "
        f"docs={len(docs)}"
    )

    if checkpoint_provider is not None:
        checkpoint_provider.manager.save_checkpoint()

    rag_query_dump = rag_query.model_dump()

    rag_context_dump = rag_context.model_dump()

    filters = rag_query_dump.get(
        "filters",
        {},
    ) or {}

    intent = rag_query_dump.get(
        "intent",
        state.get(
            "intent",
            "dog_info",
        ),
    )

    top_k = rag_query_dump.get(
        "top_k",
        state.get(
            "top_k",
            5,
        ),
    )

    diagnostic_state = {
        **state,
        "rag_query": rag_query_dump,
        "rag_context": rag_context_dump,
        "filters": filters,
        "intent": intent,
        "top_k": top_k,
    }

    retrieval_quality = build_retrieval_diagnostics(
        state=diagnostic_state,
        stage="retrieve",
        docs=docs,
        rag_context=rag_context,
        failure_type="",
        decision="evaluate",
        reason=(
            "retrieve_node 完成初始召回，"
            f"rag_context.status={rag_context.status}，"
            f"retrieved_chunks={len(rag_context.chunks)}，"
            f"docs={len(docs)}。"
        ),
    )

    logger.debug(
        "retrieve_node 检索诊断摘要: "
        f"{json.dumps(
            build_retrieval_quality_log_summary(
                retrieval_quality=retrieval_quality,
            ),
            ensure_ascii=False,
        )}"
    )
    return {
        # 新版 RAG 结构化字段
        "rag_query": rag_query_dump,
        "rag_context": rag_context_dump,

        # v1.5 filters 字段补齐重点：
        # 将 parser / query_builder 最终生成的 filters 回写到 DogState。
        # 这样后面的旧节点、debug 日志、evaluate、retry、generate 都可以继续读取 state["filters"]。
        "filters": filters,

        # 兼容旧链路字段
        "intent": intent,

        "top_k": top_k,

        "docs": docs,

        # 注意：
        # retrieve_node 只负责召回，不负责判断召回质量。
        # 所以这里不能直接设置 retrieval_ok=True。
        "retrieval_ok": False,

        # 表示当前 rag_context 还没有经过 evaluate_node 质量评估。
        "retrieval_evaluated": False,

        # retrieve 阶段先写入基础诊断信息。
        # evaluate_node 后续会在这个基础上 merge 质量评估结果。
        "retrieval_quality": retrieval_quality,

        # retrieve 阶段还没有失败类型判断。
        "retrieval_failure_type": "",
    }