from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document

from src.graph.states.dog_state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


# 导入rag诊断工具
from src.rag.observation.diagnostics import (
    build_retrieval_diagnostics,
    merge_retrieval_diagnostics,
    build_retrieval_quality_log_summary,
)


RetrievedChunkDict = dict[str, Any]
RagContextDict = dict[str, Any]


def build_rerank_node(
        reranker_provider=None,
        checkpoint_provider=None,
):
    """
    构建新版 RAG rerank_node。

    功能：
        使用 Provider Injection（提供者注入）的方式构建 rerank 节点。

        v1.5 设计目标：
        1. 优先对 state["rag_context"]["chunks"] 做 rerank。
        2. 同步更新 state["rag_context"]["context_text"]。
        3. 同步更新 state["docs"]，兼容旧节点。
        4. 避免节点内部直接 import container。
        5. 为后续企业级 RAG 评估保留 rerank_score / final_score。

    技术名词：
        Rerank：
            重排。对 Retriever 初步召回的结果进行二次排序。

        Retriever：
            召回器。负责从向量数据库中取回候选 chunks。

        Cross-Encoder：
            交叉编码模型。把 query 和 document 拼在一起输入模型，
            输出一个相关性分数，通常比普通向量召回更精确。

        Provider Injection：
            提供者注入。依赖对象从外部传入，节点内部不直接 import container。

    参数：
        reranker_provider:
            RerankerProvider 实例。
            中文释义：用于提供 reranker 模型。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint。

    返回值：
        callable:
            返回 LangGraph 可注册的 rerank_node 函数。
    """

    def rerank_node(
            state: DogState,
    ) -> dict[str, Any]:
        """
        执行 RAG 重排节点。

        功能：
            对 retrieve_node 召回出来的结果进行 rerank。

            优先级：
            1. 如果 state["rag_context"]["chunks"] 存在，则重排 chunks。
            2. 如果 rag_context 不存在，则回退到旧版 state["docs"]。
            3. 重排后同时返回 rag_context 和 docs，保持新旧链路兼容。

        参数：
            state:
                当前 DogState。
                中文释义：包含 question、rag_context、docs、top_k 等字段。

        返回值：
            dict[str, Any]:
                需要合并回 DogState 的字段。
                包含：
                - rag_context：重排后的新版 RAG 上下文。
                - docs：重排后的旧版 Document 列表。
        """

        return execute_rerank(
            state=state,
            reranker_provider=reranker_provider,
            checkpoint_provider=checkpoint_provider,
        )

    return rerank_node


def rerank_node(
        state: DogState,
) -> dict[str, Any]:
    """
    旧版兼容 rerank_node。

    功能：
        保留旧函数名，避免旧代码直接导入 rerank_node 时报错。

        注意：
            v1.5 新代码推荐使用 build_rerank_node 注入 reranker_provider。
            当前函数只是兼容入口。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            重排后的状态更新。
    """

    from src.runtime.container.init import container

    reranker_provider = container.get(
        "reranker"
    )

    checkpoint_provider = container.get(
        "checkpoint"
    )

    return execute_rerank(
        state=state,
        reranker_provider=reranker_provider,
        checkpoint_provider=checkpoint_provider,
    )


def execute_rerank(
        state: DogState,
        reranker_provider=None,
        checkpoint_provider=None,
) -> dict[str, Any]:
    """
    执行 rerank 核心逻辑。

    功能：
        根据当前 state 执行重排。

        执行流程：
        1. 设置 runtime 当前节点。
        2. 读取 question。
        3. 优先读取 rag_context.chunks。
        4. 如果 chunks 存在，则对 chunks 做 rerank。
        5. 如果 chunks 不存在，则回退到 docs rerank。
        6. 返回更新后的 rag_context 和 docs。
        7. 保存 checkpoint。

    参数：
        state:
            当前 DogState。

        reranker_provider:
            RerankerProvider 实例。

        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        dict[str, Any]:
            状态更新字典。
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "rerank_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="rerank_node",
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
            "rerank_node 缺少 question"
        )

    logger.info(
        "进入新版 rerank_node，"
        f"question={question}"
    )

    reranker_model = resolve_reranker_model(
        reranker_provider=reranker_provider
    )

    rag_context = get_rag_context_from_state(
        state=state
    )

    chunks = get_chunks_from_rag_context(
        rag_context=rag_context
    )

    if chunks:
        result = rerank_rag_context_chunks(
            question=question,
            rag_context=rag_context,
            chunks=chunks,
            reranker_model=reranker_model,
            top_k=resolve_top_k_from_state(
                state=state
            ),
        )

        reranked_chunks_count = len(
            result.get(
                "rag_context",
                {},
            ).get(
                "chunks",
                [],
            )
        )

        reranked_docs_count = len(
            result.get(
                "docs",
                [],
            )
        )

        result = attach_rerank_diagnostics(
            state=state,
            result=result,
            reason=(
                "rerank_node 完成新版 rag_context.chunks 重排序，"
                f"reranked_chunks={reranked_chunks_count}，"
                f"reranked_docs={reranked_docs_count}。"
            ),
        )

        if checkpoint_provider is not None:
            checkpoint_provider.manager.save_checkpoint()

        logger.info(
            "新版 rerank_node 执行完成，"
            f"reranked_chunks={reranked_chunks_count}, "
            f"docs={reranked_docs_count}"
        )

        return result

    docs = list(
        state.get(
            "docs",
            []
        )
        or []
    )

    result = rerank_legacy_docs(
        question=question,
        docs=docs,
        reranker_model=reranker_model,
        top_k=resolve_top_k_from_state(
            state=state
        ),
    )

    reranked_docs_count = len(
        result.get(
            "docs",
            [],
        )
    )

    result = attach_rerank_diagnostics(
        state=state,
        result=result,
        reason=(
            "rerank_node 完成旧版 docs 重排序，"
            f"reranked_docs={reranked_docs_count}。"
        ),
    )

    if checkpoint_provider is not None:
        checkpoint_provider.manager.save_checkpoint()

    logger.info(
        "旧版 docs rerank 执行完成，"
        f"docs={reranked_docs_count}"
    )

    return result

def attach_rerank_diagnostics(
        state: DogState,
        result: dict[str, Any],
        reason: str,
) -> dict[str, Any]:
    """
    给 rerank 结果附加检索诊断信息。

    功能：
        rerank_node 完成重排序后，会返回新的 rag_context。
        本函数负责基于新版 RagContext 构建 rerank 阶段 diagnostics。

        数据优先级：
        1. result["rag_context"] 是 rerank 后的新版 RAG 主结果。
        2. state["rag_context"] 是 rerank 前的新版 RAG 结果。
        3. result["docs"] / state["docs"] 只作为旧版兼容 fallback。

    参数：
        state:
            当前 DogState。

        result:
            rerank_node 返回的状态更新。
            新版链路中通常包含 result["rag_context"]。

        reason:
            rerank 阶段诊断原因。

    返回值：
        dict[str, Any]:
            附加 retrieval_quality 后的 result。
    """

    original_rag_context = state.get(
        "rag_context",
        {},
    ) or {}

    reranked_rag_context = result.get(
        "rag_context",
        {},
    ) or {}

    original_docs = list(
        state.get(
            "docs",
            [],
        )
        or []
    )

    reranked_docs = list(
        result.get(
            "docs",
            [],
        )
        or []
    )

    rerank_diagnostics = build_retrieval_diagnostics(
        state=state,
        stage="rerank",
        docs=original_docs,
        reranked_docs=reranked_docs,
        rag_context=original_rag_context,
        reranked_rag_context=reranked_rag_context,
        failure_type=str(
            state.get(
                "retrieval_failure_type",
                "",
            )
            or ""
        ),
        decision="generate",
        reason=reason,
    )

    retrieval_quality = merge_retrieval_diagnostics(
        old_diagnostics=state.get(
            "retrieval_quality",
            {},
        ),
        new_diagnostics=rerank_diagnostics,
    )

    result[
        "retrieval_quality"
    ] = retrieval_quality

    logger.debug(
        "rerank_node 检索诊断摘要: "
        f"{json.dumps(
            build_retrieval_quality_log_summary(
                retrieval_quality=retrieval_quality,
            ),
            ensure_ascii=False,
        )}"
    )

    return result


def resolve_reranker_model(
        reranker_provider=None,
) -> Any:
    """
    从 RerankerProvider 中解析 reranker 模型。

    功能：
        兼容不同 Provider 写法，尽量自动找到 reranker 模型。

        支持的常见属性 / 方法：
        1. reranker_provider.reranker
        2. reranker_provider.model
        3. reranker_provider.reranker_model
        4. reranker_provider.get_reranker()
        5. reranker_provider.get_model()

    技术名词：
        Reranker Model：
            重排模型。通常是 Cross-Encoder，用于计算 query-document 相关性。

    参数：
        reranker_provider:
            RerankerProvider 实例。

    返回值：
        Any:
            reranker 模型对象。
            需要支持 predict 方法。
    """

    if reranker_provider is None:
        raise RuntimeError(
            "rerank_node 缺少 reranker_provider"
        )

    for attr_name in (
            "reranker",
            "model",
            "reranker_model",
    ):
        if hasattr(
                reranker_provider,
                attr_name
        ):
            model = getattr(
                reranker_provider,
                attr_name
            )

            if model is not None:
                return model

    for method_name in (
            "get_reranker",
            "get_model",
    ):
        if hasattr(
                reranker_provider,
                method_name
        ):
            method = getattr(
                reranker_provider,
                method_name
            )

            if callable(
                    method
            ):
                model = method()

                if model is not None:
                    return model

    raise RuntimeError(
        "无法从 reranker_provider 中获取 reranker 模型。"
        "请检查 RerankerProvider 是否提供 reranker / model / reranker_model / get_reranker。"
    )


def get_rag_context_from_state(
        state: Mapping[str, Any],
) -> RagContextDict:
    """
    从 state 中读取 rag_context。

    功能：
        兼容 dict 和 Pydantic 对象两种情况。

        因为当前项目使用 SQLite checkpoint，
        所以 rag_context 在 state 中通常会保存成 dict。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            rag_context 字典。
            不存在时返回空 dict。
    """

    rag_context = state.get(
        "rag_context",
        {}
    ) or {}

    if isinstance(
            rag_context,
            Mapping
    ):
        return dict(
            rag_context
        )

    if hasattr(
            rag_context,
            "model_dump"
    ):
        return rag_context.model_dump()

    return {}


def get_chunks_from_rag_context(
        rag_context: RagContextDict,
) -> list[RetrievedChunkDict]:
    """
    从 rag_context 中读取 chunks。

    功能：
        读取 rag_context["chunks"]。
        如果不存在或不是 list，则返回空列表。

    参数：
        rag_context:
            RagContext 字典。

    返回值：
        list[dict[str, Any]]:
            检索结果 chunk 列表。
    """

    chunks = rag_context.get(
        "chunks",
        []
    )

    if not isinstance(
            chunks,
            list
    ):
        return []

    return [
        dict(
            chunk
        )
        for chunk in chunks
        if isinstance(
            chunk,
            Mapping
        )
    ]


def rerank_rag_context_chunks(
        question: str,
        rag_context: RagContextDict,
        chunks: list[RetrievedChunkDict],
        reranker_model: Any,
        top_k: int,
) -> dict[str, Any]:
    """
    对 rag_context.chunks 执行 rerank。

    功能：
        使用 reranker 模型对每个 retrieved chunk 打分，
        然后按 raw rerank_score 从高到低排序。

        v1.5 当前设计：
        1. rerank_score 保存模型原始分数。
           注意：Cross-Encoder 的原始分数可能是负数，这是正常现象。
        2. final_score 保存归一化后的 0~1 分数。
           这个分数更适合展示、日志和后续评估。
        3. 排序仍然使用 raw rerank_score。
           因为 raw score 的相对大小最能反映模型原始判断。
        4. context_text 中尽量使用 final_score，避免把负数直接暴露给 LLM。

    技术名词：
        Raw Score：
            原始分数。模型直接输出的分数，可能为负数。

        Normalization：
            归一化。把不同范围的分数转换到统一范围，例如 0~1。

        Final Score：
            最终分数。项目内部用于排序后展示、评估和可观测分析的分数。

    参数：
        question:
            用户问题。

        rag_context:
            当前 RagContext 字典。

        chunks:
            待重排的 retrieved chunks。

        reranker_model:
            reranker 模型。

        top_k:
            最终保留数量。

    返回值：
        dict[str, Any]:
            包含 rag_context 和 docs 的状态更新。
    """

    pairs = build_rerank_pairs_from_chunks(
        question=question,
        chunks=chunks,
    )

    raw_scores = predict_rerank_scores(
        reranker_model=reranker_model,
        pairs=pairs,
    )

    scored_items: list[tuple[RetrievedChunkDict, float]] = []

    for index, retrieved_chunk in enumerate(
            chunks
    ):
        raw_rerank_score = safe_float(
            value=raw_scores[index],
            default=0.0,
        )

        scored_items.append(
            (
                dict(
                    retrieved_chunk
                ),
                raw_rerank_score,
            )
        )

    # 注意：
    # raw rerank_score 即使是负数，也可以排序。
    # 分数越大越相关，例如 -2.1 > -5.6。
    scored_items.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    scored_items = scored_items[
        :top_k
    ]

    normalized_scores = normalize_scores_to_0_1(
        scores=[
            raw_score
            for _, raw_score in scored_items
        ]
    )

    final_chunks: list[RetrievedChunkDict] = []

    for index, item in enumerate(
            scored_items
    ):
        retrieved_chunk, raw_rerank_score = item

        normalized_score = normalized_scores[
            index
        ]

        updated_chunk = update_retrieved_chunk_score(
            retrieved_chunk=retrieved_chunk,
            raw_rerank_score=raw_rerank_score,
            normalized_rerank_score=normalized_score,
        )

        final_chunks.append(
            updated_chunk
        )

    updated_rag_context = dict(
        rag_context
    )

    updated_rag_context[
        "chunks"
    ] = final_chunks

    updated_rag_context[
        "source_count"
    ] = count_sources_from_chunks(
        chunks=final_chunks
    )

    updated_rag_context[
        "status"
    ] = (
        "success"
        if final_chunks
        else "empty"
    )

    updated_rag_context[
        "context_text"
    ] = build_context_text_from_chunks(
        chunks=final_chunks
    )

    docs = build_documents_from_chunks(
        chunks=final_chunks
    )

    return {
        "rag_context": updated_rag_context,
        "docs": docs,
    }


def build_rerank_pairs_from_chunks(
        question: str,
        chunks: list[RetrievedChunkDict],
) -> list[tuple[str, str]]:
    """
    从 chunks 构建 reranker 输入 pairs。

    功能：
        Cross-Encoder reranker 通常需要输入：
        [
            (query, document_text),
            (query, document_text),
        ]

    参数：
        question:
            用户问题。

        chunks:
            retrieved chunks。

    返回值：
        list[tuple[str, str]]:
            reranker 输入对。
    """

    pairs: list[tuple[str, str]] = []

    for retrieved_chunk in chunks:

        content = get_retrieved_chunk_content(
            retrieved_chunk=retrieved_chunk
        )

        pairs.append(
            (
                question,
                content,
            )
        )

    return pairs


def predict_rerank_scores(
        reranker_model: Any,
        pairs: list[tuple[str, str]],
) -> list[float]:
    """
    调用 reranker 模型预测相关性分数。

    功能：
        支持常见 Cross-Encoder 的 predict(pairs) 调用方式。

    参数：
        reranker_model:
            reranker 模型对象。

        pairs:
            query-document 输入对。

    返回值：
        list[float]:
            每个候选 chunk 的 rerank 分数。
    """

    if not pairs:
        return []

    if not hasattr(
            reranker_model,
            "predict"
    ):
        raise AttributeError(
            "reranker_model 必须提供 predict(pairs) 方法"
        )

    raw_scores = reranker_model.predict(
        pairs
    )

    return [
        safe_float(
            value=score,
            default=0.0,
        )
        for score in raw_scores
    ]


def update_retrieved_chunk_score(
        retrieved_chunk: RetrievedChunkDict,
        raw_rerank_score: float,
        normalized_rerank_score: float,
) -> RetrievedChunkDict:
    """
    更新单个 retrieved chunk 的 rerank 分数。

    功能：
        给 retrieved chunk 写入：
        1. rerank_score：
           模型原始分数，可能是负数。
        2. normalized_rerank_score：
           归一化后的 0~1 分数。
        3. final_score：
           当前等于 normalized_rerank_score。
        4. reason：
           追加 rerank 信息。

    设计说明：
        rerank_score 保留原始模型输出，方便调试。
        final_score 使用归一化分数，方便展示和后续评估。

    参数：
        retrieved_chunk:
            原始 retrieved chunk。

        raw_rerank_score:
            reranker 模型输出的原始分数。

        normalized_rerank_score:
            归一化后的 rerank 分数，范围 0~1。

    返回值：
        dict[str, Any]:
            更新后的 retrieved chunk。
    """

    updated_chunk = dict(
        retrieved_chunk
    )

    updated_chunk[
        "rerank_score"
    ] = raw_rerank_score

    updated_chunk[
        "normalized_rerank_score"
    ] = normalized_rerank_score

    updated_chunk[
        "final_score"
    ] = normalized_rerank_score

    old_reason = str(
        updated_chunk.get(
            "reason",
            ""
        )
        or ""
    ).strip()

    rerank_reason = (
        "Rerank 信息：该 chunk 经过 reranker 二次排序，"
        f"raw_rerank_score={raw_rerank_score}，"
        f"normalized_rerank_score={normalized_rerank_score}。"
    )

    if old_reason:
        updated_chunk[
            "reason"
        ] = old_reason + " " + rerank_reason
    else:
        updated_chunk[
            "reason"
        ] = rerank_reason

    return updated_chunk


def get_retrieved_chunk_content(
        retrieved_chunk: RetrievedChunkDict,
) -> str:
    """
    从 retrieved chunk 中读取正文内容。

    功能：
        支持 RagRetrievedChunk 的 dict 结构：
        {
            "chunk": {
                "content": "..."
            }
        }

    参数：
        retrieved_chunk:
            retrieved chunk 字典。

    返回值：
        str:
            chunk 正文。
    """

    chunk = retrieved_chunk.get(
        "chunk",
        {}
    ) or {}

    if isinstance(
            chunk,
            Mapping
    ):
        return str(
            chunk.get(
                "content",
                ""
            )
            or ""
        )

    return ""


def get_retrieved_chunk_metadata(
        retrieved_chunk: RetrievedChunkDict,
) -> dict[str, Any]:
    """
    从 retrieved chunk 中读取 metadata。

    功能：
        读取 retrieved_chunk["chunk"]["metadata"]。

    参数：
        retrieved_chunk:
            retrieved chunk 字典。

    返回值：
        dict[str, Any]:
            metadata 字典。
    """

    chunk = retrieved_chunk.get(
        "chunk",
        {}
    ) or {}

    if not isinstance(
            chunk,
            Mapping
    ):
        return {}

    metadata = chunk.get(
        "metadata",
        {}
    ) or {}

    if isinstance(
            metadata,
            Mapping
    ):
        return dict(
            metadata
        )

    return {}


def get_retrieved_chunk_source(
        retrieved_chunk: RetrievedChunkDict,
) -> str:
    """
    从 retrieved chunk 中读取 source。

    功能：
        优先读取 chunk.source；
        如果没有，则从 metadata.source / relative_path 兜底。

    参数：
        retrieved_chunk:
            retrieved chunk 字典。

    返回值：
        str:
            来源路径。
    """

    chunk = retrieved_chunk.get(
        "chunk",
        {}
    ) or {}

    if isinstance(
            chunk,
            Mapping
    ):
        source = chunk.get(
            "source",
            ""
        )

        if source:
            return str(
                source
            )

    metadata = get_retrieved_chunk_metadata(
        retrieved_chunk=retrieved_chunk
    )

    return str(
        metadata.get(
            "source",
            ""
        )
        or metadata.get(
            "relative_path",
            ""
        )
        or ""
    )


def build_documents_from_chunks(
        chunks: list[RetrievedChunkDict],
) -> list[Document]:
    """
    将 rerank 后的 chunks 转换成旧版 docs。

    功能：
        这是向后兼容逻辑。
        因为部分旧节点仍然读取 state["docs"]，
        所以 rerank 后需要同步生成 docs。

        当前会把以下分数同步到 Document.metadata：
        1. retrieval_score
        2. rerank_score
        3. normalized_rerank_score
        4. final_score

    参数：
        chunks:
            rerank 后的 retrieved chunks。

    返回值：
        list[Document]:
            LangChain Document 列表。
    """

    docs: list[Document] = []

    for retrieved_chunk in chunks:

        content = get_retrieved_chunk_content(
            retrieved_chunk=retrieved_chunk
        )

        metadata = get_retrieved_chunk_metadata(
            retrieved_chunk=retrieved_chunk
        )

        metadata[
            "retrieval_score"
        ] = retrieved_chunk.get(
            "retrieval_score",
            0.0,
        )

        metadata[
            "rerank_score"
        ] = retrieved_chunk.get(
            "rerank_score",
            0.0,
        )

        metadata[
            "normalized_rerank_score"
        ] = retrieved_chunk.get(
            "normalized_rerank_score",
            0.0,
        )

        metadata[
            "final_score"
        ] = retrieved_chunk.get(
            "final_score",
            0.0,
        )

        metadata[
            "retrieval_reason"
        ] = retrieved_chunk.get(
            "reason",
            "",
        )

        docs.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return docs


def build_context_text_from_chunks(
        chunks: list[RetrievedChunkDict],
) -> str:
    """
    根据 rerank 后的 chunks 重新构建 context_text。

    功能：
        generate_node 会优先使用 rag_context.context_text。
        所以 rerank 之后必须重新拼接上下文，
        否则虽然 chunks 排序变了，但 prompt 中的上下文顺序没有变化。

        当前设计：
        1. 不把 raw rerank_score 直接放进 context_text。
        2. 只展示 final_score。
        3. 避免 LLM 看到负数分数后产生误解。
        4. raw rerank_score 仍然保存在 chunks 和 docs metadata 中，方便调试。

    参数：
        chunks:
            rerank 后的 chunks。

    返回值：
        str:
            重新拼接后的上下文文本。
    """

    blocks: list[str] = []

    for index, retrieved_chunk in enumerate(
            chunks
    ):
        metadata = get_retrieved_chunk_metadata(
            retrieved_chunk=retrieved_chunk
        )

        dog_name = metadata.get(
            "dog_name",
            "unknown",
        )

        source = get_retrieved_chunk_source(
            retrieved_chunk=retrieved_chunk
        )

        content = get_retrieved_chunk_content(
            retrieved_chunk=retrieved_chunk
        )

        final_score = retrieved_chunk.get(
            "final_score",
            0.0,
        )

        blocks.append(
            "\n".join(
                [
                    f"[Chunk {index + 1}]",
                    f"dog_name: {dog_name}",
                    f"source: {source}",
                    f"final_score: {final_score}",
                    content,
                ]
            )
        )

    return "\n\n".join(
        blocks
    )


def count_sources_from_chunks(
        chunks: list[RetrievedChunkDict],
) -> int:
    """
    统计 rerank 后 chunks 的来源数量。

    功能：
        根据 doc_id / source / relative_path 统计不同来源数量。

    参数：
        chunks:
            rerank 后的 chunks。

    返回值：
        int:
            不同来源数量。
    """

    sources: set[str] = set()

    for retrieved_chunk in chunks:

        chunk = retrieved_chunk.get(
            "chunk",
            {}
        ) or {}

        metadata = get_retrieved_chunk_metadata(
            retrieved_chunk=retrieved_chunk
        )

        source_key = ""

        if isinstance(
                chunk,
                Mapping
        ):
            source_key = str(
                chunk.get(
                    "doc_id",
                    ""
                )
                or chunk.get(
                    "source",
                    ""
                )
                or ""
            )

        if not source_key:
            source_key = str(
                metadata.get(
                    "doc_id",
                    ""
                )
                or metadata.get(
                    "source",
                    ""
                )
                or metadata.get(
                    "relative_path",
                    ""
                )
                or ""
            )

        if source_key:
            sources.add(
                source_key
            )

    return len(
        sources
    )


def rerank_legacy_docs(
        question: str,
        docs: list[Document],
        reranker_model: Any,
        top_k: int,
) -> dict[str, Any]:
    """
    对旧版 docs 执行 rerank。

    功能：
        当 state 中没有 rag_context.chunks 时，
        回退到旧版 docs 重排逻辑。

        当前会保留：
        1. rerank_score：原始模型分数，可能为负数。
        2. normalized_rerank_score：归一化分数。
        3. final_score：当前等于 normalized_rerank_score。

    参数：
        question:
            用户问题。

        docs:
            LangChain Document 列表。

        reranker_model:
            reranker 模型。

        top_k:
            最终保留数量。

    返回值：
        dict[str, Any]:
            只返回 docs 字段。
    """

    if not docs:
        return {
            "docs": []
        }

    pairs = [
        (
            question,
            doc.page_content,
        )
        for doc in docs
    ]

    raw_scores = predict_rerank_scores(
        reranker_model=reranker_model,
        pairs=pairs,
    )

    scored_items: list[tuple[Document, float]] = []

    for index, doc in enumerate(
            docs
    ):
        raw_score = safe_float(
            value=raw_scores[index],
            default=0.0,
        )

        scored_items.append(
            (
                doc,
                raw_score,
            )
        )

    scored_items.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    scored_items = scored_items[
        :top_k
    ]

    normalized_scores = normalize_scores_to_0_1(
        scores=[
            raw_score
            for _, raw_score in scored_items
        ]
    )

    reranked_docs: list[Document] = []

    for index, item in enumerate(
            scored_items
    ):
        doc, raw_score = item

        normalized_score = normalized_scores[
            index
        ]

        metadata = dict(
            doc.metadata
            or {}
        )

        metadata[
            "rerank_score"
        ] = raw_score

        metadata[
            "normalized_rerank_score"
        ] = normalized_score

        metadata[
            "final_score"
        ] = normalized_score

        reranked_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    return {
        "docs": reranked_docs
    }


def resolve_top_k_from_state(
        state: Mapping[str, Any],
) -> int:
    """
    从 state 中解析 top_k。

    功能：
        top_k 表示最终保留多少条 rerank 结果。
        如果 state 中没有 top_k，则默认使用 5。

    参数：
        state:
            当前 DogState。

    返回值：
        int:
            合法 top_k。
    """

    raw_top_k = state.get(
        "top_k",
        5
    )

    try:
        top_k = int(
            raw_top_k
        )
    except (
            TypeError,
            ValueError,
    ):
        top_k = 5

    if top_k <= 0:
        return 5

    return top_k


def safe_float(
        value: Any,
        default: float = 0.0,
) -> float:
    """
    安全转换 float。

    功能：
        将任意值转换成 float。
        如果转换失败，则返回默认值。

    参数：
        value:
            待转换的值。

        default:
            转换失败时使用的默认值。

    返回值：
        float:
            转换后的浮点数。
    """

    if value is None:
        return default

    try:
        return float(
            value
        )
    except (
            TypeError,
            ValueError,
    ):
        return default

def normalize_scores_to_0_1(
        scores: list[float],
) -> list[float]:
    """
    将 rerank 原始分数归一化到 0~1。

    功能：
        使用 min-max normalization（最小最大归一化）将一组分数转换成 0~1。

        示例：
            原始分数：
                [-2.1, -3.5, -5.0]

            归一化后：
                [1.0, 0.517, 0.0]

        注意：
            这个归一化结果只适合在同一次 query 的候选结果之间比较。
            不建议跨 query 比较。

    技术名词：
        Min-Max Normalization：
            最小最大归一化。
            公式：
                normalized = (score - min_score) / (max_score - min_score)

    参数：
        scores:
            reranker 模型输出的原始分数列表。

    返回值：
        list[float]:
            归一化后的 0~1 分数列表。
    """

    if not scores:
        return []

    min_score = min(
        scores
    )

    max_score = max(
        scores
    )

    if max_score == min_score:
        return [
            1.0
            for _ in scores
        ]

    return [
        round(
            (
                score - min_score
            )
            /
            (
                max_score - min_score
            ),
            6,
        )
        for score in scores
    ]