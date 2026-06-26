import json
from typing import Any


def build_context_from_docs(
        docs: list[Any],
) -> str:
    """
    根据旧版 docs 构建 Prompt 上下文。

    功能：
        将旧版 LangChain Document 列表转换成 JSON 字符串。
        这个函数主要用于兼容旧版 RAG 链路。

        在 v1.5 新版链路中，generate_node 会优先使用 rag_context.context_text。
        只有当 state 中没有 rag_context，或者 rag_context 无法生成上下文时，
        才会回退使用 docs。

    技术名词：
        Document：
            文档对象。这里主要指 LangChain Document，
            通常包含 page_content 和 metadata 两个核心字段。

        Metadata：
            元数据。用于描述文档或 chunk 的结构化信息，
            例如 dog_name、size、energy_level、barking_level 等。

        Fallback：
            兜底机制。当新版 rag_context 不可用时，
            使用旧版 docs 继续保证链路可运行。

    参数：
        docs:
            LangChain Document 列表。
            中文释义：旧版检索节点返回的文档列表。

    返回值：
        str:
            JSON 字符串格式的上下文。
            该字符串会被注入 Prompt，供 LLM 生成回答。
    """

    context_items: list[dict[str, Any]] = []

    for doc in docs:

        metadata = getattr(
            doc,
            "metadata",
            {}
        ) or {}

        page_content = str(
            getattr(
                doc,
                "page_content",
                ""
            )
            or ""
        )

        item = {
            "name": (
                    metadata.get(
                        "dog_name"
                    )
                    or metadata.get(
                "name"
            )
            ),
            "source": (
                    metadata.get(
                        "source"
                    )
                    or metadata.get(
                "relative_path"
            )
                    or metadata.get(
                "file_name"
            )
            ),
            "structured": {
                "size": metadata.get(
                    "size"
                ),
                "barking_level": metadata.get(
                    "barking_level"
                ),
                "energy_level": metadata.get(
                    "energy_level"
                ),
                "trainability_level": metadata.get(
                    "trainability_level"
                ),
                "shedding_level": metadata.get(
                    "shedding_level"
                ),
                "good_for_apartment": metadata.get(
                    "good_for_apartment"
                ),
                "good_for_beginner": metadata.get(
                    "good_for_beginner"
                ),
            },
            "retrieval": {
                "retrieval_score": metadata.get(
                    "retrieval_score"
                ),
                "rerank_score": metadata.get(
                    "rerank_score"
                ),
                "final_score": metadata.get(
                    "final_score"
                ),
                "reason": metadata.get(
                    "retrieval_reason"
                ),
            },
            "text": page_content[
                    :500
                    ],
        }

        context_items.append(
            item
        )

    return json.dumps(
        context_items,
        ensure_ascii=False,
        indent=2
    )


def build_context_from_rag_context(
        rag_context: dict[str, Any] | Any,
) -> str:
    """
    根据新版 RagContext 构建 Prompt 上下文。

    功能：
        优先读取 rag_context["context_text"]。
        context_text 是 MetadataFilterRetriever 已经整理好的上下文字符串，
        最适合直接注入 Prompt。

        如果 context_text 为空，则从 rag_context["chunks"] 中兜底拼接上下文。

    技术名词：
        RagContext：
            RAG 上下文对象。
            包含 question、context_text、chunks、source_count、status 等字段。

        context_text：
            已经整理好的 Prompt 上下文字符串。
            LLM 可以直接读取这段字符串进行回答。

        chunks：
            结构化召回结果列表。
            每个元素通常包含 chunk、retrieval_score、final_score、reason 等信息。

    参数：
        rag_context:
            新版 RAG 上下文。
            在 LangGraph state 中通常是 dict。
            如果传入的是 Pydantic 对象，也会尝试转换成 dict。

    返回值：
        str:
            用于注入 Prompt 的上下文字符串。
    """

    normalized_context = _normalize_mapping(
        value=rag_context
    )

    if not normalized_context:

        return ""

    context_text = str(
        normalized_context.get(
            "context_text",
            ""
        )
        or ""
    ).strip()

    if context_text:

        return context_text

    chunks = normalized_context.get(
        "chunks",
        []
    )

    if not isinstance(
            chunks,
            list
    ):

        return ""

    blocks: list[str] = []

    for index, retrieved_chunk in enumerate(
            chunks
    ):

        retrieved_chunk_dict = _normalize_mapping(
            value=retrieved_chunk
        )

        if not retrieved_chunk_dict:

            continue

        chunk = _normalize_mapping(
            value=retrieved_chunk_dict.get(
                "chunk",
                {}
            )
        )

        if not chunk:

            continue

        metadata = _normalize_mapping(
            value=chunk.get(
                "metadata",
                {}
            )
        )

        dog_name = (
                metadata.get(
                    "dog_name"
                )
                or chunk.get(
            "title"
        )
                or "unknown"
        )

        source = (
                chunk.get(
                    "source"
                )
                or metadata.get(
            "relative_path"
        )
                or metadata.get(
            "file_name"
        )
                or "unknown"
        )

        content = str(
            chunk.get(
                "content",
                ""
            )
            or ""
        )

        reason = str(
            retrieved_chunk_dict.get(
                "reason",
                ""
            )
            or ""
        )

        retrieval_score = retrieved_chunk_dict.get(
            "retrieval_score"
        )

        final_score = retrieved_chunk_dict.get(
            "final_score"
        )

        block = "\n".join(
            [
                f"[Chunk {index + 1}]",
                f"dog_name: {dog_name}",
                f"source: {source}",
                f"retrieval_score: {retrieval_score}",
                f"final_score: {final_score}",
                f"reason: {reason}",
                content,
            ]
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


def resolve_generation_context(
        state: dict[str, Any],
) -> tuple[str, str]:
    """
    从 LangGraph state 中解析生成节点需要使用的上下文。

    功能：
        按优先级选择上下文来源：

        1. 优先使用新版 state["rag_context"]。
        2. 如果没有 rag_context，回退使用旧版 state["docs"]。
        3. 如果两者都没有，返回空上下文。

    技术名词：
        State：
            状态。LangGraph 中节点之间共享和传递的数据字典。

        Context：
            上下文。这里指最终注入 Prompt 的检索内容。

        Context Source：
            上下文来源。用于日志和调试，例如 rag_context、docs、empty。

        Backward Compatibility：
            向后兼容。表示新版代码仍然支持旧版 docs 链路。

    参数：
        state:
            当前 LangGraph 状态。
            通常包含 question、rag_context、docs、messages 等字段。

    返回值：
        tuple[str, str]:
            第一个元素是上下文字符串。
            第二个元素是上下文来源标识：
            - rag_context
            - docs
            - empty
    """

    rag_context = state.get(
        "rag_context"
    )

    if rag_context:

        context = build_context_from_rag_context(
            rag_context=rag_context
        )

        if context.strip():

            return (
                context,
                "rag_context"
            )

    docs = state.get(
        "docs",
        []
    )

    if docs:

        context = build_context_from_docs(
            docs=docs
        )

        if context.strip():

            return (
                context,
                "docs"
            )

    return (
        "",
        "empty"
    )


def _normalize_mapping(
        value: Any,
) -> dict[str, Any]:
    """
    将不同类型的对象归一化成 dict。

    功能：
        兼容以下几种情况：

        1. value 本身就是 dict。
        2. value 是 Pydantic BaseModel，支持 model_dump。
        3. value 是其他对象，无法转换则返回空 dict。

    技术名词：
        Normalize：
            归一化。把不同形态的数据转换成统一格式。

        Pydantic BaseModel：
            Pydantic 数据模型对象。
            例如 RagContext、RagRetrievedChunk、RagChunk 等。

        model_dump：
            Pydantic v2 中把模型对象转换成 dict 的方法。

    参数：
        value:
            任意对象。

    返回值：
        dict[str, Any]:
            转换后的字典。
            如果无法转换，返回空 dict。
    """

    if isinstance(
            value,
            dict
    ):

        return value

    if hasattr(
            value,
            "model_dump"
    ):

        try:

            dumped = value.model_dump()

            if isinstance(
                    dumped,
                    dict
            ):

                return dumped

        except Exception:

            return {}

    return {}