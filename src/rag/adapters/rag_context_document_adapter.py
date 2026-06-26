from typing import Any

from langchain_core.documents import Document


def rag_context_to_documents(
        rag_context: Any,
) -> list[Document]:
    """
    将 RagContext 转换成 LangChain Document 列表。

    功能：
        旧版 exact_search_agent 的部分节点仍然依赖 state["docs"]。
        新版 RAG 返回的是 RagContext。
        为了兼容旧链路，需要把 RagContext.chunks 转换成
        LangChain Document 列表。

    技术名词：
        Adapter：
            适配器。负责在不同对象结构之间转换。

        RagContext：
            新版 RAG 上下文对象，包含 chunks、context_text、status 等字段。

        LangChain Document：
            LangChain 文档对象，包含 page_content 和 metadata。

        Backward Compatibility：
            向后兼容。这里表示新版 RAG 结果仍然可以适配旧版 docs 链路。

    参数：
        rag_context:
            RagContext 对象或 dict。
            中文释义：新版 RAG 检索上下文。

    返回值：
        list[Document]:
            兼容旧链路的 LangChain Document 列表。
    """

    normalized_context = normalize_mapping(
        value=rag_context
    )

    if not normalized_context:
        return []

    chunks = normalized_context.get(
        "chunks",
        []
    )

    if not isinstance(
            chunks,
            list
    ):
        return []

    documents: list[Document] = []

    for retrieved_chunk in chunks:

        retrieved_chunk_dict = normalize_mapping(
            value=retrieved_chunk
        )

        if not retrieved_chunk_dict:
            continue

        chunk = normalize_mapping(
            value=retrieved_chunk_dict.get(
                "chunk",
                {}
            )
        )

        if not chunk:
            continue

        metadata = normalize_mapping(
            value=chunk.get(
                "metadata",
                {}
            )
        ).copy()

        metadata.update(
            {
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "doc_id": chunk.get(
                    "doc_id"
                ),
                "source": chunk.get(
                    "source"
                ),
                "title": chunk.get(
                    "title"
                ),
                "chunk_index": chunk.get(
                    "chunk_index"
                ),
                "retrieval_score": (
                    retrieved_chunk_dict.get(
                        "retrieval_score"
                    )
                ),
                "rerank_score": (
                    retrieved_chunk_dict.get(
                        "rerank_score"
                    )
                ),
                "final_score": (
                    retrieved_chunk_dict.get(
                        "final_score"
                    )
                ),
                "retrieval_reason": (
                    retrieved_chunk_dict.get(
                        "reason"
                    )
                ),
            }
        )

        content = str(
            chunk.get(
                "content",
                ""
            )
            or ""
        )

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def normalize_mapping(
        value: Any,
) -> dict[str, Any]:
    """
    将对象归一化成 dict。

    功能：
        兼容以下输入：
        1. dict。
        2. Pydantic BaseModel，例如 RagContext / RagRetrievedChunk / RagChunk。
        3. 其他对象，无法转换则返回空 dict。

    技术名词：
        Normalize：
            归一化。把不同形态的数据转换成统一格式。

        Pydantic BaseModel：
            Pydantic 数据模型对象。

        model_dump：
            Pydantic v2 中把模型对象转换成 dict 的方法。

    参数：
        value:
            任意对象。

    返回值：
        dict[str, Any]:
            转换后的字典。
            无法转换时返回空 dict。
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