import pytest

from src.rag.schemas import (
    RagDocument,
    RagChunk,
    RagQuery,
    RagRetrievedChunk,
    RagContext,
)


def test_create_rag_document():
    """
    测试创建 RAG 原始文档模型。

    功能：
        验证 RagDocument 是否可以正确保存文档 ID、来源、标题、正文和元数据。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    document = RagDocument(
        doc_id="doc_001",
        source="data/dog_markdown/golden_retriever.md",
        title="Golden Retriever",
        content="Golden Retriever is a friendly dog breed.",
        metadata={
            "category": "dog_breed",
            "language": "en"
        }
    )

    assert document.doc_id == "doc_001"
    assert document.source.endswith("golden_retriever.md")
    assert document.title == "Golden Retriever"
    assert document.metadata["category"] == "dog_breed"


def test_create_rag_chunk():
    """
    测试创建 RAG 文本块模型。

    功能：
        验证 RagChunk 是否可以正确保存文本块信息。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunk = RagChunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        content="Golden Retriever is friendly and intelligent.",
        chunk_index=0,
        source="data/dog_markdown/golden_retriever.md",
        title="Golden Retriever",
        metadata={
            "section": "overview"
        }
    )

    assert chunk.chunk_id == "chunk_001"
    assert chunk.doc_id == "doc_001"
    assert chunk.chunk_index == 0
    assert chunk.metadata["section"] == "overview"


def test_create_rag_query_with_default_values():
    """
    测试创建 RAG 检索请求模型。

    功能：
        验证 RagQuery 的默认值是否符合预期。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    query = RagQuery(
        question="金毛适合新手养吗？"
    )

    assert query.question == "金毛适合新手养吗？"
    assert query.user_id == "default"
    assert query.top_k == 5
    assert query.intent == "general"
    assert query.filters == {}


def test_create_rag_retrieved_chunk():
    """
    测试创建 RAG 检索结果模型。

    功能：
        验证 RagRetrievedChunk 是否可以正确保存检索分数和文本块。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunk = RagChunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        content="Golden Retriever is friendly.",
        chunk_index=0
    )

    retrieved_chunk = RagRetrievedChunk(
        chunk=chunk,
        retrieval_score=0.82,
        rerank_score=0.91,
        final_score=0.91,
        reason="语义相似度较高"
    )

    assert retrieved_chunk.chunk.chunk_id == "chunk_001"
    assert retrieved_chunk.retrieval_score == 0.82
    assert retrieved_chunk.rerank_score == 0.91
    assert retrieved_chunk.final_score == 0.91


def test_create_empty_rag_context():
    """
    测试创建空 RAG 上下文模型。

    功能：
        验证没有检索结果时 RagContext 的默认状态是否为 empty。

    参数：
        无。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    context = RagContext(
        question="边牧聪明吗？"
    )

    assert context.question == "边牧聪明吗？"
    assert context.context_text == ""
    assert context.chunks == []
    assert context.source_count == 0
    assert context.status == "empty"


def test_rag_query_top_k_must_be_positive():
    """
    测试 RagQuery 的 top_k 参数校验。

    功能：
        验证 top_k 小于 1 时，Pydantic 是否会抛出校验错误。

    参数：
        无。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    with pytest.raises(Exception):
        RagQuery(
            question="拉布拉多适合家庭吗？",
            top_k=0
        )