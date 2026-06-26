"""
RagChromaIndexer 单元测试。

本测试只测试 indexer 自身逻辑：
1. 不连接真实 Chroma
2. 不调用 Embedding
3. 不访问网络
4. 不依赖 RuntimeContainer
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.indexers import RagChromaIndexer
from src.rag.schemas import RagChunk


class FakeVectorStore:
    """
    Fake Vector Store（假的向量库）。

    功能：
        模拟 Chroma 的 add_documents 和 delete 方法。
        用于测试 RagChromaIndexer 是否正确调用向量库。

    字段：
        added_documents:
            记录被写入的 Document。

        added_ids:
            记录被写入的 ids。

        deleted_ids:
            记录被删除的 ids。
    """

    def __init__(self):
        """
        初始化 FakeVectorStore。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.added_documents = []
        self.added_ids = []
        self.deleted_ids = []

    def add_documents(
            self,
            documents,
            ids=None,
    ):
        """
        模拟 Chroma.add_documents。

        参数：
            documents:
                要写入的 Document 列表。

            ids:
                与 documents 对应的 id 列表。

        返回值：
            list:
                返回传入的 ids，模拟真实写入结果。
        """

        self.added_documents.extend(
            documents,
        )

        self.added_ids.extend(
            ids
            or []
        )

        return ids

    def delete(
            self,
            ids,
    ):
        """
        模拟 Chroma.delete。

        参数：
            ids:
                要删除的 id 列表。

        返回值：
            None：
                删除方法无返回值。
        """

        self.deleted_ids.extend(
            ids,
        )


def build_test_rag_chunk(
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk_id: str | None = "chunk-001",
) -> RagChunk:
    """
    构建测试用 RagChunk。

    功能：
        根据当前项目 RagChunk schema 动态构建测试对象。
        如果你的 RagChunk 有 doc_id、source_path、chunk_index 等字段，
        这里会根据 model_fields 自动补充常见字段。

    参数：
        content: str
            chunk 正文。

        metadata: dict[str, Any] | None
            chunk metadata。

        chunk_id: str | None
            chunk id。
            如果传 None，则测试 indexer 的兜底 id 构造逻辑。

    返回值：
        RagChunk：
            测试用 RAG chunk 对象。
    """

    payload: dict[str, Any] = {
        "content": content,
        "metadata": metadata or {},
    }

    model_fields = getattr(
        RagChunk,
        "model_fields",
        {},
    )

    if "chunk_id" in model_fields and chunk_id is not None:
        payload[
            "chunk_id"
        ] = chunk_id

    if "id" in model_fields and chunk_id is not None:
        payload[
            "id"
        ] = chunk_id

    if "doc_id" in model_fields:
        payload[
            "doc_id"
        ] = "doc-001"

    if "document_id" in model_fields:
        payload[
            "document_id"
        ] = "doc-001"

    if "source" in model_fields:
        payload[
            "source"
        ] = "affenpinscher.md"

    if "source_path" in model_fields:
        payload[
            "source_path"
        ] = "affenpinscher.md"

    if "chunk_index" in model_fields:
        payload[
            "chunk_index"
        ] = 0

    if "start_index" in model_fields:
        payload[
            "start_index"
        ] = 0

    if "end_index" in model_fields:
        payload[
            "end_index"
        ] = len(
            content,
        )

    return RagChunk(
        **payload,
    )


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """
    构建 FakeVectorStore 测试对象。

    参数：
        无。

    返回值：
        FakeVectorStore：
            用于替代真实 Chroma 的假向量库。
    """

    return FakeVectorStore()


@pytest.fixture
def indexer(
        fake_vector_store,
) -> RagChromaIndexer:
    """
    构建 RagChromaIndexer 测试对象。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        RagChromaIndexer：
            用于测试的 Chroma indexer。
    """

    return RagChromaIndexer(
        vector_store=fake_vector_store,
        batch_size=2,
        overwrite_existing=True,
    )


@pytest.fixture
def affenpinscher_chunk() -> RagChunk:
    """
    构建 Affenpinscher 测试 chunk。

    参数：
        无。

    返回值：
        RagChunk：
            带有狗狗 metadata 的测试 chunk。
    """

    return build_test_rag_chunk(
        content="Affenpinschers are loyal, curious, and famously funny.",
        metadata={
            "dog_name": "Affenpinscher",
            "dog_tags": "confident / famously funny / fearless",
            "size": "small",
            "energy_level": 3,
            "barking_level": 3,
            "good_for_apartment": True,
            "invalid_list": [
                "a",
                "b",
            ],
            "invalid_dict": {
                "x": 1,
            },
            "none_value": None,
        },
        chunk_id="affenpinscher-001",
    )


def test_index_chunk_should_write_document_to_vector_store(
        indexer,
        fake_vector_store,
        affenpinscher_chunk,
):
    """
    测试 index_chunk 是否能写入单个 chunk。

    参数：
        indexer:
            pytest fixture 注入的 RagChromaIndexer。

        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

        affenpinscher_chunk:
            pytest fixture 注入的 RagChunk。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = indexer.index_chunk(
        chunk=affenpinscher_chunk,
    )

    assert result[
        "total_chunks"
    ] == 1

    assert result[
        "indexed_chunks"
    ] == 1

    assert result[
        "skipped_chunks"
    ] == 0

    assert fake_vector_store.added_ids == [
        "affenpinscher-001",
    ]

    assert fake_vector_store.deleted_ids == [
        "affenpinscher-001",
    ]

    added_document = fake_vector_store.added_documents[
        0
    ]

    assert added_document.page_content == (
        "Affenpinschers are loyal, curious, and famously funny."
    )

    assert added_document.metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert added_document.metadata[
        "energy_level"
    ] == 3

    assert added_document.metadata[
        "good_for_apartment"
    ] is True


def test_index_chunk_should_clean_invalid_metadata(
        indexer,
        fake_vector_store,
        affenpinscher_chunk,
):
    """
    测试 indexer 是否会过滤 Chroma 不支持的 metadata 类型。

    参数：
        indexer:
            pytest fixture 注入的 RagChromaIndexer。

        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

        affenpinscher_chunk:
            pytest fixture 注入的 RagChunk。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    indexer.index_chunk(
        chunk=affenpinscher_chunk,
    )

    added_document = fake_vector_store.added_documents[
        0
    ]

    assert "invalid_list" not in added_document.metadata

    assert "invalid_dict" not in added_document.metadata

    assert "none_value" not in added_document.metadata

    assert added_document.metadata[
        "indexer"
    ] == "rag_chroma_indexer_v1"


def test_index_chunks_should_skip_empty_content(
        indexer,
        fake_vector_store,
):
    """
    测试空 content 的 chunk 是否会被跳过。

    参数：
        indexer:
            pytest fixture 注入的 RagChromaIndexer。

        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    empty_chunk = build_test_rag_chunk(
        content="   ",
        metadata={
            "dog_name": "Empty Dog",
        },
        chunk_id="empty-001",
    )

    result = indexer.index_chunks(
        chunks=[
            empty_chunk,
        ],
    )

    assert result[
        "total_chunks"
    ] == 1

    assert result[
        "indexed_chunks"
    ] == 0

    assert result[
        "skipped_chunks"
    ] == 1

    assert fake_vector_store.added_documents == []

    assert fake_vector_store.added_ids == []


def test_index_chunks_should_write_in_batches(
        fake_vector_store,
):
    """
    测试 indexer 是否按 batch_size 分批写入。

    参数：
        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    indexer = RagChromaIndexer(
        vector_store=fake_vector_store,
        batch_size=2,
        overwrite_existing=True,
    )

    chunks = [
        build_test_rag_chunk(
            content="chunk 1",
            chunk_id="chunk-1",
        ),
        build_test_rag_chunk(
            content="chunk 2",
            chunk_id="chunk-2",
        ),
        build_test_rag_chunk(
            content="chunk 3",
            chunk_id="chunk-3",
        ),
    ]

    result = indexer.index_chunks(
        chunks=chunks,
    )

    assert result[
        "indexed_chunks"
    ] == 3

    assert result[
        "batch_count"
    ] == 2

    assert fake_vector_store.added_ids == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]

    assert fake_vector_store.deleted_ids == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


def test_indexer_should_not_delete_when_overwrite_disabled(
        fake_vector_store,
        affenpinscher_chunk,
):
    """
    测试 overwrite_existing=False 时不会删除旧数据。

    参数：
        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

        affenpinscher_chunk:
            pytest fixture 注入的 RagChunk。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    indexer = RagChromaIndexer(
        vector_store=fake_vector_store,
        overwrite_existing=False,
    )

    indexer.index_chunk(
        chunk=affenpinscher_chunk,
    )

    assert fake_vector_store.deleted_ids == []

    assert fake_vector_store.added_ids == [
        "affenpinscher-001",
    ]


def test_indexer_should_raise_error_when_batch_size_invalid(
        fake_vector_store,
):
    """
    测试 batch_size 非法时是否抛出异常。

    参数：
        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    with pytest.raises(
            ValueError,
    ):
        RagChromaIndexer(
            vector_store=fake_vector_store,
            batch_size=0,
        )