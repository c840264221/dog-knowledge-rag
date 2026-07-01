"""
MetadataFilterRetriever 单元测试。

本测试只验证 Retriever 自身逻辑：
1. 不连接真实 Chroma
2. 不调用真实 Embedding
3. 不访问网络
4. 不依赖 RuntimeContainer
5. 使用 FakeVectorStore 模拟 similarity_search
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.retrievers import (
    MetadataFilterRetriever,
)


class FakeDocument:
    """
    Fake Document（假的 LangChain Document）。

    功能：
        模拟 LangChain Document 对象。
    """

    def __init__(
            self,
            page_content: str,
            metadata: dict[str, Any],
    ):
        """
        初始化 FakeDocument。

        参数：
            page_content: str
                文档正文。

            metadata: dict[str, Any]
                文档 metadata。

        返回值：
            None：
                构造函数无返回值。
        """

        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    """
    Fake Vector Store（假的向量库）。

    功能：
        模拟 Chroma similarity_search_with_score。
    """

    def __init__(
            self,
            documents: list[FakeDocument],
    ):
        """
        初始化 FakeVectorStore。

        参数：
            documents: list[FakeDocument]
                假文档列表。

        返回值：
            None：
                构造函数无返回值。
        """

        self.documents = documents
        self.received_query = None
        self.received_k = None
        self.received_filter = None

    def similarity_search_with_score(
            self,
            query: str,
            k: int,
            filter: dict[str, Any] | None = None,
    ) -> list[tuple[FakeDocument, float]]:
        """
        模拟 Chroma similarity_search_with_score。

        参数：
            query: str
                查询文本。

            k: int
                返回条数。

            filter: dict[str, Any] | None
                metadata filter。

        返回值：
            list[tuple[FakeDocument, float]]：
                模拟检索结果和分数。
        """

        self.received_query = query
        self.received_k = k
        self.received_filter = filter

        matched_documents = [
            document
            for document in self.documents
            if match_where(
                metadata=document.metadata,
                where=filter,
            )
        ]

        return [
            (
                document,
                float(
                    index,
                ),
            )
            for index, document in enumerate(
                matched_documents[
                    :k
                ]
            )
        ]


def match_where(
        metadata: dict[str, Any],
        where: dict[str, Any] | None,
) -> bool:
    """
    匹配简化版 Chroma where filter。

    功能：
        支持测试中用到的：
        1. $and
        2. $eq
        3. $lte
        4. $gte

    参数：
        metadata: dict[str, Any]
            文档 metadata。

        where: dict[str, Any] | None
            metadata filter。

    返回值：
        bool：
            True 表示匹配；
            False 表示不匹配。
    """

    if not where:
        return True

    if "$and" in where:
        return all(
            match_where(
                metadata=metadata,
                where=condition,
            )
            for condition in where[
                "$and"
            ]
        )

    for key, condition in where.items():

        actual_value = metadata.get(
            key,
        )

        if isinstance(
                condition,
                dict,
        ):

            if "$eq" in condition and actual_value != condition[
                "$eq"
            ]:
                return False

            if "$lte" in condition and actual_value > condition[
                "$lte"
            ]:
                return False

            if "$gte" in condition and actual_value < condition[
                "$gte"
            ]:
                return False

            continue

        if actual_value != condition:
            return False

    return True


def get_context_chunks(
        context: Any,
) -> list[Any]:
    """
    从 RagContext 中读取 chunks。

    功能：
        兼容不同 RagContext schema 字段名：
        1. retrieved_chunks
        2. chunks
        3. results

    参数：
        context: Any
            RagContext 对象。

    返回值：
        list[Any]：
            chunk 列表。
    """

    for attr_name in [
        "retrieved_chunks",
        "chunks",
        "results",
    ]:

        value = getattr(
            context,
            attr_name,
            None,
        )

        if value is not None:
            return list(
                value,
            )

    return []


def get_chunk_content(
        chunk: Any,
) -> str:
    """
    从 RagRetrievedChunk 中读取正文内容。

    功能：
        兼容不同 RagRetrievedChunk schema 字段名：
        1. content
        2. page_content
        3. text

    参数：
        chunk: Any
            RagRetrievedChunk 对象。

    返回值：
        str：
            chunk 正文。
    """

    for attr_name in [
        "content",
        "page_content",
        "text",
    ]:

        value = getattr(
            chunk,
            attr_name,
            None,
        )

        if value is not None:
            return str(
                value,
            )

    nested_chunk = getattr(
        chunk,
        "chunk",
        None,
    )

    if nested_chunk is not None:
        return get_chunk_content(
            chunk=nested_chunk,
        )

    return ""


@pytest.fixture
def fake_documents() -> list[FakeDocument]:
    """
    构建测试用假文档列表。

    参数：
        无。

    返回值：
        list[FakeDocument]：
            假 LangChain Document 列表。
    """

    return [
        FakeDocument(
            page_content="Affenpinscher is a small confident dog.",
            metadata={
                "dog_name": "Affenpinscher",
                "size": "small",
                "energy_level": 3,
                "barking_level": 3,
                "trainability_level": 3,
                "good_for_apartment": True,
            },
        ),
        FakeDocument(
            page_content="Beagle is friendly but can be vocal.",
            metadata={
                "dog_name": "Beagle",
                "size": "small",
                "energy_level": 4,
                "barking_level": 5,
                "trainability_level": 3,
                "good_for_apartment": False,
            },
        ),
        FakeDocument(
            page_content="Golden Retriever is friendly and trainable.",
            metadata={
                "dog_name": "Golden Retriever",
                "size": "large",
                "energy_level": 4,
                "barking_level": 3,
                "trainability_level": 5,
                "good_for_apartment": False,
            },
        ),
    ]


@pytest.fixture
def fake_vector_store(
        fake_documents,
) -> FakeVectorStore:
    """
    构建 FakeVectorStore。

    参数：
        fake_documents:
            pytest fixture 注入的假文档列表。

    返回值：
        FakeVectorStore：
            假向量库。
    """

    return FakeVectorStore(
        documents=fake_documents,
    )


@pytest.fixture
def retriever(
        fake_vector_store,
) -> MetadataFilterRetriever:
    """
    构建 MetadataFilterRetriever。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        MetadataFilterRetriever：
            测试用 Retriever。
    """

    return MetadataFilterRetriever(
        vector_store=fake_vector_store,
        default_top_k=5,
    )


def test_build_dog_metadata_filter_should_create_and_filter(
        retriever,
):
    """
    测试狗狗 metadata filter 构造。

    参数：
        retriever:
            pytest fixture 注入的 MetadataFilterRetriever。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = retriever.build_dog_metadata_filter(
        size="small",
        max_energy=3,
        max_barking=3,
        min_trainability=3,
        good_for_apartment=True,
    )

    assert result == {
        "$and": [
            {
                "size": {
                    "$eq": "small",
                }
            },
            {
                "energy_level": {
                    "$lte": 3,
                }
            },
            {
                "barking_level": {
                    "$lte": 3,
                }
            },
            {
                "trainability_level": {
                    "$gte": 3,
                }
            },
            {
                "good_for_apartment": {
                    "$eq": True,
                }
            },
        ]
    }


def test_retrieve_should_pass_query_top_k_and_filter_to_vector_store(
        retriever,
        fake_vector_store,
):
    """
    测试 retrieve 是否把 query、top_k、filter 传给 vector_store。

    参数：
        retriever:
            pytest fixture 注入的 MetadataFilterRetriever。

        fake_vector_store:
            pytest fixture 注入的 FakeVectorStore。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    metadata_filter = retriever.build_dog_metadata_filter(
        size="small",
        max_energy=3,
        max_barking=3,
    )

    retriever.retrieve(
        query="small apartment dog",
        metadata_filter=metadata_filter,
        top_k=2,
    )

    assert fake_vector_store.received_query == "small apartment dog"
    assert fake_vector_store.received_k == 2
    assert fake_vector_store.received_filter == metadata_filter


def test_retrieve_should_return_rag_context_with_filtered_chunks(
        retriever,
):
    """
    测试 retrieve 是否返回带过滤结果的 RagContext。

    参数：
        retriever:
            pytest fixture 注入的 MetadataFilterRetriever。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    metadata_filter = retriever.build_dog_metadata_filter(
        size="small",
        max_energy=3,
        max_barking=3,
    )

    context = retriever.retrieve(
        query="small apartment dog",
        metadata_filter=metadata_filter,
        top_k=5,
    )

    chunks = get_context_chunks(
        context=context,
    )

    assert len(
        chunks,
    ) == 1

    assert "Affenpinscher" in get_chunk_content(
        chunk=chunks[0],
    )


def test_retrieve_chunks_should_return_chunk_list(
        retriever,
):
    """
    测试 retrieve_chunks 是否只返回 chunk 列表。

    参数：
        retriever:
            pytest fixture 注入的 MetadataFilterRetriever。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    metadata_filter = retriever.build_dog_metadata_filter(
        dog_name="Golden Retriever",
    )

    chunks = retriever.retrieve_chunks(
        query="friendly trainable dog",
        metadata_filter=metadata_filter,
    )

    assert len(
        chunks,
    ) == 1

    assert "Golden Retriever" in get_chunk_content(
        chunk=chunks[0],
    )


def test_retriever_should_raise_error_when_top_k_invalid(
        fake_vector_store,
):
    """
    测试 default_top_k 非法时是否抛出异常。

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
        MetadataFilterRetriever(
            vector_store=fake_vector_store,
            default_top_k=0,
        )
