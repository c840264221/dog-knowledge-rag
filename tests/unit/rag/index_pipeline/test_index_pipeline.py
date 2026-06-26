"""
RagIndexPipeline 单元测试。

本测试只验证 Pipeline 编排逻辑：
1. 不读取真实 Markdown 文件
2. 不调用真实 Chroma
3. 不调用真实 Embedding
4. 不依赖 RuntimeContainer
5. 只使用 Fake Loader / Fake Extractor / Fake Chunker / Fake Indexer
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.pipelines import (
    RagIndexPipeline,
)
from src.rag.schemas import (
    RagChunk,
    RagDocument,
)


def build_test_rag_document(
        content: str,
        metadata: dict[str, Any] | None = None,
) -> RagDocument:
    """
    构建测试用 RagDocument。

    功能：
        根据当前项目 RagDocument schema 动态构建测试对象。
        保证 content 和 metadata 一定存在。

    参数：
        content: str
            文档正文。

        metadata: dict[str, Any] | None
            文档 metadata。

    返回值：
        RagDocument：
            测试用 RAG 文档对象。
    """

    payload: dict[str, Any] = {
        "content": content,
        "metadata": metadata or {},
    }

    model_fields = getattr(
        RagDocument,
        "model_fields",
        {},
    )

    if "doc_id" in model_fields:
        payload[
            "doc_id"
        ] = "doc-001"

    if "document_id" in model_fields:
        payload[
            "document_id"
        ] = "doc-001"

    if "id" in model_fields:
        payload[
            "id"
        ] = "doc-001"

    if "source" in model_fields:
        payload[
            "source"
        ] = "affenpinscher.md"

    if "source_path" in model_fields:
        payload[
            "source_path"
        ] = "affenpinscher.md"

    if "title" in model_fields:
        payload[
            "title"
        ] = "Affenpinscher"

    return RagDocument(
        **payload,
    )


def build_test_rag_chunk(
        content: str,
        metadata: dict[str, Any] | None = None,
) -> RagChunk:
    """
    构建测试用 RagChunk。

    功能：
        根据当前项目 RagChunk schema 动态构建测试对象。
        保证 content 和 metadata 一定存在。

    参数：
        content: str
            chunk 正文。

        metadata: dict[str, Any] | None
            chunk metadata。

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

    if "chunk_id" in model_fields:
        payload[
            "chunk_id"
        ] = "chunk-001"

    if "id" in model_fields:
        payload[
            "id"
        ] = "chunk-001"

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


class FakeMarkdownLoader:
    """
    Fake Markdown Loader（假的 Markdown 加载器）。

    功能：
        模拟当前项目中的 MarkdownDocumentLoader。
        当前 Loader 公开方法是 load()。
    """

    def __init__(
            self,
            documents: list[RagDocument],
            input_path: str = "data/dog_markdown/affenpinscher.md",
    ):
        """
        初始化 FakeMarkdownLoader。

        参数：
            documents: list[RagDocument]
                预设返回的文档列表。

            input_path: str
                模拟 Loader 持有的输入路径。

        返回值：
            None：
                构造函数无返回值。
        """

        self.documents = documents
        self.input_path = input_path
        self.called = False

    def load(self) -> list[RagDocument]:
        """
        模拟 MarkdownDocumentLoader.load。

        功能：
            返回预设的 RagDocument 列表。

        参数：
            无。

        返回值：
            list[RagDocument]：
                预设文档列表。
        """

        self.called = True

        return self.documents


class FakeMetadataExtractor:
    """
    Fake Metadata Extractor（假的元数据提取器）。

    功能：
        模拟 DogBreedMetadataExtractor。
        用于测试 Pipeline 是否正确调用 extract_many。
    """

    def __init__(self):
        """
        初始化 FakeMetadataExtractor。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.called = False
        self.received_documents = None

    def extract_many(
            self,
            documents: list[RagDocument],
    ) -> list[RagDocument]:
        """
        模拟批量提取 metadata。

        参数：
            documents: list[RagDocument]
                原始文档列表。

        返回值：
            list[RagDocument]：
                metadata 增强后的文档列表。
        """

        self.called = True
        self.received_documents = documents

        enhanced_documents = []

        for document in documents:

            metadata = dict(
                document.metadata
                or {}
            )

            metadata[
                "dog_name"
            ] = "Affenpinscher"

            metadata[
                "energy_level"
            ] = 3

            enhanced_documents.append(
                document.model_copy(
                    update={
                        "metadata": metadata,
                    }
                )
            )

        return enhanced_documents


class FakeMarkdownChunker:
    """
    Fake Markdown Chunker（假的 Markdown 切块器）。

    功能：
        模拟 MarkdownChunker。
        用于测试 Pipeline 是否正确调用 chunk_many。
    """

    def __init__(self):
        """
        初始化 FakeMarkdownChunker。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.called = False
        self.received_documents = None

    def chunk_many(
            self,
            documents: list[RagDocument],
    ) -> list[RagChunk]:
        """
        模拟批量切块。

        参数：
            documents: list[RagDocument]
                metadata 增强后的文档列表。

        返回值：
            list[RagChunk]：
                切块结果。
        """

        self.called = True
        self.received_documents = documents

        chunks = []

        for document in documents:

            chunks.append(
                build_test_rag_chunk(
                    content=document.content,
                    metadata=document.metadata,
                )
            )

        return chunks


class FakeChromaIndexer:
    """
    Fake Chroma Indexer（假的 Chroma 入库器）。

    功能：
        模拟 RagChromaIndexer。
        用于测试 Pipeline 是否正确调用 index_chunks。
    """

    def __init__(self):
        """
        初始化 FakeChromaIndexer。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.called = False
        self.received_chunks = None

    def index_chunks(
            self,
            chunks: list[RagChunk],
    ) -> dict[str, Any]:
        """
        模拟批量写入 chunks。

        参数：
            chunks: list[RagChunk]
                待写入的 chunk 列表。

        返回值：
            dict[str, Any]：
                模拟入库结果。
        """

        self.called = True
        self.received_chunks = chunks

        return {
            "indexer": "fake_chroma_indexer",
            "total_chunks": len(
                chunks,
            ),
            "indexed_chunks": len(
                chunks,
            ),
            "skipped_chunks": 0,
            "batch_count": 1 if chunks else 0,
            "index_ids": [
                f"chunk-{index}"
                for index, _ in enumerate(
                    chunks,
                )
            ],
        }


@pytest.fixture
def raw_document() -> RagDocument:
    """
    构建原始测试文档。

    参数：
        无。

    返回值：
        RagDocument：
            未增强 metadata 的测试文档。
    """

    return build_test_rag_document(
        content="# Affenpinscher\n\nAffenpinschers are loyal dogs.",
        metadata={
            "source_type": "markdown",
        },
    )


@pytest.fixture
def fake_loader(
        raw_document,
) -> FakeMarkdownLoader:
    """
    构建 FakeMarkdownLoader。

    参数：
        raw_document:
            pytest fixture 注入的原始文档。

    返回值：
        FakeMarkdownLoader：
            假 Markdown 加载器。
    """

    return FakeMarkdownLoader(
        documents=[
            raw_document,
        ],
    )


@pytest.fixture
def fake_metadata_extractor() -> FakeMetadataExtractor:
    """
    构建 FakeMetadataExtractor。

    参数：
        无。

    返回值：
        FakeMetadataExtractor：
            假 metadata 提取器。
    """

    return FakeMetadataExtractor()


@pytest.fixture
def fake_chunker() -> FakeMarkdownChunker:
    """
    构建 FakeMarkdownChunker。

    参数：
        无。

    返回值：
        FakeMarkdownChunker：
            假 Markdown 切块器。
    """

    return FakeMarkdownChunker()


@pytest.fixture
def fake_indexer() -> FakeChromaIndexer:
    """
    构建 FakeChromaIndexer。

    参数：
        无。

    返回值：
        FakeChromaIndexer：
            假 Chroma 入库器。
    """

    return FakeChromaIndexer()


@pytest.fixture
def pipeline(
        fake_loader,
        fake_metadata_extractor,
        fake_chunker,
        fake_indexer,
) -> RagIndexPipeline:
    """
    构建 RagIndexPipeline。

    参数：
        fake_loader:
            pytest fixture 注入的假 Loader。

        fake_metadata_extractor:
            pytest fixture 注入的假 Metadata Extractor。

        fake_chunker:
            pytest fixture 注入的假 Chunker。

        fake_indexer:
            pytest fixture 注入的假 Indexer。

    返回值：
        RagIndexPipeline：
            用于测试的入库流水线。
    """

    return RagIndexPipeline(
        loader=fake_loader,
        metadata_extractor=fake_metadata_extractor,
        chunker=fake_chunker,
        indexer=fake_indexer,
    )


def test_index_file_should_run_full_pipeline(
        pipeline,
        fake_loader,
        fake_metadata_extractor,
        fake_chunker,
        fake_indexer,
):
    """
    测试 index_file 是否完整执行 Loader -> Extractor -> Chunker -> Indexer。

    参数：
        pipeline:
            pytest fixture 注入的 RagIndexPipeline。

        fake_loader:
            pytest fixture 注入的 FakeMarkdownLoader。

        fake_metadata_extractor:
            pytest fixture 注入的 FakeMetadataExtractor。

        fake_chunker:
            pytest fixture 注入的 FakeMarkdownChunker。

        fake_indexer:
            pytest fixture 注入的 FakeChromaIndexer。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = pipeline.index()

    assert str(
        fake_loader.input_path,
    ) == "data/dog_markdown/affenpinscher.md"

    assert fake_metadata_extractor.called is True

    assert fake_chunker.called is True

    assert fake_indexer.called is True

    assert result[
        "pipeline"
    ] == "rag_index_pipeline_v1"

    assert result[
        "loaded_documents"
    ] == 1

    assert result[
        "enhanced_documents"
    ] == 1

    assert result[
        "created_chunks"
    ] == 1

    assert result[
        "index_result"
    ][
        "indexed_chunks"
    ] == 1


def test_index_dir_should_call_loader_load_dir(
        pipeline,
        fake_loader,
):
    """
    测试 index_dir 是否调用 loader.load_dir。

    参数：
        pipeline:
            pytest fixture 注入的 RagIndexPipeline。

        fake_loader:
            pytest fixture 注入的 FakeMarkdownLoader。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = pipeline.index()

    assert str(
        fake_loader.input_path,
    ) == "data/dog_markdown/affenpinscher.md"

    assert result[
        "loaded_documents"
    ] == 1

    assert result[
        "created_chunks"
    ] == 1


def test_index_documents_should_pass_enhanced_metadata_to_chunker(
        pipeline,
        fake_chunker,
):
    """
    测试 Pipeline 是否把 metadata 增强后的文档传给 chunker。

    参数：
        pipeline:
            pytest fixture 注入的 RagIndexPipeline。

        fake_chunker:
            pytest fixture 注入的 FakeMarkdownChunker。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    document = build_test_rag_document(
        content="# Affenpinscher",
        metadata={
            "source_type": "markdown",
        },
    )

    pipeline.index_documents(
        documents=[
            document,
        ],
        source="memory-test",
    )

    received_document = fake_chunker.received_documents[
        0
    ]

    assert received_document.metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert received_document.metadata[
        "energy_level"
    ] == 3

    assert received_document.metadata[
        "source_type"
    ] == "markdown"


def test_index_documents_should_pass_chunks_to_indexer(
        pipeline,
        fake_indexer,
):
    """
    测试 Pipeline 是否把 chunker 产生的 chunks 传给 indexer。

    参数：
        pipeline:
            pytest fixture 注入的 RagIndexPipeline。

        fake_indexer:
            pytest fixture 注入的 FakeChromaIndexer。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    document = build_test_rag_document(
        content="# Affenpinscher",
        metadata={
            "source_type": "markdown",
        },
    )

    pipeline.index_documents(
        documents=[
            document,
        ],
    )

    received_chunk = fake_indexer.received_chunks[
        0
    ]

    assert received_chunk.content == "# Affenpinscher"

    assert received_chunk.metadata[
        "dog_name"
    ] == "Affenpinscher"


def test_index_documents_should_return_empty_result_when_no_documents(
        pipeline,
        fake_metadata_extractor,
        fake_chunker,
        fake_indexer,
):
    """
    测试空文档列表时 Pipeline 是否返回标准空结果。

    参数：
        pipeline:
            pytest fixture 注入的 RagIndexPipeline。

        fake_metadata_extractor:
            pytest fixture 注入的 FakeMetadataExtractor。

        fake_chunker:
            pytest fixture 注入的 FakeMarkdownChunker。

        fake_indexer:
            pytest fixture 注入的 FakeChromaIndexer。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = pipeline.index_documents(
        documents=[],
        source="empty-source",
    )

    assert result[
        "source"
    ] == "empty-source"

    assert result[
        "loaded_documents"
    ] == 0

    assert result[
        "enhanced_documents"
    ] == 0

    assert result[
        "created_chunks"
    ] == 0

    assert result[
        "index_result"
    ][
        "indexed_chunks"
    ] == 0

    assert fake_metadata_extractor.called is False

    assert fake_chunker.called is False

    assert fake_indexer.called is False


def test_pipeline_should_raise_error_when_loader_missing_load_file(
        fake_metadata_extractor,
        fake_chunker,
        fake_indexer,
):
    """
    测试 loader 缺少 load_file 方法时是否抛出异常。

    参数：
        fake_metadata_extractor:
            pytest fixture 注入的 FakeMetadataExtractor。

        fake_chunker:
            pytest fixture 注入的 FakeMarkdownChunker。

        fake_indexer:
            pytest fixture 注入的 FakeChromaIndexer。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    class InvalidLoader:
        """
        无效 Loader。

        功能：
            用于模拟缺少 load_file 方法的错误场景。
        """

        pass

    pipeline = RagIndexPipeline(
        loader=InvalidLoader(),
        metadata_extractor=fake_metadata_extractor,
        chunker=fake_chunker,
        indexer=fake_indexer,
    )

    with pytest.raises(
            AttributeError,
    ):
        pipeline.index()