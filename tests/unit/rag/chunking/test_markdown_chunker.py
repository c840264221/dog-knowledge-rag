import pytest

from src.rag.chunking.markdown_chunker import MarkdownChunker
from src.rag.schemas import RagDocument


@pytest.fixture
def sample_markdown_document() -> RagDocument:
    """
    测试用 Markdown 文档 fixture。

    功能：
        创建一篇带有 Markdown 标题和正文的 RagDocument，
        用于测试 MarkdownChunker。

    参数：
        无。

    返回值：
        RagDocument:
            测试用 RAG 原始文档。
    """

    return RagDocument(
        doc_id="doc_test_001",
        source="data/dog_markdown/golden_retriever.md",
        title="Golden Retriever",
        content=(
            "# Golden Retriever\n\n"
            "Golden Retriever is friendly and intelligent.\n\n"
            "## Care\n\n"
            "Golden Retriever needs regular exercise."
        ),
        metadata={
            "loader_type": "markdown",
            "source_type": "local_file",
            "file_name": "golden_retriever.md",
        }
    )


@pytest.fixture
def long_markdown_document() -> RagDocument:
    """
    测试用长 Markdown 文档 fixture。

    功能：
        创建一篇内容较长的 RagDocument，
        用于测试 MarkdownChunker 是否会按照 chunk_size 切分文本。

    参数：
        无。

    返回值：
        RagDocument:
            测试用长文档。
    """

    long_text = "Golden Retriever is friendly. " * 100

    return RagDocument(
        doc_id="doc_long_001",
        source="data/dog_markdown/long.md",
        title="Long Dog Document",
        content=f"# Long Dog Document\n\n{long_text}",
        metadata={
            "loader_type": "markdown",
            "source_type": "local_file",
            "file_name": "long.md",
        }
    )


@pytest.fixture
def empty_markdown_document() -> RagDocument:
    """
    测试用空 Markdown 文档 fixture。

    功能：
        创建一篇正文为空的 RagDocument，
        用于测试 skip_empty=True 时是否返回空 Chunk 列表。

    参数：
        无。

    返回值：
        RagDocument:
            测试用空文档。
    """

    return RagDocument(
        doc_id="doc_empty_001",
        source="data/dog_markdown/empty.md",
        title="Empty",
        content="",
        metadata={
            "loader_type": "markdown",
            "source_type": "local_file",
            "file_name": "empty.md",
        }
    )


@pytest.fixture
def second_markdown_document() -> RagDocument:
    """
    第二篇测试用 Markdown 文档 fixture。

    功能：
        创建另一篇 RagDocument，
        用于测试 chunk_many 是否可以批量切分多篇文档。

    参数：
        无。

    返回值：
        RagDocument:
            第二篇测试文档。
    """

    return RagDocument(
        doc_id="doc_test_002",
        source="data/dog_markdown/border_collie.md",
        title="Border Collie",
        content=(
            "# Border Collie\n\n"
            "Border Collie is very smart."
        ),
        metadata={
            "loader_type": "markdown",
            "source_type": "local_file",
            "file_name": "border_collie.md",
        }
    )


def test_chunk_single_markdown_document(
    sample_markdown_document: RagDocument
):
    """
    测试切分单篇 Markdown 文档。

    功能：
        验证 MarkdownChunker 可以把 RagDocument 切分成 RagChunk 列表。

    参数：
        sample_markdown_document:
            pytest fixture 注入的测试文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = chunker.chunk(
        sample_markdown_document
    )

    assert len(chunks) == 2

    first_chunk = chunks[0]

    assert first_chunk.chunk_id.startswith("chunk_")
    assert first_chunk.doc_id == "doc_test_001"
    assert first_chunk.chunk_index == 0
    assert first_chunk.source == sample_markdown_document.source
    assert first_chunk.title == sample_markdown_document.title
    assert "Golden Retriever" in first_chunk.content


def test_chunk_preserves_document_metadata(
    sample_markdown_document: RagDocument
):
    """
    测试 Chunk 是否继承文档元数据。

    功能：
        验证 RagDocument.metadata 中的信息是否会传递到 RagChunk.metadata。

    参数：
        sample_markdown_document:
            pytest fixture 注入的测试文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = chunker.chunk(
        sample_markdown_document
    )

    first_chunk = chunks[0]

    assert first_chunk.metadata["loader_type"] == "markdown"
    assert first_chunk.metadata["source_type"] == "local_file"
    assert first_chunk.metadata["file_name"] == "golden_retriever.md"
    assert first_chunk.metadata["chunker_type"] == "markdown"


def test_chunk_adds_markdown_section_metadata(
    sample_markdown_document: RagDocument
):
    """
    测试 Chunk 是否添加 Markdown 章节元数据。

    功能：
        验证 MarkdownChunker 是否会在 metadata 中记录 section_title 和 section_level。

    参数：
        sample_markdown_document:
            pytest fixture 注入的测试文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = chunker.chunk(
        sample_markdown_document
    )

    section_titles = {
        chunk.metadata["section_title"]
        for chunk in chunks
    }

    assert "Golden Retriever" in section_titles
    assert "Care" in section_titles

    assert chunks[0].metadata["section_level"] == 1
    assert chunks[1].metadata["section_level"] == 2


def test_chunk_splits_long_document_by_chunk_size(
    long_markdown_document: RagDocument
):
    """
    测试长文档是否会被切分成多个 Chunk。

    功能：
        验证文档长度超过 chunk_size 时，
        MarkdownChunker 是否会生成多个 RagChunk。

    参数：
        long_markdown_document:
            pytest fixture 注入的长文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = chunker.chunk(
        long_markdown_document
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.content) <= 300


def test_chunk_overlap_keeps_repeated_boundary_text(
    long_markdown_document: RagDocument
):
    """
    测试 Chunk Overlap 是否生效。

    功能：
        验证相邻 Chunk 之间是否存在重叠文本。
        Overlap（重叠）可以降低切分边界造成的上下文丢失。

    参数：
        long_markdown_document:
            pytest fixture 注入的长文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = chunker.chunk(
        long_markdown_document
    )

    assert len(chunks) > 1

    first_chunk_tail = chunks[0].content[-50:]
    second_chunk = chunks[1].content

    assert first_chunk_tail in second_chunk


def test_chunk_empty_document_returns_empty_list(
    empty_markdown_document: RagDocument
):
    """
    测试空文档返回空 Chunk 列表。

    功能：
        验证 skip_empty=True 时，空文档不会生成 RagChunk。

    参数：
        empty_markdown_document:
            pytest fixture 注入的空文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker()

    chunks = chunker.chunk(
        empty_markdown_document
    )

    assert chunks == []


def test_chunk_many_documents(
    sample_markdown_document: RagDocument,
    second_markdown_document: RagDocument,
):
    """
    测试批量切分多篇文档。

    功能：
        验证 chunk_many 可以把多个 RagDocument 批量切分成 RagChunk 列表。

    参数：
        sample_markdown_document:
            pytest fixture 注入的第一篇测试文档。

        second_markdown_document:
            pytest fixture 注入的第二篇测试文档。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    chunker = MarkdownChunker(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = chunker.chunk_many(
        [
            sample_markdown_document,
            second_markdown_document,
        ]
    )

    doc_ids = {
        chunk.doc_id
        for chunk in chunks
    }

    assert "doc_test_001" in doc_ids
    assert "doc_test_002" in doc_ids
    assert len(chunks) == 3


def test_chunker_raises_error_when_chunk_size_invalid():
    """
    测试 chunk_size 参数非法时抛出异常。

    功能：
        验证 chunk_size 小于等于 0 时，MarkdownChunker 会抛出 ValueError。

    参数：
        无。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    with pytest.raises(ValueError):
        MarkdownChunker(
            chunk_size=0,
            chunk_overlap=0
        )


def test_chunker_raises_error_when_overlap_is_negative():
    """
    测试 chunk_overlap 为负数时抛出异常。

    功能：
        验证 chunk_overlap 小于 0 时，MarkdownChunker 会抛出 ValueError。

    参数：
        无。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    with pytest.raises(ValueError):
        MarkdownChunker(
            chunk_size=100,
            chunk_overlap=-1
        )


def test_chunker_raises_error_when_overlap_greater_than_chunk_size():
    """
    测试 chunk_overlap 大于等于 chunk_size 时抛出异常。

    功能：
        验证 chunk_overlap 大于等于 chunk_size 时，
        MarkdownChunker 会抛出 ValueError。

    参数：
        无。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    with pytest.raises(ValueError):
        MarkdownChunker(
            chunk_size=100,
            chunk_overlap=100
        )