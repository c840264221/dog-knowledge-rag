import pytest

from src.rag.loaders.markdown_loader import MarkdownDocumentLoader


def test_load_single_markdown_file(tmp_path):
    """
    测试加载单个 Markdown 文件。

    功能：
        验证 MarkdownDocumentLoader 能否把单个 .md 文件转换成 RagDocument。

    参数：
        tmp_path:
            pytest 提供的临时目录对象，用于创建测试文件。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    markdown_file = tmp_path / "golden_retriever.md"
    markdown_file.write_text(
        "# Golden Retriever\n\nGolden Retriever is friendly.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=markdown_file
    )

    documents = loader.load()

    assert len(documents) == 1

    document = documents[0]

    assert document.title == "Golden Retriever"
    assert document.content == "# Golden Retriever\n\nGolden Retriever is friendly."
    assert document.source.endswith("golden_retriever.md")
    assert document.doc_id.startswith("doc_")
    assert document.metadata["loader_type"] == "markdown"
    assert document.metadata["source_type"] == "local_file"
    assert document.metadata["file_name"] == "golden_retriever.md"
    assert document.metadata["file_suffix"] == ".md"
    assert "content_hash" in document.metadata


def test_load_markdown_directory_recursive(tmp_path):
    """
    测试递归加载 Markdown 目录。

    功能：
        验证 recursive=True 时，MarkdownDocumentLoader 能否扫描子目录中的 .md 文件。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    root_file = tmp_path / "border_collie.md"
    root_file.write_text(
        "# Border Collie\n\nBorder Collie is intelligent.",
        encoding="utf-8"
    )

    sub_dir = tmp_path / "large_dogs"
    sub_dir.mkdir()

    sub_file = sub_dir / "german_shepherd.md"
    sub_file.write_text(
        "# German Shepherd\n\nGerman Shepherd is loyal.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=tmp_path,
        recursive=True
    )

    documents = loader.load()

    titles = {
        document.title
        for document in documents
    }

    assert len(documents) == 2
    assert "Border Collie" in titles
    assert "German Shepherd" in titles


def test_load_markdown_directory_non_recursive(tmp_path):
    """
    测试非递归加载 Markdown 目录。

    功能：
        验证 recursive=False 时，MarkdownDocumentLoader 不会扫描子目录。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    root_file = tmp_path / "poodle.md"
    root_file.write_text(
        "# Poodle\n\nPoodle is smart.",
        encoding="utf-8"
    )

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()

    sub_file = sub_dir / "husky.md"
    sub_file.write_text(
        "# Husky\n\nHusky is energetic.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=tmp_path,
        recursive=False
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "Poodle"


def test_markdown_loader_ignores_non_markdown_files(tmp_path):
    """
    测试目录加载时忽略非 Markdown 文件。

    功能：
        验证 MarkdownDocumentLoader 在扫描目录时，只会加载 .md 文件。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    markdown_file = tmp_path / "beagle.md"
    markdown_file.write_text(
        "# Beagle\n\nBeagle is curious.",
        encoding="utf-8"
    )

    text_file = tmp_path / "notes.txt"
    text_file.write_text(
        "This should not be loaded.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=tmp_path
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "Beagle"


def test_markdown_loader_uses_file_stem_when_no_h1_title(tmp_path):
    """
    测试没有一级标题时使用文件名作为标题。

    功能：
        验证 Markdown 文件没有 # 一级标题时，
        MarkdownDocumentLoader 会使用文件名生成标题。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    markdown_file = tmp_path / "labrador_retriever.md"
    markdown_file.write_text(
        "Labrador Retriever is suitable for families.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=markdown_file
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "labrador retriever"


def test_markdown_loader_skips_empty_file_by_default(tmp_path):
    """
    测试默认跳过空 Markdown 文件。

    功能：
        验证 skip_empty=True 时，空文件不会生成 RagDocument。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    markdown_file = tmp_path / "empty.md"
    markdown_file.write_text(
        "",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=markdown_file
    )

    documents = loader.load()

    assert documents == []


def test_markdown_loader_can_keep_empty_file(tmp_path):
    """
    测试可以保留空 Markdown 文件。

    功能：
        验证 skip_empty=False 时，即使文件内容为空，也会生成 RagDocument。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    markdown_file = tmp_path / "empty.md"
    markdown_file.write_text(
        "",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=markdown_file,
        skip_empty=False
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "empty"
    assert documents[0].content == ""


def test_markdown_loader_raises_error_when_path_not_exists(tmp_path):
    """
    测试路径不存在时抛出异常。

    功能：
        验证 input_path 不存在时，MarkdownDocumentLoader 会抛出 FileNotFoundError。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    missing_path = tmp_path / "missing"

    loader = MarkdownDocumentLoader(
        input_path=missing_path
    )

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_markdown_loader_raises_error_when_single_file_is_not_markdown(tmp_path):
    """
    测试单文件不是 Markdown 时抛出异常。

    功能：
        验证 input_path 是单个非 .md 文件时，MarkdownDocumentLoader 会抛出 ValueError。

    参数：
        tmp_path:
            pytest 提供的临时目录对象。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    text_file = tmp_path / "notes.txt"
    text_file.write_text(
        "This is not markdown.",
        encoding="utf-8"
    )

    loader = MarkdownDocumentLoader(
        input_path=text_file
    )

    with pytest.raises(ValueError):
        loader.load()