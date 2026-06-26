from pathlib import Path
from typing import Callable

import pytest

from src.rag.loaders.markdown_loader import MarkdownDocumentLoader


@pytest.fixture
def test_file_factory(tmp_path) -> Callable[[str, str], Path]:
    """
    测试文件工厂 fixture。

    功能：
        在 pytest 临时目录中创建测试文件。
        Factory（工厂）在这里表示“用来批量创建对象或资源的辅助函数”。

    参数：
        tmp_path:
            pytest 内置 fixture，用于提供临时目录。

    返回值：
        Callable[[str, str], Path]:
            返回一个内部函数，用于创建指定名称和内容的测试文件。
    """

    def _create_file(
        relative_path: str,
        content: str = "",
        encoding: str = "utf-8"
    ) -> Path:
        """
        创建测试文件。

        功能：
            根据相对路径，在 tmp_path 临时目录下创建文件。
            如果文件所在目录不存在，会自动创建父目录。

        参数：
            relative_path:
                文件相对于 tmp_path 的路径，例如 golden.md 或 sub/husky.md。

            content:
                文件内容。

            encoding:
                文件编码格式，默认是 utf-8。

        返回值：
            Path:
                创建完成后的文件路径。
        """

        file_path = tmp_path / relative_path

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path.write_text(
            content,
            encoding=encoding
        )

        return file_path

    return _create_file


@pytest.fixture
def golden_markdown_file(
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    金毛 Markdown 文件 fixture。

    功能：
        创建一个带有一级标题的 Markdown 测试文件。

    参数：
        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            创建完成的 golden_retriever.md 文件路径。
    """

    return test_file_factory(
        "golden_retriever.md",
        "# Golden Retriever\n\nGolden Retriever is friendly."
    )


@pytest.fixture
def recursive_markdown_dir(
    tmp_path,
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    递归目录 fixture。

    功能：
        创建一个包含根目录 Markdown 文件和子目录 Markdown 文件的测试目录。

    参数：
        tmp_path:
            pytest 内置临时目录 fixture。

        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            包含多个 Markdown 文件的临时目录路径。
    """

    test_file_factory(
        "border_collie.md",
        "# Border Collie\n\nBorder Collie is intelligent."
    )

    test_file_factory(
        "large_dogs/german_shepherd.md",
        "# German Shepherd\n\nGerman Shepherd is loyal."
    )

    return tmp_path


@pytest.fixture
def non_recursive_markdown_dir(
    tmp_path,
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    非递归目录 fixture。

    功能：
        创建一个根目录 Markdown 文件和一个子目录 Markdown 文件，
        用于测试 recursive=False 时是否只加载根目录文件。

    参数：
        tmp_path:
            pytest 内置临时目录 fixture。

        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            测试目录路径。
    """

    test_file_factory(
        "poodle.md",
        "# Poodle\n\nPoodle is smart."
    )

    test_file_factory(
        "sub/husky.md",
        "# Husky\n\nHusky is energetic."
    )

    return tmp_path


@pytest.fixture
def mixed_file_dir(
    tmp_path,
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    混合文件目录 fixture。

    功能：
        创建一个同时包含 Markdown 文件和非 Markdown 文件的测试目录，
        用于验证 Loader 是否只加载 .md 文件。

    参数：
        tmp_path:
            pytest 内置临时目录 fixture。

        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            混合文件测试目录路径。
    """

    test_file_factory(
        "beagle.md",
        "# Beagle\n\nBeagle is curious."
    )

    test_file_factory(
        "notes.txt",
        "This should not be loaded."
    )

    return tmp_path


@pytest.fixture
def markdown_without_h1_file(
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    无一级标题 Markdown 文件 fixture。

    功能：
        创建一个没有 # 一级标题的 Markdown 文件，
        用于测试 Loader 是否会使用文件名作为标题。

    参数：
        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            创建完成的 Markdown 文件路径。
    """

    return test_file_factory(
        "labrador_retriever.md",
        "Labrador Retriever is suitable for families."
    )


@pytest.fixture
def empty_markdown_file(
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    空 Markdown 文件 fixture。

    功能：
        创建一个内容为空的 Markdown 文件。

    参数：
        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            创建完成的空 Markdown 文件路径。
    """

    return test_file_factory(
        "empty.md",
        ""
    )


@pytest.fixture
def missing_path(tmp_path) -> Path:
    """
    不存在路径 fixture。

    功能：
        构造一个位于 pytest 临时目录下、但没有实际创建的路径，
        用于测试路径不存在时 Loader 是否抛出异常。

    参数：
        tmp_path:
            pytest 内置临时目录 fixture。

    返回值：
        Path:
            一个不存在的路径对象。
    """

    return tmp_path / "missing"


@pytest.fixture
def non_markdown_file(
    test_file_factory: Callable[[str, str], Path]
) -> Path:
    """
    非 Markdown 文件 fixture。

    功能：
        创建一个 .txt 文件，用于测试单文件路径不是 .md 时是否抛出异常。

    参数：
        test_file_factory:
            测试文件工厂 fixture，用于创建临时文件。

    返回值：
        Path:
            创建完成的 notes.txt 文件路径。
    """

    return test_file_factory(
        "notes.txt",
        "This is not markdown."
    )


def test_load_single_markdown_file(golden_markdown_file: Path):
    """
    测试加载单个 Markdown 文件。

    功能：
        验证 MarkdownDocumentLoader 能否把单个 .md 文件转换成 RagDocument。

    参数：
        golden_markdown_file:
            pytest fixture 注入的 Markdown 文件路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=golden_markdown_file
    )

    documents = loader.load()

    assert len(documents) == 1

    document = documents[0]

    assert document.title == "Golden Retriever"
    assert document.content == (
        "# Golden Retriever\n\nGolden Retriever is friendly."
    )
    assert document.source.endswith("golden_retriever.md")
    assert document.doc_id.startswith("doc_")
    assert document.metadata["loader_type"] == "markdown"
    assert document.metadata["source_type"] == "local_file"
    assert document.metadata["file_name"] == "golden_retriever.md"
    assert document.metadata["file_suffix"] == ".md"
    assert "content_hash" in document.metadata


def test_load_markdown_directory_recursive(
    recursive_markdown_dir: Path
):
    """
    测试递归加载 Markdown 目录。

    功能：
        验证 recursive=True 时，MarkdownDocumentLoader 能否扫描子目录中的 .md 文件。

    参数：
        recursive_markdown_dir:
            pytest fixture 注入的递归测试目录路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=recursive_markdown_dir,
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


def test_load_markdown_directory_non_recursive(
    non_recursive_markdown_dir: Path
):
    """
    测试非递归加载 Markdown 目录。

    功能：
        验证 recursive=False 时，MarkdownDocumentLoader 不会扫描子目录。

    参数：
        non_recursive_markdown_dir:
            pytest fixture 注入的非递归测试目录路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=non_recursive_markdown_dir,
        recursive=False
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "Poodle"


def test_markdown_loader_ignores_non_markdown_files(
    mixed_file_dir: Path
):
    """
    测试目录加载时忽略非 Markdown 文件。

    功能：
        验证 MarkdownDocumentLoader 在扫描目录时，只会加载 .md 文件。

    参数：
        mixed_file_dir:
            pytest fixture 注入的混合文件目录路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=mixed_file_dir
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "Beagle"


def test_markdown_loader_uses_file_stem_when_no_h1_title(
    markdown_without_h1_file: Path
):
    """
    测试没有一级标题时使用文件名作为标题。

    功能：
        验证 Markdown 文件没有 # 一级标题时，
        MarkdownDocumentLoader 会使用文件名生成标题。

    参数：
        markdown_without_h1_file:
            pytest fixture 注入的无一级标题 Markdown 文件路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=markdown_without_h1_file
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "labrador retriever"


def test_markdown_loader_skips_empty_file_by_default(
    empty_markdown_file: Path
):
    """
    测试默认跳过空 Markdown 文件。

    功能：
        验证 skip_empty=True 时，空文件不会生成 RagDocument。

    参数：
        empty_markdown_file:
            pytest fixture 注入的空 Markdown 文件路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=empty_markdown_file
    )

    documents = loader.load()

    assert documents == []


def test_markdown_loader_can_keep_empty_file(
    empty_markdown_file: Path
):
    """
    测试可以保留空 Markdown 文件。

    功能：
        验证 skip_empty=False 时，即使文件内容为空，也会生成 RagDocument。

    参数：
        empty_markdown_file:
            pytest fixture 注入的空 Markdown 文件路径。

    返回值：
        无。pytest 会根据断言结果判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=empty_markdown_file,
        skip_empty=False
    )

    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].title == "empty"
    assert documents[0].content == ""


def test_markdown_loader_raises_error_when_path_not_exists(
    missing_path: Path
):
    """
    测试路径不存在时抛出异常。

    功能：
        验证 input_path 不存在时，MarkdownDocumentLoader 会抛出 FileNotFoundError。

    参数：
        missing_path:
            pytest fixture 注入的不存在路径对象。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=missing_path
    )

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_markdown_loader_raises_error_when_single_file_is_not_markdown(
    non_markdown_file: Path
):
    """
    测试单文件不是 Markdown 时抛出异常。

    功能：
        验证 input_path 是单个非 .md 文件时，
        MarkdownDocumentLoader 会抛出 ValueError。

    参数：
        non_markdown_file:
            pytest fixture 注入的非 Markdown 文件路径。

    返回值：
        无。pytest 会根据是否抛出异常判断测试是否通过。
    """

    loader = MarkdownDocumentLoader(
        input_path=non_markdown_file
    )

    with pytest.raises(ValueError):
        loader.load()