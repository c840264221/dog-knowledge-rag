import hashlib
from pathlib import Path

from src.rag.loaders.base_loader import BaseDocumentLoader
from src.rag.schemas import RagDocument


class MarkdownDocumentLoader(BaseDocumentLoader):
    """
    Markdown 文档加载器。

    功能：
        从本地 Markdown（轻量级标记语言）文件或目录中读取 .md 文件，
        并转换成 RAG 系统内部统一的 RagDocument 模型。

    参数：
        input_path:
            Markdown 文件路径或 Markdown 目录路径。

        encoding:
            文件编码格式，默认使用 utf-8。

        recursive:
            是否递归扫描子目录。
            True 表示扫描当前目录和所有子目录；
            False 表示只扫描当前目录。

        skip_empty:
            是否跳过空文件。
            True 表示空 Markdown 文件不会生成 RagDocument；
            False 表示即使内容为空也会生成 RagDocument。

    返回值：
        MarkdownDocumentLoader 实例。
    """

    def __init__(
        self,
        input_path: str | Path,
        encoding: str = "utf-8",
        recursive: bool = True,
        skip_empty: bool = True,
    ):
        """
        初始化 Markdown 文档加载器。

        功能：
            保存 Markdown 文件路径、编码格式、递归扫描配置等参数。

        参数：
            input_path:
                Markdown 文件路径或目录路径。

            encoding:
                文件编码格式，默认是 utf-8。

            recursive:
                是否递归扫描子目录。

            skip_empty:
                是否跳过空 Markdown 文件。

        返回值：
            无。该方法用于初始化对象。
        """

        self.input_path = Path(input_path)
        self.encoding = encoding
        self.recursive = recursive
        self.skip_empty = skip_empty

    def load(self) -> list[RagDocument]:
        """
        加载 Markdown 文档。

        功能：
            判断 input_path 是单个 Markdown 文件还是目录。
            如果是文件，则加载单个文件；
            如果是目录，则加载目录下的所有 .md 文件。

        参数：
            无。

        返回值：
            list[RagDocument]:
                加载完成后的 RAG 原始文档列表。
        """

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Markdown 路径不存在: {self.input_path}"
            )

        if self.input_path.is_file():
            document = self._load_file(self.input_path)
            return [document] if document is not None else []

        documents: list[RagDocument] = []

        for file_path in self._iter_markdown_files():
            document = self._load_file(file_path)

            if document is not None:
                documents.append(document)

        documents.sort(
            key=lambda item: item.source
        )

        return documents

    def _iter_markdown_files(self) -> list[Path]:
        """
        遍历 Markdown 文件。

        功能：
            根据 recursive 参数，查找 input_path 目录下的 Markdown 文件。

        参数：
            无。

        返回值：
            list[Path]:
                Markdown 文件路径列表。
        """

        if self.recursive:
            return list(self.input_path.rglob("*.md"))

        return list(self.input_path.glob("*.md"))

    def _load_file(self, file_path: Path) -> RagDocument | None:
        """
        加载单个 Markdown 文件。

        功能：
            读取一个 .md 文件的正文内容，提取标题，生成文档 ID，
            并构造 RagDocument 对象。

        参数：
            file_path:
                Markdown 文件路径。

        返回值：
            RagDocument | None:
                如果文件内容有效，返回 RagDocument；
                如果 skip_empty=True 且文件为空，返回 None。
        """

        if file_path.suffix.lower() != ".md":
            raise ValueError(
                f"MarkdownDocumentLoader 只支持 .md 文件: {file_path}"
            )

        raw_content = file_path.read_text(
            encoding=self.encoding
        )

        content = raw_content.strip()

        if self.skip_empty and not content:
            return None

        source = self._normalize_path(file_path)
        title = self._extract_title(content, file_path)
        doc_id = self._build_doc_id(source)
        content_hash = self._build_content_hash(content)
        relative_path = self._build_relative_path(file_path)

        return RagDocument(
            doc_id=doc_id,
            source=source,
            title=title,
            content=content,
            metadata={
                "loader_type": "markdown",
                "source_type": "local_file",
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "file_suffix": file_path.suffix,
                "relative_path": relative_path,
                "content_hash": content_hash,
            }
        )

    def _extract_title(
        self,
        content: str,
        file_path: Path
    ) -> str:
        """
        提取 Markdown 标题。

        功能：
            优先从 Markdown 一级标题中提取文档标题。
            如果没有一级标题，则使用文件名作为标题。

        参数：
            content:
                Markdown 文档正文。

            file_path:
                Markdown 文件路径。

        返回值：
            str:
                文档标题。
        """

        for line in content.splitlines():
            stripped_line = line.strip()

            if stripped_line.startswith("# "):
                return stripped_line.removeprefix("# ").strip()

        return file_path.stem.replace("_", " ").strip()

    def _build_doc_id(self, source: str) -> str:
        """
        生成文档 ID。

        功能：
            根据文档来源路径生成稳定的 doc_id。
            Hash（哈希）可以把较长路径转换成固定长度字符串。

        参数：
            source:
                文档来源路径。

        返回值：
            str:
                文档唯一 ID。
        """

        digest = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()

        return f"doc_{digest[:16]}"

    def _build_content_hash(self, content: str) -> str:
        """
        生成文档内容 Hash。

        功能：
            根据 Markdown 正文内容生成 hash，用于判断文档内容是否变化。

        参数：
            content:
                Markdown 文档正文。

        返回值：
            str:
                文档内容 hash。
        """

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def _build_relative_path(self, file_path: Path) -> str:
        """
        生成相对路径。

        功能：
            尽量生成相对于 input_path 的相对路径，方便后续展示和 metadata 记录。

        参数：
            file_path:
                Markdown 文件路径。

        返回值：
            str:
                相对路径字符串。
        """

        base_path = (
            self.input_path
            if self.input_path.is_dir()
            else self.input_path.parent
        )

        try:
            return file_path.relative_to(base_path).as_posix()
        except ValueError:
            return file_path.name

    def _normalize_path(self, file_path: Path) -> str:
        """
        规范化文件路径。

        功能：
            把 Path 对象转换成统一的 posix 风格路径字符串。
            posix 风格路径使用 /，可以减少 Windows 和 Linux 路径差异。

        参数：
            file_path:
                文件路径。

        返回值：
            str:
                规范化后的路径字符串。
        """

        return file_path.resolve().as_posix()