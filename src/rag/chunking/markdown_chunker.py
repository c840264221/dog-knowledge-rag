import hashlib
import re

from src.rag.chunking.base_chunker import BaseDocumentChunker
from src.rag.schemas import (
    RagDocument,
    RagChunk,
)


class MarkdownChunker(BaseDocumentChunker):
    """
    Markdown 文档切分器。

    功能：
        把 Markdown 格式的 RagDocument 切分成多个 RagChunk。
        当前版本支持 Markdown 标题感知切分，并在每个 Chunk 的 metadata 中记录章节信息。

    技术名词：
        Markdown:
            一种轻量级标记语言，常用于 .md 文档。

        Heading:
            标题，例如 # 一级标题、## 二级标题。

        Chunk Size:
            文本块最大长度。

        Chunk Overlap:
            文本块重叠长度，用于减少切分边界造成的语义丢失。

        Metadata:
            元数据，用来保存文本块的来源、章节、切分参数等附加信息。

    参数：
        chunk_size:
            每个文本块的最大字符数。

        chunk_overlap:
            相邻文本块之间的重叠字符数。

        skip_empty:
            是否跳过空文本块。

    返回值：
        MarkdownChunker 实例。
    """

    HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        skip_empty: bool = True,
    ):
        """
        初始化 Markdown 文档切分器。

        功能：
            保存切分参数，并检查 chunk_size 和 chunk_overlap 是否合法。

        参数：
            chunk_size:
                每个文本块的最大字符数，必须大于 0。

            chunk_overlap:
                相邻文本块之间的重叠字符数，必须大于等于 0，
                并且必须小于 chunk_size。

            skip_empty:
                是否跳过空文本块。

        返回值：
            无。该方法用于初始化对象。
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size 必须大于 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap 必须大于等于 0"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap 必须小于 chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.skip_empty = skip_empty

    def chunk(
        self,
        document: RagDocument
    ) -> list[RagChunk]:
        """
        切分单篇 Markdown 文档。

        功能：
            先按照 Markdown 标题把文档拆成章节，
            再按照 chunk_size 和 chunk_overlap 把章节内容切成多个 RagChunk。

        参数：
            document:
                需要切分的 RAG 原始文档。

        返回值：
            list[RagChunk]:
                切分后的文本块列表。
        """

        content = document.content.strip()

        if self.skip_empty and not content:
            return []

        sections = self._split_markdown_sections(
            content=content,
            default_title=document.title
        )

        chunks: list[RagChunk] = []
        global_chunk_index = 0

        for section_index, section in enumerate(sections):
            section_chunks = self._split_text_by_size(
                section["content"]
            )

            for section_chunk_index, chunk_content in enumerate(section_chunks):
                normalized_content = chunk_content.strip()

                if self.skip_empty and not normalized_content:
                    continue

                chunk_id = self._build_chunk_id(
                    document=document,
                    chunk_index=global_chunk_index,
                    content=normalized_content
                )

                chunk_metadata = self._build_chunk_metadata(
                    document=document,
                    section=section,
                    section_index=section_index,
                    section_chunk_index=section_chunk_index
                )

                chunks.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        content=normalized_content,
                        chunk_index=global_chunk_index,
                        source=document.source,
                        title=document.title,
                        metadata=chunk_metadata
                    )
                )

                global_chunk_index += 1

        return chunks

    def _split_markdown_sections(
        self,
        content: str,
        default_title: str = "",
    ) -> list[dict]:
        """
        按 Markdown 标题切分章节。

        功能：
            扫描 Markdown 文本中的标题行。
            每遇到一个标题，就开始一个新的章节。
            如果文档开头没有标题，则使用 default_title 作为章节标题。

        参数：
            content:
                Markdown 文档正文。

            default_title:
                默认章节标题，通常使用 RagDocument.title。

        返回值：
            list[dict]:
                章节列表。
                每个章节包含 section_title、section_level 和 content。
        """

        sections: list[dict] = []

        current_lines: list[str] = []
        current_title = default_title or "untitled"
        current_level = 0

        for line in content.splitlines():
            heading_match = self.HEADING_PATTERN.match(
                line.strip()
            )

            if heading_match and current_lines:
                sections.append(
                    {
                        "section_title": current_title,
                        "section_level": current_level,
                        "content": "\n".join(current_lines).strip(),
                    }
                )

                current_lines = []

            if heading_match:
                current_level = len(
                    heading_match.group(1)
                )
                current_title = heading_match.group(2).strip()

            current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "section_title": current_title,
                    "section_level": current_level,
                    "content": "\n".join(current_lines).strip(),
                }
            )

        return sections

    def _split_text_by_size(
        self,
        text: str
    ) -> list[str]:
        """
        按固定长度切分文本。

        功能：
            根据 chunk_size 把长文本切成多个片段。
            如果设置了 chunk_overlap，则相邻片段之间保留一部分重叠文本。

        参数：
            text:
                需要切分的文本。

        返回值：
            list[str]:
                切分后的文本片段列表。
        """

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            chunk_text = text[start:end]

            chunks.append(
                chunk_text
            )

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _build_chunk_id(
        self,
        document: RagDocument,
        chunk_index: int,
        content: str,
    ) -> str:
        """
        生成文本块 ID。

        功能：
            根据文档 ID、文本块顺序编号和文本内容生成稳定的 chunk_id。
            Hash（哈希）可以把不同内容转换成固定长度字符串。

        参数：
            document:
                当前 Chunk 所属的原始文档。

            chunk_index:
                当前 Chunk 在文档中的全局顺序编号。

            content:
                当前 Chunk 的正文内容。

        返回值：
            str:
                文本块唯一 ID。
        """

        raw_value = (
            f"{document.doc_id}:{chunk_index}:{content}"
        )

        digest = hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()

        return f"chunk_{digest[:16]}"

    def _build_chunk_metadata(
        self,
        document: RagDocument,
        section: dict,
        section_index: int,
        section_chunk_index: int,
    ) -> dict:
        """
        构建文本块元数据。

        功能：
            合并文档 metadata，并追加 Chunker 生成的章节信息和切分参数。

        参数：
            document:
                当前 Chunk 所属的原始文档。

            section:
                当前 Chunk 所属的 Markdown 章节信息。

            section_index:
                当前章节在文档中的顺序编号。

            section_chunk_index:
                当前 Chunk 在章节中的顺序编号。

        返回值：
            dict:
                文本块元数据。
        """

        metadata = dict(
            document.metadata
        )

        metadata.update(
            {
                "chunker_type": "markdown",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "section_title": section["section_title"],
                "section_level": section["section_level"],
                "section_index": section_index,
                "section_chunk_index": section_chunk_index,
                "source_doc_id": document.doc_id,
                "source_title": document.title,
            }
        )

        return metadata