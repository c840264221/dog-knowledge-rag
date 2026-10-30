"""按PDF物理页和文字边界切块；不把普通文本冒充标题结构化结果。"""

import hashlib

from src.rag.chunking.base_chunker import BaseDocumentChunker
from src.rag.schemas import RagChunk, RagDocument


class PdfTextChunker(BaseDocumentChunker):
    """PDF首版切块策略：页内优先在换行或空格处分割，保留一定重叠和页码。"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        """设置字符预算；chunk_size为最大字符数，chunk_overlap为重叠字符数；无返回值。"""
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("块长度必须为正，重叠须非负且小于块长度。")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RagDocument) -> list[RagChunk]:
        """切分一份PDF文档，文档级元数据继承到每个块。

        参数：document为PdfDocumentLoader返回的文档。
        返回值：按页序排列的RagChunk列表；每块保留page_number（物理页码）和
            source_doc_id（源文档编号），不把整份逐页原文复制进每块元数据。
            本版不跨页合并、不识别标题、不保证一个块包含全部条件，作为评测基线。
        """
        pages = document.metadata["pdf_pages"]
        selected = set(document.metadata["selected_page_numbers"])
        metadata = {key: value for key, value in document.metadata.items()
                    if key not in {"pdf_pages", "selected_page_numbers"}}
        chunks = []
        for page in pages:
            number = page["page_number"]
            if number not in selected:
                continue
            for page_index, text in enumerate(self._split_text(page["text"])):
                identity = f"{document.doc_id}:{number}:{page_index}:{text}"
                chunks.append(RagChunk(
                    chunk_id="chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    doc_id=document.doc_id, content=text, chunk_index=len(chunks),
                    source=document.source, title=document.title,
                    metadata={
                        **metadata, "page_number": number, "page_chunk_index": page_index,
                        "source_doc_id": document.doc_id, "chunker_type": "pdf_page_text_v1",
                        "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                    },
                ))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """按字符上限切分页内文字，优先保留词边界；参数text为原文；返回非空片段列表。"""
        parts = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                lower = start + self.chunk_size // 2
                boundary = text.rfind("\n", lower, end)
                if boundary < lower:
                    boundary = text.rfind(" ", lower, end)
                if boundary >= lower:
                    end = boundary + 1
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end == len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return parts
