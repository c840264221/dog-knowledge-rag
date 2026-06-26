"""
RAG Chunking 模块初始化文件。

功能：
    标记 src.rag.chunking 为 Python Package（Python 包），
    并统一导出常用的 Chunker（文本切分器）。

说明：
    Chunker 负责把 RagDocument 切分成 RagChunk，
    是 RAG 索引管线中的重要步骤。
"""

from src.rag.chunking.base_chunker import BaseDocumentChunker
from src.rag.chunking.markdown_chunker import MarkdownChunker

__all__ = [
    "BaseDocumentChunker",
    "MarkdownChunker",
]