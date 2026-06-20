"""
RAG Loaders 模块初始化文件。

功能：
    标记 src.rag.loaders 为 Python Package（Python 包），
    并统一导出常用的 Loader（加载器）。

说明：
    Loader 负责把外部数据源转换成 RAG 系统内部标准文档结构。
"""

from src.rag.loaders.base_loader import BaseDocumentLoader
from src.rag.loaders.markdown_loader import MarkdownDocumentLoader

__all__ = [
    "BaseDocumentLoader",
    "MarkdownDocumentLoader",
]