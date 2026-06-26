"""
RAG Debug 工具模块。

当前包用于存放 RAG 调试相关工具，例如：
1. Retriever Debug Report（检索调试报告）
2. Parser Debug Report（查询解析调试报告）
3. Index Debug Report（索引调试报告）
"""

from src.rag.debug.retriever_debug_report import (
    build_retriever_debug_report,
)


__all__ = [
    "build_retriever_debug_report",
]