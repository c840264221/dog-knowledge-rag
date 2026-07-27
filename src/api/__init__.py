"""
Dog Agent Framework 的 HTTP API 入口包。

该包只负责把外部 HTTP 请求适配到现有 Runtime 和 Main Graph，
不在 API 层重复实现 Agent、RAG、Memory 或多 Agent 业务逻辑。
"""

from src.api.app import create_app

__all__ = [
    "create_app",
]
