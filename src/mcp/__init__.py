"""
MCP 学习模块。

功能：
    收拢 Model Context Protocol（模型上下文协议）相关的学习型 schema 和适配代码。

当前阶段：
    V1.9.2 只定义 Mock MCP 数据结构，不连接真实 MCP Server。
"""

from src.mcp.schemas import (
    MockMcpToolDefinition,
)
from src.mcp.mock_client import (
    MockMcpClient,
    MockMcpToolCallRecord,
)
from src.mcp.sqlite import (
    SQLiteMcpReadonlyClient,
)

__all__ = [
    "MockMcpClient",
    "MockMcpToolCallRecord",
    "MockMcpToolDefinition",
    "SQLiteMcpReadonlyClient",
]
