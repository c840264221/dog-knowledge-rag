"""
Mock MCP Client。

功能：
    提供 V1.9.3 阶段学习用的 MCP 客户端模拟实现。
    它不连接真实 MCP Server，只在内存中保存工具返回值，方便测试执行适配链路。

专业名词：
    MCP Client：
        Model Context Protocol Client，模型上下文协议客户端。
        真实项目中负责连接 MCP Server 并调用外部工具。
    Mock：
        模拟对象，用于测试，不依赖真实外部服务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MockMcpToolCallRecord:
    """
    Mock MCP 工具调用记录。

    功能：
        保存一次 Mock MCP 工具调用的工具名和参数，方便单元测试断言。

    参数：
        name:
            被调用的 MCP 工具名称。

        arguments:
            调用 MCP 工具时传入的参数。

    返回值：
        MockMcpToolCallRecord:
            dataclass 数据对象，无额外方法返回值。
    """

    name: str
    arguments: dict[str, Any] = field(
        default_factory=dict
    )


class MockMcpClient:
    """
    Mock MCP 客户端。

    功能：
        模拟 MCP Client 的 call_tool 能力。
        测试中可以预先注册工具返回值，然后通过 call_tool 获取结果。

    参数：
        无。

    返回值：
        MockMcpClient:
            可用于单元测试的 MCP 客户端模拟对象。
    """

    def __init__(self) -> None:
        """
        初始化 Mock MCP 客户端。

        功能：
            创建内存工具结果表和调用记录列表。

        参数：
            无。

        返回值：
            None。
        """

        self._tool_results: dict[str, Any] = {}
        self.call_records: list[MockMcpToolCallRecord] = []

    def register_tool_result(
        self,
        name: str,
        result: Any,
    ) -> None:
        """
        注册 Mock MCP 工具返回值。

        功能：
            将指定工具名和返回值保存到内存中，后续 call_tool 会返回该值。

        参数：
            name:
                MCP 工具名称。

            result:
                MCP 工具模拟返回内容。

        返回值：
            None。
        """

        self._tool_results[name] = result

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """
        调用 Mock MCP 工具。

        功能：
            记录工具调用参数，并返回预先注册的工具结果。
            如果工具未注册，则抛出 KeyError，模拟 MCP 工具不存在。

        参数：
            name:
                MCP 工具名称。

            arguments:
                MCP 工具调用参数。

        返回值：
            Any:
                预先注册的工具返回内容。
        """

        resolved_arguments = dict(
            arguments
            or {}
        )
        self.call_records.append(
            MockMcpToolCallRecord(
                name=name,
                arguments=resolved_arguments,
            )
        )

        if name not in self._tool_results:
            raise KeyError(
                f"Mock MCP 工具不存在: {name}"
            )

        return self._tool_results[name]

    def get_last_call_record(self) -> MockMcpToolCallRecord | None:
        """
        获取最后一次工具调用记录。

        功能：
            返回 call_records 中最后一条记录，方便测试检查调用参数。

        参数：
            无。

        返回值：
            MockMcpToolCallRecord | None:
                有调用记录时返回最后一条；没有调用记录时返回 None。
        """

        if not self.call_records:
            return None

        return self.call_records[-1]
