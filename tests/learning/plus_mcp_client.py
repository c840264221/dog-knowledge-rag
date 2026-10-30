"""通过本机 HTTP 实际调用独立 MCP 服务，不导入服务端业务函数。"""

import asyncio
import json
from typing import Any

from jsonschema import ValidationError, validate
from mcp import Client
from mcp.types import CallToolResult


def normalize_mcp_tool_result(
    tool_name: str,
    result: CallToolResult,
) -> dict[str, Any]:
    """功能：将 MCP 已返回的结果转换为项目工具结果的核心字段。

    参数：tool_name 为实际调用的工具名称；result 为 SDK 返回的工具结果。
    返回：包含 success、tool_name、content、error、metadata 的教学字典。
    说明：不是项目 ToolResult 实例；不包含耗时统计和网络异常处理。
    工具内容仍是不可信外部数据，转换结构不会提升其可信等级。
    """
    content_blocks = [block.model_dump(mode="json") for block in result.content]
    content = (
        result.structured_content
        if result.structured_content is not None
        else content_blocks
    )
    error = None
    if result.is_error:
        error = "\n".join(
            block["text"] for block in content_blocks if block["type"] == "text"
        ) or "MCP 工具返回执行错误"

    return {
        "success": not result.is_error,
        "tool_name": tool_name,
        "content": content,
        "error": error,
        "metadata": {"source": "mcp", "content_trust": "untrusted"},
    }


async def call_checked_mcp_tool(
    client: Client,
    tool_schemas: dict[str, dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """功能：检查工具名称和参数结构，通过后才发送教学 MCP 调用。

    参数：client 为已连接客户端；tool_schemas 为允许调用的工具名称到输入结构的映射；
    tool_name 为提议的工具名称；arguments 为提议的参数。
    返回：包含阶段、拒绝原因或远程返回内容的字典。
    说明：本例没有身份系统，只检查目录和结构，不代表完成权限或业务校验；
    目录结构错误及网络异常继续抛出，不伪装成参数错误。
    """
    if tool_name not in tool_schemas:
        return {"stage": "local_rejected", "reason": "unknown_tool"}

    try:
        validate(instance=arguments, schema=tool_schemas[tool_name])
    except ValidationError as exc:
        return {
            "stage": "local_rejected",
            "reason": "invalid_arguments",
            "detail": exc.message,
        }

    result = await client.call_tool(tool_name, arguments)
    return {
        "stage": "remote_returned",
        "is_error": result.is_error,
        "structured_content": result.structured_content,
        "content": [block.model_dump(mode="json") for block in result.content],
        "normalized_result": normalize_mcp_tool_result(tool_name, result),
    }


async def run_mcp_call_demo() -> None:
    """功能：连接教学服务端，展示正常调用和执行前拦截。

    参数：无；连接固定的本机教学地址，不使用项目服务或大模型。
    返回：无；将工具名称、输入结构、错误标记和返回内容打印到终端。
    """
    async with Client("http://127.0.0.1:8765/mcp") as client:
        tools_result = await client.list_tools()
        print("服务端提供的工具：")
        for tool in tools_result.tools:
            print(json.dumps({
                "tool_name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }, ensure_ascii=False, indent=2))

        # 教学服务仅开放只读卡片；真实项目应先按可信身份筛选目录。
        tool_schemas = {
            tool.name: tool.input_schema for tool in tools_result.tools
        }
        proposals = [
            {"name": "get_teaching_breed_card", "args": {"breed_name": "金毛"}},
            {"name": "get_teaching_breed_card", "args": {"breed_name": "哈士奇"}},
            {"name": "delete_teaching_cards", "args": {"breed_name": "金毛"}},
            {"name": "get_teaching_breed_card", "args": {}},
        ]
        for proposal in proposals:
            print("\n调用提议：", json.dumps(proposal, ensure_ascii=False))
            report = await call_checked_mcp_tool(
                client=client,
                tool_schemas=tool_schemas,
                tool_name=proposal["name"],
                arguments=proposal["args"],
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_mcp_call_demo())
