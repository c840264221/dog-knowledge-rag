"""只读查询客户端：同一MCP服务发现记账与查询工具，但只执行查询，不新增支出。

先重启plus_expense_mcp_server，再运行：
python -m tests.learning.plus_expense_mcp_query_client --category 宠物 --limit 20
"""

import argparse
import asyncio
import json

from jsonschema import validate
from mcp import Client

from src.mcp.expenses.schemas import EXPENSE_CATEGORIES, ExpenseQueryInput, ExpenseQueryResult


async def query_expense_demo(arguments: dict) -> dict:
    """发现同服务工具并执行受限查询。

    参数：arguments为已校验的category与limit，不含用户身份。
    返回值：服务工具名称列表和结构化查询结果；协议错误或服务错误抛出，不伪装成查无数据。
    """
    async with Client("http://127.0.0.1:8766/mcp") as client:
        tools = (await client.list_tools()).tools
        schemas = {tool.name: tool.input_schema for tool in tools}
        if not {"record_expense", "query_expenses"}.issubset(schemas):
            raise ValueError("没有发现两个工具，请确认已重启最新记账服务。")
        validate(instance=arguments, schema=schemas["query_expenses"])
        result = await client.call_tool("query_expenses", arguments)
        if result.is_error:
            detail = "\n".join(block.text for block in result.content if block.type == "text")
            raise RuntimeError(f"MCP查询失败：{detail}")
        payload = result.structured_content
        if payload is None:
            texts = [block.text for block in result.content if block.type == "text"]
            if len(texts) != 1:
                raise ValueError("无法确定唯一查询结果。")
            payload = json.loads(texts[0])
        parsed = ExpenseQueryResult.model_validate(payload)
        return {
            "available_tools": sorted(schemas),
            "called_tool": "query_expenses",
            "query_result": parsed.model_dump(mode="json"),
            "notice": "本次仅查询已有账本，没有调用记账工具；无自动分类或生产认证。",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询教学账本，展示同一MCP服务的多个工具。")
    parser.add_argument("--category", choices=list(EXPENSE_CATEGORIES), default="宠物", help="查询的一级分类")
    parser.add_argument("--limit", type=int, default=20, help="最近记录条数，1至100")
    args = parser.parse_args()
    request = ExpenseQueryInput(category=args.category, limit=args.limit).model_dump(mode="json")
    print(json.dumps(asyncio.run(query_expense_demo(request)), ensure_ascii=False, indent=2))
