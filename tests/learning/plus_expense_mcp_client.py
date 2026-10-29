"""通过真实本机HTTP调用记账MCP工具；客户端不导入存储或服务端业务函数。

先启动plus_expense_mcp_server，再运行：
python -m tests.learning.plus_expense_mcp_client --expense-date 2026-10-23
默认演示内容是宠物洗护80元，只供测试；确认后才写入独立教学账本。
"""

import argparse
import asyncio
import json
from uuid import uuid4

from jsonschema import validate
from mcp import Client

from src.mcp.expenses.schemas import ExpenseBatchWriteResult, ExpenseInput, ExpenseWriteResult


def parse_expense_result(result, result_model=ExpenseWriteResult) -> ExpenseWriteResult | ExpenseBatchWriteResult:
    """解析MCP回执；result为SDK结果，result_model为单笔或批量契约；返回校验后回执，错误抛出。"""
    if result.is_error:
        detail = "\n".join(block.text for block in result.content if block.type == "text")
        raise RuntimeError(f"MCP记账失败：{detail}")
    payload = result.structured_content
    if payload is None:
        texts = [block.text for block in result.content if block.type == "text"]
        if len(texts) != 1:
            raise ValueError("无法从MCP结果确定唯一记账回执。")
        payload = json.loads(texts[0])
    return result_model.model_validate(payload)


async def submit_expense_demo(arguments: dict) -> dict:
    """连接服务并重复提交同一动作；arguments为已确认的参数；返回两次回执及比较结果。

    此函数只负责通信，终端入口在调用前取得人工确认；没有自动重试或模型选择。
    网络错误继续抛出，不把“没收到响应”解释成“没有保存”。重试必须复用原幂等键。
    """
    async with Client("http://127.0.0.1:8766/mcp") as client:
        tools = (await client.list_tools()).tools
        schemas = {tool.name: tool.input_schema for tool in tools}
        tool_name = "record_expense"
        if tool_name not in schemas:
            raise ValueError("当前服务没有提供record_expense，拒绝调用。")
        validate(instance=arguments, schema=schemas[tool_name])
        print("已发现记账工具，参数结构检查通过；现在提交已确认内容。", flush=True)
        first = parse_expense_result(await client.call_tool(tool_name, arguments))
        print(f"首次返回：{first.status}；记录编号：{first.record.expense_id}", flush=True)
        second = parse_expense_result(await client.call_tool(tool_name, arguments))
        if second.status != "already_recorded" or first.record != second.record:
            raise RuntimeError("重复提交没有返回同一记录，请检查服务端。")
        return {
            "first_submission": first.model_dump(mode="json"),
            "duplicate_submission": second.model_dump(mode="json"),
            "same_expense_id": first.record.expense_id == second.record.expense_id,
            "checks_passed": True,
            "notice": "重复调用返回同一持久化记录；尚未接入自然语言分类、主图或生产认证。",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本机MCP记账教学；确认后产生真实教学数据库写入。")
    parser.add_argument("--idempotency-key", help="同一次提交重试时传入原键；不填则生成新的动作编号")
    parser.add_argument("--expense-date", required=True, help="实际账单日期，YYYY-MM-DD；补记旧账时填写当时日期")
    args = parser.parse_args()
    request = ExpenseInput(
        amount="80.00", category="宠物", subcategory="洗护",
        description="教学演示：狗子洗澡", expense_date=args.expense_date,
        idempotency_key=args.idempotency_key or "expense-demo:" + uuid4().hex,
    ).model_dump(mode="json")
    print("即将向独立教学账本写入以下演示支出：", flush=True)
    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)
    print("本次将使用同一个键提交两次以验证幂等；网络异常后请保留原键，勿换键重试。", flush=True)
    confirmation = input("确认写入这笔教学记录？输入 y 继续，其他输入取消：").strip().casefold()
    if confirmation == "y":
        print(json.dumps(asyncio.run(submit_expense_demo(request)), ensure_ascii=False, indent=2))
    else:
        print("已取消，没有调用记账工具。")
