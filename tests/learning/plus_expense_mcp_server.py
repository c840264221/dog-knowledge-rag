"""本机记账MCP服务：分模块注册记账与查询工具，固定教学身份，无认证，不得对外开放。

在项目根目录运行：python -m tests.learning.plus_expense_mcp_server
依赖与原MCP教学示例相同，使用已安装的MCP Python SDK 2.x。
"""

from pathlib import Path
from mcp.server import MCPServer

from src.mcp.expenses.query_tools import register_expense_query_tools
from src.mcp.expenses.schemas import EXPENSE_CATEGORIES
from src.mcp.expenses.store import ExpenseStore
from src.mcp.expenses.write_tools import register_expense_write_tools


def build_expense_server(database_path: Path) -> MCPServer:
    """装配独立服务；database_path为教学数据库路径；返回已注册记账和查询工具的同一服务。"""
    store = ExpenseStore(database_path, user_id="expense_learning_user")
    server = MCPServer("ExpenseLearningDemo")

    register_expense_write_tools(server, store)
    register_expense_query_tools(server, store)
    return server


if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="本机教学记账服务，可指定独立数据库和端口。")
    parser.add_argument("--database", type=Path, default=root / "output/learning/expenses/expenses.sqlite3")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须为1至65535。")
    database_path = args.database.resolve()
    server = build_expense_server(database_path)
    print(f"教学记账数据库：{database_path}", flush=True)
    print(f"允许分类：{EXPENSE_CATEGORIES}", flush=True)
    print(f"仅监听127.0.0.1:{args.port}，无认证；只供本机教学，按Ctrl+C结束。", flush=True)
    server.run(transport="streamable-http", host="127.0.0.1", port=args.port)
