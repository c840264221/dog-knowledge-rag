"""查询工具注册模块；复用已有服务和绑定身份的存储，只开放受限只读查询。"""

from typing import TYPE_CHECKING, Literal

from src.mcp.expenses.schemas import ExpenseQueryInput, ExpenseQueryResult
from src.mcp.expenses.store import ExpenseStore

if TYPE_CHECKING:
    from mcp.server import MCPServer


def register_expense_query_tools(server: "MCPServer", store: ExpenseStore) -> None:
    """向已有服务注册查询工具。

    参数：server为与记账工具共用的MCP服务；store为绑定用户身份的存储。
    返回值：None；注册query_expenses，不开放客户端SQL、文件路径或任意用户查询。
    """
    @server.tool()
    def query_expenses(
        category: Literal["生活", "出行", "娱乐", "宠物", "其他"],
        limit: int = 20,
    ) -> ExpenseQueryResult:
        """只读查询当前绑定用户某分类的最近支出，不新增或修改记录。

        参数：category为一级分类；limit为返回条数上限，1至100，默认20。
        返回值：found表示有记录，not_found表示没有匹配记录；has_more为真说明仍有更早记录。
            结果按新增顺序由近到远，不是全部历史，不提供全量金额统计。
        """
        return store.query(ExpenseQueryInput(category=category, limit=limit))
