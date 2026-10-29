"""记账工具注册模块；不创建服务器、不配置数据库、不启动端口。"""

from typing import TYPE_CHECKING, Literal

from src.mcp.expenses.schemas import (
    ExpenseBatchInput, ExpenseBatchWriteResult, ExpenseDetails, ExpenseInput, ExpenseWriteResult,
)
from src.mcp.expenses.store import ExpenseStore

if TYPE_CHECKING:
    from mcp.server import MCPServer


def register_expense_write_tools(server: "MCPServer", store: ExpenseStore) -> None:
    """向已有服务注册记账工具。

    参数：server为调用方创建的MCP服务；store为已绑定身份的支出存储对象。
    返回值：None；直接向server注册工具，不创建第二个服务，不复制业务保存逻辑。
    """
    @server.tool()
    def record_expense(
        amount: str,
        category: Literal["生活", "出行", "娱乐", "宠物", "其他"],
        subcategory: str,
        description: str,
        idempotency_key: str,
        expense_date: str,
        currency: Literal["CNY"] = "CNY",
    ) -> ExpenseWriteResult:
        """记录实际已经发生的支出，有持久化写入副作用，不把计划预算记成消费。

        参数：amount为正金额字符串；category与subcategory为目录中合法分类组合；
            description为实际用途；idempotency_key为调用方生成并在重试时复用的动作编号；
            expense_date为实际支出所属日期，必须明确提供YYYY-MM-DD，不是系统创建日期；
            currency仅支持CNY人民币。教学客户端须先展示并取得用户确认。
        返回值：recorded表示新增；already_recorded表示同键同内容已保存，返回原记录。
            不接受用户编号；身份由本机服务绑定。同键不同内容报错，不执行覆盖。
        """
        request = ExpenseInput(
            amount=amount, category=category, subcategory=subcategory,
            description=description, idempotency_key=idempotency_key,
            expense_date=expense_date, currency=currency,
        )
        return store.record(request)

    @server.tool()
    def record_expenses(items: list[ExpenseDetails], idempotency_key: str) -> ExpenseBatchWriteResult:
        """整批记录1至20笔实际支出，有写入副作用，必须先确认完整清单。

        参数：items为逐笔日期、金额、分类和用途；idempotency_key为整批动作编号，重试复用。
        返回值：整批回执与按输入顺序返回的账单；同键同内容返回原批次，任何写入失败整批回滚。
            不接收模型指定的用户身份，不把计划或缺参账目混入提交。
        """
        return store.record_batch(ExpenseBatchInput(items=items, idempotency_key=idempotency_key))
