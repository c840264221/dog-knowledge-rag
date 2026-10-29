"""本地用品预算工具：只核算，不推荐商品、不查价、不下单、不写文件。"""

from decimal import Decimal

from src.graph.tools.base.base_tool import BaseTool
from src.graph.tools.schemas.supply_budget_args import SupplyBudgetArgs
from src.graph.tools.schemas.tool_metadata import ToolMetadata


class SupplyBudgetTool(BaseTool):
    """基于显式价格核算预算，供既有工具注册表和工具运行时复用。"""

    args_schema = SupplyBudgetArgs
    metadata = ToolMetadata(
        name="calculate_supply_budget",
        description=(
            "按提供的人民币单价和数量核算用品预算，返回合计、余额及超支金额。"
            "金额必须使用字符串；价格来源必须明确。不查询实时价格，不下单。"
        ),
        timeout=3,
        retries=0,
        require_confirm=False,
        input_schema=SupplyBudgetArgs.model_json_schema(),
    )

    async def run(self, args: dict) -> dict:
        """校验输入并精确核算，无外部副作用。

        参数：args包含budget（预算）、items（单价与数量）、price_source（价格来源）等。
        返回值：业务结果字典，金额统一为两位小数字符串；is_over_budget表示是否超支。
            超支属于正常计算结果，不代表工具失败；非法参数抛出校验异常，交由调用层处理。
            返回业务内容，不在此重复封装运行时负责生成的ToolResult（统一工具结果）。
        """
        validated = self.args_schema.model_validate(args)
        budget = Decimal(validated.budget)
        total = Decimal("0.00")
        items = []
        for item in validated.items:
            unit_price = Decimal(item.unit_price)
            subtotal = unit_price * item.quantity
            total += subtotal
            items.append({
                "name": item.name,
                "unit_price": format(unit_price, ".2f"),
                "quantity": item.quantity,
                "subtotal": format(subtotal, ".2f"),
            })
        balance = budget - total
        return {
            "currency": validated.currency,
            "price_source": validated.price_source,
            "items": items,
            "budget": format(budget, ".2f"),
            "total": format(total, ".2f"),
            "remaining_budget": format(max(balance, Decimal("0.00")), ".2f"),
            "over_budget_amount": format(max(-balance, Decimal("0.00")), ".2f"),
            "is_over_budget": balance < 0,
            "notice": "仅基于输入价格核算；不验证市场价格、库存或用品适用性，不含未输入的运费等费用。",
        }
