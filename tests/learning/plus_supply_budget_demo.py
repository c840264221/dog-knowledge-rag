"""新猫到家案例第一步：本地预算工具，不调用模型或联网，不修改默认工具目录。

在项目根目录运行：python -m tests.learning.plus_supply_budget_demo
演示价仅供学习；本步直接注册并调用工具，不经过ToolAgent、运行时中间件或调度器。
"""

import asyncio
from copy import deepcopy
import json

from pydantic import ValidationError

from src.graph.tools.implementations.supply_budget_tool import SupplyBudgetTool
from src.graph.tools.registry.tool_registry import ToolRegistry


async def run_budget_demo() -> dict:
    """演示工具注册、两种预算结果及非法参数拒绝。

    参数：无，使用明确标注的演示单价与数量，不是推荐清单或实时报价。
    返回值：含工具名称、预算内和超支结果、校验结果的报告；不持久化报告或产生外部操作。
    """
    registry = ToolRegistry()
    registry.register(SupplyBudgetTool())
    tool = registry.get_tool("calculate_supply_budget")
    if tool is None:
        raise RuntimeError("预算工具未注册成功。")
    request = {
        "budget": "300.00",
        "currency": "CNY",
        "price_source": "demo_catalog",
        "items": [
            {"name": "猫砂盆", "unit_price": "59.90", "quantity": 1},
            {"name": "食水碗", "unit_price": "19.90", "quantity": 2},
            {"name": "猫抓板", "unit_price": "29.90", "quantity": 1},
            {"name": "躲避窝", "unit_price": "49.90", "quantity": 1},
        ],
    }
    within_budget = await tool.run(request)
    assert within_budget["total"] == "179.50"
    assert within_budget["remaining_budget"] == "120.50"
    assert within_budget["is_over_budget"] is False

    lower_budget_request = {**request, "budget": "150.00"}
    over_budget = await tool.run(lower_budget_request)
    assert over_budget["over_budget_amount"] == "29.50"
    assert over_budget["remaining_budget"] == "0.00"
    assert over_budget["is_over_budget"] is True

    invalid_quantity_request = deepcopy(request)
    invalid_quantity_request["items"][0]["quantity"] = -1
    try:
        await tool.run(invalid_quantity_request)
    except ValidationError as error:
        validation_result = {
            "rejected": True,
            "field": ".".join(map(str, error.errors()[0]["loc"])),
            "reason": "数量必须为1至999的整数，非法输入不会继续计算。",
        }
    else:
        raise AssertionError("负数数量应被拒绝。")

    return {
        "tool_name": tool.metadata.name,
        "require_confirm": tool.metadata.require_confirm,
        "within_budget": within_budget,
        "over_budget": over_budget,
        "invalid_quantity": validation_result,
        "checks_passed": True,
        "notice": "本步只验证本地工具，不代表模型已经能够选择工具或执行完整多步骤任务。",
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_budget_demo()), ensure_ascii=False, indent=2))
