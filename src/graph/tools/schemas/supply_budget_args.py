"""用品预算工具的输入契约；金额使用人民币元的字符串，不接受浮点金额。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SupplyBudgetItem(BaseModel):
    """校验一项用品；参数为名称、人民币单价字符串和整数数量；返回校验后的模型。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100, description="用品名称")
    unit_price: str = Field(
        pattern=r"^[0-9]{1,8}(\.[0-9]{1,2})?$",
        description="单价，单位为人民币元，字符串格式，例如 39.90；最多两位小数",
    )
    quantity: int = Field(strict=True, ge=1, le=999, description="购买数量，1至999的整数")


class SupplyBudgetArgs(BaseModel):
    """校验预算输入；参数为预算、币种、价格来源及用品；返回标准输入模型，不验证市场报价。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    budget: str = Field(
        pattern=r"^[0-9]{1,8}(\.[0-9]{1,2})?$",
        description="总预算，单位为人民币元，字符串格式，例如 300.00",
    )
    currency: Literal["CNY"] = Field(default="CNY", description="币种，仅支持人民币")
    price_source: Literal["demo_catalog", "user_provided"] = Field(
        description="价格来源：demo_catalog为演示价格；user_provided为用户提供，均非实时报价验证",
    )
    items: list[SupplyBudgetItem] = Field(min_length=1, max_length=100, description="用品明细，1至100项")
