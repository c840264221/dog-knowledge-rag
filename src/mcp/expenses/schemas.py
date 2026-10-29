"""记账服务的数据契约；自然语言分类后续再接入，本步只接受明确结构化参数。"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.mcp.expenses.skill_adapter import load_expense_categories


# 保留原公开名称和元组格式；服务端校验与模型参考共用同一份分类目录。
EXPENSE_CATEGORIES = load_expense_categories()


def validate_expense_date(value: str) -> str:
    """校验实际账单日期；value为通过YYYY-MM-DD格式检查的字符串；返回原值，非法日期抛异常。"""
    date.fromisoformat(value)
    return value


ExpenseDate = Annotated[
    str,
    Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", description="支出发生日期，YYYY-MM-DD，不是创建时间"),
    AfterValidator(validate_expense_date),
]


class ExpenseDetails(BaseModel):
    """单笔账单业务字段；参数为日期、金额、分类及用途；返回规范明细，不包含动作身份。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    amount: str = Field(pattern=r"^[0-9]{1,8}(\.[0-9]{1,2})?$", description="支出金额，人民币元的字符串，必须大于零")
    currency: Literal["CNY"] = "CNY"
    category: Literal["生活", "出行", "娱乐", "宠物", "其他"]
    subcategory: str = Field(min_length=1, max_length=30)
    description: str = Field(min_length=1, max_length=500)
    expense_date: ExpenseDate

    @field_validator("amount")
    @classmethod
    def normalize_amount(cls, value: str) -> str:
        """规范金额；value为已通过格式校验的字符串；返回两位小数，零金额抛出异常。"""
        amount = Decimal(value)
        if amount <= 0:
            raise ValueError("支出金额必须大于零。")
        return format(amount, ".2f")

    @model_validator(mode="after")
    def check_category_pair(self) -> "ExpenseDetails":
        """校验分类组合；无外部参数；组合合法返回本对象，否则拒绝写入。"""
        if self.subcategory not in EXPENSE_CATEGORIES[self.category]:
            raise ValueError(f"分类不匹配：{self.category}允许的用途为{EXPENSE_CATEGORIES[self.category]}。")
        return self


class ExpenseInput(ExpenseDetails):
    """兼容原单笔输入；参数为账单业务字段及幂等键；返回可提交的单笔输入。"""

    idempotency_key: str = Field(min_length=8, max_length=128, pattern=r"^[a-zA-Z0-9:_-]+$")


class ExpenseBatchInput(BaseModel):
    """批量记账输入；items为1至20笔完整明细，idempotency_key为整批动作编号；返回校验后批次。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    items: list[ExpenseDetails] = Field(min_length=1, max_length=20)
    idempotency_key: str = Field(min_length=8, max_length=128, pattern=r"^[a-zA-Z0-9:_-]+$")


class ExpenseRecord(BaseModel):
    """支出记录；参数含账单日期与独立UTC创建时间；旧记录日期缺失时返回None，不自动补写。"""

    model_config = ConfigDict(extra="forbid")

    expense_id: str
    user_id: str
    amount: str
    currency: Literal["CNY"]
    category: str
    subcategory: str
    description: str
    expense_date: ExpenseDate | None = None
    idempotency_key: str
    created_at: str


class ExpenseWriteResult(BaseModel):
    """记账回执；参数为新增或重复状态及原记录；返回可由MCP序列化的业务结果。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["recorded", "already_recorded"]
    record: ExpenseRecord


class ExpenseBatchWriteResult(BaseModel):
    """整批回执；参数为状态、批次编号、原键、数量及有序记录；返回整批保存或原批次结果。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["recorded", "already_recorded"]
    batch_id: str
    idempotency_key: str
    record_count: int
    records: list[ExpenseRecord]


class ExpenseQueryInput(BaseModel):
    """校验分类查询；参数为一级分类及最多返回条数；返回标准查询输入，不允许指定用户或SQL。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal["生活", "出行", "娱乐", "宠物", "其他"]
    limit: int = Field(default=20, strict=True, ge=1, le=100, description="最多返回的最近记录数，1至100")


class ExpenseQueryResult(BaseModel):
    """分类查询结果；参数为分类、入选记录、数量及是否还有更多；返回只读结果，不含全量统计。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["found", "not_found"]
    category: str
    records: list[ExpenseRecord]
    returned_count: int
    limit: int
    has_more: bool
