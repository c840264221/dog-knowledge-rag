"""自然语言支出提取：模型只提出账目，不生成身份、幂等键或执行授权。"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.mcp.expenses.schemas import ExpenseBatchInput, ExpenseDate, ExpenseDetails, ExpenseInput
from src.mcp.expenses.skill_adapter import build_expense_skill_context


class ExpenseProposal(BaseModel):
    """约束单笔支出提议；参数为状态、实际支出标记、账单字段和说明；返回提议，不代表确认。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    status: Literal["ready", "needs_clarification", "unsupported"]
    is_actual_expense: bool
    amount: str | None = Field(default=None, description="人民币金额字符串；缺失时为null，不猜价格")
    expense_date: ExpenseDate | None = None
    category: Literal["生活", "出行", "娱乐", "宠物", "其他"] | None = None
    subcategory: str | None = None
    description: str | None = None
    message: str = ""

    @model_validator(mode="after")
    def check_status_contract(self) -> "ExpenseProposal":
        """检查提议状态；无外部参数；ready须通过业务契约，其他状态必须有中文说明。"""
        if self.status == "ready":
            if not self.is_actual_expense:
                raise ValueError("非实际支出不能声明为可记账。")
            # 这里只使用固定占位键校验字段，不创建实际动作，也不保存任何账单。
            build_expense_request(self, idempotency_key="validation:only")
        elif not self.message:
            raise ValueError("需要澄清或不支持时必须提供说明。")
        return self


def build_expense_request(proposal: ExpenseProposal, *, idempotency_key: str) -> ExpenseInput:
    """将可记账提议适配为现有工具输入。

    参数：proposal为模型提议；idempotency_key由可信调用程序提供，不来自模型。
    返回值：通过现有金额、日期和分类组合校验的ExpenseInput；不执行工具、不代表用户已确认。
    """
    if proposal.status != "ready" or not proposal.is_actual_expense:
        raise ValueError("当前提议不能转成记账请求，需要先澄清或说明限制。")
    return ExpenseInput(
        amount=proposal.amount,
        expense_date=proposal.expense_date,
        category=proposal.category,
        subcategory=proposal.subcategory,
        description=proposal.description,
        idempotency_key=idempotency_key,
    )


class ExpenseBatchProposal(BaseModel):
    """批量提议；参数为整批状态、逐笔提议与说明；返回待审核清单，不授予整批执行权限。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ready", "needs_clarification", "unsupported"]
    items: list[ExpenseProposal] = Field(default_factory=list, max_length=20)
    message: str = ""

    @model_validator(mode="after")
    def check_batch_status(self) -> "ExpenseBatchProposal":
        """校验整批状态；无外部参数；仅非空且每笔均ready时允许整批ready，其他状态须说明。"""
        if self.status == "ready":
            if not self.items or any(item.status != "ready" for item in self.items):
                raise ValueError("整批ready必须包含至少一笔账，且每笔完整合法。")
        elif not self.message.strip():
            raise ValueError("整批需要澄清或不支持时必须说明原因。")
        return self


def build_expense_batch_request(proposal: ExpenseBatchProposal, *, idempotency_key: str) -> ExpenseBatchInput:
    """将完整批量提议转换成批次参数；键由程序分配；返回校验后批次，缺参不部分提交。"""
    if proposal.status != "ready":
        raise ValueError("批次尚未齐全或不支持，不能提交其中一部分。")
    items = [ExpenseDetails.model_validate(
        build_expense_request(item, idempotency_key="validation:only").model_dump(exclude={"idempotency_key"}),
    ) for item in proposal.items]
    return ExpenseBatchInput(items=items, idempotency_key=idempotency_key)


def build_expense_extraction_instructions(skill_context: str | None = None, *, batch: bool = False) -> str:
    """构建提取提示；skill_context为可选已加载技能内容；返回技能与输出契约，不包含用户输入。

    不传时读取文件式技能；batch选择批量契约，默认False保持原单笔调用兼容。
    """
    if skill_context is None:
        skill_context, _ = build_expense_skill_context()
    return (
        "本轮执行模式：仅提取，不调用工具。只输出一个合法JSON对象，缺失字段使用null。\n"
        + ("本轮为批量：逐笔保留用户顺序；共同日期只有语义明确覆盖多笔时才共享；任何一笔不完整则整批澄清，不能漏掉该笔只返回其余账。\n"
           if batch else "本轮为单笔：多笔请求请说明需使用批量模式，不合并金额冒充单笔。\n")
        + skill_context
        + "\n\n以下是程序提供的提议输出契约：\n"
        + json.dumps((ExpenseBatchProposal if batch else ExpenseProposal).model_json_schema(), ensure_ascii=False)
    )


async def extract_expense_proposal(provider, question: str, *, reference_date: str, report: dict, batch: bool = False) -> ExpenseProposal | ExpenseBatchProposal:
    """调用项目模型服务提取单笔或批量支出，保留原始响应，不执行记账。

    参数：provider为注入的模型提供者；question为用户原话；reference_date为程序日期基准；batch选择批量；
        report为调用方的报告字典，写入原始输出和词元用量，不包含API密钥。
    返回值：通过结构与业务契约检查的提议；语义仍需人工核对，不自动修复或重复调用模型。
        底层Provider仍可能按项目配置重试或切备用模型，不能承诺只有一次网络请求。
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from src.runtime.observability.llm_call_records import LLMCallPurpose, build_llm_call_metadata

    if not question.strip():
        raise ValueError("用户输入不能为空。")
    skill_context, skill_metadata = build_expense_skill_context()
    report["loaded_skill"] = skill_metadata
    report["skill_context"] = skill_context
    report["extraction_mode"] = "batch" if batch else "single"
    parser = PydanticOutputParser(pydantic_object=ExpenseBatchProposal if batch else ExpenseProposal)
    messages = [
        SystemMessage(content=build_expense_extraction_instructions(skill_context, batch=batch)),
        HumanMessage(content=json.dumps({
            "reference_date": reference_date,
            "timezone": "Asia/Hong_Kong（UTC+08:00）",
            "user_text": question,
        }, ensure_ascii=False)),
    ]
    output = await provider.safe_ainvoke(
        llm=provider.main_llm,
        prompt=messages,
        call_metadata=build_llm_call_metadata(
            purpose=LLMCallPurpose.TOOL_PLANNING,
            component="expense_extraction_demo",
            extra={"output_mode": "json_object", "prompt_version": "expense_extraction_skill_v2", "skill_id": skill_metadata["skill_id"], "batch": batch},
        ),
    )
    report["raw_model_content"] = output.content
    report["usage_metadata"] = getattr(output, "usage_metadata", None)
    return parser.invoke(output)
