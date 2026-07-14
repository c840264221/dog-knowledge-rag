from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)


class AgentEvaluationCase(BaseModel):
    """
    定义一条通用 Agent 评估用例。

    功能：
        统一描述评估输入、所属业务类别和期望结果，使 RootAgent、
        ToolAgent、Memory、DogKnowledgeAgent 等模块可以共享同一种用例外壳。

    参数含义：
        case_id:
            评估用例唯一编号。
        category:
            评估类别，例如 root_route、tool_call、memory_recall。
        question:
            交给被评估链路处理的用户问题。
        expected:
            当前领域的期望结果，例如期望路由、工具名或答案关键词。
        input_state:
            除 question 外需要预先注入的 state 数据。
        tags:
            用于筛选和分类统计的标签。
        note:
            用例存在原因或业务背景说明。
        metadata:
            不参与核心判断的扩展元数据。

    返回值含义：
        AgentEvaluationCase:
            经过 Pydantic 校验和归一化的统一评估用例对象。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    case_id: str = Field(
        ...,
        description="评估用例唯一编号。",
    )
    category: str = Field(
        ...,
        description="评估类别，例如 root_route、tool_call、memory_recall。",
    )
    question: str = Field(
        ...,
        description="交给被评估链路处理的用户问题。",
    )
    expected: dict[str, Any] = Field(
        ...,
        description="当前业务领域需要验证的期望结果。",
    )
    input_state: dict[str, Any] = Field(
        default_factory=dict,
        description="运行前需要额外注入的 state 数据。",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="用于筛选和分类统计的标签。",
    )
    note: str | None = Field(
        default=None,
        description="用例存在原因或业务背景说明。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="不参与核心判断的扩展元数据。",
    )

    @field_validator("case_id", "category", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        校验并清洗评估用例的必填文本。

        参数含义：
            value:
                待校验的文本字段。

        返回值含义：
            str:
                去除首尾空格后的非空文本。
        """

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("评估用例必填文本不能为空")

        return normalized_value

    @field_validator("expected")
    @classmethod
    def validate_expected_not_empty(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """
        校验评估用例至少声明一项期望结果。

        参数含义：
            value:
                当前用例声明的期望结果字典。

        返回值含义：
            dict[str, Any]:
                校验通过后的原始期望结果。
        """

        if not value:
            raise ValueError("评估用例 expected 不能为空")

        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        """
        清洗标签并按首次出现顺序去重。

        参数含义：
            values:
                原始标签列表。

        返回值含义：
            list[str]:
                去除空白、空字符串和重复项后的标签列表。
        """

        normalized_tags: list[str] = []
        seen_tags: set[str] = set()

        for value in values:
            normalized_value = value.strip()

            if not normalized_value or normalized_value in seen_tags:
                continue

            normalized_tags.append(normalized_value)
            seen_tags.add(normalized_value)

        return normalized_tags


class EvaluationCheckResult(BaseModel):
    """
    定义一项具体字段或行为的评估检查结果。

    功能：
        保存某一个检查点的期望值、实际值和判断结果，让失败报告可以明确指出
        是路由、工具、参数、记忆还是答案字段不符合预期。

    参数含义：
        check_name:
            检查项名称，例如 route、tool_names、answer_keywords。
        passed:
            当前检查项是否通过。
        expected:
            当前检查项的期望值。
        actual:
            系统运行后得到的实际值。
        message:
            对检查结果的中文解释。
        metadata:
            检查项的扩展调试信息。

    返回值含义：
        EvaluationCheckResult:
            一项结构化评估检查结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    check_name: str = Field(
        ...,
        description="检查项名称。",
    )
    passed: bool = Field(
        ...,
        description="当前检查项是否通过。",
    )
    expected: Any = Field(
        default=None,
        description="当前检查项的期望值。",
    )
    actual: Any = Field(
        default=None,
        description="系统运行后得到的实际值。",
    )
    message: str = Field(
        default="",
        description="对检查结果的中文解释。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="检查项的扩展调试信息。",
    )

    @field_validator("check_name")
    @classmethod
    def validate_check_name(cls, value: str) -> str:
        """
        校验并清洗检查项名称。

        参数含义：
            value:
                原始检查项名称。

        返回值含义：
            str:
                去除首尾空格后的非空名称。
        """

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("评估检查项名称不能为空")

        return normalized_value

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        """
        清洗检查结果说明文本。

        参数含义：
            value:
                原始检查结果说明。

        返回值含义：
            str:
                去除首尾空格后的说明文本。
        """

        return value.strip()


class AgentEvaluationResult(BaseModel):
    """
    定义一条 Agent 评估用例的完整执行结果。

    功能：
        汇总所有检查项、真实输出、耗时和异常信息，并自动计算整条用例是否通过。

    参数含义：
        case_id:
            对应 AgentEvaluationCase 的唯一编号。
        category:
            当前结果所属评估类别。
        checks:
            本条用例产生的所有字段或行为检查结果。
        latency_ms:
            本条用例执行耗时，单位为毫秒。
        output:
            被评估链路返回的实际输出摘要。
        error_message:
            执行阶段发生的异常信息；没有异常时为 None。
        metadata:
            本次运行的扩展信息。

    返回值含义：
        AgentEvaluationResult:
            可序列化的单条统一评估结果；passed 字段由检查结果自动计算。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    case_id: str = Field(
        ...,
        description="对应评估用例的唯一编号。",
    )
    category: str = Field(
        ...,
        description="当前结果所属评估类别。",
    )
    checks: list[EvaluationCheckResult] = Field(
        default_factory=list,
        description="本条用例产生的所有检查结果。",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="本条用例执行耗时，单位毫秒。",
    )
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="被评估链路返回的实际输出摘要。",
    )
    error_message: str | None = Field(
        default=None,
        description="执行阶段的异常信息。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="本次运行的扩展信息。",
    )

    @field_validator("case_id", "category")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        校验并清洗评估结果中的必填文本。

        参数含义：
            value:
                待校验的文本字段。

        返回值含义：
            str:
                去除首尾空格后的非空文本。
        """

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("评估结果必填文本不能为空")

        return normalized_value

    @field_validator("error_message")
    @classmethod
    def normalize_error_message(
        cls,
        value: str | None,
    ) -> str | None:
        """
        清洗错误信息并把空字符串归一化成 None。

        参数含义：
            value:
                原始错误信息或 None。

        返回值含义：
            str | None:
                清洗后的错误信息；空文本返回 None。
        """

        if value is None:
            return None

        normalized_value = value.strip()
        return normalized_value or None

    @computed_field(
        return_type=bool,
        description="本条用例是否整体通过。",
    )
    @property
    def passed(self) -> bool:
        """
        根据错误信息和全部检查项自动计算用例是否通过。

        参数含义：
            无。

        返回值含义：
            bool:
                至少存在一项检查、全部检查通过且没有错误时返回 True；
                否则返回 False。
        """

        return (
            self.error_message is None
            and bool(self.checks)
            and all(check.passed for check in self.checks)
        )

    def failed_checks(self) -> list[EvaluationCheckResult]:
        """
        返回当前用例中未通过的检查项。

        参数含义：
            无。

        返回值含义：
            list[EvaluationCheckResult]:
                所有 passed=False 的检查项；全部通过时返回空列表。
        """

        return [
            check
            for check in self.checks
            if not check.passed
        ]
