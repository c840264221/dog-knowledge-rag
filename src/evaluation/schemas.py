from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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


class EvaluationCategorySummary(BaseModel):
    """
    定义单个 Evaluation（评估）类别的成绩汇总。

    功能：
        把同一类别下的多条 AgentEvaluationResult 汇总成用例数量、
        通过率、异常数量、耗时和失败用例编号，回答“某一科考了多少分”。

    参数含义：
        category:
            评估类别，例如 tool_behavior。
        dataset_path:
            当前类别使用的黄金数据集路径。
        total_cases、passed_cases、failed_cases、error_cases:
            用例总数、通过数、失败数和执行异常数。
        pass_rate:
            当前类别通过率，范围为 0.0 到 1.0。
        average_latency_ms:
            当前类别所有有效耗时的平均值，单位毫秒。
        failed_case_ids、error_case_ids:
            未通过用例和发生执行异常的用例编号。
        execution_error:
            数据集加载或评估器初始化失败等类别级异常。
        metrics:
            当前类别专属指标，例如 RAG 的 hit_at_k 和 top1_accuracy。

    返回值含义：
        EvaluationCategorySummary:
            一门评估科目的结构化成绩单。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    category: str = Field(..., description="评估类别名称。")
    dataset_path: str = Field(..., description="黄金数据集路径。")
    total_cases: int = Field(default=0, ge=0, description="用例总数。")
    passed_cases: int = Field(default=0, ge=0, description="通过用例数。")
    failed_cases: int = Field(default=0, ge=0, description="失败用例数。")
    error_cases: int = Field(default=0, ge=0, description="执行异常用例数。")
    pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="当前类别通过率。",
    )
    average_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="当前类别平均耗时，单位毫秒。",
    )
    failed_case_ids: list[str] = Field(
        default_factory=list,
        description="未通过用例编号。",
    )
    error_case_ids: list[str] = Field(
        default_factory=list,
        description="执行异常用例编号。",
    )
    execution_error: str | None = Field(
        default=None,
        description="类别级加载或执行异常。",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="当前评估类别的专属汇总指标。",
    )


class EvaluationMetricGateResult(BaseModel):
    """
    定义一项类别总体指标的质量门禁结果。

    功能：
        保存指标所属类别、指标名称、比较方向、门禁阈值、实际值和判断结果，
        回答“某个 RAG 总体指标是否达到发布要求”。

    参数含义：
        category:
            指标所属评估类别，例如 rag_retrieval_behavior。
        metric_name:
            指标名称，例如 hit_at_k。
        operator:
            gte 表示实际值必须大于等于阈值；lte 表示必须小于等于阈值。
        threshold、actual:
            门禁要求值和本次评估实际值。
        passed:
            当前指标是否通过门禁。
        message:
            当前指标判断结果的中文说明。

    返回值含义：
        EvaluationMetricGateResult:
            一项可序列化的类别总体指标门禁成绩。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    category: str = Field(..., description="指标所属评估类别。")
    metric_name: str = Field(..., description="总体指标名称。")
    operator: Literal["gte", "lte"] = Field(
        ...,
        description="指标阈值比较方向。",
    )
    threshold: float = Field(..., description="质量门禁要求值。")
    actual: float | None = Field(
        default=None,
        description="本次评估产生的实际指标值。",
    )
    passed: bool = Field(..., description="当前指标是否通过门禁。")
    message: str = Field(default="", description="指标门禁中文说明。")


class EvaluationQualityGate(BaseModel):
    """
    定义 Evaluation Quality Gate（评估质量门禁）结论。

    功能：
        保存门禁策略、实际成绩和违规原因，回答“所有科目的成绩是否及格”。

    参数含义：
        policy_name:
            门禁策略名称，例如 v112_strict。
        passed:
            当前整套评估是否通过门禁。
        required_overall_pass_rate、actual_overall_pass_rate:
            最低要求通过率和实际整体通过率。
        require_all_categories_passed:
            是否要求每个评估类别都全部通过。
        maximum_error_cases、actual_error_cases:
            最多允许的执行异常数和实际异常数。
        require_checks:
            是否要求每条结果至少产生一个结构化检查项。
        failed_categories:
            未达到门禁要求的评估类别。
        violations:
            违反门禁规则的中文原因列表。
        metric_results:
            各类别专业总体指标对应的结构化门禁结果。

    返回值含义：
        EvaluationQualityGate:
            可以直接用于 CI/CD 退出码判断的质量结论。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    policy_name: str = Field(..., description="质量门禁策略名称。")
    passed: bool = Field(..., description="整套评估是否通过门禁。")
    required_overall_pass_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="最低整体通过率。",
    )
    actual_overall_pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="实际整体通过率。",
    )
    require_all_categories_passed: bool = Field(
        default=True,
        description="是否要求每个类别全部通过。",
    )
    maximum_error_cases: int = Field(
        default=0,
        ge=0,
        description="最多允许的执行异常数。",
    )
    actual_error_cases: int = Field(
        default=0,
        ge=0,
        description="实际执行异常数。",
    )
    require_checks: bool = Field(
        default=True,
        description="是否要求每条用例产生检查项。",
    )
    failed_categories: list[str] = Field(
        default_factory=list,
        description="未通过门禁的类别名称。",
    )
    violations: list[str] = Field(
        default_factory=list,
        description="违反门禁规则的中文原因。",
    )
    metric_results: list[EvaluationMetricGateResult] = Field(
        default_factory=list,
        description="类别专业总体指标的质量门禁结果。",
    )


class EvaluationSuiteReport(BaseModel):
    """
    定义一整套 Agent Evaluation（智能体评估）的完整成绩单。

    功能：
        保存总体成绩、每个类别汇总、质量门禁和全部单条结果，
        回答“所有科目、所有成绩以及最终是否及格分别是什么”。

    参数含义：
        suite_name、version:
            评估套件名称和版本。
        generated_at:
            报告生成时间。
        duration_ms:
            整套评估执行总耗时，单位毫秒。
        total_cases、passed_cases、failed_cases、error_cases、pass_rate:
            整体用例统计数据。
        category_summaries:
            每个评估类别的成绩汇总列表。
        quality_gate:
            整套成绩对应的质量门禁结论。
        results:
            所有单条 AgentEvaluationResult 原始成绩。
        runner_errors:
            数据集加载或评估器执行产生的套件级错误。
        metadata:
            执行环境等扩展信息。

    返回值含义：
        EvaluationSuiteReport:
            可输出为 JSON 和 Markdown 的完整评估报告。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    suite_name: str = Field(..., description="评估套件名称。")
    version: str = Field(..., description="评估套件版本。")
    generated_at: datetime = Field(..., description="报告生成时间。")
    duration_ms: float = Field(default=0.0, ge=0.0, description="总耗时毫秒数。")
    total_cases: int = Field(default=0, ge=0, description="用例总数。")
    passed_cases: int = Field(default=0, ge=0, description="通过用例数。")
    failed_cases: int = Field(default=0, ge=0, description="失败用例数。")
    error_cases: int = Field(default=0, ge=0, description="异常用例数。")
    pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="整体通过率。",
    )
    category_summaries: list[EvaluationCategorySummary] = Field(
        default_factory=list,
        description="各类别成绩汇总。",
    )
    quality_gate: EvaluationQualityGate = Field(
        ...,
        description="质量门禁结论。",
    )
    results: list[AgentEvaluationResult] = Field(
        default_factory=list,
        description="所有单条评估结果。",
    )
    runner_errors: list[str] = Field(
        default_factory=list,
        description="评估套件运行错误。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="报告扩展元数据。",
    )


class EvaluationBaselineMetric(BaseModel):
    """
    记录某一个指标以前考了多少分，以及这个指标应该越高越好还是越低越好
    某个具体指标在 V1.12 时是多少，以及以后允许它下降多少。

    功能：
        保存指标所属类别、基线值、质量方向和允许退步幅度，供回归比较器
        判断当前成绩是否比已发布版本明显下降。

    参数含义：
        category：这个指标属于哪个评估类别。
        metric_name：具体比较哪个指标。
        baseline_value：历史基线成绩。
        direction：这个指标应该越高越好，还是越低越好。
        maximum_regression：最多允许比基线差多少。
    返回值含义：
        EvaluationBaselineMetric:
            一项可持久化并参与回归判断的指标基线。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    category: str = Field(..., description="指标所属评估类别。")
    metric_name: str = Field(..., description="专业总体指标名称。")
    baseline_value: float = Field(..., description="已验证的指标基线值。")
    direction: Literal[
        "higher_is_better",
        "lower_is_better",
    ] = Field(..., description="指标质量方向。")
    maximum_regression: float = Field(
        default=0.0,
        ge=0.0,
        description="最多允许的绝对退步幅度。",
    )


class EvaluationBaselineSnapshot(BaseModel):
    """
    保存 V1.12 整套评估的历史成绩，作为以后比较的标准答案。
    某个已经验证通过的版本的完整历史成绩。

    功能：
        固化已发布版本的整体通过率、各类别通过率和专业总体指标，作为后续
        代码变更执行 Regression Detection（回归检测）时的比较标准。

    参数含义：
        baseline_name：这份基线的唯一名称。
        source_suite_name：它来自哪套评估。
        source_version：它来自哪个版本。
        overall_pass_rate：当时整套评估的通过率。
        category_pass_rates：当时每个类别的通过率。
        metrics：当时需要重点比较的 RAG 等专业指标。
        metadata：Tag、用例数量、提交编号等补充信息。

    返回值含义：
        EvaluationBaselineSnapshot:
            可写入 JSON 并由比较器重复加载的版本成绩基线。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    baseline_name: str = Field(..., description="评估基线名称。")
    source_suite_name: str = Field(..., description="来源评估套件名称。")
    source_version: str = Field(..., description="来源评估版本。")
    overall_pass_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="来源版本整体通过率。",
    )
    category_pass_rates: dict[str, float] = Field(
        default_factory=dict,
        description="各评估类别的基线通过率。",
    )
    metrics: list[EvaluationBaselineMetric] = Field(
        default_factory=list,
        description="需要比较的专业总体指标基线。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="基线来源扩展信息。",
    )

    @field_validator("category_pass_rates")
    @classmethod
    def validate_category_pass_rates(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """
        校验所有类别通过率均位于合法百分比范围。

        参数含义：
            value:
                类别名称到通过率的映射。

        返回值含义：
            dict[str, float]:
                校验通过后的原始映射。
        """

        invalid_categories = [
            category
            for category, pass_rate in value.items()
            if not 0.0 <= pass_rate <= 1.0
        ]
        if invalid_categories:
            raise ValueError(
                "类别通过率必须位于 0.0 到 1.0: "
                f"{sorted(invalid_categories)}"
            )
        return value


class EvaluationRegressionCheck(BaseModel):
    """
    拿当前的一个成绩和历史成绩比较，记录这一项有没有退步。
    当前版本的某一个成绩，与 V1.12 对比后的结果。

    功能：
        记录比较范围、基线值、当前值、变化量、允许退步幅度和判断结论。

    参数含义：
        scope：比较的是整体、某个类别，还是某个专业指标。
        category：属于哪个类别；整体通过率不需要类别。
        metric_name：比较的成绩名称。
        baseline_value：V1.12 的历史值。
        current_value：当前版本的实际值。
        delta：当前值减去基线值。
        direction：越高越好还是越低越好。
        maximum_regression：最多允许退步多少。
        passed：这一项是否没有发生不可接受的退步。
        message：人工可读的比较说明。

    返回值含义：
        EvaluationRegressionCheck:
            一项结构化的版本成绩回归检查结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    scope: Literal["overall", "category", "metric"] = Field(
        ...,
        description="回归检查范围。",
    )
    category: str | None = Field(default=None, description="关联评估类别。")
    metric_name: str = Field(..., description="参与比较的指标名称。")
    baseline_value: float = Field(..., description="基线指标值。")
    current_value: float | None = Field(
        default=None,
        description="当前报告指标值。",
    )
    delta: float | None = Field(
        default=None,
        description="当前值减去基线值。",
    )
    direction: Literal[
        "higher_is_better",
        "lower_is_better",
    ] = Field(..., description="指标质量方向。")
    maximum_regression: float = Field(
        default=0.0,
        ge=0.0,
        description="最多允许的绝对退步幅度。",
    )
    passed: bool = Field(..., description="当前回归检查是否通过。")
    message: str = Field(default="", description="回归检查中文说明。")


class EvaluationRegressionReport(BaseModel):
    """
    收集所有比较结果，给出这次版本整体有没有质量回退。
    当前版本与 V1.12 完整比较后的回归检测报告。

    功能：
        汇总整体、类别和专业指标检查，并提供是否发生质量回归的总判断。

    参数含义：
        baseline_name：使用了哪份历史基线。
        current_suite_name：当前运行的是哪套评估。
        current_version：当前报告版本。
        checks：所有 EvaluationRegressionCheck。
        passed：是否所有回归检查都通过。

    返回值含义：
        EvaluationRegressionReport:
            可供后续 CLI 和 CI 使用的结构化回归检测报告。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    baseline_name: str = Field(..., description="本次使用的基线名称。")
    current_suite_name: str = Field(..., description="当前评估套件名称。")
    current_version: str = Field(..., description="当前评估报告版本。")
    checks: list[EvaluationRegressionCheck] = Field(
        default_factory=list,
        description="全部回归检查结果。",
    )

    @computed_field
    @property
    def passed(self) -> bool:
        """
        计算整份回归报告是否通过。

        参数含义：
            无。

        返回值含义：
            bool:
                至少产生一项检查且全部检查通过时返回 True。
        """

        return bool(self.checks) and all(check.passed for check in self.checks)

    def failed_checks(self) -> list[EvaluationRegressionCheck]:
        """
        返回所有未通过的回归检查。

        参数含义：
            无。

        返回值含义：
            list[EvaluationRegressionCheck]:
                当前报告中发生质量退步或指标缺失的检查项。
        """

        return [check for check in self.checks if not check.passed]
