from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class RagEvalCase(BaseModel):
    """
    RAG 评估用例模型。

    用于描述一条需要被 RAG 系统执行和验证的测试问题。

    参数含义：
        case_id: 评估用例唯一编号，例如 dog_eval_001。
        question: 用户问题，也就是要交给 RAG 检索链路的问题。
        expected_dog_names: 期望召回到的犬种名称列表。
        expected_filters: 期望 Query Parser 解析出的结构化过滤条件。
        top_k: 本条评估用例期望召回的最大 chunk 数量。
        tags: 用例标签，例如 easy、metadata、small_dog。
        note: 用例备注，用于解释这个 case 为什么存在。
        metadata: 扩展元数据，方便未来保存更多评估相关信息。

    返回值含义：
        RagEvalCase 实例，用于后续 DatasetLoader、Evaluator、ReportWriter 使用。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    case_id: str = Field(
        ...,
        description="评估用例唯一编号，例如 dog_eval_001。",
    )

    question: str = Field(
        ...,
        description="用户问题，也就是需要被 RAG 检索系统处理的问题。",
    )

    expected_dog_names: list[str] = Field(
        default_factory=list,
        description="期望召回到的犬种名称列表。",
    )

    expected_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="期望 Query Parser 解析出的结构化过滤条件。",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="本条评估用例期望召回的最大 chunk 数量。",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="评估用例标签，用于分类统计和筛选。",
    )

    note: str | None = Field(
        default=None,
        description="评估用例备注。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据，方便未来兼容更多评估信息。",
    )

    @field_validator("case_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        校验必填文本字段不能为空。

        参数含义：
            value: 需要校验的字符串字段值。

        返回值含义：
            去除首尾空格后的字符串。

        异常：
            ValueError: 当字符串为空时抛出。
        """

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("字段不能为空字符串")

        return normalized_value

    @field_validator("expected_dog_names", "tags")
    @classmethod
    def normalize_string_list(cls, values: list[str]) -> list[str]:
        """
        规范化字符串列表。

        会去除每个字符串的首尾空格，并过滤空字符串。

        参数含义：
            values: 原始字符串列表。

        返回值含义：
            清洗后的字符串列表。
        """

        normalized_values: list[str] = []

        for value in values:
            normalized_value = value.strip()

            if normalized_value:
                normalized_values.append(normalized_value)

        return normalized_values


class RagEvalRetrievedItem(BaseModel):
    """
    RAG 评估召回结果项模型。

    用于保存 Retriever 每召回一个 chunk 后，需要参与评估和报告展示的信息。

    参数含义：
        rank: 当前 chunk 在召回结果中的排序，从 1 开始。
        chunk_id: chunk 唯一编号。
        dog_name: 当前 chunk 所属犬种名称。
        score: 检索分数或者重排序分数。
        source: chunk 来源文件路径或者来源标识。
        section_title: chunk 所属 Markdown 标题。
        content_preview: chunk 内容预览，避免报告里输出太长文本。
        metadata: chunk 原始 metadata，方便调试。

    返回值含义：
        RagEvalRetrievedItem 实例，用于 RagEvalResult 保存召回详情。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    rank: int = Field(
        ...,
        ge=1,
        description="召回结果排序，从 1 开始。",
    )

    chunk_id: str | None = Field(
        default=None,
        description="chunk 唯一编号。",
    )

    dog_name: str | None = Field(
        default=None,
        description="当前 chunk 所属犬种名称。",
    )

    score: float | None = Field(
        default=None,
        description="检索分数或者重排序分数。",
    )

    source: str | None = Field(
        default=None,
        description="chunk 来源文件路径或者来源标识。",
    )

    section_title: str | None = Field(
        default=None,
        description="chunk 所属 Markdown 标题。",
    )

    content_preview: str | None = Field(
        default=None,
        description="chunk 内容预览。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="chunk 原始 metadata。",
    )


class RagEvalResult(BaseModel):
    """
    单条 RAG 评估结果模型。

    用于保存一条 RagEvalCase 执行后的完整评估结果。

    参数含义：
        case_id: 评估用例唯一编号。
        question: 用户问题。
        expected_dog_names: 期望命中的犬种名称。
        expected_filters: 期望解析出的过滤条件。
        parsed_filters: 实际 Query Parser 解析出的过滤条件。
        retrieved_items: 实际召回到的 chunk 详情。
        retrieved_dog_names: 实际召回结果中的犬种名称列表。
        hit: top_k 结果中是否命中期望犬种。
        hit_rank: 第一次命中期望犬种的排序位置。
        top1_hit: 第一个召回结果是否命中期望犬种。
        filter_matched: 实际 filters 是否符合 expected_filters。
        empty_retrieval: 是否为空召回。
        passed: 当前评估用例是否通过。
        error_message: 当前评估用例执行过程中的错误信息。
        latency_ms: 当前评估用例执行耗时，单位毫秒。
        debug_report_path: 当前用例关联的 RAG debug report 路径。
        extra: 额外调试信息。

    返回值含义：
        RagEvalResult 实例，用于统计指标和生成报告。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    case_id: str = Field(
        ...,
        description="评估用例唯一编号。",
    )

    question: str = Field(
        ...,
        description="用户问题。",
    )

    expected_dog_names: list[str] = Field(
        default_factory=list,
        description="期望命中的犬种名称。",
    )

    expected_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="期望解析出的过滤条件。",
    )

    parsed_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="实际 Query Parser 解析出的过滤条件。",
    )

    retrieved_items: list[RagEvalRetrievedItem] = Field(
        default_factory=list,
        description="实际召回到的 chunk 详情。",
    )

    retrieved_dog_names: list[str] = Field(
        default_factory=list,
        description="实际召回结果中的犬种名称列表。",
    )

    hit: bool = Field(
        default=False,
        description="top_k 结果中是否命中期望犬种。",
    )

    hit_rank: int | None = Field(
        default=None,
        ge=1,
        description="第一次命中期望犬种的排序位置。",
    )

    top1_hit: bool = Field(
        default=False,
        description="第一个召回结果是否命中期望犬种。",
    )

    filter_matched: bool = Field(
        default=False,
        description="实际 filters 是否符合 expected_filters。",
    )

    empty_retrieval: bool = Field(
        default=False,
        description="是否为空召回。",
    )

    passed: bool = Field(
        default=False,
        description="当前评估用例是否通过。",
    )

    error_message: str | None = Field(
        default=None,
        description="当前评估用例执行过程中的错误信息。",
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="当前评估用例执行耗时，单位毫秒。",
    )

    debug_report_path: str | None = Field(
        default=None,
        description="当前用例关联的 RAG debug report 路径。",
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="额外调试信息。",
    )

    @classmethod
    def from_failed_case(
        cls,
        eval_case: RagEvalCase,
        error_message: str,
    ) -> "RagEvalResult":
        """
        根据评估用例创建失败结果。

        当某条 case 执行过程中发生异常时，可以用这个方法快速构造失败结果，
        避免整个评估流程因为单条异常而中断。

        参数含义：
            eval_case: 原始 RAG 评估用例。
            error_message: 异常信息。

        返回值含义：
            RagEvalResult 失败结果实例。
        """

        return cls(
            case_id=eval_case.case_id,
            question=eval_case.question,
            expected_dog_names=eval_case.expected_dog_names,
            expected_filters=eval_case.expected_filters,
            parsed_filters={},
            retrieved_items=[],
            retrieved_dog_names=[],
            hit=False,
            hit_rank=None,
            top1_hit=False,
            filter_matched=False,
            empty_retrieval=True,
            passed=False,
            error_message=error_message,
        )

    def has_error(self) -> bool:
        """
        判断当前评估结果是否存在错误。

        参数含义：
            无。

        返回值含义：
            bool:
                True 表示存在错误。
                False 表示不存在错误。
        """

        return self.error_message is not None

    def is_successful(self) -> bool:
        """
        判断当前评估结果是否成功通过。

        参数含义：
            无。

        返回值含义：
            bool:
                True 表示 passed=True 且没有 error_message。
                False 表示未通过或者存在错误。
        """

        return self.passed and not self.has_error()


class RagEvalMetrics(BaseModel):
    """
    RAG 评估指标模型。

    用于保存整批评估用例执行后的统计指标。

    参数含义：
        total_cases: 总用例数。
        passed_cases: 通过用例数。
        failed_cases: 失败用例数。
        hit_at_k: top_k 命中率。
        top1_accuracy: top1 准确率。
        filter_match_rate: filter 匹配率。
        empty_retrieval_rate: 空召回率。
        average_latency_ms: 平均耗时，单位毫秒。

    返回值含义：
        RagEvalMetrics 实例，用于 RagEvalReport 汇总展示。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    total_cases: int = Field(
        default=0,
        ge=0,
        description="总用例数。",
    )

    passed_cases: int = Field(
        default=0,
        ge=0,
        description="通过用例数。",
    )

    failed_cases: int = Field(
        default=0,
        ge=0,
        description="失败用例数。",
    )

    hit_at_k: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="top_k 命中率。",
    )

    top1_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="top1 准确率。",
    )

    filter_match_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="filter 匹配率。",
    )

    empty_retrieval_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="空召回率。",
    )

    average_latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="平均耗时，单位毫秒。",
    )


class RagEvalReport(BaseModel):
    """
    RAG 评估报告模型。

    用于保存一次完整 RAG Evaluation Run 的结果。

    参数含义：
        run_id: 本次评估运行 ID。
        created_at: 报告创建时间。
        dataset_path: 评估数据集路径。
        report_path: Markdown 报告输出路径。
        metrics: 汇总指标。
        results: 每条评估用例的执行结果。
        metadata: 扩展信息，例如模型名、embedding 名、向量库路径等。

    返回值含义：
        RagEvalReport 实例，用于后续 Markdown ReportWriter 输出报告。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    run_id: str = Field(
        ...,
        description="本次评估运行 ID。",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="报告创建时间。",
    )

    dataset_path: str | None = Field(
        default=None,
        description="评估数据集路径。",
    )

    report_path: str | None = Field(
        default=None,
        description="Markdown 报告输出路径。",
    )

    metrics: RagEvalMetrics = Field(
        default_factory=RagEvalMetrics,
        description="汇总评估指标。",
    )

    results: list[RagEvalResult] = Field(
        default_factory=list,
        description="每条评估用例的执行结果。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展信息，例如模型名、embedding 名、向量库路径等。",
    )

    def failed_results(self) -> list[RagEvalResult]:
        """
        获取失败的评估结果列表。

        参数含义：
            无。

        返回值含义：
            list[RagEvalResult]: 所有未通过或者发生错误的评估结果。
        """

        return [
            result
            for result in self.results
            if not result.is_successful()
        ]

    def passed_results(self) -> list[RagEvalResult]:
        """
        获取通过的评估结果列表。

        参数含义：
            无。

        返回值含义：
            list[RagEvalResult]: 所有成功通过的评估结果。
        """

        return [
            result
            for result in self.results
            if result.is_successful()
        ]