from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswerStatus,
    DogKnowledgeEvidence,
    DogKnowledgeQueryType,
    DogKnowledgeRecommendationItem,
)


class DogQueryLayerOutput(BaseModel):
    """
    DogKnowledgeAgent 查询理解层输出契约。

    功能：
        描述用户问题被 DogKnowledgeAgent 理解成什么业务类型，
        以及后续检索、推荐、生成层需要使用的查询条件。

    字段说明：
        question:
            用户原始问题。

        query_type:
            标准问题类型，例如 exact_lookup、recommendation、fallback。

        task_intent:
            更细粒度的任务意图。
            例如 dog_attribute_lookup、beginner_recommendation。

        dog_names:
            从问题中识别出的犬种标准名列表。

        target_fields:
            用户想查询的犬种属性字段。
            例如 lifespan、temperament、size。

        filters:
            给 RAG 或 metadata 检索使用的结构化过滤条件。

        confidence:
            查询理解层对本次解析结果的置信度，范围 0 到 1。

        reason:
            查询理解层给出该解析结果的原因。

        metadata:
            扩展信息，用于保存暂时不固定的中间字段。
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="用户原始问题。",
    )

    query_type: DogKnowledgeQueryType = Field(
        ...,
        description="标准问题类型，例如 exact_lookup、recommendation、fallback。",
    )

    task_intent: str | None = Field(
        default=None,
        description="更细粒度的任务意图，避免继续复用旧 intent 字段。",
    )

    dog_names: list[str] = Field(
        default_factory=list,
        description="识别出的犬种标准名列表。",
    )

    target_fields: list[str] = Field(
        default_factory=list,
        description="用户想查询的犬种属性字段列表。",
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="用于检索或 metadata 查询的结构化过滤条件。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="查询理解置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="查询理解层生成该结果的原因。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="查询理解层扩展信息。",
    )


class DogRetrievalLayerOutput(BaseModel):
    """
    DogKnowledgeAgent 检索层输出契约。

    功能：
        描述 RAG 或 metadata 检索层召回了哪些标准证据。
        检索层只负责提供证据，不负责生成最终答案或推荐表达。

    字段说明：
        query_type:
            当前检索服务的问题类型。

        evidences:
            标准答案证据列表，复用 DogKnowledgeEvidence。

        retrieved_count:
            本次检索召回的证据数量。

        confidence:
            检索层对召回结果可靠性的置信度，范围 0 到 1。

        reason:
            检索层说明为什么认为这些证据可用。

        metadata:
            扩展信息，例如 retriever 名称、filter、top_k。
    """

    model_config = ConfigDict(extra="forbid")

    query_type: DogKnowledgeQueryType = Field(
        ...,
        description="当前检索服务的问题类型。",
    )

    evidences: list[DogKnowledgeEvidence] = Field(
        default_factory=list,
        description="检索层召回的标准证据列表。",
    )

    retrieved_count: int = Field(
        default=0,
        ge=0,
        description="本次检索召回的证据数量。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="检索结果置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="检索层输出该结果的原因。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="检索层扩展信息。",
    )


class DogRecommendationLayerOutput(BaseModel):
    """
    DogKnowledgeAgent 推荐层输出契约。

    功能：
        描述推荐层产出的犬种推荐列表。
        推荐层只负责推荐结果，不负责最终自然语言表达。

    字段说明：
        recommended_breeds:
            标准推荐犬种列表，复用 DogKnowledgeRecommendationItem。

        confidence:
            推荐层对推荐结果的置信度，范围 0 到 1。

        reason:
            推荐层说明为什么推荐这些犬种。

        metadata:
            扩展信息，例如推荐策略、规则命中情况。
    """

    model_config = ConfigDict(extra="forbid")

    recommended_breeds: list[DogKnowledgeRecommendationItem] = Field(
        default_factory=list,
        description="推荐层产出的标准犬种推荐列表。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="推荐结果置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="推荐层输出该结果的原因。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="推荐层扩展信息。",
    )


class DogGenerationLayerOutput(BaseModel):
    """
    DogKnowledgeAgent 生成层输出契约。

    功能：
        描述生成层产出的自然语言答案。
        生成层只负责“怎么说”，不重新决定问题类型、推荐列表或证据列表。

    字段说明：
        generated_answer:
            生成层产出的自然语言答案。

        confidence:
            生成层对答案表达可靠性的置信度，范围 0 到 1。

        reason:
            生成层说明为什么这样组织答案。

        used_evidence_ids:
            生成答案时使用的证据 ID 列表。

        metadata:
            扩展信息，例如模型名、prompt 版本。
    """

    model_config = ConfigDict(extra="forbid")

    generated_answer: str = Field(
        ...,
        description="生成层产出的自然语言答案。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="生成层置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="生成层输出该答案的原因。",
    )

    used_evidence_ids: list[str] = Field(
        default_factory=list,
        description="生成答案时使用的证据 ID 列表。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="生成层扩展信息。",
    )


class DogFallbackLayerOutput(BaseModel):
    """
    DogKnowledgeAgent 兜底层输出契约。

    功能：
        描述 DogKnowledgeAgent 无法可靠回答时的兜底结果。
        兜底层负责给出安全答案、兜底原因和低置信度标记。

    字段说明：
        is_fallback:
            是否进入兜底流程。

        fallback_reason:
            进入兜底流程的原因。

        generated_answer:
            兜底自然语言答案。

        confidence:
            兜底答案置信度，范围 0 到 1。

        reason:
            兜底层输出该结果的原因。

        metadata:
            扩展信息，例如失败类型、缺失证据说明。
    """

    model_config = ConfigDict(extra="forbid")

    is_fallback: bool = Field(
        default=True,
        description="是否进入兜底流程。",
    )

    fallback_reason: str = Field(
        ...,
        description="进入兜底流程的原因。",
    )

    generated_answer: str = Field(
        ...,
        description="兜底自然语言答案。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="兜底答案置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="兜底层输出该结果的原因。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="兜底层扩展信息。",
    )


class DogKnowledgePipelineResult(BaseModel):
    """
    DogKnowledgeAgent 聚合层输出契约。

    功能：
        作为各 layer output 汇总后的标准 pipeline_result。
        它不是最终对外响应对象，但字段会尽量贴近 DogKnowledgeAnswer，
        方便 finalize_answer_node 继续生成 dog_knowledge_answer、
        dog_knowledge_answer_public 和 final_answer。

    字段说明：
        question:
            用户原始问题。

        query_type:
            标准问题类型。

        status:
            聚合后的答案状态。

        answer:
            聚合后的最终自然语言答案文本。

        recommended_breeds:
            聚合后的推荐犬种列表。

        evidences:
            聚合后的证据列表。

        confidence:
            聚合后的整体置信度。

        reason:
            聚合层给出的整体原因。

        is_fallback:
            是否进入兜底。

        fallback_reason:
            兜底原因。

        metadata:
            聚合层扩展信息。

        debug:
            聚合层调试信息。
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="用户原始问题。",
    )

    query_type: DogKnowledgeQueryType = Field(
        ...,
        description="标准问题类型。",
    )

    status: DogKnowledgeAnswerStatus = Field(
        ...,
        description="聚合后的答案状态。",
    )

    answer: str = Field(
        ...,
        description="聚合后的最终自然语言答案文本。",
    )

    recommended_breeds: list[DogKnowledgeRecommendationItem] = Field(
        default_factory=list,
        description="聚合后的推荐犬种列表。",
    )

    evidences: list[DogKnowledgeEvidence] = Field(
        default_factory=list,
        description="聚合后的证据列表。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="聚合后的整体置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="聚合层给出的整体原因。",
    )

    is_fallback: bool = Field(
        default=False,
        description="是否进入兜底。",
    )

    fallback_reason: str | None = Field(
        default=None,
        description="兜底原因。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="聚合层扩展信息。",
    )

    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="聚合层调试信息。",
    )
