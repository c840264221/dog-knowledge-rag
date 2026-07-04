from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DogKnowledgeQueryType = Literal[
    "exact_lookup",
    "recommendation",
    "comparison",
    "general_qa",
    "fallback",
]

DogKnowledgeAnswerStatus = Literal[
    "success",
    "partial",
    "fallback",
    "empty",
    "error",
]

DogKnowledgeSourceKind = Literal[
    "rag_chunk",
    "metadata",
    "llm",
    "rule",
    "fallback",
    "user_input",
]


class DogKnowledgeEvidence(BaseModel):
    """
    DogKnowledgeAgent 答案证据结构。

    功能：
        用来描述 DogKnowledgeAgent 生成答案时参考了哪些证据。
        例如 RAG 召回的 chunk、犬种 metadata、规则命中的结果等。

    字段说明：
        evidence_id:
            证据唯一 ID。
            可以是 chunk_id、metadata_id，也可以是内部生成的字符串。

        source_kind:
            证据来源类型。
            例如 rag_chunk 表示来自 RAG 检索片段，metadata 表示来自结构化犬种信息。

        title:
            证据标题。
            例如犬种名、文档标题、chunk 标题。

        content:
            证据正文内容。
            建议只放摘要或片段内容，不要放过长全文。

        score:
            证据分数。
            通常来自 retriever score、rerank score 或规则匹配分数。

        metadata:
            扩展信息。
            用来保存 source_path、dog_name、chunk_index 等暂时不固定的字段。
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        ...,
        description="证据唯一 ID，例如 chunk_id、metadata_id 或内部生成 ID。",
    )

    source_kind: DogKnowledgeSourceKind = Field(
        ...,
        description="证据来源类型，例如 rag_chunk、metadata、llm、rule、fallback。",
    )

    title: str | None = Field(
        default=None,
        description="证据标题，例如犬种名、文档标题或 chunk 标题。",
    )

    content: str = Field(
        ...,
        description="证据正文内容，建议放摘要或关键片段。",
    )

    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="证据相关性分数，范围 0 到 1。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="证据扩展信息，例如 source_path、dog_name、chunk_index。",
    )


class DogKnowledgeRecommendationItem(BaseModel):
    """
    DogKnowledgeAgent 犬种推荐项结构。

    功能：
        用来描述一次推荐答案中的单个犬种推荐结果。
        例如用户问“新手适合养什么狗”，这里每一项就是一个推荐犬种。

    字段说明：
        breed_name:
            犬种英文名或系统内部标准名。
            例如 golden_retriever、labrador_retriever。

        display_name:
            展示给用户看的名称。
            例如 Golden Retriever / 金毛寻回犬。

        reason:
            推荐这个犬种的原因。
            例如“性格友好、训练难度低、适合新手”。

        matched_traits:
            命中的用户需求特征。
            例如 ["新手友好", "容易训练", "中等运动量"]。

        warnings:
            需要提醒用户注意的点。
            例如 ["掉毛较多", "需要每天运动"]。

        evidence_ids:
            支撑该推荐的证据 ID 列表。
            对应 DogKnowledgeEvidence.evidence_id。

        score:
            推荐分数。
            范围 0 到 1，用来表示推荐匹配度。

        metadata:
            扩展信息。
            可以保存 energy、barking、trainability、size 等结构化字段。
    """

    model_config = ConfigDict(extra="forbid")

    breed_name: str = Field(
        ...,
        description="犬种标准名，建议使用系统内部统一名称。",
    )

    display_name: str | None = Field(
        default=None,
        description="展示名称，可以包含中文名或更适合用户阅读的名称。",
    )

    reason: str = Field(
        ...,
        description="推荐该犬种的原因。",
    )

    matched_traits: list[str] = Field(
        default_factory=list,
        description="命中的用户需求特征列表。",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="养该犬种需要注意的事项。",
    )

    evidence_ids: list[str] = Field(
        default_factory=list,
        description="支撑该推荐结果的证据 ID 列表。",
    )

    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="推荐匹配分数，范围 0 到 1。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="推荐项扩展信息，例如犬种体型、运动量、训练难度。",
    )


class DogKnowledgeAnswer(BaseModel):
    """
    DogKnowledgeAgent 统一答案输出结构。

    功能：
        作为 DogKnowledgeAgent 的最终 Response Contract（响应协议）。
        不管内部经过 RAG 检索、metadata 过滤、犬种推荐、fallback 兜底，
        最后都应该尽量转换成这个结构。

    字段说明：
        question:
            用户原始问题。

        query_type:
            问题类型。
            exact_lookup 表示精确查询；
            recommendation 表示推荐类问题；
            comparison 表示对比类问题；
            general_qa 表示普通问答；
            fallback 表示无法可靠回答时的兜底问题。

        status:
            答案状态。
            success 表示成功；
            partial 表示部分成功；
            fallback 表示走了兜底；
            empty 表示没有有效结果；
            error 表示内部异常。

        answer:
            最终给用户看的自然语言答案。

        recommended_breeds:
            推荐犬种列表。
            只有推荐类问题通常会有内容。

        evidences:
            答案证据列表。
            可以来自 RAG chunk、metadata、规则命中结果。

        confidence:
            答案置信度。
            范围 0 到 1。
            不是数学绝对概率，而是系统对答案可靠性的工程评分。

        reason:
            系统生成这个答案的主要原因。
            用于 debug、日志、可观测、后续评估。

        is_fallback:
            是否走了 fallback 兜底逻辑。

        fallback_reason:
            如果走了 fallback，这里记录 fallback 原因。

        debug:
            调试信息。
            可以保存内部 pipeline 节点结果、retriever 信息、formatter 信息。
            对外返回时可以隐藏。

        metadata:
            扩展信息。
            保存本次回答相关的非固定字段。
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="用户原始问题。",
    )

    query_type: DogKnowledgeQueryType = Field(
        ...,
        description="问题类型，例如 exact_lookup、recommendation、fallback。",
    )

    status: DogKnowledgeAnswerStatus = Field(
        ...,
        description="答案状态，例如 success、partial、fallback、empty、error。",
    )

    answer: str = Field(
        ...,
        description="最终给用户看的自然语言答案。",
    )

    recommended_breeds: list[DogKnowledgeRecommendationItem] = Field(
        default_factory=list,
        description="推荐犬种列表，推荐类问题通常会使用。",
    )

    evidences: list[DogKnowledgeEvidence] = Field(
        default_factory=list,
        description="答案证据列表，记录回答参考了哪些内容。",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="答案置信度，范围 0 到 1。",
    )

    reason: str | None = Field(
        default=None,
        description="生成该答案的原因说明。",
    )

    is_fallback: bool = Field(
        default=False,
        description="是否走了 fallback 兜底逻辑。",
    )

    fallback_reason: str | None = Field(
        default=None,
        description="fallback 原因，仅在 is_fallback=True 时通常有值。",
    )

    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="内部调试信息，对外返回时可以隐藏。",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展信息，用于保存暂时不固定的字段。",
    )

    def has_recommendations(self) -> bool:
        """
        判断当前答案是否包含推荐犬种。

        参数：
            无。

        返回值：
            bool:
                True 表示 recommended_breeds 不为空；
                False 表示没有推荐犬种。
        """

        return len(self.recommended_breeds) > 0

    def has_evidences(self) -> bool:
        """
        判断当前答案是否包含证据。

        参数：
            无。

        返回值：
            bool:
                True 表示 evidences 不为空；
                False 表示没有证据。
        """

        return len(self.evidences) > 0

    def to_public_dict(self, include_debug: bool = False) -> dict[str, Any]:
        """
        将 DogKnowledgeAnswer 转换成可对外返回的 dict。

        功能：
            默认隐藏 debug 字段，避免把内部链路、节点状态、调试数据直接暴露给前端或用户。
            如果需要排查问题，可以传 include_debug=True。

        参数：
            include_debug:
                是否包含 debug 调试信息。
                True 表示保留 debug；
                False 表示移除 debug。

        返回值：
            dict[str, Any]:
                可以直接返回给 API、WebUI 或日志系统的字典结构。
        """

        data = self.model_dump(mode="json", exclude_none=True)

        if not include_debug:
            data.pop("debug", None)

        return data

    @classmethod
    def build_fallback(
        cls,
        question: str,
        answer: str,
        fallback_reason: str,
        confidence: float = 0.0,
        debug: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DogKnowledgeAnswer":
        """
        快速构建 fallback 类型答案。

        功能：
            当 DogKnowledgeAgent 无法可靠回答、没有检索结果、内部异常或用户问题超出范围时，
            使用这个方法统一生成 fallback 答案。

        参数：
            question:
                用户原始问题。

            answer:
                最终返回给用户的兜底答案。

            fallback_reason:
                触发 fallback 的原因。

            confidence:
                fallback 答案的置信度。
                默认是 0.0。

            debug:
                可选调试信息。

            metadata:
                可选扩展信息。

        返回值：
            DogKnowledgeAnswer:
                一个标准 fallback 答案对象。
        """

        return cls(
            question=question,
            query_type="fallback",
            status="fallback",
            answer=answer,
            confidence=confidence,
            reason=fallback_reason,
            is_fallback=True,
            fallback_reason=fallback_reason,
            debug=debug or {},
            metadata=metadata or {},
        )