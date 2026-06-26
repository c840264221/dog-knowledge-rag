from typing import Any, Literal

from pydantic import BaseModel
from pydantic import Field

from src.settings import settings


RetrievalQualityStatus = Literal[
    "good",
    "weak",
    "bad",
]

RetrievalFailureType = Literal[
    "empty",
    "metadata_mismatch",
    "section_mismatch",
    "insufficient_context",
    "score_too_low",
    "low_quality",
    "unknown",
]


QUESTION_TOPIC_KEYWORDS = {
    "temperament": {
        "question_keywords": [
            "性格",
            "脾气",
            "温顺",
            "友好",
            "亲人",
            "适合孩子",
            "和其他狗",
            "temperament",
            "personality",
        ],
        "section_keywords": [
            "性格",
            "性格特征",
            "temperament",
            "personality",
            "affectionate",
            "good with",
        ],
    },
    "training": {
        "question_keywords": [
            "训练",
            "可训练",
            "聪明",
            "服从",
            "train",
            "training",
            "trainability",
        ],
        "section_keywords": [
            "训练",
            "可训练",
            "trainability",
            "training",
        ],
    },
    "shedding": {
        "question_keywords": [
            "掉毛",
            "脱毛",
            "毛发",
            "shedding",
            "coat",
        ],
        "section_keywords": [
            "掉毛",
            "毛发",
            "shedding",
            "coat",
            "grooming",
        ],
    },
    "energy": {
        "question_keywords": [
            "精力",
            "运动",
            "活跃",
            "energy",
            "exercise",
        ],
        "section_keywords": [
            "精力",
            "运动",
            "energy",
            "exercise",
        ],
    },
    "basic_info": {
        "question_keywords": [
            "身高",
            "体重",
            "寿命",
            "多大",
            "height",
            "weight",
            "lifespan",
        ],
        "section_keywords": [
            "基本信息",
            "height",
            "weight",
            "lifespan",
            "basic",
        ],
    },
}


class RetrievalQualityResult(BaseModel):
    """
    RAG 召回质量评估结果。

    功能：
        保存 RAG 召回质量评估后的结构化结果。
        evaluate_node 会把该对象写入 DogState。

    技术名词：
        Retrieval Quality：
            召回质量。表示当前 RAG 召回内容是否足够支持回答。

        Failure Type：
            失败类型。表示召回不可用或质量不足的主要原因。

        Quality Score：
            质量分。当前是项目自定义分数，不是向量库原始置信度。

    字段说明：
        status:
            good / weak / bad。

        is_usable:
            是否允许进入 generate_node。

        failure_type:
            失败类型。

        quality_score:
            综合质量分，范围 0 到 1。

        reasons:
            质量评估原因。

        metrics:
            评估指标明细。
    """

    status: RetrievalQualityStatus = "bad"

    is_usable: bool = False

    failure_type: RetrievalFailureType | None = None

    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    reasons: list[str] = Field(
        default_factory=list
    )

    metrics: dict[str, Any] = Field(
        default_factory=dict
    )


def evaluate_retrieval_quality(
        state: dict[str, Any],
) -> RetrievalQualityResult:
    """
    从 LangGraph State 中评估 RAG 召回质量。

    功能：
        这是给 evaluate_node 使用的入口函数。
        它从 state 中取出 rag_context、question、dog_name、filters，
        然后调用 evaluate_rag_context_quality 执行核心评估。

    注意：
        当前主路径只评估 rag_context。
        docs 是旧版兼容字段，不作为新版质量评估主依据。

    参数：
        state:
            LangGraph 当前状态。

    返回值：
        RetrievalQualityResult:
            召回质量评估结果。
    """

    rag_context = normalize_mapping(
        state.get(
            "rag_context"
        )
    )

    question = str(
        state.get(
            "question",
            ""
        )
        or ""
    )

    expected_dog_name = extract_expected_dog_name(
        state=state
    )

    return evaluate_rag_context_quality(
        rag_context=rag_context,
        question=question,
        expected_dog_name=expected_dog_name,
    )


def evaluate_rag_context_quality(
        rag_context: dict[str, Any],
        question: str,
        expected_dog_name: str | None = None,
) -> RetrievalQualityResult:
    """
    评估 RagContext 召回质量。

    功能：
        基于新版 RagContext 进行质量判断。
        该函数不依赖旧版 docs 字段。

    判断维度：
        1. rag_context 是否为空。
        2. chunks 是否存在。
        3. dog_name metadata 是否命中。
        4. section 是否匹配用户问题主题。
        5. retrieval_score 是否在合理范围。
        6. context_text 是否足够长。

    参数：
        rag_context:
            新版 RAG 上下文。

        question:
            用户问题。

        expected_dog_name:
            期望命中的犬种英文名。

    返回值：
        RetrievalQualityResult:
            召回质量评估结果。
    """

    min_quality_score = settings.rag.quality_min_score

    min_context_chars = settings.rag.quality_min_context_chars

    max_distance_threshold = settings.rag.quality_max_distance

    chunks = extract_chunks_from_rag_context(
        rag_context=rag_context
    )

    context_text = str(
        rag_context.get(
            "context_text",
            ""
        )
        or ""
    )

    context_length = len(
        context_text.strip()
    )

    chunks_count = len(
        chunks
    )

    reasons: list[str] = []

    metrics: dict[str, Any] = {
        "chunks_count": chunks_count,
        "context_length": context_length,
        "expected_dog_name": expected_dog_name,
        "min_quality_score": min_quality_score,
        "min_context_chars": min_context_chars,
        "max_distance_threshold": max_distance_threshold,
    }

    if not rag_context:
        reasons.append(
            "rag_context 不存在，无法评估召回质量。"
        )

        return RetrievalQualityResult(
            status="bad",
            is_usable=False,
            failure_type="empty",
            quality_score=0.0,
            reasons=reasons,
            metrics=metrics,
        )

    rag_status = str(
        rag_context.get(
            "status",
            "empty"
        )
        or "empty"
    )

    metrics[
        "rag_status"
    ] = rag_status

    if rag_status == "empty" or chunks_count == 0:
        reasons.append(
            f"rag_context.status={rag_status}，chunks_count={chunks_count}，召回为空。"
        )

        return RetrievalQualityResult(
            status="bad",
            is_usable=False,
            failure_type="empty",
            quality_score=0.0,
            reasons=reasons,
            metrics=metrics,
        )

    quality_score = 0.0

    quality_score += settings.rag.quality_non_empty_weight

    reasons.append(
        f"召回结果非空，基础分 +{settings.rag.quality_non_empty_weight}。"
    )

    if settings.rag.quality_enable_metadata_check:
        dog_name_hit_count = count_dog_name_hits_from_chunks(
            chunks=chunks,
            expected_dog_name=expected_dog_name,
        )

        metrics[
            "dog_name_hit_count"
        ] = dog_name_hit_count

        if expected_dog_name:
            if dog_name_hit_count > 0:
                quality_score += settings.rag.quality_metadata_hit_weight

                reasons.append(
                    f"召回结果命中目标 dog_name={expected_dog_name}，metadata 分 +{settings.rag.quality_metadata_hit_weight}。"
                )
            else:
                reasons.append(
                    f"期望 dog_name={expected_dog_name}，但 RagContext.chunks 中没有命中。"
                )

                return RetrievalQualityResult(
                    status="bad",
                    is_usable=False,
                    failure_type="metadata_mismatch",
                    quality_score=round(
                        quality_score,
                        3
                    ),
                    reasons=reasons,
                    metrics=metrics,
                )
        else:
            neutral_score = settings.rag.quality_metadata_hit_weight / 2

            quality_score += neutral_score

            reasons.append(
                f"当前问题没有明确 dog_name，metadata 检查给中性分 +{neutral_score}。"
            )

    expected_topics = infer_question_topics(
        question=question
    )

    metrics[
        "expected_topics"
    ] = expected_topics

    if settings.rag.quality_enable_section_check:
        section_hit_count = count_section_hits_from_chunks(
            chunks=chunks,
            expected_topics=expected_topics,
        )

        metrics[
            "section_hit_count"
        ] = section_hit_count

        if expected_topics:
            if section_hit_count > 0:
                quality_score += settings.rag.quality_section_hit_weight

                reasons.append(
                    f"召回 section 命中问题主题 {expected_topics}，section 分 +{settings.rag.quality_section_hit_weight}。"
                )
            else:
                reasons.append(
                    f"问题主题为 {expected_topics}，但召回 section 没有明显匹配。"
                )
        else:
            neutral_score = settings.rag.quality_section_hit_weight / 2

            quality_score += neutral_score

            reasons.append(
                f"无法明确问题主题，section 检查给中性分 +{neutral_score}。"
            )

    best_distance = extract_best_retrieval_distance_from_chunks(
        chunks=chunks
    )

    metrics[
        "best_retrieval_score"
    ] = best_distance

    if settings.rag.quality_enable_score_check:
        if max_distance_threshold is None:
            neutral_score = settings.rag.quality_score_weight / 2

            quality_score += neutral_score

            reasons.append(
                f"未配置最大 distance 阈值，score 检查给中性分 +{neutral_score}。"
            )
        elif best_distance is None:
            neutral_score = settings.rag.quality_score_weight / 4

            quality_score += neutral_score

            reasons.append(
                f"召回结果没有 retrieval_score，score 检查给较低中性分 +{neutral_score}。"
            )
        elif best_distance <= max_distance_threshold:
            quality_score += settings.rag.quality_score_weight

            reasons.append(
                f"最佳 retrieval_score={best_distance}，小于等于阈值 {max_distance_threshold}，score 分 +{settings.rag.quality_score_weight}。"
            )
        else:
            reasons.append(
                f"最佳 retrieval_score={best_distance}，大于阈值 {max_distance_threshold}，相关性偏弱。"
            )

    if context_length >= min_context_chars:
        quality_score += settings.rag.quality_context_length_weight

        reasons.append(
            f"context_text 长度为 {context_length}，达到最小要求 {min_context_chars}，上下文长度分 +{settings.rag.quality_context_length_weight}。"
        )
    else:
        reasons.append(
            f"context_text 长度为 {context_length}，低于最小要求 {min_context_chars}。"
        )

    quality_score = min(
        round(
            quality_score,
            3
        ),
        1.0
    )

    metrics[
        "quality_score"
    ] = quality_score

    if quality_score >= min_quality_score:
        reasons.append(
            f"综合质量分 {quality_score} >= {min_quality_score}，召回结果可用。"
        )

        return RetrievalQualityResult(
            status="good",
            is_usable=True,
            failure_type=None,
            quality_score=quality_score,
            reasons=reasons,
            metrics=metrics,
        )

    failure_type = decide_failure_type(
        expected_topics=expected_topics,
        section_hit_count=metrics.get(
            "section_hit_count",
            0
        ),
        context_length=context_length,
        min_context_chars=min_context_chars,
        best_distance=best_distance,
        max_distance_threshold=max_distance_threshold,
    )

    reasons.append(
        f"综合质量分 {quality_score} < {min_quality_score}，召回结果质量不足。"
    )

    return RetrievalQualityResult(
        status="weak",
        is_usable=False,
        failure_type=failure_type,
        quality_score=quality_score,
        reasons=reasons,
        metrics=metrics,
    )


def normalize_mapping(
        value: Any,
) -> dict[str, Any]:
    """
    将对象归一化成 dict。

    功能：
        兼容 dict 和 Pydantic BaseModel。
        如果 value 是 RagContext / RagRetrievedChunk / RagChunk 等 Pydantic 对象，
        会尝试调用 model_dump 转成 dict。

    参数：
        value:
            任意对象。

    返回值：
        dict[str, Any]:
            转换后的字典。
    """

    if isinstance(
            value,
            dict
    ):
        return value

    if hasattr(
            value,
            "model_dump"
    ):
        try:
            dumped = value.model_dump()

            if isinstance(
                    dumped,
                    dict
            ):
                return dumped

        except Exception:
            return {}

    return {}


def extract_chunks_from_rag_context(
        rag_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    从 RagContext 中提取 chunks。

    功能：
        将 rag_context["chunks"] 中的每个 RagRetrievedChunk 归一化成 dict。

    参数：
        rag_context:
            RagContext 字典。

    返回值：
        list[dict[str, Any]]:
            召回 chunk 列表。
    """

    chunks = rag_context.get(
        "chunks",
        []
    )

    if not isinstance(
            chunks,
            list
    ):
        return []

    normalized_chunks = []

    for item in chunks:
        normalized_item = normalize_mapping(
            item
        )

        if normalized_item:
            normalized_chunks.append(
                normalized_item
            )

    return normalized_chunks


def extract_expected_dog_name(
        state: dict[str, Any],
) -> str | None:
    """
    从 state 中提取期望命中的 dog_name。

    功能：
        优先读取 state["dog_name"]。
        如果没有，则从 filters 中尝试解析 dog_name。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        str | None:
            期望命中的犬种英文名。
    """

    dog_name = state.get(
        "dog_name"
    )

    if dog_name:
        return str(
            dog_name
        ).strip()

    filters = state.get(
        "filters",
        {}
    ) or {}

    if not isinstance(
            filters,
            dict
    ):
        return None

    raw_value = filters.get(
        "dog_name"
    ) or filters.get(
        "name"
    )

    if isinstance(
            raw_value,
            dict
    ):
        value = raw_value.get(
            "$eq"
        )

        if value:
            return str(
                value
            ).strip()

    if raw_value:
        return str(
            raw_value
        ).strip()

    return None


def count_dog_name_hits_from_chunks(
        chunks: list[dict[str, Any]],
        expected_dog_name: str | None,
) -> int:
    """
    统计 RagContext.chunks 中命中 dog_name 的数量。

    功能：
        基于新版 RagContext 结构判断召回结果是否命中目标犬种。
        不依赖旧版 docs。

    参数：
        chunks:
            RagRetrievedChunk 字典列表。

        expected_dog_name:
            期望命中的犬种英文名。

    返回值：
        int:
            命中数量。
    """

    if not expected_dog_name:
        return 0

    expected = expected_dog_name.lower()

    hit_count = 0

    for retrieved_chunk in chunks:
        chunk = normalize_mapping(
            retrieved_chunk.get(
                "chunk",
                {}
            )
        )

        metadata = normalize_mapping(
            chunk.get(
                "metadata",
                {}
            )
        )

        dog_name = str(
            metadata.get(
                "dog_name",
                ""
            )
            or ""
        ).lower()

        if dog_name == expected:
            hit_count += 1

    return hit_count


def infer_question_topics(
        question: str,
) -> list[str]:
    """
    根据用户问题推断主题。

    功能：
        使用关键词规则判断用户问题关注的主题，
        例如性格、训练、掉毛、精力、基本信息等。

    参数：
        question:
            用户问题。

    返回值：
        list[str]:
            主题列表。
    """

    normalized_question = question.lower()

    topics = []

    for topic_name, config in QUESTION_TOPIC_KEYWORDS.items():
        for keyword in config[
            "question_keywords"
        ]:
            if keyword.lower() in normalized_question:
                topics.append(
                    topic_name
                )

                break

    return topics


def count_section_hits_from_chunks(
        chunks: list[dict[str, Any]],
        expected_topics: list[str],
) -> int:
    """
    统计 RagContext.chunks 中 section 是否命中问题主题。

    功能：
        通过 chunk.title、metadata.section_title、metadata.heading、content 前 200 字
        判断召回 chunk 是否与用户问题主题匹配。

    参数：
        chunks:
            RagRetrievedChunk 字典列表。

        expected_topics:
            问题主题列表。

    返回值：
        int:
            命中数量。
    """

    if not expected_topics:
        return 0

    section_keywords = collect_section_keywords(
        expected_topics=expected_topics
    )

    hit_count = 0

    for retrieved_chunk in chunks:
        chunk = normalize_mapping(
            retrieved_chunk.get(
                "chunk",
                {}
            )
        )

        metadata = normalize_mapping(
            chunk.get(
                "metadata",
                {}
            )
        )

        searchable_text = " ".join(
            [
                str(
                    chunk.get(
                        "title",
                        ""
                    )
                    or ""
                ),
                str(
                    metadata.get(
                        "section_title",
                        ""
                    )
                    or ""
                ),
                str(
                    metadata.get(
                        "heading",
                        ""
                    )
                    or ""
                ),
                str(
                    chunk.get(
                        "content",
                        ""
                    )
                    or ""
                )[:200],
            ]
        ).lower()

        if contains_any_keyword(
                text=searchable_text,
                keywords=section_keywords,
        ):
            hit_count += 1

    return hit_count


def collect_section_keywords(
        expected_topics: list[str],
) -> list[str]:
    """
    根据问题主题收集 section 关键词。

    参数:
        expected_topics:
            问题主题列表。

    返回值:
        list[str]:
            section 关键词列表。
    """

    keywords = []

    for topic in expected_topics:
        config = QUESTION_TOPIC_KEYWORDS.get(
            topic,
            {}
        )

        keywords.extend(
            config.get(
                "section_keywords",
                []
            )
        )

    return keywords


def contains_any_keyword(
        text: str,
        keywords: list[str],
) -> bool:
    """
    判断文本是否包含任意关键词。

    参数:
        text:
            待检查文本。

        keywords:
            关键词列表。

    返回值:
        bool:
            True 表示命中任意关键词。
    """

    normalized_text = text.lower()

    for keyword in keywords:
        if keyword.lower() in normalized_text:
            return True

    return False


def extract_best_retrieval_distance_from_chunks(
        chunks: list[dict[str, Any]],
) -> float | None:
    """
    从 RagContext.chunks 中提取最佳 retrieval_score。

    功能：
        当前默认 retrieval_score 是 distance 距离分数。
        因此分数越小越好，返回最小值。

    参数:
        chunks:
            RagRetrievedChunk 字典列表。

    返回值:
        float | None:
            最小 retrieval_score。
    """

    scores = []

    for retrieved_chunk in chunks:
        score = retrieved_chunk.get(
            "retrieval_score"
        )

        if score is None:
            continue

        try:
            scores.append(
                float(
                    score
                )
            )
        except (
                TypeError,
                ValueError,
        ):
            pass

    if not scores:
        return None

    return min(
        scores
    )


def decide_failure_type(
        expected_topics: list[str],
        section_hit_count: int,
        context_length: int,
        min_context_chars: int,
        best_distance: float | None,
        max_distance_threshold: float | None,
) -> RetrievalFailureType:
    """
    判断召回失败类型。

    功能：
        根据质量评估指标判断主要失败原因。
        retry_node 后续会根据 failure_type 选择不同策略。

    参数:
        expected_topics:
            问题主题列表。

        section_hit_count:
            section 命中数量。

        context_length:
            context_text 长度。

        min_context_chars:
            最小上下文长度。

        best_distance:
            最佳 retrieval_score。

        max_distance_threshold:
            最大允许 distance。

    返回值:
        RetrievalFailureType:
            失败类型。
    """

    if context_length < min_context_chars:
        return "insufficient_context"

    if expected_topics and section_hit_count == 0:
        return "section_mismatch"

    if (
            best_distance is not None
            and max_distance_threshold is not None
            and best_distance > max_distance_threshold
    ):
        return "score_too_low"

    return "low_quality"