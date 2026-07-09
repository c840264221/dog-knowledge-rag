from collections.abc import Mapping, Sequence
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogFallbackLayerOutput,
    DogGenerationLayerOutput,
    DogRecommendationLayerOutput,
    DogRetrievalLayerOutput,
)
from src.agents.dog_knowledge_agent.nodes.query_layer_output_node import (
    build_dog_query_layer_output_from_state,
)
from src.agents.dog_knowledge_agent.contracts.schemas import (
    DogKnowledgeEvidence,
    DogKnowledgeRecommendationItem,
)


def build_legacy_state_to_dog_knowledge_layer_outputs_node():
    """
    构建旧 state 到 DogKnowledgeAgent 分层契约的适配节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点读取当前真实主图已经存在的旧字段，
        例如 rag_query、rag_context、final_answer、answer_strategy，
        并转换成 v1.7.4 标准 layer output 字段。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回 dog_query_result、dog_retrieval_result 等字段。
    """

    def legacy_state_to_dog_knowledge_layer_outputs_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        旧 state 到分层契约的适配节点。

        功能：
            将当前已有 DogState 字段转换成 V1.7.4 layer outputs。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update。
        """

        return build_layer_outputs_from_legacy_state(state)

    return legacy_state_to_dog_knowledge_layer_outputs_node


def build_layer_outputs_from_legacy_state(
    state: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """
    从当前旧 DogState 构建 V1.7.4 分层输出。

    功能：
        作为过渡期适配器，从旧字段中提取信息并生成：
        dog_query_result、dog_retrieval_result、dog_recommendation_result、
        dog_generation_result、dog_fallback_result。

    参数：
        state:
            LangGraph state 或普通状态对象。

    返回值：
        dict[str, Any]:
            包含分层输出字段的 state update。
    """

    state_data = _to_dict(state)
    existing_query_result = _to_dict(state_data.get("dog_query_result"))
    existing_retrieval_result = _to_dict(state_data.get("dog_retrieval_result"))
    existing_generation_result = _to_dict(state_data.get("dog_generation_result"))
    existing_fallback_result = _to_dict(state_data.get("dog_fallback_result"))
    query_result = build_dog_query_layer_output_from_state(state_data)
    query_type = _first_non_empty_str(
        existing_query_result.get("query_type"),
        query_result.query_type,
    )
    retrieval_result = _build_retrieval_layer_output(
        state_data=state_data,
        query_type=query_type,
    )
    recommendation_result = _build_recommendation_layer_output(state_data)
    generation_result = _build_generation_layer_output(state_data)
    fallback_result = _build_fallback_layer_output(state_data)

    update: dict[str, Any] = {}

    if not existing_query_result:
        update["dog_query_result"] = query_result.model_dump(mode="python")

    if not existing_retrieval_result:
        update["dog_retrieval_result"] = retrieval_result.model_dump(mode="python")

    if recommendation_result is not None:
        update["dog_recommendation_result"] = recommendation_result.model_dump(
            mode="python",
        )

    if generation_result is not None and not existing_generation_result:
        update["dog_generation_result"] = generation_result.model_dump(
            mode="python",
        )

    if fallback_result is not None and not existing_fallback_result:
        update["dog_fallback_result"] = fallback_result.model_dump(mode="python")

    return update


def _build_retrieval_layer_output(
    state_data: dict[str, Any],
    query_type: str,
) -> DogRetrievalLayerOutput:
    """
    从旧 state 构建检索层输出。

    参数：
        state_data:
            当前状态字典。

        query_type:
            标准问题类型。

    返回值：
        DogRetrievalLayerOutput:
            检索层标准输出。
    """

    rag_context = _to_dict(state_data.get("rag_context"))
    chunks = _as_list(rag_context.get("chunks"))
    evidences = [
        evidence
        for evidence in [
            _build_evidence_from_chunk(
                raw_chunk=raw_chunk,
                index=index,
            )
            for index, raw_chunk in enumerate(chunks)
        ]
        if evidence is not None
    ]

    return DogRetrievalLayerOutput(
        query_type=query_type,
        evidences=evidences,
        retrieved_count=len(evidences),
        confidence=_resolve_retrieval_confidence(
            rag_context=rag_context,
            evidences=evidences,
        ),
        reason=(
            f"从旧 rag_context 适配生成检索层输出，"
            f"evidences={len(evidences)}。"
        ),
        metadata={
            "source": "legacy_state",
            "rag_context_status": rag_context.get("status"),
            "source_count": rag_context.get("source_count"),
        },
    )


def _build_recommendation_layer_output(
    state_data: dict[str, Any],
) -> DogRecommendationLayerOutput | None:
    """
    从旧 state 构建推荐层输出。

    参数：
        state_data:
            当前状态字典。

    返回值：
        DogRecommendationLayerOutput | None:
            推荐层标准输出；如果没有推荐结果则返回 None。
    """

    raw_items = _as_list(
        state_data.get("recommended_breeds")
        or state_data.get("recommendations")
    )

    recommendations = [
        recommendation
        for recommendation in [
            _build_recommendation_item(raw_item)
            for raw_item in raw_items
        ]
        if recommendation is not None
    ]

    if not recommendations:
        return None

    return DogRecommendationLayerOutput(
        recommended_breeds=recommendations,
        confidence=_max_score_or_default(
            [
                item.score
                for item in recommendations
            ],
            default=0.0,
        ),
        reason="从旧推荐字段适配生成推荐层输出。",
        metadata={
            "source": "legacy_state",
        },
    )


def _build_generation_layer_output(
    state_data: dict[str, Any],
) -> DogGenerationLayerOutput | None:
    """
    从旧 state 构建生成层输出。

    参数：
        state_data:
            当前状态字典。

    返回值：
        DogGenerationLayerOutput | None:
            生成层标准输出；如果没有答案文本则返回 None。
    """

    answer = _first_non_empty_str(
        state_data.get("final_answer"),
        state_data.get("answer"),
    )

    if not answer:
        return None

    return DogGenerationLayerOutput(
        generated_answer=answer,
        confidence=0.0,
        reason="从旧 final_answer / answer 字段适配生成生成层输出。",
        used_evidence_ids=[],
        metadata={
            "source": "legacy_state",
        },
    )


def _build_fallback_layer_output(
    state_data: dict[str, Any],
) -> DogFallbackLayerOutput | None:
    """
    从旧 state 构建兜底层输出。

    参数：
        state_data:
            当前状态字典。

    返回值：
        DogFallbackLayerOutput | None:
            兜底层标准输出；如果没有兜底迹象则返回 None。
    """

    error = _first_non_empty_str(
        state_data.get("error"),
        state_data.get("fallback_reason"),
    )

    retrieval_failure_type = _first_non_empty_str(
        state_data.get("retrieval_failure_type"),
    )

    if not error and not retrieval_failure_type:
        return None

    fallback_reason = _first_non_empty_str(
        error,
        retrieval_failure_type,
        "DogKnowledgeAgent 当前无法可靠回答该问题。",
    )

    return DogFallbackLayerOutput(
        fallback_reason=fallback_reason,
        generated_answer=_first_non_empty_str(
            state_data.get("final_answer"),
            state_data.get("answer"),
            "我暂时无法基于当前犬种知识库可靠回答这个问题。",
        ),
        confidence=0.1,
        reason=fallback_reason,
        metadata={
            "source": "legacy_state",
            "retrieval_failure_type": retrieval_failure_type,
        },
    )


def _build_evidence_from_chunk(
    raw_chunk: Any,
    index: int,
) -> DogKnowledgeEvidence | None:
    """
    从 RagContext chunk 构建标准证据。

    参数：
        raw_chunk:
            原始 RAG chunk，可能是 dict 或 Pydantic 对象。

        index:
            当前 chunk 下标。

    返回值：
        DogKnowledgeEvidence | None:
            标准证据；如果没有内容则返回 None。
    """

    item_data = _to_dict(raw_chunk)
    chunk_data = _to_dict(item_data.get("chunk"))
    metadata = _to_dict(chunk_data.get("metadata"))
    content = _first_non_empty_str(
        chunk_data.get("content"),
        item_data.get("content"),
        item_data.get("page_content"),
    )

    if not content:
        return None

    evidence_id = _first_non_empty_str(
        chunk_data.get("chunk_id"),
        item_data.get("evidence_id"),
        item_data.get("chunk_id"),
        f"legacy-rag-evidence-{index + 1}",
    )

    return DogKnowledgeEvidence(
        evidence_id=evidence_id,
        source_kind="rag_chunk",
        title=_first_non_empty_str(
            chunk_data.get("title"),
            metadata.get("dog_name"),
        ) or None,
        content=content,
        score=_normalize_score(
            item_data.get("final_score")
            or item_data.get("rerank_score")
            or item_data.get("retrieval_score")
        ),
        metadata=metadata,
    )


def _build_recommendation_item(
    raw_item: Any,
) -> DogKnowledgeRecommendationItem | None:
    """
    从旧推荐字段构建标准推荐项。

    参数：
        raw_item:
            原始推荐项。

    返回值：
        DogKnowledgeRecommendationItem | None:
            标准推荐项；无法识别犬种名时返回 None。
    """

    item_data = _to_dict(raw_item)
    breed_name = _first_non_empty_str(
        item_data.get("breed_name"),
        item_data.get("dog_name"),
        item_data.get("name"),
    )

    if not breed_name:
        return None

    return DogKnowledgeRecommendationItem(
        breed_name=breed_name,
        display_name=_first_non_empty_str(
            item_data.get("display_name"),
            item_data.get("title"),
            item_data.get("name_cn"),
        ) or None,
        reason=_first_non_empty_str(
            item_data.get("reason"),
            item_data.get("description"),
            "该犬种与用户需求存在一定匹配。",
        ),
        matched_traits=_as_str_list(
            item_data.get("matched_traits")
            or item_data.get("traits")
        ),
        warnings=_as_str_list(
            item_data.get("warnings")
            or item_data.get("notes")
        ),
        evidence_ids=_as_str_list(
            item_data.get("evidence_ids")
            or item_data.get("chunk_ids")
        ),
        score=_normalize_score(
            item_data.get("score")
            or item_data.get("confidence")
        ),
        metadata=_to_dict(item_data.get("metadata")),
    )


def _resolve_retrieval_confidence(
    rag_context: dict[str, Any],
    evidences: list[DogKnowledgeEvidence],
) -> float:
    """
    解析检索层置信度。

    参数：
        rag_context:
            RAG 上下文字典。

        evidences:
            证据列表。

    返回值：
        float:
            检索层置信度。
    """

    if not evidences:
        return 0.0

    scores = [
        item.score
        for item in evidences
        if item.score is not None
    ]

    return _max_score_or_default(
        scores,
        default=0.7 if rag_context.get("status") == "success" else 0.5,
    )


def _normalize_score(
    value: Any,
) -> float | None:
    """
    将分数归一化到 0 到 1。

    参数：
        value:
            原始分数。

    返回值：
        float | None:
            归一化后的分数。
    """

    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score > 1.0 and score <= 100.0:
        score = score / 100.0

    if score < 0.0:
        return 0.0

    if score > 1.0:
        return 1.0

    return score


def _max_score_or_default(
    values: list[float | None],
    default: float,
) -> float:
    """
    返回最大有效分数或默认值。

    参数：
        values:
            候选分数列表。

        default:
            默认值。

    返回值：
        float:
            最大有效分数或默认值。
    """

    normalized_values = [
        value
        for value in values
        if value is not None
    ]

    if not normalized_values:
        return default

    return max(normalized_values)


def _as_list(
    value: Any,
) -> list[Any]:
    """
    将任意值转换成列表。

    参数：
        value:
            原始值。

    返回值：
        list[Any]:
            列表形式的值。
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [
            value,
        ]

    if isinstance(value, Sequence):
        return list(value)

    return [
        value,
    ]


def _as_str_list(
    value: Any,
) -> list[str]:
    """
    将任意值转换成字符串列表。

    参数：
        value:
            原始值。

    返回值：
        list[str]:
            字符串列表。
    """

    result = []

    for item in _as_list(value):
        text = str(item or "").strip()

        if text:
            result.append(text)

    return result


def _first_non_empty_str(
    *values: Any,
) -> str:
    """
    返回第一个非空字符串。

    参数：
        values:
            候选值列表。

    返回值：
        str:
            第一个非空字符串；如果没有找到则返回空字符串。
    """

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    将任意对象转换成 dict。

    参数：
        value:
            原始对象，可以是 dict、Pydantic Model 或普通对象。

    返回值：
        dict[str, Any]:
            转换后的字典；无法转换时返回空字典。
    """

    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(value, "dict"):
        dumped = value.dict()

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(value, "__dict__"):
        return dict(value.__dict__)

    return {}


legacy_state_to_dog_knowledge_layer_outputs_node = (
    build_legacy_state_to_dog_knowledge_layer_outputs_node()
)

