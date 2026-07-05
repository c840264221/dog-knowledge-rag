from collections.abc import Mapping
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogKnowledgePipelineResult,
)


def build_aggregate_dog_knowledge_layer_outputs_node():
    """
    构建 DogKnowledgeAgent 分层输出聚合节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点读取 DogKnowledgeAgent 各 layer 的标准中间产物，
        汇总生成 dog_knowledge_pipeline_result。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回包含 dog_knowledge_pipeline_result 的 state update。
    """

    def aggregate_dog_knowledge_layer_outputs_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 分层输出聚合节点。

        功能：
            从 state 中读取 dog_query_result、dog_retrieval_result、
            dog_recommendation_result、dog_generation_result、dog_fallback_result，
            聚合成 dog_knowledge_pipeline_result。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update。
        """

        return aggregate_dog_knowledge_layer_outputs(state)

    return aggregate_dog_knowledge_layer_outputs_node


def aggregate_dog_knowledge_layer_outputs(
    state: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """
    聚合 DogKnowledgeAgent 各层标准中间产物。

    功能：
        将查询理解层、检索层、推荐层、生成层、兜底层的输出汇总成
        DogKnowledgePipelineResult，并以 LangGraph state update 形式返回。

    参数：
        state:
            LangGraph state 或普通状态对象。

    返回值：
        dict[str, Any]:
            包含 dog_knowledge_pipeline_result 的 state update。
    """

    state_data = _to_dict(state)

    query_result = _to_dict(state_data.get("dog_query_result"))
    retrieval_result = _to_dict(state_data.get("dog_retrieval_result"))
    recommendation_result = _to_dict(state_data.get("dog_recommendation_result"))
    generation_result = _to_dict(state_data.get("dog_generation_result"))
    fallback_result = _to_dict(state_data.get("dog_fallback_result"))

    is_fallback = bool(fallback_result.get("is_fallback"))

    question = _first_non_empty_str(
        query_result.get("question"),
        state_data.get("question"),
        state_data.get("user_question"),
    )

    query_type = _resolve_query_type(
        is_fallback=is_fallback,
        query_result=query_result,
        retrieval_result=retrieval_result,
        recommendation_result=recommendation_result,
    )

    answer = _resolve_answer(
        is_fallback=is_fallback,
        fallback_result=fallback_result,
        generation_result=generation_result,
        state_data=state_data,
    )

    recommended_breeds = list(
        recommendation_result.get("recommended_breeds") or []
    )

    evidences = list(
        retrieval_result.get("evidences") or []
    )

    fallback_reason = None
    if is_fallback:
        fallback_reason = _first_non_empty_str(
            fallback_result.get("fallback_reason"),
            fallback_result.get("reason"),
            "DogKnowledgeAgent 当前无法基于已有犬种知识库可靠回答该问题。",
        )

    status = _resolve_status(
        is_fallback=is_fallback,
        answer=answer,
        recommended_breeds=recommended_breeds,
        evidences=evidences,
    )

    confidence = _resolve_confidence(
        is_fallback=is_fallback,
        fallback_result=fallback_result,
        generation_result=generation_result,
        recommendation_result=recommendation_result,
        retrieval_result=retrieval_result,
        query_result=query_result,
    )

    reason = _resolve_reason(
        is_fallback=is_fallback,
        fallback_result=fallback_result,
        generation_result=generation_result,
        recommendation_result=recommendation_result,
        retrieval_result=retrieval_result,
        query_result=query_result,
    )

    pipeline_result = DogKnowledgePipelineResult(
        question=question,
        query_type=query_type,
        status=status,
        answer=answer,
        recommended_breeds=recommended_breeds,
        evidences=evidences,
        confidence=confidence,
        reason=reason,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        metadata=_build_metadata(
            query_result=query_result,
            retrieval_result=retrieval_result,
            recommendation_result=recommendation_result,
            generation_result=generation_result,
            fallback_result=fallback_result,
        ),
        debug=_build_debug(
            query_result=query_result,
            retrieval_result=retrieval_result,
            recommendation_result=recommendation_result,
            generation_result=generation_result,
            fallback_result=fallback_result,
        ),
    )

    return {
        "dog_knowledge_pipeline_result": pipeline_result.model_dump(mode="python"),
    }


def _resolve_query_type(
    is_fallback: bool,
    query_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    recommendation_result: dict[str, Any],
) -> str:
    """
    解析聚合后的问题类型。

    参数：
        is_fallback:
            是否走兜底流程。

        query_result:
            查询理解层输出。

        retrieval_result:
            检索层输出。

        recommendation_result:
            推荐层输出。

    返回值：
        str:
            聚合后的标准问题类型。
    """

    if is_fallback:
        return "fallback"

    query_type = _first_non_empty_str(
        query_result.get("query_type"),
        retrieval_result.get("query_type"),
    )

    if query_type:
        return query_type

    if recommendation_result.get("recommended_breeds"):
        return "recommendation"

    return "general_qa"


def _resolve_answer(
    is_fallback: bool,
    fallback_result: dict[str, Any],
    generation_result: dict[str, Any],
    state_data: dict[str, Any],
) -> str:
    """
    解析聚合后的自然语言答案。

    参数：
        is_fallback:
            是否走兜底流程。

        fallback_result:
            兜底层输出。

        generation_result:
            生成层输出。

        state_data:
            当前完整 state。

    返回值：
        str:
            聚合后的自然语言答案。
    """

    if is_fallback:
        return _first_non_empty_str(
            fallback_result.get("generated_answer"),
            "我暂时无法基于当前犬种知识库可靠回答这个问题。",
        )

    return _first_non_empty_str(
        generation_result.get("generated_answer"),
        state_data.get("final_answer"),
        state_data.get("answer"),
        "当前没有生成可用的狗狗知识答案。",
    )


def _resolve_status(
    is_fallback: bool,
    answer: str,
    recommended_breeds: list[Any],
    evidences: list[Any],
) -> str:
    """
    解析聚合后的答案状态。

    参数：
        is_fallback:
            是否走兜底流程。

        answer:
            聚合后的自然语言答案。

        recommended_breeds:
            推荐犬种列表。

        evidences:
            证据列表。

    返回值：
        str:
            聚合后的答案状态。
    """

    if is_fallback:
        return "fallback"

    if answer:
        return "success"

    if recommended_breeds or evidences:
        return "partial"

    return "empty"


def _resolve_confidence(
    is_fallback: bool,
    fallback_result: dict[str, Any],
    generation_result: dict[str, Any],
    recommendation_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    query_result: dict[str, Any],
) -> float:
    """
    解析聚合后的整体置信度。

    参数：
        is_fallback:
            是否走兜底流程。

        fallback_result:
            兜底层输出。

        generation_result:
            生成层输出。

        recommendation_result:
            推荐层输出。

        retrieval_result:
            检索层输出。

        query_result:
            查询理解层输出。

    返回值：
        float:
            0 到 1 之间的整体置信度。
    """

    if is_fallback:
        return _normalize_confidence(
            fallback_result.get("confidence"),
            default=0.1,
        )

    for value in [
        generation_result.get("confidence"),
        recommendation_result.get("confidence"),
        retrieval_result.get("confidence"),
        query_result.get("confidence"),
    ]:
        confidence = _normalize_confidence(value)

        if confidence is not None:
            return confidence

    return 0.0


def _resolve_reason(
    is_fallback: bool,
    fallback_result: dict[str, Any],
    generation_result: dict[str, Any],
    recommendation_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    query_result: dict[str, Any],
) -> str | None:
    """
    解析聚合层原因说明。

    参数：
        is_fallback:
            是否走兜底流程。

        fallback_result:
            兜底层输出。

        generation_result:
            生成层输出。

        recommendation_result:
            推荐层输出。

        retrieval_result:
            检索层输出。

        query_result:
            查询理解层输出。

    返回值：
        str | None:
            聚合后的原因说明。
    """

    if is_fallback:
        return _first_non_empty_str(
            fallback_result.get("reason"),
            fallback_result.get("fallback_reason"),
        )

    return _first_non_empty_str(
        generation_result.get("reason"),
        recommendation_result.get("reason"),
        retrieval_result.get("reason"),
        query_result.get("reason"),
    )


def _build_metadata(
    query_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    recommendation_result: dict[str, Any],
    generation_result: dict[str, Any],
    fallback_result: dict[str, Any],
) -> dict[str, Any]:
    """
    构建聚合层 metadata。

    参数：
        query_result:
            查询理解层输出。

        retrieval_result:
            检索层输出。

        recommendation_result:
            推荐层输出。

        generation_result:
            生成层输出。

        fallback_result:
            兜底层输出。

    返回值：
        dict[str, Any]:
            聚合后的 metadata。
    """

    return {
        "layers": {
            "query": _to_dict(query_result.get("metadata")),
            "retrieval": _to_dict(retrieval_result.get("metadata")),
            "recommendation": _to_dict(recommendation_result.get("metadata")),
            "generation": _to_dict(generation_result.get("metadata")),
            "fallback": _to_dict(fallback_result.get("metadata")),
        }
    }


def _build_debug(
    query_result: dict[str, Any],
    retrieval_result: dict[str, Any],
    recommendation_result: dict[str, Any],
    generation_result: dict[str, Any],
    fallback_result: dict[str, Any],
) -> dict[str, Any]:
    """
    构建聚合层 debug 信息。

    参数：
        query_result:
            查询理解层输出。

        retrieval_result:
            检索层输出。

        recommendation_result:
            推荐层输出。

        generation_result:
            生成层输出。

        fallback_result:
            兜底层输出。

    返回值：
        dict[str, Any]:
            聚合层调试信息。
    """

    return {
        "aggregator": {
            "name": "aggregate_dog_knowledge_layer_outputs_node",
            "version": "v1.7.4",
            "available_layers": [
                name
                for name, value in {
                    "query": query_result,
                    "retrieval": retrieval_result,
                    "recommendation": recommendation_result,
                    "generation": generation_result,
                    "fallback": fallback_result,
                }.items()
                if value
            ],
        }
    }


def _normalize_confidence(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    将置信度归一化到 0 到 1。

    参数：
        value:
            原始置信度。

        default:
            无法转换时的默认值。

    返回值：
        float | None:
            0 到 1 之间的置信度；无法转换且无默认值时返回 None。
    """

    if value is None:
        return default

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default

    if confidence < 0.0:
        return 0.0

    if confidence > 1.0:
        return 1.0

    return confidence


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


aggregate_dog_knowledge_layer_outputs_node = (
    build_aggregate_dog_knowledge_layer_outputs_node()
)
