from collections.abc import Mapping
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogGenerationLayerOutput,
)


def build_dog_knowledge_generation_layer_output_node():
    """
    构建 DogKnowledgeAgent 生成层输出节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点在 generate_node 之后执行，
        将 final_answer / answer 转换成 V1.7.4 标准生成层产物 dog_generation_result。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回包含 dog_generation_result 的 state update。
    """

    def dog_knowledge_generation_layer_output_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 生成层输出节点。

        功能：
            从当前 DogState 中读取 final_answer、answer、dog_retrieval_result，
            生成 dog_generation_result。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update，包含 dog_generation_result。
        """

        generation_result = build_dog_generation_layer_output_from_state(state)

        return {
            "dog_generation_result": generation_result.model_dump(mode="python"),
        }

    return dog_knowledge_generation_layer_output_node


def build_dog_generation_layer_output_from_state(
    state: Mapping[str, Any] | Any,
) -> DogGenerationLayerOutput:
    """
    从当前 state 构建生成层标准输出。

    功能：
        读取 final_answer、answer、dog_retrieval_result 等字段，
        归一化生成 DogGenerationLayerOutput。

    参数：
        state:
            LangGraph state、普通 dict 或带属性对象。

    返回值：
        DogGenerationLayerOutput:
            生成层标准中间产物。
    """

    state_data = _to_dict(state)
    retrieval_result = _to_dict(state_data.get("dog_retrieval_result"))
    answer = _first_non_empty_str(
        state_data.get("final_answer"),
        state_data.get("answer"),
        "当前没有生成可用的狗狗知识答案。",
    )

    return DogGenerationLayerOutput(
        generated_answer=answer,
        confidence=_resolve_generation_confidence(retrieval_result),
        reason="从 generate_node 输出生成生成层契约。",
        used_evidence_ids=_extract_used_evidence_ids(retrieval_result),
        metadata={
            "source": "generation_layer_output_node",
            "has_final_answer": bool(
                _first_non_empty_str(
                    state_data.get("final_answer"),
                )
            ),
            "has_answer": bool(
                _first_non_empty_str(
                    state_data.get("answer"),
                )
            ),
            "confidence_source": _resolve_confidence_source(retrieval_result),
        },
    )


def _extract_used_evidence_ids(
    retrieval_result: dict[str, Any],
) -> list[str]:
    """
    从检索层结果中提取生成答案使用的证据 ID。

    参数：
        retrieval_result:
            dog_retrieval_result 字典。

    返回值：
        list[str]:
            证据 ID 列表。
    """

    evidence_ids = []

    for evidence in _as_list(retrieval_result.get("evidences")):
        evidence_data = _to_dict(evidence)
        evidence_id = _first_non_empty_str(
            evidence_data.get("evidence_id"),
        )

        if evidence_id:
            evidence_ids.append(evidence_id)

    return _unique_strings(evidence_ids)


def _resolve_generation_confidence(
    retrieval_result: dict[str, Any],
) -> float:
    """
    解析生成层置信度。

    参数：
        retrieval_result:
            dog_retrieval_result 字典。

    返回值：
        float:
            生成层置信度。
    """

    confidence = _normalize_confidence(
        retrieval_result.get("confidence"),
    )

    if confidence is not None:
        return confidence

    return 0.0


def _resolve_confidence_source(
    retrieval_result: dict[str, Any],
) -> str:
    """
    解析生成层置信度来源。

    参数：
        retrieval_result:
            dog_retrieval_result 字典。

    返回值：
        str:
            置信度来源说明。
    """

    if _normalize_confidence(
        retrieval_result.get("confidence"),
    ) is not None:
        return "dog_retrieval_result.confidence"

    return "default"


def _normalize_confidence(
    value: Any,
) -> float | None:
    """
    将置信度归一化到 0 到 1。

    参数：
        value:
            原始置信度。

    返回值：
        float | None:
            归一化后的置信度；无法转换时返回 None。
    """

    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence < 0.0:
        return 0.0

    if confidence > 1.0:
        return 1.0

    return confidence


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

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [
        value,
    ]


def _unique_strings(
    values: list[str],
) -> list[str]:
    """
    对字符串列表去重并保持原顺序。

    参数：
        values:
            原始字符串列表。

    返回值：
        list[str]:
            去重后的字符串列表。
    """

    result = []
    seen = set()

    for value in values:
        text = str(value or "").strip()

        if not text or text in seen:
            continue

        seen.add(text)
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


dog_knowledge_generation_layer_output_node = (
    build_dog_knowledge_generation_layer_output_node()
)
