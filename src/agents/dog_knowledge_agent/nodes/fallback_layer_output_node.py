from collections.abc import Mapping
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogFallbackLayerOutput,
)


def build_dog_knowledge_fallback_layer_output_node():
    """
    构建 DogKnowledgeAgent 兜底层输出节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点在 generation_layer_output 之后执行，
        当 state 中存在兜底迹象时生成 dog_fallback_result。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，按需返回包含 dog_fallback_result 的 state update。
    """

    def dog_knowledge_fallback_layer_output_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 兜底层输出节点。

        功能：
            判断当前 state 是否存在兜底迹象；
            如果存在，则生成 dog_fallback_result；
            如果不存在，则返回空 dict，不污染成功路径 state。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update。
                有兜底迹象时包含 dog_fallback_result，否则为空 dict。
        """

        fallback_result = build_dog_fallback_layer_output_from_state(state)

        if fallback_result is None:
            return {}

        return {
            "dog_fallback_result": fallback_result.model_dump(mode="python"),
        }

    return dog_knowledge_fallback_layer_output_node


def build_dog_fallback_layer_output_from_state(
    state: Mapping[str, Any] | Any,
) -> DogFallbackLayerOutput | None:
    """
    从当前 state 构建兜底层标准输出。

    功能：
        读取 error、fallback_reason、retrieval_ok、retrieval_failure_type、
        dog_generation_result、final_answer、answer 等字段。
        如果没有兜底迹象，则返回 None。

    参数：
        state:
            LangGraph state、普通 dict 或带属性对象。

    返回值：
        DogFallbackLayerOutput | None:
            兜底层标准中间产物；没有兜底迹象时返回 None。
    """

    state_data = _to_dict(state)
    fallback_reason = _resolve_fallback_reason(state_data)

    if not fallback_reason:
        return None

    return DogFallbackLayerOutput(
        fallback_reason=fallback_reason,
        generated_answer=_resolve_fallback_answer(state_data),
        confidence=0.1,
        reason=fallback_reason,
        metadata={
            "source": "fallback_layer_output_node",
            "retrieval_ok": state_data.get("retrieval_ok"),
            "retrieval_evaluated": state_data.get("retrieval_evaluated"),
            "retrieval_failure_type": state_data.get("retrieval_failure_type"),
            "has_error": bool(
                _first_non_empty_str(
                    state_data.get("error"),
                )
            ),
            "has_fallback_reason": bool(
                _first_non_empty_str(
                    state_data.get("fallback_reason"),
                )
            ),
        },
    )


def _resolve_fallback_reason(
    state_data: dict[str, Any],
) -> str:
    """
    解析兜底原因。

    参数：
        state_data:
            当前状态字典。

    返回值：
        str:
            兜底原因；没有兜底迹象时返回空字符串。
    """

    explicit_reason = _first_non_empty_str(
        state_data.get("error"),
        state_data.get("fallback_reason"),
    )

    if explicit_reason:
        return explicit_reason

    retrieval_evaluated = bool(
        state_data.get("retrieval_evaluated")
    )
    retrieval_ok = state_data.get("retrieval_ok")
    retrieval_failure_type = _first_non_empty_str(
        state_data.get("retrieval_failure_type"),
    )

    if (
        retrieval_evaluated
        and retrieval_ok is False
        and retrieval_failure_type
    ):
        return retrieval_failure_type

    return ""


def _resolve_fallback_answer(
    state_data: dict[str, Any],
) -> str:
    """
    解析兜底答案文本。

    参数：
        state_data:
            当前状态字典。

    返回值：
        str:
            兜底答案文本。
    """

    generation_result = _to_dict(
        state_data.get("dog_generation_result")
    )

    return _first_non_empty_str(
        generation_result.get("generated_answer"),
        state_data.get("final_answer"),
        state_data.get("answer"),
        "我暂时无法基于当前狗狗知识库可靠回答这个问题。",
    )


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


dog_knowledge_fallback_layer_output_node = (
    build_dog_knowledge_fallback_layer_output_node()
)
