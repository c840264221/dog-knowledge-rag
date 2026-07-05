from collections.abc import Mapping, Sequence
from typing import Any

from src.agents.dog_knowledge_agent.contracts.layer_outputs import (
    DogRetrievalLayerOutput,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeEvidence,
)


def build_dog_knowledge_retrieval_layer_output_node():
    """
    构建 DogKnowledgeAgent 检索层输出节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点在 evaluate 判断检索结果可用后执行，
        将 rag_context 转换成 V1.7.4 标准检索层产物 dog_retrieval_result。

    参数：
        无。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回包含 dog_retrieval_result 的 state update。
    """

    def dog_knowledge_retrieval_layer_output_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 检索层输出节点。

        功能：
            从当前 DogState 中读取 rag_context、dog_query_result、
            retrieval_quality 等字段，生成 dog_retrieval_result。

        参数：
            state:
                LangGraph 当前状态。

        返回值：
            dict[str, Any]:
                LangGraph state update，包含 dog_retrieval_result。
        """

        retrieval_result = build_dog_retrieval_layer_output_from_state(state)

        return {
            "dog_retrieval_result": retrieval_result.model_dump(mode="python"),
        }

    return dog_knowledge_retrieval_layer_output_node


def build_dog_retrieval_layer_output_from_state(
    state: Mapping[str, Any] | Any,
) -> DogRetrievalLayerOutput:
    """
    从当前 state 构建检索层标准输出。

    功能：
        读取 dog_query_result、rag_context、retrieval_quality 等字段，
        归一化生成 DogRetrievalLayerOutput。

    参数：
        state:
            LangGraph state、普通 dict 或带属性对象。

    返回值：
        DogRetrievalLayerOutput:
            检索层标准中间产物。
    """

    state_data = _to_dict(state)
    query_result = _to_dict(state_data.get("dog_query_result"))
    rag_context = _to_dict(state_data.get("rag_context"))
    retrieval_quality = _to_dict(state_data.get("retrieval_quality"))
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
        query_type=_first_non_empty_str(
            query_result.get("query_type"),
            "general_qa",
        ),
        evidences=evidences,
        retrieved_count=len(evidences),
        confidence=_resolve_retrieval_confidence(
            rag_context=rag_context,
            retrieval_quality=retrieval_quality,
            evidences=evidences,
        ),
        reason=(
            "evaluate 已确认检索结果可用，"
            f"从 rag_context 生成检索层输出，evidences={len(evidences)}。"
        ),
        metadata={
            "source": "retrieval_layer_output_node",
            "rag_context_status": rag_context.get("status"),
            "source_count": rag_context.get("source_count"),
            "retrieval_ok": state_data.get("retrieval_ok"),
            "retrieval_evaluated": state_data.get("retrieval_evaluated"),
            "retrieval_failure_type": state_data.get("retrieval_failure_type"),
            "retrieval_quality": retrieval_quality,
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
        f"rag-evidence-{index + 1}",
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


def _resolve_retrieval_confidence(
    rag_context: dict[str, Any],
    retrieval_quality: dict[str, Any],
    evidences: list[DogKnowledgeEvidence],
) -> float:
    """
    解析检索层置信度。

    参数：
        rag_context:
            RAG 上下文字典。

        retrieval_quality:
            evaluate 阶段写入的检索质量诊断信息。

        evidences:
            证据列表。

    返回值：
        float:
            检索层置信度。
    """

    quality_score = _normalize_score(
        retrieval_quality.get("quality_score")
    )

    if quality_score is not None:
        return quality_score

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


dog_knowledge_retrieval_layer_output_node = (
    build_dog_knowledge_retrieval_layer_output_node()
)
