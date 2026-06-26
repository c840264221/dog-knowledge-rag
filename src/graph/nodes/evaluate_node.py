import json
from typing import Any

from src.graph.states.state import DogState
from src.logger import logger
from src.rag.evaluators import (
    evaluate_retrieval_quality
)
from src.runtime.context import runtime_ctx


def evaluate_retrieval_node(
        state: DogState,
) -> dict[str, Any]:
    """
    评估 RAG 召回结果质量。

    功能：
        判断 retrieve_node 返回的 rag_context / docs 是否足够支持 generate_node 生成答案。

        v1.5 改造点：
        1. 不再只依赖 docs 数量。
        2. 优先评估新版 rag_context。
        3. 使用 RetrievalQualityEvaluator 进行规则化质量评分。
        4. 返回 retrieval_ok、retrieval_quality、retrieval_failure_type。
        5. 为 retry_node 提供明确失败原因。

    技术名词：
        Evaluate：
            评估。这里指判断召回结果是否可用。

        Retrieval Quality：
            召回质量。表示召回内容是否足够支持回答。

        Failure Type：
            失败类型。表示召回失败或质量不足的原因。

    参数：
        state:
            当前 LangGraph 状态。
            中文释义：包含 question、rag_context、docs、filters、dog_name 等字段。

    返回值：
        dict[str, Any]:
            返回需要合并进 DogState 的字段：
            - retrieval_ok
            - retrieval_quality
            - retrieval_failure_type
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "evaluate_retrieval_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="evaluate_retrieval_node"
    )

    logger.info(
        "进入 evaluate_retrieval_node 节点，"
        f"question={state.get('question')}, "
        f"dog_name={state.get('dog_name')}, "
        f"filters={state.get('filters')}"
    )

    quality_result = evaluate_retrieval_quality(
        state=state
    )

    logger.info(
        "RAG 召回质量评估完成，"
        f"status={quality_result.status}, "
        f"is_usable={quality_result.is_usable}, "
        f"failure_type={quality_result.failure_type}, "
        f"quality_score={quality_result.quality_score}"
    )

    logger.debug(
        f"RAG 召回质量评估详情: {json.dumps(quality_result.model_dump(),indent=4, ensure_ascii=False)}"
    )

    return {
        "retrieval_ok": quality_result.is_usable,
        "retrieval_evaluated": True,
        "retrieval_quality": quality_result.model_dump(),
        "retrieval_failure_type": quality_result.failure_type,
    }