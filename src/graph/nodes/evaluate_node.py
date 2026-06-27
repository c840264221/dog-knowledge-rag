import json
from typing import Any

from src.graph.states.state import DogState
from src.logger import logger
from src.rag.evaluators import (
    evaluate_retrieval_quality
)
from src.runtime.context import runtime_ctx

# 导入rag诊断工具
from src.rag.observation.diagnostics import (
    build_retrieval_diagnostics,
    merge_retrieval_diagnostics,
    build_retrieval_quality_log_summary,
)


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

    failure_type = str(
        quality_result.failure_type
        or ""
    )

    logger.info(
        "RAG 召回质量评估完成，"
        f"status={quality_result.status}, "
        f"is_usable={quality_result.is_usable}, "
        f"failure_type={failure_type}, "
        f"quality_score={quality_result.quality_score}"
    )

    quality_result_dump = quality_result.model_dump()

    logger.debug(
        "RAG 召回质量评估摘要: "
        f"status={quality_result.status}, "
        f"is_usable={quality_result.is_usable}, "
        f"failure_type={failure_type}, "
        f"quality_score={quality_result.quality_score}, "
        f"reasons_count={len(quality_result_dump.get('reasons', []))}, "
        f"metrics={quality_result_dump.get('metrics', {})}"
    )

    decision = resolve_evaluate_decision(
        state=state,
        is_usable=quality_result.is_usable,
        failure_type=failure_type,
    )

    evaluate_diagnostics = build_retrieval_diagnostics(
        state=state,
        stage="evaluate",
        docs=state.get(
            "docs",
            [],
        ) or [],
        rag_context=state.get(
            "rag_context",
            {},
        ) or {},
        failure_type=failure_type,
        decision=decision,
        reason=(
            "evaluate_retrieval_node 完成召回质量评估，"
            f"status={quality_result.status}，"
            f"is_usable={quality_result.is_usable}，"
            f"failure_type={failure_type}，"
            f"quality_score={quality_result.quality_score}，"
            f"decision={decision}。"
        ),
    )

    retrieval_quality = merge_retrieval_diagnostics(
        old_diagnostics=state.get(
            "retrieval_quality",
            {},
        ),
        new_diagnostics={
            **evaluate_diagnostics,
            "quality_evaluation": quality_result_dump,
            "quality_status": str(
                quality_result.status
                or ""
            ),
            "quality_score": quality_result.quality_score,
            "is_usable": quality_result.is_usable,
        },
    )

    logger.debug(
        "evaluate_retrieval_node 检索诊断摘要: "
        f"{json.dumps(
            build_retrieval_quality_log_summary(
                retrieval_quality=retrieval_quality,
            ),
            ensure_ascii=False,
        )}"
    )

    return {
        "retrieval_ok": quality_result.is_usable,
        "retrieval_evaluated": True,
        "retrieval_quality": retrieval_quality,
        "retrieval_failure_type": failure_type,
    }

def resolve_evaluate_decision(
        state: DogState,
        is_usable: bool,
        failure_type: str,
) -> str:
    """
    根据召回质量评估结果推断 evaluate 后的建议动作。

    功能：
        这个函数只用于写入 retrieval_quality["decision"]，
        不直接控制 LangGraph 路由。

        真正的路由仍然由 dog_knowledge_agent.routes 中的
        route_after_dog_knowledge_evaluate 决定。

        当前逻辑尽量和 dog_knowledge_agent 的 evaluate 后路由保持一致：
        1. 如果召回可用，建议进入 rerank。
        2. 如果问题模糊且还没问过用户，建议 ask_user。
        3. 如果还没超过 retry 次数，建议 retry。
        4. 否则建议 generate，让 generate_node 用有限上下文兜底回答。

    参数：
        state:
            当前 DogState。

        is_usable:
            召回结果是否可用。

        failure_type:
            召回失败类型。

    返回值：
        str:
            建议动作：
            - rerank
            - ask_user
            - retry
            - generate

    专业名词：
        Decision：
            决策。这里表示 evaluate 后建议下一步走哪个节点。

        Quality Gate：
            质量门控。根据质量评估结果决定是否继续生成或重试。
    """
    failure_type = str(
        failure_type
        or ""
    )

    if is_usable:
        return "rerank"

    has_asked_user = bool(
        state.get(
            "has_asked_user",
            False,
        )
    )

    if (
            failure_type
            in {
                "ambiguous_query",
                "need_user_clarification",
            }
            and not has_asked_user
    ):
        return "ask_user"

    retry_count = int(
        state.get(
            "retry_count",
            0,
        )
        or 0
    )

    if retry_count < 2:
        return "retry"

    return "generate"