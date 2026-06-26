from typing import Any

import  json

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


VALID_WORKERS = {
    "retrieve",
    "evaluate",
    "retry",
    "rerank",
    "generate",
    "finish",
}


def build_exact_search_supervisor_node(
        llm_provider=None,
):
    """
    构建 exact_search_agent 的 Supervisor 节点。

    功能：
        使用闭包方式注入 LLMProvider，避免 Supervisor 节点内部直接 import container。
        这是 v1.5 Graph Redesign 中对旧 exact_search_agent 的兼容迁移改造。

    技术名词：
        Supervisor：
            监督者节点，负责根据当前 state 判断下一步应该执行哪个 worker。

        Provider：
            提供者，负责统一创建和管理服务对象。
            这里的 llm_provider 负责提供 LLM 调用能力。

        Closure：
            闭包，内部函数可以访问外部函数传入的变量。
            这里 supervisor_node 可以访问 build_exact_search_supervisor_node 传入的 llm_provider。

        Dependency Injection：
            依赖注入，表示依赖对象从外部传入，而不是在函数内部直接创建或获取。

    参数：
        llm_provider:
            LLMProvider 实例。
            中文释义：用于获取 backup_llm，并调用 safe_ainvoke 完成 Supervisor 决策。

    返回值：
        callable:
            返回一个 async supervisor_node 函数，供 LangGraph 注册使用。
    """

    async def supervisor_node(
            state: DogState,
    ) -> dict[str, Any]:
        """
        exact_search_agent 的 Supervisor 节点函数。

        功能：
            根据当前 DogState 判断 exact_search_agent 下一步应该执行哪个 worker。

            当前 v1.5 改造点：
            1. 使用外部注入的 llm_provider。
            2. 不再直接 import container。
            3. 不再把 Supervisor 决策写入 messages。
            4. 优先读取新版 rag_context 状态。
            5. 如果已经有 answer / final_answer，直接 finish，减少一次 LLM 调用。
            6. 继续返回 next_worker，兼容 route_exact_worker。

        参数：
            state:
                当前 DogState。
                中文释义：LangGraph 子图中共享的状态数据。

        返回值：
            dict[str, Any]:
                返回需要合并进 state 的字段。
                当前主要返回：
                - next_worker
        """

        if llm_provider is None:
            raise RuntimeError(
                "build_exact_search_supervisor_node 缺少 llm_provider，"
                "请确认 build_exact_search_agent 已传入 llm_provider。"
            )

        return await execute_exact_search_supervisor(
            state=state,
            llm_provider=llm_provider,
        )

    return supervisor_node


async def exact_search_supervisor_node(
        state: DogState,
) -> dict[str, Any]:
    """
    旧版兼容 Supervisor 节点。

    功能：
        保留旧函数名，避免旧代码仍然直接导入 exact_search_supervisor_node 时立刻报错。

        注意：
            v1.5 新代码推荐使用 build_exact_search_supervisor_node 注入 llm_provider。
            这个函数只是兼容入口，不建议新代码继续依赖它。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            返回 next_worker。
    """

    from src.runtime.container.init import container

    llm_provider = container.get(
        "llm"
    )

    return await execute_exact_search_supervisor(
        state=state,
        llm_provider=llm_provider,
    )


async def execute_exact_search_supervisor(
        state: DogState,
        llm_provider,
) -> dict[str, Any]:
    """
    执行 exact_search_agent Supervisor 的核心决策逻辑。

    功能：
        这是 Supervisor 的核心实现。
        之所以单独抽出来，是为了同时支持：
        1. build_exact_search_supervisor_node 的 Provider 注入方式。
        2. exact_search_supervisor_node 的旧版兼容方式。

    执行流程：
        1. 设置 runtime 当前节点。
        2. 写入 timeline。
        3. 构建 state summary。
        4. 如果已经有 answer / final_answer，直接 finish。
        5. 否则调用 backup_llm 判断下一步 worker。
        6. 清洗 LLM 决策结果。
        7. 返回 next_worker。

    参数：
        state:
            当前 DogState。

        llm_provider:
            LLMProvider 实例。

    返回值：
        dict[str, Any]:
            返回 next_worker。
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "exact_search_supervisor_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="exact_search_supervisor_node"
    )

    logger.info(
        "进入 exact_search supervisor"
    )

    summary = build_exact_state_summary(
        state=state
    )

    logger.debug(
        f"exact_search supervisor state summary: {json.dumps(summary, indent=4, ensure_ascii=False)}"
    )

    if should_finish_without_llm(
            summary=summary
    ):
        logger.info(
            "exact_search supervisor 检测到已有 answer/final_answer，直接 finish"
        )

        return {
            "next_worker": "finish"
        }

    # todo:加了规则短路  这种偏固定流程的尽量使用规则路由 但是我现在属于学习阶段 所以用LLM来做一下supervisor
    rule_decision = decide_next_worker_by_rule(
        summary=summary
    )

    if rule_decision:
        logger.info(
            f"exact_search supervisor 规则决策: {rule_decision}"
        )

        return {
            "next_worker": rule_decision
        }

    backup_llm = llm_provider.backup_llm

    prompt = build_supervisor_prompt()

    response = await llm_provider.safe_ainvoke(
        llm=backup_llm,
        prompt=prompt.format(
            summary=summary
        ),
        fallback_response="finish"
    )

    logger.debug(
        f"exact_search supervisor LLM 原始决策 response={response}"
    )

    decision = normalize_worker_decision(
        raw_decision=response
    )

    logger.info(
        f"Supervisor决策: {decision}"
    )

    return {
        "next_worker": decision
    }


def normalize_worker_decision(
        raw_decision: Any,
) -> str:
    """
    归一化 Supervisor 决策结果。

    功能：
        将 LLM 返回的原始决策文本清洗成合法 worker 名称。
        如果 LLM 返回非法结果，则兜底为 finish，避免子图死循环或路由报错。

    技术名词：
        Normalize：
            归一化，把不稳定格式转换成稳定格式。

        Worker：
            工作者节点，例如 filter、retrieve、evaluate、retry、generate、finish。

        Fallback：
            兜底策略。当 LLM 输出不合法时，使用安全默认值。

    参数：
        raw_decision:
            LLM 返回的原始结果。
            可能是字符串，也可能是 AIMessage 等带 content 字段的对象。

    返回值：
        str:
            合法 worker 名称。
    """

    if hasattr(
            raw_decision,
            "content"
    ):
        decision = str(
            raw_decision.content
        ).strip()
    else:
        decision = str(
            raw_decision
        ).strip()

    decision = decision.lower()

    decision = decision.replace(
        "`",
        ""
    ).replace(
        "\"",
        ""
    ).replace(
        "'",
        ""
    ).strip()

    if decision not in VALID_WORKERS:
        logger.warning(
            f"exact_search_supervisor_node 收到非法决策: {decision}，兜底 finish"
        )

        return "retrieve"

    return decision

def detect_rerank_done_from_chunks(
        chunks: list[Any],
) -> bool:
    """
    判断当前 chunks 是否已经经过 rerank。

    功能：
        通过检查 chunk 中是否存在 rerank_score 或 normalized_rerank_score，
        判断 rerank_node 是否已经执行过。

    设计说明：
        当前不额外新增 DogState 字段，而是从 rag_context.chunks 中推断。
        这样改动更小，也更兼容已有 checkpoint。

    技术名词：
        Rerank Done：
            重排完成标记。
            表示当前 RAG chunks 已经被 reranker 二次排序。

        normalized_rerank_score：
            归一化后的重排分数，通常为 0 到 1。

    参数：
        chunks:
            rag_context 中的 chunks 列表。

    返回值：
        bool:
            True 表示已经 rerank。
            False 表示还没有 rerank。
    """

    for chunk in chunks:

        if not isinstance(
                chunk,
                dict
        ):
            continue

        if chunk.get(
                "normalized_rerank_score"
        ) is not None:
            return True

        if chunk.get(
                "rerank_score"
        ) is not None:
            return True

    return False


def build_exact_state_summary(
        state: DogState,
) -> dict[str, Any]:
    """
    构建 exact_search_supervisor_node 的状态摘要。

    功能：
        从复杂 DogState 中提取 Supervisor 决策真正需要看的关键信息。
        v1.5 优先读取新版 rag_context，同时保留旧版 docs / answer 兼容。

    技术名词：
        State Summary：
            状态摘要，把复杂 state 简化成适合 LLM 决策的小字典。

        RagContext：
            新版 RAG 上下文对象，包含 context_text、chunks、source_count、status 等字段。

        Backward Compatibility：
            向后兼容，表示新版逻辑仍然支持旧版 docs / answer 字段。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            Supervisor 使用的状态摘要。
    """

    rag_context = state.get(
        "rag_context"
    )

    rag_status = "missing"
    rag_chunks_count = 0
    rag_source_count = 0
    has_rag_context_text = False

    has_reranked = False

    if isinstance(
            rag_context,
            dict
    ):
        rag_status = str(
            rag_context.get(
                "status",
                "empty"
            )
            or "empty"
        )

        chunks = rag_context.get(
            "chunks",
            []
        )

        if isinstance(
                chunks,
                list
        ):
            rag_chunks_count = len(
                chunks
            )

            has_reranked = detect_rerank_done_from_chunks(
                chunks=chunks
            )

        rag_source_count = int(
            rag_context.get(
                "source_count",
                0
            )
            or 0
        )

        has_rag_context_text = bool(
            str(
                rag_context.get(
                    "context_text",
                    ""
                )
                or ""
            ).strip()
        )

    docs = state.get(
        "docs",
        []
    )

    docs_count = 0

    if isinstance(
            docs,
            list
    ):
        docs_count = len(
            docs
        )

    answer = str(
        state.get(
            "answer",
            ""
        )
        or ""
    ).strip()

    final_answer = str(
        state.get(
            "final_answer",
            ""
        )
        or ""
    ).strip()

    retrieval_evaluated = bool(
        state.get(
            "retrieval_evaluated",
            False
        )
    )

    summary = {
        "question": state.get(
            "question"
        ),
        "intent": state.get(
            "intent"
        ),
        "filters": state.get(
            "filters"
        ),
        "dog_name": state.get(
            "dog_name"
        ),
        "rag_status": rag_status,
        "rag_chunks_count": rag_chunks_count,
        "rag_source_count": rag_source_count,
        "has_rag_context_text": has_rag_context_text,
        "docs_count": docs_count,
        "retrieval_ok": state.get(
            "retrieval_ok"
        ),
        "retrieval_evaluated": retrieval_evaluated,
        "retrieval_failure_type": state.get(
            "retrieval_failure_type"
        ),
        "retry_count": state.get(
            "retry_count",
            0
        ),
        "has_answer": bool(
            answer
        ),
        "has_final_answer": bool(
            final_answer
        ),
        "next_worker": state.get(
            "next_worker"
        ),
        "has_reranked": has_reranked
    }

    return summary


def should_finish_without_llm(
        summary: dict[str, Any],
) -> bool:
    """
    判断是否可以不调用 LLM，直接 finish。

    功能：
        如果当前 state 已经存在 answer 或 final_answer，
        说明 generate_node 已经生成过最终答案。
        这时 Supervisor 不需要再调用 LLM 判断下一步，
        可以直接返回 finish。

    技术名词：
        Short-circuit：
            短路逻辑。满足明确条件时跳过后续复杂判断。

        Finish：
            exact_search_agent 子图结束标记。

    参数：
        summary:
            build_exact_state_summary 构建出来的状态摘要。

    返回值：
        bool:
            True 表示可以直接 finish。
            False 表示还需要继续决策。
    """

    return bool(
        summary.get(
            "has_answer"
        )
        or summary.get(
            "has_final_answer"
        )
    )


def build_supervisor_prompt() -> str:
    """
    构建 exact_search_supervisor_node 使用的 Prompt 模板。

    功能：
        让 LLM 根据当前状态摘要，选择下一步 worker。

    返回值：
        str:
            Supervisor Prompt 模板字符串。
    """

    return """
你是 Dog Agent Framework 中 exact_search_agent 的 Supervisor。

你的任务是根据当前状态摘要，选择下一个 worker。

只能返回以下一个词：
retrieve
evaluate
retry
rerank
generate
finish

# worker 含义

retrieve:
用于执行 RAG 检索，召回相关文档。

evaluate:
用于判断检索结果质量是否足够回答问题。

retry:
用于在检索质量不足时，根据 failure_type 调整下一轮检索策略。

rerank:
用于对已召回的 RAG chunks 进行二次排序，让最相关的上下文排在前面。

generate:
用于根据检索上下文生成最终答案。

finish:
用于结束 exact_search_agent。

# 决策规则

1. 如果 has_answer 或 has_final_answer 为 true，返回 finish。

2. 如果 rag_status 是 missing 且 docs_count 为 0，返回 retrieve。

3. 如果 rag_status 是 success，rag_chunks_count > 0，且 retrieval_evaluated 为 false，返回 evaluate。

4. 如果 retrieval_evaluated 为 true，retrieval_ok 为 true，且 has_reranked 为 false，返回 rerank。

5. 如果 retrieval_evaluated 为 true，retrieval_ok 为 true，且 has_reranked 为 true，并且还没有 answer / final_answer，返回 generate。

6. 如果 retrieval_evaluated 为 true，retrieval_ok 为 false，且 retry_count 小于 3，返回 retry。

7. 如果 retrieval_evaluated 为 true，retrieval_ok 为 false，且 retry_count 大于等于 3，返回 generate。

8. 如果 rag_status 是 empty 且 retry_count 小于 3，返回 retry。

9. 其他情况返回 finish。

# 当前状态摘要

{summary}

请只返回一个 worker 名称，不要解释。
"""

def decide_next_worker_by_rule(
        summary: dict[str, Any],
) -> str | None:
    """
    使用规则判断 exact_search_agent 下一步 worker。

    功能：
        对确定性强的状态，直接返回 worker。
        如果规则无法判断，再返回 None，让 LLM supervisor 兜底。

        v1.5 当前设计：
        1. 不再返回 filter。
        2. 如果没有 rag_context，直接 retrieve。
        3. 如果 retrieve 成功但未 evaluate，进入 evaluate。
        4. 如果 evaluate 成功但还没 rerank，进入 rerank。
        5. 如果已经 rerank，进入 generate。
        6. 如果已经生成 answer / final_answer，进入 finish。

    参数：
        summary:
            当前状态摘要。

    返回值:
        str | None:
            worker 名称。
            None 表示交给 LLM 判断。
    """

    if (
            summary.get("has_answer")
            or summary.get("has_final_answer")
    ):
        return "finish"

    rag_status = summary.get(
        "rag_status"
    )

    rag_chunks_count = int(
        summary.get("rag_chunks_count")
        or 0
    )

    docs_count = int(
        summary.get("docs_count")
        or 0
    )

    retrieval_evaluated = bool(
        summary.get("retrieval_evaluated")
    )

    retrieval_ok = bool(
        summary.get("retrieval_ok")
    )

    has_reranked = bool(
        summary.get("has_reranked")
    )

    retry_count = int(
        summary.get("retry_count")
        or 0
    )

    if (
            rag_status == "missing"
            and docs_count == 0
    ):
        return "retrieve"

    if (
            rag_status == "success"
            and rag_chunks_count > 0
            and not retrieval_evaluated
    ):
        return "evaluate"

    if (
            retrieval_evaluated
            and retrieval_ok
            and not has_reranked
    ):
        return "rerank"

    if (
            retrieval_evaluated
            and retrieval_ok
            and has_reranked
    ):
        return "generate"

    if (
            retrieval_evaluated
            and not retrieval_ok
            and retry_count < 3
    ):
        return "retry"

    if (
            retrieval_evaluated
            and not retrieval_ok
            and retry_count >= 3
    ):
        return "generate"

    if (
            rag_status == "empty"
            and retry_count < 3
    ):
        return "retry"

    if (
            rag_status == "empty"
            and retry_count >= 3
    ):
        return "generate"

    return None