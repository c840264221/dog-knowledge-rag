"""人工运行查询改写对照：真实查询理解一次，固定过滤与重排问题，只读知识库。"""

import asyncio
from copy import deepcopy
import json
from typing import Any
from uuid import uuid4

from scripts.rag.compare_candidate_counts import FILTERS, summarize_candidates
from src.agents.dog_knowledge_agent.nodes.query_understanding_node import (
    build_dog_query_understanding_node,
)
from src.graph.nodes.rerank_node import rerank_rag_context_chunks
from src.rag.schemas import RagQuery
from src.runtime.context import RuntimeContext, runtime_ctx


QUESTION = "成都现在这个天气，带金毛出去运动需要注意什么？"
USER_ID = "rag_rewrite_comparison"


def compare_retrieval_texts(
    retriever: Any, reranker: Any, rewritten_question: str,
) -> dict[str, Any]:
    """只改变向量检索文本，对比召回及后续重排排名。

    参数：retriever 为项目检索器；reranker 为同一重排模型；
        rewritten_question 为人工确认过的真实模型改写文本。
    返回：两组候选、完整重排排名和最终五条；不生成答案或修改索引。
    """
    groups = []
    for name, text in (("original", QUESTION), ("rewritten", rewritten_question)):
        query = RagQuery(
            question=text, user_id=USER_ID, filters=deepcopy(FILTERS),
            top_k=10, intent="exact_info",
        )
        context = retriever.retrieve(query=query).model_dump(mode="python")
        candidates = context["chunks"]
        if not candidates or any(
            item["chunk"]["metadata"].get("dog_name") != "Golden Retriever"
            for item in candidates
        ):
            raise ValueError(f"{name} 组候选为空或犬种不符，停止对照")
        retrieval_summary = summarize_candidates(candidates)
        # 两组重排问题相同，避免同时改变检索文本与重排文本。
        reranked = rerank_rag_context_chunks(
            question=QUESTION, rag_context=deepcopy(context),
            chunks=deepcopy(candidates), reranker_model=reranker,
            top_k=len(candidates),
        )["rag_context"]["chunks"]
        selected = reranked[:5]
        groups.append({
            "group": name,
            "retrieval_question": text,
            "rerank_question": QUESTION,
            "actual_candidate_count": len(candidates),
            "retrieval": retrieval_summary,
            "reranked": summarize_candidates(reranked),
            "selected": summarize_candidates(selected),
            "exercise_in_final_top_k": bool(
                summarize_candidates(selected)["exercise_ranks"]
            ),
        })
    return {
        "original_question": QUESTION,
        "rewritten_question": rewritten_question,
        "fixed_filters": FILTERS,
        "requested_candidate_count": 10,
        "final_top_k": 5,
        "groups": groups,
        "limitations": [
            "模型建议的过滤条件仅记录，不应用；两组强制使用同一犬种过滤。",
            "只改变检索文本；重排统一使用原问题。这是隔离实验，不是完整主图测试。",
            "运动章节命中仅为本样本定位指标，不代表全部问题得到覆盖。",
            "不调用天气服务、不生成答案；不以排名代替改写忠实性检查。",
        ],
    }


async def main() -> None:
    """复用真实查询理解节点，人工核对改写后进行检索对照。

    参数：无，读取项目服务配置；用户需在终端确认文本忠实性。
    返回：无，打印模型分析和对照报告，结束时释放运行资源。
    """
    from src.runtime.container.init import container
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.settings import settings

    context = RuntimeContext(
        trace_id=uuid4().hex, user_id=USER_ID,
        session_id=f"rewrite-{uuid4().hex}", component="compare_query_rewrite",
    )
    previous_context = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        await context.startup()
        retrieval_provider = container.get("retriever")
        node = build_dog_query_understanding_node(
            llm_provider=container.get("llm"),
            query_filter_parser=retrieval_provider.dog_query_filter_parser,
        )
        print("即将调用真实查询理解模型；可能产生费用，不调用回答模型。", flush=True)
        update = await node({
            "question": QUESTION, "user_id": USER_ID,
            "top_k": 10, "intent": "exact_info",
        })
        audit = update["dog_query_understanding_result"]
        rewritten = update["retrieval_question"]
        print(json.dumps({
            "original_question": QUESTION,
            "actual_retrieval_question": rewritten,
            "query_understanding": audit,
        }, ensure_ascii=False, indent=2), flush=True)
        if rewritten == QUESTION:
            print("检索文本未变化，可能未改写或已降级；本次不重复检索，不算改写优化成功。")
            return
        print("请检查是否保留金毛、运动注意事项，是否编造天气；过滤建议不参与本次对照。")
        approval = await asyncio.to_thread(input, "确认改写忠实后输入 继续，其他输入退出：")
        if approval.strip() != "继续":
            print("未确认改写，本次不进行检索对照。")
            return

        model_name = "BAAI/bge-reranker-v2-m3"
        container.register("reranker_rewrite_comparison", RerankerProvider(
            reranker_settings=settings.reranker.model_copy(update={"model_name": model_name})
        ))
        print(f"正在加载 {model_name}，随后进行两组检索。", flush=True)
        model = await asyncio.to_thread(
            lambda: container.get("reranker_rewrite_comparison").reranker
        )
        report = await asyncio.to_thread(
            compare_retrieval_texts,
            retrieval_provider.metadata_filter_retriever, model, rewritten,
        )
        report["query_understanding"] = audit
        report["configured_reranker"] = model_name
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous_context)


if __name__ == "__main__":
    asyncio.run(main())
