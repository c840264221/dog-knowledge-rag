"""手动验证新重排模型到现有回答节点的链路；不改默认配置或知识库。"""

import asyncio
import json
from typing import Any
from uuid import uuid4

from scripts.rag.compare_candidate_counts import summarize_candidates
from scripts.rag.compare_rerank_languages import CHINESE_QUESTION, FILTERS
from src.graph.nodes.generate_node import build_generate_node
from src.graph.nodes.rerank_node import rerank_rag_context_chunks
from src.rag.context_builders.generation_context_builder import resolve_generation_context
from src.rag.schemas import RagQuery
from src.runtime.context import RuntimeContext, runtime_ctx


async def verify_reranked_answer(retriever: Any, reranker: Any, llm_provider: Any) -> dict[str, Any]:
    """召回十条、重排保留五条，再复用项目回答节点生成答案。

    参数：retriever 为检索器；reranker 为已加载的重排模型；
        llm_provider 为模型服务提供者。调用方须设置并启动运行上下文。
    返回：实际上下文、入选原文、回答和人工检查提示，不自动判定答案正确。
    """
    query = RagQuery(
        question=CHINESE_QUESTION, filters=FILTERS, top_k=10,
        intent="exact_info", user_id="rag_answer_verification",
    )
    context = await asyncio.to_thread(
        lambda: retriever.retrieve(query=query).model_dump(mode="python")
    )
    candidates = context["chunks"]
    if not candidates or any(
        item["chunk"]["metadata"].get("dog_name") != "Golden Retriever"
        for item in candidates
    ):
        raise ValueError("候选为空或犬种过滤不符，停止生成")

    reranked = await asyncio.to_thread(
        rerank_rag_context_chunks,
        question=CHINESE_QUESTION, rag_context=context, chunks=candidates,
        reranker_model=reranker, top_k=5,
    )
    state = {
        "question": CHINESE_QUESTION,
        "user_id": query.user_id,
        "intent": "exact_info",
        "current_agent": "dog_knowledge_agent",
        "rag_query": query.model_dump(mode="python"),
        "rag_context": reranked["rag_context"],
        "docs": reranked["docs"],
        "messages": [],
        "memory_context": "",
    }
    actual_context, context_source = resolve_generation_context(state=state)
    print("重排完成，即将调用现有回答节点生成答案（会产生模型调用费用）。", flush=True)
    generate = build_generate_node(llm_provider=llm_provider)
    update = await generate(state)
    return {
        "question": CHINESE_QUESTION,
        "retrieval": summarize_candidates(candidates),
        "selected": summarize_candidates(reranked["rag_context"]["chunks"]),
        "selected_evidence": reranked["rag_context"]["chunks"],
        "context_source": context_source,
        "actual_context": actual_context,
        "final_answer": update["final_answer"],
        "answer_strategy": update["answer_strategy"],
        "review_status": "needs_human_review",
        "review_questions": [
            "运动章节是否进入实际上下文？",
            "回答的运动需求和注意事项能否在入选原文中找到依据？",
            "是否编造运动时长、健康状态或来源？资料不足是否明确说明？",
        ],
    }


async def main() -> None:
    """使用独立配置启动本次人工验证，结束后释放资源。

    参数：无，读取项目现有检索和生成模型配置。
    返回：无，打印验证报告；只加载实验重排模型，不启动整套服务。
    """
    from src.runtime.container.init import container
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.settings import settings

    model_name = "BAAI/bge-reranker-v2-m3"
    container.register("reranker_answer_verification", RerankerProvider(
        reranker_settings=settings.reranker.model_copy(update={"model_name": model_name})
    ))
    context = RuntimeContext(
        trace_id=uuid4().hex, user_id="rag_answer_verification",
        session_id=f"rag-answer-{uuid4().hex}", component="verify_reranked_answer",
    )
    previous_context = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        await context.startup()
        print(f"正在加载实验重排模型：{model_name}", flush=True)
        model = await asyncio.to_thread(
            lambda: container.get("reranker_answer_verification").reranker
        )
        report = await verify_reranked_answer(
            container.get("retriever").metadata_filter_retriever,
            model, container.get("llm"),
        )
        report["configured_reranker"] = model_name
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
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
