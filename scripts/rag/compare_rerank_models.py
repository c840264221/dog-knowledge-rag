"""固定中文问题和候选，对比两个重排模型；不改索引或全局配置。"""

from copy import deepcopy
import json
from time import perf_counter
from typing import Any

from scripts.rag.compare_candidate_counts import summarize_candidates
from scripts.rag.compare_rerank_languages import CHINESE_QUESTION, FILTERS
from src.graph.nodes.rerank_node import rerank_rag_context_chunks
from src.rag.schemas import RagQuery


def compare_rerank_models(retriever: Any, providers: dict[str, Any]) -> dict[str, Any]:
    """召回一次，使用不同服务提供者在相同中文问题和正文上重排。

    参数：retriever 为项目检索器；providers 为模型名称到服务提供者的映射。
    返回：初始候选、各模型排名、加载耗时和单次重排耗时；不生成业务答案。
    """
    query = RagQuery(
        question=CHINESE_QUESTION, filters=FILTERS, top_k=10,
        intent="exact_info", user_id="rag_rerank_model_comparison",
    )
    context = retriever.retrieve(query=query).model_dump(mode="python")
    candidates = context["chunks"]
    if not candidates or any(
        item["chunk"]["metadata"].get("dog_name") != "Golden Retriever"
        for item in candidates
    ):
        raise ValueError("候选为空或犬种过滤不符，停止模型对照")

    groups = []
    for model_name, provider in providers.items():
        print(f"正在准备重排模型：{model_name}", flush=True)
        started = perf_counter()
        model = provider.reranker
        load_seconds = perf_counter() - started
        group_context = deepcopy(context)
        started = perf_counter()
        reranked = rerank_rag_context_chunks(
            question=CHINESE_QUESTION,
            rag_context=group_context,
            chunks=group_context["chunks"],
            reranker_model=model,
            top_k=len(candidates),
        )["rag_context"]["chunks"]
        rerank_seconds = perf_counter() - started
        selected = reranked[:5]
        groups.append({
            "configured_model": model_name,
            "load_seconds": round(load_seconds, 3),
            "single_rerank_seconds": round(rerank_seconds, 3),
            "input_chunk_ids": [item["chunk"]["chunk_id"] for item in candidates],
            "reranked": summarize_candidates(reranked),
            "selected_chunk_ids": [item["chunk"]["chunk_id"] for item in selected],
            "exercise_in_final_top_k": bool(
                summarize_candidates(selected)["exercise_ranks"]
            ),
        })
    return {
        "question": CHINESE_QUESTION,
        "filters": FILTERS,
        "requested_candidate_count": 10,
        "actual_candidate_count": len(candidates),
        "final_top_k": 5,
        "retrieval": summarize_candidates(candidates),
        "groups": groups,
        "limitations": [
            "单样本结果不等于整体效果，不自动替换生产配置。",
            "加载耗时可能包含下载；单次重排耗时包含首次推理开销，不是性能基准。",
            "不同模型的分数尺度不同，不比较分数绝对大小来判断模型优劣。",
        ],
    }


def main() -> None:
    """注册实验专用提供者，比较当前模型与多语言候选模型。

    参数：无。
    返回：无；打印报告，首次运行可能下载模型；不调用生成模型或写入索引。
    """
    from src.runtime.container.init import container
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.settings import settings

    candidate_name = "BAAI/bge-reranker-v2-m3"
    if settings.reranker.model_name == candidate_name:
        raise ValueError("当前模型已是候选模型，请勿用同一模型冒充两组对照")
    candidate_settings = settings.reranker.model_copy(
        update={"model_name": candidate_name}
    )
    container.register(
        "reranker_model_comparison",
        RerankerProvider(reranker_settings=candidate_settings),
    )
    report = compare_rerank_models(
        container.get("retriever").metadata_filter_retriever,
        {
            settings.reranker.model_name: container.get("reranker"),
            candidate_name: container.get("reranker_model_comparison"),
        },
    )
    report["device"] = settings.reranker.device
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
