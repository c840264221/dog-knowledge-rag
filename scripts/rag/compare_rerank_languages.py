"""只召回一次候选，对比中文、英文等义问题的重排结果；不写索引。"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from scripts.rag.compare_candidate_counts import summarize_candidates
from src.graph.nodes.rerank_node import rerank_rag_context_chunks
from src.rag.schemas import RagQuery


CHINESE_QUESTION = "金毛的日常运动需求，以及运动时的注意事项"
ENGLISH_QUESTION = (
    "What are a Golden Retriever's daily exercise needs, "
    "and what precautions should be taken during exercise?"
)
FILTERS = {"dog_name": {"$eq": "Golden Retriever"}}


def compare_rerank_languages(retriever: Any, reranker: Any) -> dict[str, Any]:
    """用同一批正文和同一模型，对比两种问题语言的重排位置。

    参数：retriever 为项目检索器；reranker 为现有重排模型实例。
    返回：初始候选、两组完整排名及最终前五条；不生成回答或自动认定根因。
    """
    query = RagQuery(
        question=CHINESE_QUESTION,
        filters=FILTERS,
        top_k=10,
        intent="exact_info",
        user_id="rag_rerank_language_comparison",
    )
    # 召回放在循环外，避免英文组改变候选集合。
    context = retriever.retrieve(query=query).model_dump(mode="python")
    candidates = context["chunks"]
    if not candidates:
        raise ValueError("没有候选资料，无法进行重排语言对照")
    if any(
        item["chunk"]["metadata"].get("dog_name") != "Golden Retriever"
        for item in candidates
    ):
        raise ValueError("候选包含其他犬种，请先检查过滤条件或降级逻辑")

    groups = []
    for language, question in (
        ("zh", CHINESE_QUESTION),
        ("en", ENGLISH_QUESTION),
    ):
        # 每组使用相同内容的独立副本，防止前一组结果污染后一组输入。
        group_context = deepcopy(context)
        group_context["question"] = question
        reranked = rerank_rag_context_chunks(
            question=question,
            rag_context=group_context,
            chunks=group_context["chunks"],
            reranker_model=reranker,
            top_k=len(candidates),
        )["rag_context"]["chunks"]
        selected = reranked[:5]
        groups.append({
            "query_language": language,
            "rerank_question": question,
            "input_chunk_ids": [item["chunk"]["chunk_id"] for item in candidates],
            "reranked": summarize_candidates(reranked),
            "selected_chunk_ids": [item["chunk"]["chunk_id"] for item in selected],
            "exercise_in_final_top_k": bool(
                summarize_candidates(selected)["exercise_ranks"]
            ),
        })
    return {
        "mode": "same_candidates_different_query_languages",
        "retrieval_question": CHINESE_QUESTION,
        "filters": FILTERS,
        "requested_candidate_count": 10,
        "actual_candidate_count": len(candidates),
        "retrieval": summarize_candidates(candidates),
        "final_top_k": 5,
        "groups": groups,
        "limitations": [
            "英文问题为人工准备的等义表达，不调用翻译模型。",
            "单条问题的差异也可能受具体措辞影响，不能证明全部根因。",
            "章节排名和原始分数用于对照，不代表正确率。",
        ],
    }


def main() -> None:
    """经现有容器获取检索和重排服务，打印语言对照报告。

    参数：无。
    返回：无；只调用一次检索及两次重排，不启动完整链路，不写入索引。
        首次加载服务可能触发模型初始化或下载。
    """
    from src.runtime.container.init import container
    from src.settings import settings

    retriever = container.get("retriever").metadata_filter_retriever
    reranker = container.get("reranker").reranker
    report = compare_rerank_languages(retriever, reranker)
    report["configured_reranker_model"] = settings.reranker.model_name
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
