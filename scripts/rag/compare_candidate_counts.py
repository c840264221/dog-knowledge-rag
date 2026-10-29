"""固定问题比较召回 5/10 条及重排结果；只读索引，不调用生成模型。"""

from __future__ import annotations

import json
from typing import Any

from src.graph.nodes.rerank_node import rerank_rag_context_chunks
from src.rag.schemas import RagQuery


QUESTION = "金毛的日常运动需求，以及运动时的注意事项"
FILTERS = {"dog_name": {"$eq": "Golden Retriever"}}
FINAL_TOP_K = 5


def summarize_candidates(items: list[dict[str, Any]]) -> dict[str, Any]:
    """压缩候选报告，记录运动章节的位置而不输出整篇正文。

    参数：items 为按排名排列的项目检索结果字典列表。
    返回：候选明细及 Exercise 章节排名；标题命中仅作定位，不等于答案质量通过。
    """
    rows = []
    exercise_ranks = []
    for rank, item in enumerate(items, start=1):
        chunk = item["chunk"]
        metadata = chunk["metadata"]
        section_title = metadata.get("section_title", "")
        if section_title.strip().casefold() == "exercise":
            exercise_ranks.append(rank)
        rows.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "section_title": section_title,
            "heading_path": metadata.get("heading_path"),
            "retrieval_score": item.get("retrieval_score"),
            "raw_rerank_score": item.get("rerank_score"),
            "content_preview": chunk["content"][:160],
            "retrieval_reason": item.get("reason", ""),
        })
    return {"exercise_ranks": exercise_ranks, "items": rows}


def compare_candidate_counts(retriever: Any, reranker: Any) -> dict[str, Any]:
    """固定查询和过滤条件，对比两种候选数量，最终保留数量始终为五。

    参数：retriever 为项目元数据检索器；reranker 为项目重排模型。
    返回：两组召回及完整重排排名、最终保留编号和运动章节覆盖情况。
        不改配置，不写索引，不生成答案；调用失败直接抛出，不伪造评估结论。
    """
    results = []
    for candidate_count in (5, 10):
        query = RagQuery(
            question=QUESTION,
            user_id="rag_candidate_comparison",
            filters=FILTERS,
            intent="exact_info",
            top_k=candidate_count,
        )
        context = retriever.retrieve(query=query).model_dump(mode="python")
        candidates = context["chunks"]
        if any(
            item["chunk"]["metadata"].get("dog_name") != "Golden Retriever"
            for item in candidates
        ):
            raise ValueError("候选出现其他犬种，本组过滤条件未保持，请排查检索降级逻辑")

        if candidates:
            # 先保留全部重排位置供诊断，最终选择数量另行固定为五。
            reranked = rerank_rag_context_chunks(
                question=QUESTION,
                rag_context=context,
                chunks=candidates,
                reranker_model=reranker,
                top_k=len(candidates),
            )["rag_context"]["chunks"]
        else:
            reranked = []
        selected = reranked[:FINAL_TOP_K]
        selected_summary = summarize_candidates(selected)
        results.append({
            "requested_candidate_count": candidate_count,
            "actual_candidate_count": len(candidates),
            "retrieval": summarize_candidates(candidates),
            "reranked": summarize_candidates(reranked),
            "final_top_k": FINAL_TOP_K,
            "selected_chunk_ids": [item["chunk"]["chunk_id"] for item in selected],
            "exercise_in_final_top_k": bool(selected_summary["exercise_ranks"]),
        })

    first_ids = {row["chunk_id"] for row in results[0]["retrieval"]["items"]}
    second_ids = {row["chunk_id"] for row in results[1]["retrieval"]["items"]}
    return {
        "question": QUESTION,
        "filters": FILTERS,
        "candidate_counts": [5, 10],
        "final_top_k": FINAL_TOP_K,
        "five_candidates_subset_of_ten": first_ids <= second_ids,
        "groups": results,
        "limitations": [
            "这是单问题诊断，不是整体准确率评估。",
            "运动章节命中不代表已充分回答天气安全问题，仍需人工检查正文。",
            "两组使用同一服务实例；运行期间不要更新索引或模型配置。",
            "重排沿用现有实现，原始分数不是正确率，不跨组比较归一化分数。",
        ],
    }


def main() -> None:
    """按需获取项目检索和重排服务，执行固定对照并打印 JSON。

    参数：无。
    返回：无；仅输出报告，不启动整个容器，不调用天气、规划或回答模型。
        首次访问服务可能加载或下载本地向量及重排模型。
    """
    from src.runtime.container.init import container
    from src.settings import settings

    retriever = container.get("retriever").metadata_filter_retriever
    reranker = container.get("reranker").reranker
    report = compare_candidate_counts(retriever, reranker)
    report["reranker_model"] = settings.reranker.model_name
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
