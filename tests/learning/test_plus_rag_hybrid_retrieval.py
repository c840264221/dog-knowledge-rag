"""Plus RAG：先用固定候选演示排名融合，不执行真实检索或模型调用。"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

from src.rag.schemas import RagChunk


def fuse_query_retrieval_results(
    *,
    query_id: str,
    results_by_channel: dict[str, list[RagChunk]],
    channel_weights: dict[str, float],
    rank_constant: float = 60.0,
) -> list[dict[str, Any]]:
    """
    功能：对单个子问题的多路有序候选进行加权倒数排名融合。
    参数含义：
        query_id：子问题编号，用于追踪结果归属。
        results_by_channel：通道名称到片段列表的映射，各列表已按相关性降序。
        channel_weights：各通道的非负权重，至少一路权重大于零。
        rank_constant：非负平滑常数，不是召回数量；默认值仅作演示起点。
    返回值含义：
        返回按 fusion_score 降序排列的记录列表。每条保留 RagChunk、子问题
        编号、各路排名与贡献；同一通道重复编号只计一次，不截取最终上下文。
        分数相同时按片段编号排序，保证演示输出稳定。
    """
    if not query_id.strip() or not results_by_channel:
        raise ValueError("必须提供子问题编号和检索通道")
    if not math.isfinite(rank_constant) or rank_constant < 0:
        raise ValueError("平滑常数必须是有限的非负数")
    if set(results_by_channel) != set(channel_weights):
        raise ValueError("通道结果与权重必须使用相同的通道名称")
    weights = list(channel_weights.values())
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("权重必须是有限的非负数")
    if not any(value > 0 for value in weights):
        raise ValueError("至少一路权重必须大于零")

    records_by_chunk_id: dict[str, dict[str, Any]] = {}
    for channel, chunks in results_by_channel.items():
        seen_in_channel: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen_in_channel:
                continue
            seen_in_channel.add(chunk.chunk_id)
            rank = len(seen_in_channel)

            record = records_by_chunk_id.setdefault(
                chunk.chunk_id,
                {
                    "query_id": query_id,
                    "chunk": chunk,
                    "fusion_score": 0.0,
                    "channel_ranks": {},
                    "channel_contributions": {},
                },
            )
            if record["chunk"] != chunk:
                raise ValueError("同一片段编号对应了不同内容或元数据，应先核对索引版本")

            contribution = channel_weights[channel] / (rank_constant + rank)
            record["fusion_score"] += contribution
            record["channel_ranks"][channel] = rank
            record["channel_contributions"][channel] = contribution

    return sorted(
        records_by_chunk_id.values(),
        key=lambda record: (-record["fusion_score"], record["chunk"].chunk_id),
    )


def run_rank_fusion_demo() -> dict[str, Any]:
    """
    功能：用 A、B、C、D 固定候选演示多通道融合与子问题隔离。
    参数含义：无参数，不创建提供者、不访问向量库、不调用模型。
    返回值含义：返回两项子问题的原始通道编号、预期顺序、实际融合记录
        和顺序比较结果。预期结果只是检查计算，不代表真实业务检索效果。
    """
    chunks = {
        label: RagChunk(
            chunk_id=label,
            doc_id="fusion-demo-document",
            chunk_index=index,
            content=f"固定候选片段{label}，仅用于观察融合计算。",
            source="rank_fusion_demo",
        )
        for index, label in enumerate(["A", "B", "C", "D"])
    }
    cases = [
        ("q1", {"dense": ["A", "B", "B", "C"],
                "sparse": ["B", "D", "A"]}, ["B", "A", "D", "C"]),
        ("q2", {"dense": ["D", "A"],
                "sparse": ["A", "C"]}, ["A", "D", "C"]),
    ]
    results = []
    for query_id, channel_ids, expected_order in cases:
        fused = fuse_query_retrieval_results(
            query_id=query_id,
            results_by_channel={
                channel: [chunks[chunk_id] for chunk_id in ids]
                for channel, ids in channel_ids.items()
            },
            channel_weights={"dense": 1.0, "sparse": 1.0},
            rank_constant=1.0,
        )
        actual_order = [record["chunk"].chunk_id for record in fused]
        results.append({
            "query_id": query_id,
            "input_channel_ids": channel_ids,
            "expected_order": expected_order,
            "actual_order": actual_order,
            "order_matches_expected": actual_order == expected_order,
            "fused_records": [
                {**record, "chunk": record["chunk"].model_dump(mode="json")}
                for record in fused
            ],
        })
    return {"mode": "fixed_candidates_rank_fusion", "results": results}


def select_context_candidates(
    *,
    results_by_query: dict[str, list[RagChunk]],
    annotations: dict[str, dict[str, list[str]]],
    maximum_chunks: int = 3,
    maximum_characters: int = 900,
) -> dict[str, Any]:
    """
    功能：演示先覆盖目标、再跳过冗余补选，并记录每次选择的原因。
    参数含义：
        results_by_query：各子问题已排序的较大候选池，不是最终前三条。
        annotations：人工标注的 supported_query_ids（支持目标）和
            information_units（信息点），按片段编号索引。不是自动语义判断。
        maximum_chunks：最终片段数量上限。
        maximum_characters：最终拼接正文的字符上限，不是模型词元预算。
    返回值含义：
        返回选中片段、上下文文本、支持目标、未覆盖目标与选择事件。
        覆盖仅表示有人工标注的支持片段，不证明真实回答所需依据已经充分。
        这是顺序贪心选择演示，不保证在有限预算中得到最优组合。
    """
    if maximum_chunks < 1 or maximum_characters < 1:
        raise ValueError("片段数量和字符预算必须大于零")
    if any(
        chunk.chunk_id not in annotations
        for chunks in results_by_query.values()
        for chunk in chunks
    ):
        raise ValueError("演示候选缺少人工标注，不能猜测是否适用或冗余")

    selected: dict[str, dict[str, Any]] = {}
    covered_query_ids: set[str] = set()
    selected_information_units: set[str] = set()
    context_blocks: list[str] = []
    selection_events = []
    coverage_queue = [
        ("coverage", query_id, chunk)
        for query_id, chunks in results_by_query.items()
        for chunk in chunks
    ]
    # 补充阶段按各列表的名次轮流遍历，不直接比较不同子问题的原始评分。
    longest_list = max((len(chunks) for chunks in results_by_query.values()), default=0)
    supplement_queue = [
        ("supplement", query_id, chunks[index])
        for index in range(longest_list)
        for query_id, chunks in results_by_query.items()
        if index < len(chunks)
    ]

    for phase, query_id, chunk in [*coverage_queue, *supplement_queue]:
        if phase == "coverage" and query_id in covered_query_ids:
            continue
        annotation = annotations[chunk.chunk_id]
        supported_ids = set(annotation["supported_query_ids"]) & set(results_by_query)
        information_units = set(annotation["information_units"])
        new_query_ids = supported_ids - covered_query_ids
        new_information_units = information_units - selected_information_units
        proposed_blocks = [*context_blocks, f"[{chunk.chunk_id}]\n{chunk.content}"]

        if query_id not in supported_ids:
            reason = "not_supported_for_query"
        elif chunk.chunk_id in selected:
            reason = "already_selected"
        elif not new_information_units and not new_query_ids:
            reason = "no_information_gain"
        elif len(selected) >= maximum_chunks:
            reason = "chunk_limit"
        elif len("\n\n".join(proposed_blocks)) > maximum_characters:
            reason = "character_budget"
        else:
            reason = "selected"
            selected[chunk.chunk_id] = {
                "chunk": chunk,
                "supported_query_ids": sorted(supported_ids),
                "information_units": sorted(information_units),
            }
            covered_query_ids.update(supported_ids)
            selected_information_units.update(information_units)
            context_blocks = proposed_blocks

        selection_events.append({
            "phase": phase,
            "query_id": query_id,
            "chunk_id": chunk.chunk_id,
            "reason": reason,
            "selected_chunk_ids_after": list(selected),
        })

    return {
        "selected_chunks": list(selected.values()),
        "selected_chunk_ids": list(selected),
        "covered_query_ids": sorted(covered_query_ids),
        "uncovered_query_ids": sorted(set(results_by_query) - covered_query_ids),
        "context_text": "\n\n".join(context_blocks),
        "used_characters": len("\n\n".join(context_blocks)),
        "selection_events": selection_events,
    }


def run_context_selection_demo() -> dict[str, Any]:
    """
    功能：演示前三条冗余时继续补选，以及预算不足时明确返回未覆盖目标。
    参数含义：无参数；候选顺序与信息标签都是人工设定，不执行真实重排。
    返回值含义：返回正常预算、仅一条名额、首选超长三种场景的完整选择记录。
        超长场景用后面的短片段补选，但不能据此断言删去的长文本信息等价。
    """
    texts = {
        "A": "教学示意：资料A描述场所限制条件。",
        "B": "教学示意：资料B重复同一项场所限制。",
        "C": "教学示意：资料C仍然重复同一项场所限制。",
        "D": "教学示意：资料D描述日常活动安排。",
        "E": "教学示意：资料E补充活动安排的调整条件。",
    }
    chunks = {
        label: RagChunk(
            chunk_id=label, doc_id="context-demo-document", chunk_index=index,
            content=text, source="context_selection_demo",
        )
        for index, (label, text) in enumerate(texts.items())
    }
    annotations = {
        label: {
            "supported_query_ids": ["q1" if label in "ABC" else "q2"],
            "information_units": [
                "place_constraint" if label in "ABC"
                else "activity_arrangement" if label == "D" else "adjustment_condition"
            ],
        }
        for label in chunks
    }
    ranked = {"q1": [chunks[label] for label in "ABCDE"],
              "q2": [chunks[label] for label in "DE"]}
    long_d = chunks["D"].model_copy(update={"content": "超长的教学原文。" * 200})
    cases = [
        ("normal_budget", ranked, 3, ["A", "D", "E"]),
        ("one_chunk_only", ranked, 1, ["A"]),
        ("long_first_candidate", {
            "q1": [chunks["A"], chunks["B"], chunks["C"], long_d, chunks["E"]],
            "q2": [long_d, chunks["E"]],
        },
         3, ["A", "E"]),
    ]
    reports = []
    for case_id, candidates, limit, expected_ids in cases:
        result = select_context_candidates(
            results_by_query=candidates, annotations=annotations,
            maximum_chunks=limit, maximum_characters=300,
        )
        reports.append({
            "case_id": case_id,
            "expected_selected_chunk_ids": expected_ids,
            "selection_matches_expected": result["selected_chunk_ids"] == expected_ids,
            **result,
            "selected_chunks": [
                {**item, "chunk": item["chunk"].model_dump(mode="json")}
                for item in result["selected_chunks"]
            ],
        })
    return {"mode": "annotated_context_selection", "results": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plus RAG 固定候选教学演示")
    parser.add_argument("--context-demo", action="store_true", help="运行上下文选择演示")
    arguments = parser.parse_args()
    report = run_context_selection_demo() if arguments.context_demo else run_rank_fusion_demo()
    print(json.dumps(report, ensure_ascii=False, indent=2))
