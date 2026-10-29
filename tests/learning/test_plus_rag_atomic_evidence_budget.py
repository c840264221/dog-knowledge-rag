"""完整证据单元预算规划：人工定义证据边界，不调用模型、不截断原文。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def plan_atomic_evidence_context(
    *, evidence_by_id: dict[str, dict[str, Any]],
    required_ids_by_query: dict[str, list[str]],
    optional_ranked_ids: list[str], maximum_budget: int,
    count_budget_units: Callable[[str], int], budget_unit: str,
) -> dict[str, Any]:
    """
    功能：优先完整保留外部指定的必要证据，预算不足时返回处理信号而不是截断原文。
    参数含义：evidence_by_id 保存证据正文与来源；required_ids_by_query 指定各问题
        必须同时保留的证据编号；optional_ranked_ids 是已排序的可选补充证据编号；
        maximum_budget 是留给资料的预算；count_budget_units 计算拼接文本的实际成本；
        budget_unit 说明单位，例如 characters 或目标模型的 tokens。
    返回值含义：返回状态、选中/延后证据、缺失编号、必要证据未齐的问题及成本。
        仅 ready 状态提供可交给下游的 context_text，其他状态的部分文本仅用于诊断。
        必要性和证据边界来自外部，不自动理解正文；本函数不执行分阶段模型调用。
    """
    if maximum_budget <= 0:
        raise ValueError("资料预算必须大于零")
    required_ids = list(dict.fromkeys(
        evidence_id for ids in required_ids_by_query.values() for evidence_id in ids
    ))
    selected_ids, blocks, events = [], [], []
    missing_ids, oversized_ids = [], []
    for evidence_id in required_ids:
        if evidence_id not in evidence_by_id:
            missing_ids.append(evidence_id)
            events.append({"evidence_id": evidence_id, "reason": "missing_evidence"})
            continue
        block = f"[{evidence_id}]\n{evidence_by_id[evidence_id]['text']}"
        if count_budget_units(block) > maximum_budget:
            oversized_ids.append(evidence_id)
        proposed_text = "\n\n".join([*blocks, block])
        if count_budget_units(proposed_text) <= maximum_budget:
            selected_ids.append(evidence_id)
            blocks.append(block)
            reason = "required_selected"
        else:
            reason = "required_budget_exceeded"
        events.append({"evidence_id": evidence_id, "reason": reason})

    incomplete_query_ids = [
        query_id for query_id, ids in required_ids_by_query.items()
        if not ids or not set(ids).issubset(selected_ids)
    ]
    if missing_ids or any(not ids for ids in required_ids_by_query.values()):
        status = "needs_evidence"
    elif oversized_ids:
        status = "needs_smaller_evidence"
    elif incomplete_query_ids:
        status = "needs_staged_processing"
    else:
        status = "ready"

    # 必要证据未齐时不继续填充可选资料，避免掩盖预算或证据缺口。
    if status == "ready":
        for evidence_id in dict.fromkeys(optional_ranked_ids):
            if evidence_id in selected_ids:
                continue
            if evidence_id not in evidence_by_id:
                events.append({"evidence_id": evidence_id, "reason": "optional_missing"})
                continue
            block = f"[{evidence_id}]\n{evidence_by_id[evidence_id]['text']}"
            if count_budget_units("\n\n".join([*blocks, block])) <= maximum_budget:
                selected_ids.append(evidence_id)
                blocks.append(block)
                reason = "optional_selected"
            else:
                reason = "optional_budget_exceeded"
            events.append({"evidence_id": evidence_id, "reason": reason})
    candidate_text = "\n\n".join(blocks)
    return {
        "status": status, "budget_unit": budget_unit,
        "maximum_budget": maximum_budget,
        "used_budget": count_budget_units(candidate_text),
        "selected_evidence_ids": selected_ids,
        "selected_evidence": [evidence_by_id[key] for key in selected_ids],
        "deferred_required_ids": [key for key in required_ids if key not in selected_ids],
        "missing_evidence_ids": missing_ids,
        "oversized_evidence_ids": oversized_ids,
        "incomplete_query_ids": incomplete_query_ids,
        "requires_staged_processing": status == "needs_staged_processing",
        "context_text": candidate_text if status == "ready" else None,
        "diagnostic_partial_text": candidate_text if status != "ready" else None,
        "selection_events": events,
    }


def run_atomic_evidence_budget_demo() -> dict[str, Any]:
    """
    功能：一次展示必要证据完整可容纳、总预算不足、单块过大和缺少证据四种情况。
    参数含义：无参数；正文、来源和必要证据关系均为人工构造，不是养犬建议。
    返回值含义：返回各场景的规划报告；使用 len 按字符计数，不冒充模型 token 成本。
        生产接入需传入目标模型的计数器，并提前扣除提示词、问题和回答预留预算。
    """
    evidence = {
        "A": {"evidence_id": "A", "text": "安全条件原文占位。" * 40,
              "source_chunk_ids": ["chunk-A"], "locations": [{"source": "teaching", "page_number": 1}]},
        "B": {"evidence_id": "B", "text": "运动安排原文占位。" * 40,
              "source_chunk_ids": ["chunk-B"], "locations": [{"source": "teaching", "page_number": 2}]},
        "C": {"evidence_id": "C", "text": "可选补充原文占位。" * 10,
              "source_chunk_ids": ["chunk-C"], "locations": [{"source": "teaching", "page_number": 3}]},
    }
    required = {"q1": ["A"], "q2": ["A", "B"]}
    cases = (
        ("enough_budget", evidence, 1000),
        ("combined_evidence_too_large", evidence, 650),
        ("single_unit_too_large", evidence, 200),
        ("missing_required_evidence", {"A": evidence["A"], "C": evidence["C"]}, 1000),
    )
    return {
        "notice": "人工证据关系、字符预算演示；没有执行模型摘要、重排或分阶段回答。",
        "required_ids_by_query": required,
        "results": [
            {"case_id": case_id, **plan_atomic_evidence_context(
                evidence_by_id=units, required_ids_by_query=required,
                optional_ranked_ids=["C"], maximum_budget=budget,
                count_budget_units=len, budget_unit="characters",
            )}
            for case_id, units, budget in cases
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_atomic_evidence_budget_demo(), ensure_ascii=False, indent=2))
