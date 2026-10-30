"""验证证据交接与披露边界，不调用模型或向量库。"""

import json
from types import SimpleNamespace

import pytest

from src.agents.pet_knowledge_agent.evidence import (
    KnowledgeDraft, normalize_citation_format, normalize_list_excerpts, parse_knowledge_output,
    validate_evidence, validate_evidence_bound_items,
)
from src.agents.collaboration.aggregator.evidence_presentation import present_evidence
from src.rag.context_builders.cited_context_builder import check_citation_ids


def make_draft(status="direct_support", conditions=None):
    """构造测试知识草稿；参数为证据状态与未决条件；返回严格校验的数据模型。"""
    return KnowledgeDraft.model_validate({
        "final_answer": "资料包含一个配置要求。[S1]",
        "assessments": [{"name": "候选A", "status": status, "reason": "依据原文判断",
            "evidence": [{"citation_id": "S1", "quote": "original requirement"}],
            "unresolved_conditions": conditions or []}],
        "limitations": ["只覆盖已提供资料。"],
    })


def test_quotes_must_exist_in_real_source():
    """验证引用及原文核对；无参数、无返回值，不将模型自报出处当成证据。"""
    draft = make_draft()
    validate_evidence(draft, {"S1": "An original\nrequirement is listed."})
    with pytest.raises(ValueError, match="摘录不在"):
        validate_evidence(draft, {"S1": "Unrelated text."})
    with pytest.raises(ValueError, match="摘录不在"):
        validate_evidence(draft, {"S2": "original requirement"})


def test_audit_records_quote_mismatch_but_strict_still_rejects():
    """验证策略分离；无参数、无返回值，审计记录未匹配，默认严格模式仍拒绝。"""
    draft = make_draft()
    sources = {"S1": "Different extracted wording."}
    warnings = validate_evidence(draft, sources, quote_mode="audit")
    assert warnings[0]["citation_id"] == "S1"
    assert warnings[0]["status"] == "not_exactly_matched"
    with pytest.raises(ValueError, match="摘录不在"):
        validate_evidence(draft, sources)


def test_audit_still_rejects_unknown_source():
    """验证审计不放宽来源身份；无参数、无返回值，未知引用始终拒绝。"""
    with pytest.raises(ValueError, match="来源编号不存在"):
        validate_evidence(make_draft(), {}, quote_mode="audit")


def test_citation_brackets_normalized_without_changing_prose():
    """验证纯格式兼容；无参数、无返回值，普通括号内容与已有标准引用不变。"""
    assert normalize_citation_format("说明（S1），补充(S2)，已有[S3]，普通（可选）。") == (
        "说明[S1]，补充[S2]，已有[S3]，普通（可选）。")


def test_normalized_unknown_citation_remains_invalid():
    """验证格式转换不授予引用有效性；无参数、无返回值，未知编号依然拒绝。"""
    result = check_citation_ids(normalize_citation_format("结论（S999）。"), {"S1": {}})
    assert result["unknown_ids"] == ["S999"]
    assert not result["all_used_ids_exist"]


def test_noncontiguous_bullets_split_into_individual_exact_quotes():
    """验证同源列表可省略中间项；无参数、无返回值，独立摘录均精确匹配并保留原始草稿。"""
    draft = make_draft()
    draft.assessments[0].evidence[0].quote = "• first item\n• last item"
    sources = {"S1": "• first item\n• middle item\n• last item"}
    normalized = normalize_list_excerpts(draft, sources)
    validate_evidence(normalized, sources)
    assert [entry.quote for entry in normalized.assessments[0].evidence] == ["• first item", "• last item"]
    assert len(draft.assessments[0].evidence) == 1


@pytest.mark.parametrize("quote", ["• changed item\n• last item", "• last item\n• first item"])
def test_changed_or_reordered_list_excerpts_still_rejected(quote):
    """验证不放行改写和顺序错乱；quote为无效摘录；无返回值，严格校验仍拒绝。"""
    draft = make_draft()
    draft.assessments[0].evidence[0].quote = quote
    sources = {"S1": "• first item\n• middle item\n• last item"}
    with pytest.raises(ValueError, match="摘录不在"):
        validate_evidence(normalize_list_excerpts(draft, sources), sources)


def test_schema_definition_copy_does_not_discard_valid_business_data():
    """验证已知结构定义副本兼容；无参数、无返回值，保留业务值而不追加模型调用。"""
    payload = make_draft().model_dump()
    payload["$defs"] = KnowledgeDraft.model_json_schema()["$defs"]
    parsed = parse_knowledge_output(json.dumps(payload))
    assert parsed == make_draft()


def test_unknown_fields_and_schema_only_output_still_rejected():
    """验证兼容不放宽业务契约；无参数、无返回值，未知字段和纯结构定义仍拒绝。"""
    payload = make_draft().model_dump()
    payload["tool_confirmed"] = "confirmed"
    with pytest.raises(ValueError):
        parse_knowledge_output(json.dumps(payload))
    with pytest.raises(ValueError):
        parse_knowledge_output(json.dumps(KnowledgeDraft.model_json_schema()))


@pytest.mark.parametrize("status,conditions", [
    ("inferred", []), ("needs_review", []),
    ("direct_support", ["未确认适用条件"]),
])
def test_inferred_and_conditional_items_can_enter_estimated_budget(status, conditions):
    """验证暂估口径；参数为证据状态和条件；无返回值，推断与条件不再阻断预算。"""
    dependencies = {"knowledge": {"output": {"knowledge_handoff": make_draft(status, conditions).model_dump()}}}
    validate_evidence_bound_items([{"name": "候选A"}], dependencies)


def test_explicitly_uncovered_items_still_rejected():
    """验证明确未覆盖仍不计入；无参数、无返回值。"""
    dependencies = {"knowledge": {"output": {"knowledge_handoff": make_draft("not_covered").model_dump()}}}
    with pytest.raises(ValueError, match="未覆盖"):
        validate_evidence_bound_items([{"name": "候选A"}], dependencies)


def test_direct_items_pass_but_missing_items_do_not():
    """验证直接支持和漏评估的区别；无参数、无返回值，未知名称不能绕过检查。"""
    dependencies = {"knowledge": {"output": {"knowledge_handoff": make_draft().model_dump()}}}
    validate_evidence_bound_items([{"name": "候选A"}], dependencies)
    with pytest.raises(ValueError):
        validate_evidence_bound_items([{"name": "候选B"}], dependencies)


def test_aggregation_preserves_scoped_sources_and_limitations():
    """验证最终展示不丢披露；无参数、无返回值，两步骤相同局部编号也不会覆盖。"""
    task = SimpleNamespace(task_results=[
        SimpleNamespace(step_id="A", status="completed", output={
            "citation_map": {"S1": {"title": "资料A", "source": "a.pdf", "page_number": 4}},
            "limitations": ["仅适用于指定条件。"]}),
        SimpleNamespace(step_id="B", status="completed", output={
            "citation_map": {"S1": {"title": "资料B", "source": "b.pdf", "page_number": 8}},
            "tool_results": [{"content": {"notice": "仅为部分项目小计。"}}]}),
    ])
    text, limits, sources = present_evidence(task, "结论[A/S1]，补充[B/S1]。", [])
    assert sources["A/S1"]["source"] == "a.pdf"
    assert sources["B/S1"]["source"] == "b.pdf"
    assert "仅适用于指定条件。" in text and "仅为部分项目小计。" in text
    assert len(limits) == 2
    task.task_results[0].output["knowledge_handoff"] = make_draft("inferred").model_dump()
    disclosed, _, _ = present_evidence(task, "结论[A/S1]。", [])
    assert "推断用品“候选A”" in disclosed
    with pytest.raises(ValueError, match="需用"):
        present_evidence(task, "结论[S1]。", [])
    with pytest.raises(ValueError, match="未知"):
        present_evidence(task, "结论[C/S1]。", [])
