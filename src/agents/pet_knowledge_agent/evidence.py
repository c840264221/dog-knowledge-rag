"""知识结果交接契约：保留依据、推断和未决条件，不把格式检查当成语义证明。"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeOutputValidationError(ValueError):
    """保存知识输出校验的失败现场，不将无效输出作为成功结果使用。"""

    def __init__(self, message: str, diagnostics: dict):
        """初始化错误；message为原失败原因、diagnostics为本次独立诊断数据；无返回值。"""
        super().__init__(message)
        self.diagnostics = diagnostics


class EvidenceQuote(BaseModel):
    """原文摘录；citation_id为本轮来源编号，quote为该来源中的原文。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    citation_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class EvidenceAssessment(BaseModel):
    """候选判断；name沿用用户名称，status区分直接支持、推断、待核实及未覆盖，conditions记录未决条件。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    status: Literal["direct_support", "inferred", "needs_review", "not_covered"]
    reason: str = Field(min_length=1)
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    unresolved_conditions: list[str] = Field(default_factory=list)


class KnowledgeDraft(BaseModel):
    """单次知识回答；含正文、候选判断及必须传递的限制，不产生工具授权。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    final_answer: str = Field(min_length=1)
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def parse_knowledge_output(raw_text: str) -> KnowledgeDraft:
    """解析单次知识回答。

    参数raw_text为模型JSON文本；返回严格校验的知识草稿。
    仅移除与本地模型定义完全一致的顶层$defs副本；其他额外字段、缺失字段或类型错误仍拒绝，
        不自动补业务数据，不把Schema定义当作回答。
    """
    payload = json.loads(raw_text)
    if isinstance(payload, dict) and "$defs" in payload:
        if payload["$defs"] == KnowledgeDraft.model_json_schema().get("$defs"):
            payload = {key: value for key, value in payload.items() if key != "$defs"}
    return KnowledgeDraft.model_validate(payload)


def normalize_citation_format(text: str) -> str:
    """统一明确的局部引用符号。

    参数text为模型回答；返回将（S数字）或(S数字)转换为[S数字]的文本。
    不改编号、不补造引用，不转换其他括号内容；未知编号仍由后续引用校验拒绝。
    """
    text = re.sub(r"（(S[0-9]+)）", r"[\1]", text)
    return re.sub(r"\((S[0-9]+)\)", r"[\1]", text)


def normalize_list_excerpts(draft: KnowledgeDraft, source_texts: dict[str, str]) -> KnowledgeDraft:
    """将可核对的非连续列表摘录拆为独立摘录。

    参数：draft为已解析草稿，source_texts为本轮原文映射。
    返回：独立草稿副本；只有整段不连续、存在至少两个明确的•列表项，且每项均在同一原文中
        按顺序精确出现时才拆分。不改摘录措辞；无法核对时保留原值，交给严格校验拒绝。
    """
    result = draft.model_copy(deep=True)
    for item in result.assessments:
        normalized = []
        for quote in item.evidence:
            original = source_texts.get(quote.citation_id)
            source = re.sub(r"\s+", " ", original) if original is not None else ""
            text = re.sub(r"\s+", " ", quote.quote)
            parts = [part.strip() for part in re.split(r"(?m)(?=^[ \t]*•[ \t]+)", quote.quote) if part.strip()]
            valid_parts = original is not None and text not in source and len(parts) >= 2
            cursor = 0
            if valid_parts:
                for part in parts:
                    if not part.startswith("•"):
                        valid_parts = False
                        break
                    fragment = re.sub(r"\s+", " ", part)
                    position = source.find(fragment, cursor)
                    if position < 0:
                        valid_parts = False
                        break
                    cursor = position + len(fragment)
            if valid_parts:
                normalized.extend(EvidenceQuote(citation_id=quote.citation_id, quote=part) for part in parts)
            else:
                normalized.append(quote)
        item.evidence = normalized
    return result


def validate_evidence(draft: KnowledgeDraft, source_texts: dict[str, str], *,
                      quote_mode: Literal["strict", "audit"] = "strict") -> list[dict]:
    """检查证据可追溯性。

    参数：draft为模型草稿，source_texts为真实入选原文，quote_mode为宿主选择的严格或审计策略。
    返回：审计模式下未匹配摘录的记录列表；严格模式不匹配则抛出异常。
    无效模式、重复候选、缺少必要依据及未知来源始终拒绝；不证明结论被原文支持。
    """
    if quote_mode not in {"strict", "audit"}:
        raise ValueError("未知摘录校验模式。")
    warnings = []
    names = [item.name for item in draft.assessments]
    if len(names) != len(set(names)):
        raise ValueError("候选判断名称重复。")
    for item in draft.assessments:
        if item.status in {"direct_support", "inferred"} and not item.evidence:
            raise ValueError(f"{item.name}缺少原文依据。")
        for quote in item.evidence:
            original = source_texts.get(quote.citation_id)
            if original is None:
                raise ValueError(f"{item.name}的摘录不在有效引用原文中：来源编号不存在。")
            if re.sub(r"\s+", " ", quote.quote) not in re.sub(r"\s+", " ", original):
                if quote_mode == "strict":
                    raise ValueError(f"{item.name}的摘录不在引用原文中。")
                warnings.append({"name": item.name, "citation_id": quote.citation_id,
                    "quote": quote.quote, "status": "not_exactly_matched",
                    "notice": "未通过逐字匹配，仅保留来源关联，不能作为已核验原文摘录。"})
    return warnings


def validate_evidence_bound_items(items: list[dict], dependency_results: dict) -> None:
    """拦截明确未覆盖或没有前置判断的核算项目。

    参数：items为最终工具项目，dependency_results为调度器传来的可信前置结果。
    返回：已有判断且未明确标为not_covered时无返回值，否则拒绝。推断、待核实和条件项允许
        暂估并由展示层披露；名称按用户名称精确对应，该检查不保证语义或商品适用性。
    """
    assessments = {}
    for result in dependency_results.values():
        handoff = result.get("output", {}).get("knowledge_handoff")
        if handoff is None:
            continue
        draft = KnowledgeDraft.model_validate(handoff)
        for assessment in draft.assessments:
            assessments.setdefault(assessment.name, []).append(assessment)
    for item in items:
        matches = assessments.get(item.get("name"), [])
        if not matches or any(entry.status == "not_covered" for entry in matches):
            raise ValueError(f"项目{item.get('name')}没有前置判断或被明确标为未覆盖，不能计入预算。")
