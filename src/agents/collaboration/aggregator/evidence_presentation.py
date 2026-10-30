"""聚合时保留步骤来源与已有限制，不依赖模型自行复述全部披露信息。"""

from collections.abc import Mapping
import re


def collect_citation_catalog(task_result) -> dict:
    """收集带步骤身份的来源；task_result为真实步骤结果；返回全局引用到原来源的映射。"""
    catalog = {}
    for result in task_result.task_results:
        if result.status != "completed":
            continue
        for local_id, source in (result.output.get("citation_map") or {}).items():
            if not re.fullmatch(r"S\d+", local_id) or not isinstance(source, Mapping):
                raise ValueError("步骤来源映射格式无效。")
            catalog[f"{result.step_id}/{local_id}"] = dict(source)
    return catalog


def present_evidence(task_result, final_answer: str, limitations: list[str]) -> tuple[str, list[str], dict]:
    """核对引用并补充披露。

    参数：task_result为执行事实，final_answer及limitations为模型草稿。
    返回：供用户展示的回答、合并限制和实际引用映射。只检查引用身份，不证明事实受来源支持；
        自动附加已有工具notice、知识限制和模型限制，不重新执行或改变步骤状态。
    """
    catalog = collect_citation_catalog(task_result)
    used = set(re.findall(r"\[([^\[\]\s]+/S\d+)\]", final_answer))
    if used - catalog.keys():
        raise ValueError("聚合回答包含未知的步骤来源引用。")
    if catalog and (not used or re.search(r"\[S\d+\]", final_answer)):
        raise ValueError("聚合回答需用[步骤编号/S编号]引用知识来源，不能省略或使用局部编号。")
    merged = list(dict.fromkeys(limitations))
    for result in task_result.task_results:
        handoff = result.output.get("knowledge_handoff") or {}
        for assessment in handoff.get("assessments", []):
            status = assessment.get("status")
            if status in {"inferred", "needs_review"}:
                label = "推断用品" if status == "inferred" else "待核实用品"
                text = f"步骤 {result.step_id}：{label}“{assessment.get('name')}”：{assessment.get('reason')}"
                if text not in merged:
                    merged.append(text)
            conditions = assessment.get("unresolved_conditions") or []
            if conditions:
                text = f"步骤 {result.step_id}：{assessment.get('name')}的预算假设与提醒：" + "；".join(conditions)
                if text not in merged:
                    merged.append(text)
        existing = result.output.get("limitations") or []
        if not isinstance(existing, list) or any(not isinstance(item, str) for item in existing):
            raise ValueError("步骤limitations必须为字符串列表。")
        for item in existing:
            text = f"步骤 {result.step_id}：{item}"
            if item.strip() and text not in merged:
                merged.append(text)
        for tool in result.output.get("tool_results") or []:
            content = tool.get("content") if isinstance(tool, Mapping) else None
            notice = content.get("notice") if isinstance(content, Mapping) else None
            if isinstance(notice, str) and notice.strip():
                text = f"步骤 {result.step_id}：{notice}"
                if text not in merged:
                    merged.append(text)
    sections = [final_answer]
    if merged:
        sections.append("限制与核算范围（程序保留）：\n" + "\n".join(f"- {item}" for item in merged))
    selected = {key: catalog[key] for key in sorted(used)}
    if selected:
        sources = []
        for key, source in selected.items():
            page = source.get("page_number")
            sources.append(f"- [{key}] {source.get('title') or '未命名资料'}；"
                           f"页码：{page if page is not None else '未提供'}；来源：{source.get('source')}")
        sections.append("资料来源（程序映射）：\n" + "\n".join(sources))
    return "\n\n".join(sections), merged, selected
