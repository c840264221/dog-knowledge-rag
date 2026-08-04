"""Skill（技能）确定性选择器。"""

from __future__ import annotations

from src.skills.registry import SkillRegistry
from src.skills.schemas import SkillSelectionResult


class SkillSelector:
    """
    根据用户问题从启用技能中选择最匹配的一项。

    功能：
        使用简单、确定性的字符串包含规则匹配 activation_hints。匹配数量更多、
        命中短语总长度更长的技能优先；仍然相同时按 skill_id 排序。

    参数含义：
        registry:
            保存可用技能定义的 SkillRegistry。

    返回值含义：
        SkillSelector:
            可重复执行确定性技能选择的对象。
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def select(self, question: str) -> SkillSelectionResult:
        """
        为一个用户问题选择技能。

        功能：
            1. 归一化用户问题。
            2. 检查每个启用技能的 activation_hints。
            3. 对命中技能进行稳定排序。
            4. 返回最佳技能或明确的未命中结果。

        参数含义：
            question:
                用户原始问题。

        返回值含义：
            SkillSelectionResult:
                包含选中技能、命中提示、候选技能和原因的结构化结果。
        """

        normalized_question = str(question or "").strip().lower()
        if not normalized_question:
            return SkillSelectionResult(
                reason="用户问题为空，无法选择技能。",
            )

        candidates: list[tuple[str, list[str]]] = []
        for item in self.registry.list_catalog():
            matched_hints = [
                hint
                for hint in item.activation_hints
                if hint.lower() in normalized_question
            ]
            if matched_hints:
                candidates.append((item.skill_id, matched_hints))

        if not candidates:
            return SkillSelectionResult(
                reason="用户问题没有命中任何启用技能的触发提示。",
            )

        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -len(candidate[1]),
                -sum(len(hint) for hint in candidate[1]),
                candidate[0],
            ),
        )
        selected_skill_id, matched_hints = ranked_candidates[0]

        return SkillSelectionResult(
            selected_skill_id=selected_skill_id,
            matched_hints=matched_hints,
            candidate_skill_ids=sorted(
                skill_id
                for skill_id, _ in candidates
            ),
            reason=(
                f"技能 {selected_skill_id} 命中 "
                f"{len(matched_hints)} 个触发提示。"
            ),
        )
