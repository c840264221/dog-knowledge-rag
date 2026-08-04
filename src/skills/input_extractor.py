"""Skill（技能）自然语言输入提取调度层。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.skills.loader import SkillLoader
from src.skills.schemas import SkillInputExtractionResult


SkillExtractionRule = Callable[[str], Mapping[str, Any]]


class SkillInputExtractor:
    """
    根据 skill_id 调用对应的确定性输入提取规则。

    功能：
        管理技能编号到提取函数的映射，过滤 Skill 未声明的字段，并将本轮新值
        与上一轮已有输入合并。该类不判断输入是否齐全，完整性由 Checker 负责。

    参数含义：
        loader:
            用于读取技能输入契约的 SkillLoader。
        rules:
            技能编号到自然语言提取函数的映射。

    返回值含义：
        SkillInputExtractor:
            可以为已注册 Skill 提取结构化输入的对象。
    """

    def __init__(
        self,
        *,
        loader: SkillLoader,
        rules: Mapping[str, SkillExtractionRule] | None = None,
    ) -> None:
        self.loader = loader

        # 每个 Skill 对应一个独立的自然语言提取规则。
        # 复制成普通字典后，外部修改原始映射不会改变当前提取器的行为。
        self.rules = dict(rules or {})

    def extract(
        self,
        *,
        skill_id: str,
        user_text: str,
        existing_inputs: Mapping[str, Any] | None = None,
    ) -> SkillInputExtractionResult:
        """
        从本轮用户文本中提取指定 Skill 的结构化输入。

        功能：
            1. 读取 Skill 声明的合法输入字段。
            2. 调用该 Skill 已注册的确定性提取规则。
            3. 丢弃规则错误返回的未声明字段。
            4. 将本轮新字段覆盖合并到上一轮已有输入。

        参数含义：
            skill_id:
                当前准备执行的技能编号。
            user_text:
                用户本轮提供的原始自然语言。
            existing_inputs:
                前几轮已经提取并保留的结构化输入；可以为空。

        返回值含义：
            SkillInputExtractionResult:
                包含本轮提取字段和合并后字段的结构化结果。
        """

        skill = self.loader.load(skill_id)

        # Skill 明确声明允许接收的机器字段。
        # 提取规则即使返回其他字段，也不能越过这份输入契约。
        allowed_input_ids = {
            requirement.input_id
            for requirement in skill.required_inputs
        }

        # 前几轮已经获得的结构化输入，这些值尚未在本方法内重新校验。
        # 先过滤未声明字段，避免把整个 State 混进 Skill 输入。
        merged_inputs = {
            input_id: value
            for input_id, value in dict(existing_inputs or {}).items()
            if input_id in allowed_input_ids
        }

        extraction_rule = self.rules.get(skill_id)
        if extraction_rule is None:
            return SkillInputExtractionResult(
                skill_id=skill_id,
                merged_inputs=merged_inputs,
                source="no_registered_rule",
            )

        # 当前规则从本轮自然语言里识别出的原始字段。
        # 规则输出还未经过 Skill 输入白名单过滤。
        raw_extracted_inputs = dict(
            extraction_rule(str(user_text or "").strip())
        )

        # 本轮提取结果中只有 Skill 声明过的字段可以继续向后传递。
        extracted_inputs = {
            input_id: value
            for input_id, value in raw_extracted_inputs.items()
            if input_id in allowed_input_ids
        }

        # 本轮用户补充的信息优先于历史值，允许用户纠正上一轮资料。
        merged_inputs.update(extracted_inputs)

        return SkillInputExtractionResult(
            skill_id=skill_id,
            extracted_inputs=extracted_inputs,
            merged_inputs=merged_inputs,
        )
