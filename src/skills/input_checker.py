"""Skill（技能）结构化输入检查器。"""

from __future__ import annotations

from collections.abc import Mapping, Sized
from typing import Any

from src.skills.loader import SkillLoader
from src.skills.schemas import SkillInputCheckResult


class SkillInputChecker:
    """
    检查一个 Skill 的必需结构化输入是否完整。

    功能：
        从 SkillLoader 读取完整技能定义，逐项检查 required_inputs 对应的
        input_id 是否存在且非空，并生成结构化结果和用户澄清提示。

    参数含义：
        loader:
            用于读取已启用技能定义的 SkillLoader。

    返回值含义：
        SkillInputChecker:
            可检查指定技能输入是否满足执行条件的对象。
    """

    def __init__(self, loader: SkillLoader) -> None:
        self.loader = loader

    def check(
        self,
        skill_id: str,
        available_inputs: Mapping[str, Any] | None,
    ) -> SkillInputCheckResult:
        """
        检查指定技能的必需输入。

        功能：
            区分完全没有提供的字段和已经提供但内容为空的字段；只把通过检查的
            必需输入放进 accepted_inputs，避免把无关 state 数据交给 Skill。

        参数含义：
            skill_id:
                需要检查的技能编号。
            available_inputs:
                上游已经结构化的输入映射，例如
                {"breed": "金毛", "age": "6岁"}。

        返回值含义：
            SkillInputCheckResult:
                包含是否就绪、缺失字段和澄清提示的检查结果。
        """

        skill = self.loader.load(skill_id)
        provided_inputs = dict(available_inputs or {})
        available_input_ids: list[str] = []
        missing_input_ids: list[str] = []
        empty_input_ids: list[str] = []
        accepted_inputs: dict[str, Any] = {}

        for requirement in skill.required_inputs:
            input_id = requirement.input_id
            if input_id not in provided_inputs:
                missing_input_ids.append(input_id)
                continue

            value = provided_inputs[input_id]
            if self._is_empty(value):
                missing_input_ids.append(input_id)
                empty_input_ids.append(input_id)
                continue

            available_input_ids.append(input_id)
            accepted_inputs[input_id] = value

        clarification_prompt = self._build_clarification_prompt(
            skill_id=skill_id,
            missing_input_ids=missing_input_ids,
        )
        return SkillInputCheckResult(
            skill_id=skill_id,
            is_ready=not missing_input_ids,
            available_input_ids=available_input_ids,
            missing_input_ids=missing_input_ids,
            empty_input_ids=empty_input_ids,
            accepted_inputs=accepted_inputs,
            clarification_prompt=clarification_prompt,
        )

    def _build_clarification_prompt(
        self,
        *,
        skill_id: str,
        missing_input_ids: list[str],
    ) -> str:
        """
        根据缺失字段构建用户澄清提示。

        功能：
            将机器字段转换回 Skill 中定义的中文名称和说明，让用户知道需要
            补充什么，而不是直接看到 breed、age 等内部字段。

        参数含义：
            skill_id:
                当前技能编号。
            missing_input_ids:
                当前缺失或为空的输入字段编号。

        返回值含义：
            str:
                没有缺失字段时返回空字符串，否则返回中文补充提示。
        """

        if not missing_input_ids:
            return ""

        skill = self.loader.load(skill_id)
        requirement_by_id = {
            item.input_id: item
            for item in skill.required_inputs
        }
        missing_descriptions = [
            (
                requirement_by_id[input_id].name
                + (
                    f"（{requirement_by_id[input_id].description}）"
                    if requirement_by_id[input_id].description
                    else ""
                )
            )
            for input_id in missing_input_ids
        ]
        return f"请继续补充：{'；'.join(missing_descriptions)}。"

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """
        判断一个结构化输入值是否为空。

        功能：
            将 None、空白字符串和长度为零的容器视为空值；数字 0 和布尔值
            False 是合法业务值，不会被错误判空。

        参数含义：
            value:
                待检查的任意输入值。

        返回值含义：
            bool:
                True 表示输入为空，False 表示存在有效内容。
        """

        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, Sized):
            return len(value) == 0
        return False
