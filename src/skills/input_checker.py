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
        input_id 是否存在且非空，并按必须、可简化、可选三个级别生成
        结构化结果和用户澄清提示。

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
        *,
        ignored_input_ids: list[str] | None = None,
    ) -> SkillInputCheckResult:
        """
        检查指定技能的必需输入。

        功能：
            区分完全没有提供的字段和已经提供但内容为空的字段；再根据输入契约
            判断该字段会阻止执行、允许简化还是可以直接忽略。

        参数含义：
            skill_id:
                需要检查的技能编号。
            available_inputs:
                上游已经结构化的输入映射，例如
                {"breed": "金毛", "age": "6岁"}。
            ignored_input_ids:
                用户已明确同意在简化执行中忽略的输入编号。这里只接受技能
                契约中标为 degradable 的字段，不能绕过强制输入。

        返回值含义：
            SkillInputCheckResult:
                包含是否就绪、缺失字段和澄清提示的检查结果。
        """

        skill = self.loader.load(skill_id)
        provided_inputs = dict(available_inputs or {})
        requirement_by_id = {
            requirement.input_id: requirement
            for requirement in skill.required_inputs
        }
        requested_ignored_input_ids = {
            str(input_id).strip()
            for input_id in (ignored_input_ids or [])
            if str(input_id).strip()
        }
        invalid_ignored_input_ids = sorted(
            input_id
            for input_id in requested_ignored_input_ids
            if input_id not in requirement_by_id
            or requirement_by_id[input_id].requirement_level
            != "degradable"
        )
        if invalid_ignored_input_ids:
            raise ValueError(
                "只有可简化输入允许被忽略: "
                f"{invalid_ignored_input_ids}"
            )

        available_input_ids: list[str] = []
        missing_input_ids: list[str] = []
        missing_hard_required_input_ids: list[str] = []
        missing_degradable_input_ids: list[str] = []
        missing_optional_input_ids: list[str] = []
        ignored_degradable_input_ids: list[str] = []
        empty_input_ids: list[str] = []
        accepted_inputs: dict[str, Any] = {}

        for requirement in skill.required_inputs:
            input_id = requirement.input_id
            if input_id not in provided_inputs:
                if input_id in requested_ignored_input_ids:
                    ignored_degradable_input_ids.append(input_id)
                    continue
                self._record_missing_input(
                    input_id=input_id,
                    requirement_level=requirement.requirement_level,
                    missing_input_ids=missing_input_ids,
                    missing_hard_required_input_ids=(
                        missing_hard_required_input_ids
                    ),
                    missing_degradable_input_ids=(
                        missing_degradable_input_ids
                    ),
                    missing_optional_input_ids=missing_optional_input_ids,
                )
                continue

            value = provided_inputs[input_id]
            if self._is_empty(value):
                empty_input_ids.append(input_id)
                if input_id in requested_ignored_input_ids:
                    ignored_degradable_input_ids.append(input_id)
                    continue
                self._record_missing_input(
                    input_id=input_id,
                    requirement_level=requirement.requirement_level,
                    missing_input_ids=missing_input_ids,
                    missing_hard_required_input_ids=(
                        missing_hard_required_input_ids
                    ),
                    missing_degradable_input_ids=(
                        missing_degradable_input_ids
                    ),
                    missing_optional_input_ids=missing_optional_input_ids,
                )
                continue

            available_input_ids.append(input_id)
            accepted_inputs[input_id] = value

        can_run_degraded = (
            bool(missing_degradable_input_ids)
            and not missing_hard_required_input_ids
        )
        clarification_prompt = self._build_clarification_prompt(
            skill_id=skill_id,
            missing_input_ids=missing_input_ids,
            can_run_degraded=can_run_degraded,
        )
        return SkillInputCheckResult(
            skill_id=skill_id,
            is_ready=not missing_input_ids,
            available_input_ids=available_input_ids,
            missing_input_ids=missing_input_ids,
            # 保留缺失字段的完整契约，不能只留下 age、breed 这类机器编号。
            missing_input_requirements=[
                requirement_by_id[input_id]
                for input_id in missing_input_ids
            ],
            missing_hard_required_input_ids=(
                missing_hard_required_input_ids
            ),
            missing_degradable_input_ids=missing_degradable_input_ids,
            missing_optional_input_ids=missing_optional_input_ids,
            ignored_degradable_input_ids=ignored_degradable_input_ids,
            can_run_degraded=can_run_degraded,
            empty_input_ids=empty_input_ids,
            accepted_inputs=accepted_inputs,
            clarification_prompt=clarification_prompt,
        )

    @staticmethod
    def _record_missing_input(
        *,
        input_id: str,
        requirement_level: str,
        missing_input_ids: list[str],
        missing_hard_required_input_ids: list[str],
        missing_degradable_input_ids: list[str],
        missing_optional_input_ids: list[str],
    ) -> None:
        """
        按输入级别记录一个当前不可用的技能字段。

        功能：
            必须输入和可简化输入都会阻止标准模式执行；可选输入只记录下来，
            不进入阻塞列表，也不会触发用户澄清。

        参数含义：
            input_id:
                当前不可用的技能输入编号。
            requirement_level:
                输入契约声明的缺失影响级别。
            missing_input_ids:
                会阻止标准模式执行的缺失字段列表。
            missing_hard_required_input_ids:
                不能通过简化执行忽略的缺失字段列表。
            missing_degradable_input_ids:
                用户明确选择后允许忽略的缺失字段列表。
            missing_optional_input_ids:
                不影响执行的可选缺失字段列表。

        返回值含义：
            None:
                直接更新调用方传入的分类结果列表。
        """

        if requirement_level == "optional":
            missing_optional_input_ids.append(input_id)
            return

        missing_input_ids.append(input_id)
        if requirement_level == "degradable":
            missing_degradable_input_ids.append(input_id)
            return
        missing_hard_required_input_ids.append(input_id)

    def _build_clarification_prompt(
        self,
        *,
        skill_id: str,
        missing_input_ids: list[str],
        can_run_degraded: bool,
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
            can_run_degraded:
                本次是否可以向用户提供简化执行选项。

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
        missing_text = "；".join(missing_descriptions)
        if can_run_degraded:
            return (
                f"当前缺少：{missing_text}。请补充这些信息；"
                "你也可以回复“简化执行”按现有信息继续，或回复“取消”。"
            )
        return f"当前缺少：{missing_text}。请补充这些信息，或回复“取消”。"

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
