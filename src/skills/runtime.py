"""Skill（技能）选择、输入准备和上下文加载运行器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.skills.input_checker import SkillInputChecker
from src.skills.input_extractor import SkillInputExtractor
from src.skills.loader import SkillLoader
from src.skills.schemas import (
    SkillRuntimeResult,
    SkillSelectionResult,
)
from src.skills.selector import SkillSelector


class SkillRuntime:
    """
    串联一次 Skill 执行前的全部准备步骤。

    功能：
        依次完成技能选择、自然语言输入提取、必需输入检查和完整技能上下文
        加载。资料不完整时只返回澄清信息，不提前加载完整技能说明。

    参数含义：
        selector:
            根据用户问题自动选择技能的选择器。
        extractor:
            把用户自然语言转换成技能结构化输入的提取器。
        checker:
            判断技能必需输入是否齐全的检查器。
        loader:
            在输入齐全后加载并渲染完整技能说明的加载器。

    返回值含义：
        SkillRuntime:
            可以执行统一 Skill 准备流程的运行器对象。
    """

    def __init__(
        self,
        *,
        selector: SkillSelector,
        extractor: SkillInputExtractor,
        checker: SkillInputChecker,
        loader: SkillLoader,
    ) -> None:
        self.selector = selector
        self.extractor = extractor
        self.checker = checker
        self.loader = loader

    def prepare(
        self,
        *,
        user_text: str,
        existing_inputs: Mapping[str, Any] | None = None,
        selected_skill_id: str | None = None,
    ) -> SkillRuntimeResult:
        """
        为本轮用户输入准备可执行的 Skill 上下文。

        功能：
            首轮没有指定 skill_id 时自动选择技能；恢复执行时允许上游传入
            上一轮保存的 skill_id。随后提取并合并输入，缺少资料时返回
            awaiting_input，资料齐全时才渲染完整技能上下文。

        参数含义：
            user_text:
                用户本轮提供的原始自然语言。
            existing_inputs:
                前几轮已经提取并保存的技能输入；首轮可以为空。
            selected_skill_id:
                上一轮已经确定的技能编号；为空时根据本轮问题自动选择。

        返回值含义：
            SkillRuntimeResult:
                状态为 no_skill、awaiting_input 或 ready 的统一结构化结果。
        """

        # 恢复执行时，上游已经知道继续哪个 Skill，不需要用简短回答重新选择。
        if selected_skill_id is not None:
            self.loader.load(selected_skill_id)
            selection = SkillSelectionResult(
                selected_skill_id=selected_skill_id,
                candidate_skill_ids=[selected_skill_id],
                reason="使用上游提供的已选技能继续准备输入。",
                source="provided_skill_id",
            )
        else:
            selection = self.selector.select(user_text)

        # 没有匹配技能时立即结束，避免对未知业务执行输入提取和上下文加载。
        resolved_skill_id = selection.selected_skill_id
        if resolved_skill_id is None:
            return SkillRuntimeResult(
                status="no_skill",
                selection=selection,
            )

        # 将本轮自然语言提取结果与上游保存的历史技能输入合并。
        extraction = self.extractor.extract(
            skill_id=resolved_skill_id,
            user_text=user_text,
            existing_inputs=existing_inputs,
        )

        # merged_inputs 已经过字段白名单过滤，但仍需要检查必需字段是否齐全。
        input_check = self.checker.check(
            resolved_skill_id,
            extraction.merged_inputs,
        )
        if not input_check.is_ready:
            return SkillRuntimeResult(
                status="awaiting_input",
                selection=selection,
                extraction=extraction,
                input_check=input_check,
            )

        # 只有输入齐全后才加载完整说明，减少等待阶段无用的上下文 Token。
        skill_context = self.loader.render_context(resolved_skill_id)
        return SkillRuntimeResult(
            status="ready",
            selection=selection,
            extraction=extraction,
            input_check=input_check,
            skill_context=skill_context,
        )
