"""Skill（技能）选择、输入准备和上下文加载运行器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.memory.pet_profile_value_normalizer import (
    normalize_age_years_for_skill,
)
from src.skills.input_checker import SkillInputChecker
from src.skills.input_extractor import SkillInputExtractor
from src.skills.loader import SkillLoader
from src.skills.schemas import (
    SkillInputCheckResult,
    SkillInputExtractionResult,
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
        selection: SkillSelectionResult | None = None,
        available_input_sources: (
            Mapping[str, Mapping[str, Any]] | None
        ) = None,
        ignored_input_ids: list[str] | None = None,
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
            selection:
                上游已经完成的技能选择结果；用于先选择技能、再按需加载外部
                输入，避免重复执行技能选择。
            available_input_sources:
                外部结构化数据源，例如 pet_profile（宠物档案）及其事实映射。
            ignored_input_ids:
                用户明确同意在简化执行时忽略的可简化输入编号。

        返回值含义：
            SkillRuntimeResult:
                状态为 no_skill、awaiting_input 或 ready 的统一结构化结果。
        """

        # 恢复执行时，上游已经知道继续哪个 Skill，不需要用简短回答重新选择。
        if selection is not None:
            resolved_selection = selection
        elif selected_skill_id is not None:
            self.loader.load(selected_skill_id)
            resolved_selection = SkillSelectionResult(
                selected_skill_id=selected_skill_id,
                candidate_skill_ids=[selected_skill_id],
                reason="使用上游提供的已选技能继续准备输入。",
                source="provided_skill_id",
            )
        else:
            resolved_selection = self.selector.select(user_text)

        # 没有匹配技能时立即结束，避免对未知业务执行输入提取和上下文加载。
        resolved_skill_id = resolved_selection.selected_skill_id
        if resolved_skill_id is None:
            return SkillRuntimeResult(
                status="no_skill",
                selection=resolved_selection,
            )

        # 外部数据只提供默认值；上轮输入和本轮用户原话可以继续覆盖这些值。
        source_default_inputs = self._build_source_default_inputs(
            skill_id=resolved_skill_id,
            available_input_sources=available_input_sources,
        )
        merged_existing_inputs = dict(source_default_inputs)
        merged_existing_inputs.update(dict(existing_inputs or {}))

        # 将本轮自然语言提取结果与上游保存的历史技能输入合并。
        extraction = self.extractor.extract(
            skill_id=resolved_skill_id,
            user_text=user_text,
            existing_inputs=merged_existing_inputs,
        )

        # merged_inputs 已经过字段白名单过滤，但仍需要检查必需字段是否齐全。
        input_check = self.checker.check(
            resolved_skill_id,
            extraction.merged_inputs,
            ignored_input_ids=ignored_input_ids,
        )
        if not input_check.is_ready:
            return SkillRuntimeResult(
                status="awaiting_input",
                selection=resolved_selection,
                extraction=extraction,
                input_check=input_check,
            )

        # 只有输入齐全后才加载完整说明，减少等待阶段无用的上下文 Token。
        skill_context = self.loader.render_context(resolved_skill_id)
        return SkillRuntimeResult(
            status="ready",
            selection=resolved_selection,
            extraction=extraction,
            input_check=input_check,
            skill_context=skill_context,
        )

    def select(
        self,
        *,
        user_text: str,
        selected_skill_id: str | None = None,
    ) -> SkillSelectionResult:
        """
        在准备输入前单独确定本轮使用的 Skill。

        参数含义：
            user_text：用户本轮自然语言。
            selected_skill_id：恢复执行时检查点保存的技能编号。

        返回值含义：
            SkillSelectionResult：新选择或恢复使用的技能结果。
        """

        if selected_skill_id is not None:
            self.loader.load(selected_skill_id)
            return SkillSelectionResult(
                selected_skill_id=selected_skill_id,
                candidate_skill_ids=[selected_skill_id],
                reason="使用上游提供的已选技能继续准备输入。",
                source="provided_skill_id",
            )
        return self.selector.select(user_text)

    def get_source_required_fields(
        self,
        *,
        skill_id: str,
        source_name: str,
    ) -> list[str]:
        """
        读取某个 Skill 从指定外部数据源需要的字段。

        参数含义：
            skill_id：已经选中的技能编号。
            source_name：外部数据源名称，例如 pet_profile（宠物档案）。

        返回值含义：
            list[str]：Skill 输入契约映射到该数据源的字段列表，包括可用于
            丰富执行效果的可选字段。
        """

        skill = self.loader.load(skill_id)
        source_fields: list[str] = []
        for requirement in skill.required_inputs:
            source_field = str(
                requirement.source_mappings.get(source_name) or ""
            ).strip()
            if source_field and source_field not in source_fields:
                source_fields.append(source_field)
        return source_fields

    def extract_inputs(
        self,
        *,
        skill_id: str,
        user_text: str,
        existing_inputs: Mapping[str, Any] | None = None,
    ) -> SkillInputExtractionResult:
        """
        在查询外部数据源前先提取用户已经提供的 Skill 输入。

        参数含义：
            skill_id：已经选中的技能编号。
            user_text：用户本轮自然语言输入。
            existing_inputs：前几轮已经保存的技能输入。

        返回值含义：
            SkillInputExtractionResult：本轮提取值及与历史值合并后的结果。
        """

        return self.extractor.extract(
            skill_id=skill_id,
            user_text=user_text,
            existing_inputs=existing_inputs,
        )

    def check_inputs(
        self,
        *,
        skill_id: str,
        provided_inputs: Mapping[str, Any] | None = None,
        ignored_input_ids: list[str] | None = None,
    ) -> SkillInputCheckResult:
        """
        检查指定 Skill 当前仍缺少哪些必需输入。

        功能：
            对外提供统一的 Skill 输入完整性检查入口，避免任务关系门卫等
            调用方越过 SkillRuntime 直接访问内部 Checker。

        参数含义：
            skill_id：需要检查的技能编号。
            provided_inputs：当前已经收集到的技能输入。
            ignored_input_ids：用户已同意在简化执行中忽略的输入编号。

        返回值含义：
            SkillInputCheckResult：包含可用字段、缺失字段和是否就绪的结果。
        """

        return self.checker.check(
            skill_id,
            provided_inputs,
            ignored_input_ids=ignored_input_ids,
        )

    def get_missing_source_required_fields(
        self,
        *,
        skill_id: str,
        source_name: str,
        provided_inputs: Mapping[str, Any] | None = None,
        ignored_input_ids: list[str] | None = None,
    ) -> list[str]:
        """
        计算 Skill 当前仍缺少、并且允许由指定数据源补全的字段。

        参数含义：
            skill_id：已经选中的技能编号。
            source_name：外部数据源名称，例如 pet_profile（宠物档案）。
            provided_inputs：用户本轮和历史轮次已经提供的技能输入。
            ignored_input_ids：用户已经同意在简化执行中忽略的输入编号。

        返回值含义：
            list[str]：仍可从指定数据源补全的来源字段列表。可选字段也会尝试
            补全，但查询不到时不会阻止 Skill 执行。
        """

        available_inputs = dict(provided_inputs or {})
        ignored_ids = {
            str(input_id).strip()
            for input_id in (ignored_input_ids or [])
            if str(input_id).strip()
        }
        skill = self.loader.load(skill_id)
        missing_source_fields: list[str] = []
        for requirement in skill.required_inputs:
            if requirement.input_id in ignored_ids:
                continue
            current_value = available_inputs.get(requirement.input_id)
            if current_value is not None and not (
                isinstance(current_value, str) and not current_value.strip()
            ):
                continue
            source_field = str(
                requirement.source_mappings.get(source_name) or ""
            ).strip()
            if source_field and source_field not in missing_source_fields:
                missing_source_fields.append(source_field)
        return missing_source_fields

    def _build_source_default_inputs(
        self,
        *,
        skill_id: str,
        available_input_sources: (
            Mapping[str, Mapping[str, Any]] | None
        ),
    ) -> dict[str, Any]:
        """
        根据 Skill 声明把外部数据源字段映射成技能输入默认值。

        参数含义：
            skill_id：已经选中的技能编号。
            available_input_sources：数据源名称到结构化字段映射的字典。

        返回值含义：
            dict[str, Any]：以 Skill input_id（技能输入编号）为键的默认输入。
        """

        source_data = dict(available_input_sources or {})
        skill = self.loader.load(skill_id)
        default_inputs: dict[str, Any] = {}
        for requirement in skill.required_inputs:
            for source_name, source_key in requirement.source_mappings.items():
                values = source_data.get(source_name)
                if not isinstance(values, Mapping):
                    continue
                value = values.get(source_key)
                if value is None or (
                    isinstance(value, str) and not value.strip()
                ):
                    continue
                try:
                    value = self._normalize_source_input_value(
                        source_name=source_name,
                        source_key=source_key,
                        value=value,
                    )
                except (TypeError, ValueError):
                    # 历史脏数据不能让整个 Skill 准备流程失败；跳过后由
                    # 输入检查器把该字段作为缺失信息继续向用户澄清。
                    continue
                default_inputs[requirement.input_id] = value
                break
        return default_inputs

    @staticmethod
    def _normalize_source_input_value(
        *,
        source_name: str,
        source_key: str,
        value: Any,
    ) -> Any:
        """
        将外部数据源字段转换成 Skill 输入使用的格式。

        功能：
            对 pet_profile.age_years（宠物档案年龄年数）补充“岁”单位；
            其他来源字段保持原值，避免通用运行器擅自改变业务数据。

        参数含义：
            source_name：外部数据源名称。
            source_key：数据源中的字段名称。
            value：数据源提供的原始值。

        返回值含义：
            Any：可以写入 Skill 输入的值。
        """

        if source_name == "pet_profile" and source_key == "age_years":
            return normalize_age_years_for_skill(value)
        return value
