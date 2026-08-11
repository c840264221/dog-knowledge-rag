"""Skill（技能）基础数据结构。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SKILL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
SKILL_INPUT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


SkillInputRequirementLevel = Literal[
    "hard_required",
    "degradable",
    "optional",
]


class SkillInputRequirement(BaseModel):
    """
    描述 Skill 声明的一项结构化输入及其缺失影响。

    功能：
        将稳定机器字段与中文展示名称分开保存，使输入检查器可以使用
        input_id 对接 state，同时可以使用 name 向用户生成易懂的澄清问题。

    参数含义：
        input_id:
            稳定机器字段，只允许小写字母、数字和下划线。
        name:
            面向用户展示的中文字段名称。
        description:
            该输入的用途和期望内容说明。
        requirement_level:
            该输入缺失时对技能执行的影响。hard_required 表示必须补充，
            degradable 表示用户明确选择简化执行后可以缺省，optional 表示
            没有该输入也不阻止执行。
        source_mappings:
            可用于补全该输入的外部结构化数据源映射，例如
            {"pet_profile": "age_years"} 表示可以用宠物档案年龄补全。

    返回值含义：
        SkillInputRequirement:
            经过格式校验的技能输入要求。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    input_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    requirement_level: SkillInputRequirementLevel = "hard_required"
    source_mappings: dict[str, str] = Field(default_factory=dict)

    @field_validator("input_id")
    @classmethod
    def validate_input_id(cls, value: str) -> str:
        """
        校验技能输入字段编号。

        功能：
            保证输入编号可以稳定作为普通字典和 state 的键。

        参数含义：
            value:
                待校验的输入字段编号。

        返回值含义：
            str:
                格式合法的输入字段编号。
        """

        if not SKILL_INPUT_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "input_id 必须以小写字母开头，且只允许小写字母、数字和下划线"
            )
        return value

    @field_validator("source_mappings")
    @classmethod
    def validate_source_mappings(
        cls,
        values: dict[str, str],
    ) -> dict[str, str]:
        """
        校验外部输入源映射的名称和字段都不是空字符串。

        参数含义：
            values：数据源名称到来源字段名称的映射。

        返回值含义：
            dict[str, str]：去除首尾空白后的稳定映射。
        """

        normalized = {
            str(source_name).strip(): str(source_key).strip()
            for source_name, source_key in values.items()
        }
        if any(not source_name or not source_key for source_name, source_key in normalized.items()):
            raise ValueError("source_mappings 的数据源名称和字段名称不能为空")
        return normalized


class SkillDefinition(BaseModel):
    """
    描述一项可复用的 Agent 技能。

    功能：
        使用统一字段保存技能身份、触发提示、执行步骤、工具边界、
        输出约束和安全规则，为后续注册、选择和 Prompt 注入提供标准契约。

    参数含义：
        skill_id:
            技能稳定编号，只允许小写字母、数字和连字符。
        name:
            面向开发者和用户展示的技能名称。
        description:
            技能负责解决哪一类问题的简短说明。
        activation_hints:
            帮助选择器判断何时使用该技能的关键词或示例表达。
        required_inputs:
            技能声明的结构化输入要求。字段名为兼容旧代码继续保留，列表中
            每项可以分别声明必须、可简化或可选级别。
        instructions:
            Agent 执行技能时应依次遵守的步骤。
        allowed_tools:
            技能允许使用的工具名称；空列表表示不声明工具权限。
        output_contract:
            对技能最终输出内容和格式的要求。
        guardrails:
            执行过程中必须遵守的安全或业务边界。
        version:
            技能定义版本，使用 major.minor.patch 格式。
        enabled:
            当前技能是否允许被选择和加载。

    返回值含义：
        SkillDefinition:
            经过 Pydantic 校验、可被注册表保存的标准技能定义。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    skill_id: str = Field(
        ...,
        min_length=1,
        description="技能稳定编号",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="技能展示名称",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="技能职责说明",
    )
    activation_hints: list[str] = Field(
        default_factory=list,
        description="技能触发提示",
    )
    required_inputs: list[SkillInputRequirement] = Field(
        default_factory=list,
        description="技能声明的输入字段及其缺失影响级别",
    )
    instructions: list[str] = Field(
        ...,
        min_length=1,
        description="按顺序执行的技能步骤",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="技能允许使用的工具名称",
    )
    output_contract: str = Field(
        ...,
        min_length=1,
        description="技能输出约束",
    )
    guardrails: list[str] = Field(
        default_factory=list,
        description="技能安全与业务边界",
    )
    version: str = Field(
        default="1.0.0",
        description="技能定义版本",
    )
    enabled: bool = Field(
        default=True,
        description="技能是否可被选择和加载",
    )

    @field_validator("skill_id")
    @classmethod
    def validate_skill_id(cls, value: str) -> str:
        """
        校验技能编号格式。

        功能：
            保证技能编号可以稳定用于字典键、日志字段和未来文件夹名称。

        参数含义：
            value:
                待校验的技能编号。

        返回值含义：
            str:
                格式合法的原始技能编号。
        """

        if not SKILL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "skill_id 只允许小写字母、数字和单个连字符分隔"
            )
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """
        校验技能版本格式。

        功能：
            要求版本使用 major.minor.patch 三段数字格式，便于后续升级和追踪。

        参数含义：
            value:
                待校验的版本字符串。

        返回值含义：
            str:
                格式合法的版本字符串。
        """

        if not SKILL_VERSION_PATTERN.fullmatch(value):
            raise ValueError("version 必须使用 major.minor.patch 格式")
        return value

    @field_validator(
        "activation_hints",
        "instructions",
        "allowed_tools",
        "guardrails",
    )
    @classmethod
    def validate_text_items(cls, values: list[str]) -> list[str]:
        """
        校验并规范化技能中的字符串列表。

        功能：
            去除每个元素首尾空白，并拒绝空字符串和重复值，避免向 Agent
            注入无意义或相互重复的技能说明。

        参数含义：
            values:
                待校验的字符串列表。

        返回值含义：
            list[str]:
                保持原顺序、已经去除首尾空白的字符串列表。
        """

        normalized_values = [value.strip() for value in values]
        if any(not value for value in normalized_values):
            raise ValueError("技能列表字段不能包含空字符串")
        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("技能列表字段不能包含重复值")
        return normalized_values

    @field_validator("required_inputs")
    @classmethod
    def validate_required_inputs(
        cls,
        values: list[SkillInputRequirement],
    ) -> list[SkillInputRequirement]:
        """
        校验技能必需输入不存在重复编号。

        功能：
            防止一个 Skill 重复声明同一个机器字段，造成澄清问题和输入检查重复。

        参数含义：
            values:
                待校验的结构化输入要求列表。

        返回值含义：
            list[SkillInputRequirement]:
                输入编号互不重复的原始要求列表。
        """

        input_ids = [item.input_id for item in values]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("required_inputs 不能包含重复 input_id")
        return values


class SkillCatalogItem(BaseModel):
    """
    保存用于技能发现阶段的简短目录信息。

    功能：
        只暴露选择技能所需的身份、说明和触发提示，不包含完整执行步骤、
        输出契约和安全边界，从而为后续按需加载减少无关上下文。

    参数含义：
        skill_id:
            技能稳定编号。
        name:
            技能展示名称。
        description:
            技能职责的简短说明。
        activation_hints:
            帮助选择器判断适用场景的关键词或短语。
        version:
            当前技能定义版本。

    返回值含义：
        SkillCatalogItem:
            可以提供给选择器、UI 或未来 LLM Router 的精简技能目录项。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    skill_id: str
    name: str
    description: str
    activation_hints: list[str] = Field(default_factory=list)
    version: str

    @classmethod
    def from_definition(
        cls,
        skill: SkillDefinition,
    ) -> "SkillCatalogItem":
        """
        从完整技能定义提取精简目录项。

        功能：
            主动丢弃 instructions、output_contract 和 guardrails 等完整内容，
            保证发现阶段只获得必要元数据。

        参数含义：
            skill:
                需要生成目录项的完整技能定义。

        返回值含义：
            SkillCatalogItem:
                对应的精简技能目录项。
        """

        return cls(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            activation_hints=list(skill.activation_hints),
            version=skill.version,
        )


class SkillSelectionResult(BaseModel):
    """
    保存一次确定性技能选择结果。

    功能：
        记录是否选中技能、命中了哪些提示、参与匹配的候选技能以及选择原因，
        为后续主图接入和可观测性提供稳定输出。

    参数含义：
        selected_skill_id:
            最终选中的技能编号；没有匹配项时为 None。
        matched_hints:
            选中技能命中的触发提示。
        candidate_skill_ids:
            本次至少命中一个提示的候选技能编号。
        reason:
            当前选择或未选择的原因。
        source:
            选择结果来源：首轮可以来自确定性关键词选择器，恢复执行可以来自
            上游保存并重新传入的技能编号。

    返回值含义：
        SkillSelectionResult:
            可序列化、可记录、可测试的技能选择结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    selected_skill_id: str | None = None
    matched_hints: list[str] = Field(default_factory=list)
    candidate_skill_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    source: Literal[
        "deterministic_keyword_v1",
        "provided_skill_id",
    ] = (
        "deterministic_keyword_v1"
    )


class SkillInputCheckResult(BaseModel):
    """
    保存一次 Skill 必需输入检查结果。

    功能：
        记录可用输入、不同级别的缺失输入、空值输入和面向用户的澄清提示，
        使调用方可以决定完整执行、请求补充或由用户选择简化执行。

    参数含义：
        skill_id:
            当前检查的技能编号。
        is_ready:
            所有必需输入是否已经具备有效值。
        available_input_ids:
            已具备有效值的必需输入编号。
        missing_input_ids:
            当前影响完整执行的输入编号，包含必须输入和可简化输入。
        missing_input_requirements:
            与 missing_input_ids 一一对应的完整输入要求，包含中文名称、说明
            和缺失级别，供多智能体调度器准确汇总每个步骤缺少的信息。
        missing_hard_required_input_ids:
            缺失后不能执行技能的输入编号。
        missing_degradable_input_ids:
            用户明确选择简化执行后可以忽略的缺失输入编号。
        missing_optional_input_ids:
            当前未提供、但不影响执行的可选输入编号。
        ignored_degradable_input_ids:
            用户已经明确同意在简化模式中忽略的可简化输入编号。
        can_run_degraded:
            是否只缺少可简化输入，因此具备让用户选择简化执行的条件。
        empty_input_ids:
            已提供但值为空的技能输入编号，也可能包含可选输入。
        accepted_inputs:
            检查通过并允许交给 Skill 的输入值。
        clarification_prompt:
            输入不足时可以展示给用户的补充信息提示。

    返回值含义：
        SkillInputCheckResult:
            可供后续路由、checkpoint 和测试使用的结构化检查结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    skill_id: str
    is_ready: bool = False
    available_input_ids: list[str] = Field(default_factory=list)
    missing_input_ids: list[str] = Field(default_factory=list)
    missing_input_requirements: list[SkillInputRequirement] = Field(
        default_factory=list
    )
    missing_hard_required_input_ids: list[str] = Field(
        default_factory=list
    )
    missing_degradable_input_ids: list[str] = Field(
        default_factory=list
    )
    missing_optional_input_ids: list[str] = Field(
        default_factory=list
    )
    ignored_degradable_input_ids: list[str] = Field(
        default_factory=list
    )
    can_run_degraded: bool = False
    empty_input_ids: list[str] = Field(default_factory=list)
    accepted_inputs: dict[str, Any] = Field(default_factory=dict)
    clarification_prompt: str = ""


class SkillInputExtractionResult(BaseModel):
    """
    保存一次 Skill 自然语言输入提取结果。

    功能：
        区分本轮从用户文本中新提取的字段，以及与上一轮已有字段合并后的完整
        输入，供后续 SkillInputChecker 继续判断资料是否齐全。

    参数含义：
        skill_id:
            当前准备执行的技能编号。
        extracted_inputs:
            仅包含本轮用户文本中确定性规则识别出的字段。
        merged_inputs:
            上一轮已有输入与本轮提取字段合并后的结果；本轮字段优先。
        source:
            提取结果来源。没有注册规则时明确记录 no_registered_rule。

    返回值含义：
        SkillInputExtractionResult:
            可继续交给输入检查器的结构化提取结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    skill_id: str
    extracted_inputs: dict[str, Any] = Field(default_factory=dict)
    merged_inputs: dict[str, Any] = Field(default_factory=dict)
    source: Literal[
        "deterministic_rule_v1",
        "no_registered_rule",
    ] = "deterministic_rule_v1"


class SkillRuntimeResult(BaseModel):
    """
    保存一次 Skill 运行准备流程的统一结果。

    功能：
        汇总技能选择、自然语言输入提取、必需输入检查和上下文加载结果，
        让上游只根据 status 就能判断没有技能、等待补充信息或已经可以执行。

    参数含义：
        status:
            当前 Skill 流程状态：no_skill 表示未选中技能，awaiting_input
            表示还缺少必需输入，ready 表示已经可以交给 Agent 执行。
        selection:
            本轮技能选择结果。
        extraction:
            已选中技能时的输入提取与历史输入合并结果。
        input_check:
            已选中技能时的必需输入检查结果。
        skill_context:
            仅在 ready 时生成的完整技能说明，用于后续注入 Agent 上下文。

    返回值含义：
        SkillRuntimeResult:
            可供未来主图节点、checkpoint 和可观测系统使用的标准运行结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["no_skill", "awaiting_input", "ready"]
    selection: SkillSelectionResult
    extraction: SkillInputExtractionResult | None = None
    input_check: SkillInputCheckResult | None = None
    skill_context: str = ""
