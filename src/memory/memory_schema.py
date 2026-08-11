from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.memory.pet_profile_value_normalizer import (
    normalize_pet_profile_value,
)


MemoryType: TypeAlias = Literal[
    "favorite_dog",
    "preference",
    "dislike",
    "hobby",
    "profile",
]

MemoryStatus: TypeAlias = Literal[
    "active",
    "inactive",
]

MemorySource: TypeAlias = Literal[
    "conversation",
    "tool",
    "manual",
    "system",
]

MemoryRecallStatus: TypeAlias = Literal[
    "applied",
    "empty",
    "failed",
]

PetProfileRecallStatus: TypeAlias = Literal[
    "applied",
    "empty",
    "ambiguous",
    "failed",
]

PetProfileSelectionSource: TypeAlias = Literal[
    "active_pet",
    "single_pet_fallback",
    "none",
]

MemoryRetentionAction: TypeAlias = Literal[
    "accepted",
    "rejected",
]

PetProfilePersistenceAction: TypeAlias = Literal[
    "created",
    "updated",
    "ignored_stale",
    "failed",
]

PetProfileAttribute: TypeAlias = Literal[
    "breed",
    "birth_date",
    "age_years",
    "weight_kg",
    "sex",
    "neutered",
    "health_condition",
    "allergy",
    "diet_pattern",
    "activity_level",
    "training_goal",
]

VALID_MEMORY_TYPES = frozenset(
    get_args(MemoryType)
)

VALID_MEMORY_SOURCES = frozenset(
    get_args(MemorySource)
)


class MemoryOutput(BaseModel):
    """
    用于定义 LLM 输出的内容。

    作用：
    - 校验 LLM 判断出来的记忆是否值得保存
    - 约束 LLM 输出字段格式
    """

    should_save: bool

    confidence: float = Field(
        ge=0,
        le=1
    )

    memory_type: MemoryType

    content: str

    reason: str

    importance: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )


class MemoryRetentionDecision(BaseModel):
    """
    Memory Retention Decision（记忆保留决策）数据契约。

    功能：
        记录候选记忆是否通过确定性长期保存门槛，以及实际分数、要求门槛
        和最终原因，避免只依赖 LLM 的 should_save 判断。

    参数含义：
        action：accepted 表示允许保存，rejected 表示拒绝保存。
        memory_type：本次审查的记忆类型。
        confidence：候选记忆的实际可信度。
        importance：候选记忆的实际重要度。
        minimum_confidence：当前类型要求的最低可信度。
        minimum_importance：当前类型要求的最低重要度。
        reason：允许或拒绝长期保存的中文原因。

    返回值含义：
        MemoryRetentionDecision：可写入主图状态和日志的结构化审查结果。
    """

    action: MemoryRetentionAction
    memory_type: str
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    minimum_confidence: float = Field(ge=0, le=1)
    minimum_importance: float = Field(ge=0, le=1)
    reason: str

    @property
    def is_accepted(self) -> bool:
        """
        判断候选记忆是否允许进入持久化层。

        参数含义：
            无。

        返回值含义：
            bool：action 为 accepted 时返回 True，否则返回 False。
        """

        return self.action == "accepted"


class PetProfileFact(BaseModel):
    """
    Pet Profile Fact（宠物档案事实）数据契约。

    功能：
        使用“宠物标识 + 属性 + 值 + 观测时间”表示一条结构化宠物档案，
        避免把年龄、体重和健康状态混成无法更新的普通文本记忆。

    参数含义：
        user_id：宠物所属用户编号，用于隔离不同用户的数据。
        pet_key：同一用户下稳定的宠物标识，例如名字归一化后的 doudou。
        pet_name：面向用户展示的宠物名字；不知道时可以为空。
        attribute：档案属性，例如 breed、weight_kg 或 allergy。
        value：属性当前值，统一保存为简洁文本。
        confidence：该事实的可信度，范围为 0 到 1。
        source：事实来源，例如 conversation 或 manual。
        observed_at：用户确认或系统观察到该事实的时间。

    返回值含义：
        PetProfileFact：经过 Pydantic 校验的结构化宠物档案事实。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1)
    pet_key: str = Field(min_length=1)
    pet_name: str = ""
    attribute: PetProfileAttribute
    value: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    source: MemorySource = "conversation"
    observed_at: datetime


class PetProfileFactCandidate(BaseModel):
    """
    Pet Profile Fact Candidate（宠物档案候选事实）数据契约。

    功能：
        保存 LLM 从一段用户输入中拆出的单条候选事实。此时只保留用户原文
        中的对象引用，尚未绑定数据库中的稳定 pet_key。

    参数含义：
        subject_reference：用户原文中的宠物称呼，例如“豆豆”“它”或“我家狗”。
        attribute：候选档案属性，例如 breed、weight_kg 或 allergy。
        value：从用户输入中提取出的属性值。
        confidence：模型对该条提取结果的可信度。
        evidence_text：支持该事实的简短原文证据，用于审计和问题排查。

    返回值含义：
        PetProfileFactCandidate：尚待实体解析和持久化审查的候选事实。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    subject_reference: str = Field(min_length=1, max_length=100)
    attribute: PetProfileAttribute
    value: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    evidence_text: str = Field(min_length=1, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def normalize_attribute_value(cls, data: object) -> object:
        """
        在候选字段类型校验前归一化受支持的宠物档案值。

        功能：
            让数字年龄等 LLM 原始输出先转换成档案标准格式，再执行字符串
            长度和属性白名单校验。其他属性保持原始值和原有校验行为。

        参数含义：
            data：尚未通过 Pydantic 校验的候选事实原始数据。

        返回值含义：
            object：归一化后的候选映射，或无法处理时的原始对象。
        """

        if not isinstance(data, Mapping):
            return data

        normalized_data = dict(data)
        normalized_data["value"] = normalize_pet_profile_value(
            attribute=str(normalized_data.get("attribute") or ""),
            value=normalized_data.get("value"),
        )
        return normalized_data


class PetProfileExtractionResult(BaseModel):
    """
    Pet Profile Extraction Result（宠物档案批量抽取结果）数据契约。

    功能：
        汇总一轮输入中通过校验的全部宠物档案候选，并记录被拒绝的候选
        数量。单条候选格式错误时不会丢弃同一批中的其他合法事实。

    参数含义：
        facts：通过契约校验的候选事实列表。
        rejected_candidate_count：因类型、空值或分数非法而被拒绝的候选数。
        reason：对本轮是否提取到档案信息的简短说明。

    返回值含义：
        PetProfileExtractionResult：可交给实体解析器继续处理的批量结果。
    """

    facts: list[PetProfileFactCandidate] = Field(default_factory=list)
    rejected_candidate_count: int = Field(default=0, ge=0)
    reason: str = ""


class PetProfileResolutionResult(BaseModel):
    """
    Pet Profile Resolution Result（宠物档案实体解析结果）数据契约。

    功能：
        保存已经绑定稳定 pet_key 的宠物档案事实，并单独保留因对象不明确
        而无法绑定的候选事实，防止系统把多只宠物的数据错误合并。

    参数含义：
        facts：已经解析成稳定宠物标识、可以进入持久化层的档案事实。
        unresolved_facts：因宠物对象不明确而暂时不能持久化的候选事实。
        reason：本轮实体解析结果的中文说明。

    返回值含义：
        PetProfileResolutionResult：可交给档案持久化服务继续处理的解析结果。
    """

    facts: list[PetProfileFact] = Field(default_factory=list)
    unresolved_facts: list[PetProfileFactCandidate] = Field(
        default_factory=list
    )
    reason: str = ""


class PetProfilePersistenceRecord(BaseModel):
    """
    Pet Profile Persistence Record（宠物档案单条保存记录）数据契约。

    功能：
        记录一条正式宠物事实进入数据库后的动作和结果，用于统计、日志
        和问题审计；单条失败时不会丢失同批其他事实的保存结果。

    参数含义：
        action：created 为新建，updated 为更新，ignored_stale 为忽略旧数据，
        failed 为保存失败。
        fact_id：SQLite 中宠物事实记录的主键；保存失败时为空。
        pet_key：宠物稳定标识。
        pet_name：宠物展示名称。
        attribute：本次保存的宠物档案属性。
        value：本次尝试保存的属性值。
        observed_at：用户表达或系统观察该事实的时间。
        error：保存失败时的错误信息；成功时为空。

    返回值含义：
        PetProfilePersistenceRecord：一条宠物事实的结构化保存记录。
    """

    action: PetProfilePersistenceAction
    fact_id: int | None = None
    pet_key: str
    pet_name: str = ""
    attribute: PetProfileAttribute
    value: str
    observed_at: datetime
    error: str = ""


class PetProfileSaveResult(BaseModel):
    """
    Pet Profile Save Result（宠物档案批量保存结果）数据契约。

    功能：
        汇总实体解析结果、逐条数据库保存记录和不同保存动作的数量，作为
        宠物档案服务对主图或其他调用方的统一输出。

    参数含义：
        resolution：宠物实体解析结果，包含正式事实和未解析候选。
        persistence_records：已经尝试写入数据库的逐条保存记录。
        created_count：本轮新建的事实数量。
        updated_count：本轮更新的事实数量。
        ignored_stale_count：因观测时间较旧而被忽略的事实数量。
        failed_count：数据库保存失败的事实数量。
        reason：本轮解析和保存结果的中文摘要。

    返回值含义：
        PetProfileSaveResult：宠物档案解析与持久化的统一结构化结果。
    """

    resolution: PetProfileResolutionResult
    persistence_records: list[PetProfilePersistenceRecord] = Field(
        default_factory=list
    )
    created_count: int = Field(default=0, ge=0)
    updated_count: int = Field(default=0, ge=0)
    ignored_stale_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    reason: str = ""


class PetProfileRecallResult(BaseModel):
    """
    Pet Profile Recall Result（宠物档案召回结果）数据契约。

    功能：
        记录当前问题最终选中了哪只宠物、为什么这样选择，以及可以交给
        回答节点使用的结构化档案事实。多只宠物无法确定时明确返回
        ambiguous，防止系统把不同宠物的数据混在一起。

    参数含义：
        status：applied 表示成功召回，empty 表示没有档案，ambiguous 表示
        存在多只宠物但无法确定目标，failed 表示召回异常。
        pet_key：当前召回宠物的稳定标识。
        pet_name：当前召回宠物的展示名称。
        selection_source：宠物选择依据，active_pet 表示使用当前宠物标识，
        single_pet_fallback 表示用户只有一只宠物时自动选择。
        facts：属性名称到当前值的结构化映射。
        selected_attributes：本次实际返回的档案属性列表。
        reason：本次召回结果的中文说明。

    返回值含义：
        PetProfileRecallResult：可写入 DogState 并交给 Prompt 构建器的结果。
    """

    status: PetProfileRecallStatus
    pet_key: str = ""
    pet_name: str = ""
    selection_source: PetProfileSelectionSource = "none"
    facts: dict[PetProfileAttribute, str] = Field(default_factory=dict)
    selected_attributes: list[PetProfileAttribute] = Field(
        default_factory=list
    )
    reason: str = ""


class PetProfileFieldAccessDecision(BaseModel):
    """
    Pet Profile Field Access Decision（宠物档案字段访问决策）数据契约。

    功能：
        记录 Skill（技能）必需字段、上游建议字段、Agent（智能体）数据库
        读取白名单以及最终允许读取的字段，便于权限审计和问题排查。

    参数含义：
        purpose：本次读取用途，skill_prefill 表示为 Skill 补参，
        answer_context 表示为最终回答准备上下文。
        agent_name：申请读取宠物档案的 Agent 名称。
        skill_required_attributes：当前 Skill 声明需要的宠物档案字段。
        suggested_attributes：上游查询理解阶段建议使用的宠物档案字段。
        requested_attributes：两类字段去重合并后的读取申请。
        allowed_attributes：经过字段契约和 Agent 白名单校验后允许读取的字段。
        denied_skill_required_attributes：Skill 需要但不允许从数据库读取的字段；
        该兼容字段同时包含可由用户补充和完全禁止处理的字段。
        user_suppliable_skill_attributes：数据库不允许主动读取、但当前 Agent
        允许处理的 Skill 必需字段，可以通过普通业务澄清请用户补充。
        blocked_skill_attributes：不属于档案契约或当前 Agent 无权处理的 Skill
        必需字段，即使用户主动提供也不能交给该 Agent。
        processing_denied_attributes：当前 Agent 不允许处理的字段，即使用户
        直接提供也不能自动进入业务执行链路。
        invalid_suggested_attributes：不属于宠物档案契约的上游建议字段。
        skill_resolution_action：权限层对 Skill 的处理建议；proceed 表示继续，
        clarify 表示隐式澄清，degrade_or_cancel 表示降级执行或取消。
        reason：本次访问决策的中文说明。

    返回值含义：
        PetProfileFieldAccessDecision：可写入 DogState 和日志的访问决策。
    """

    purpose: Literal["skill_prefill", "answer_context"]
    agent_name: str
    skill_required_attributes: list[str] = Field(default_factory=list)
    suggested_attributes: list[str] = Field(default_factory=list)
    requested_attributes: list[str] = Field(default_factory=list)
    allowed_attributes: list[PetProfileAttribute] = Field(default_factory=list)
    denied_skill_required_attributes: list[str] = Field(default_factory=list)
    user_suppliable_skill_attributes: list[str] = Field(default_factory=list)
    blocked_skill_attributes: list[str] = Field(default_factory=list)
    processing_denied_attributes: list[str] = Field(default_factory=list)
    invalid_suggested_attributes: list[str] = Field(default_factory=list)
    skill_resolution_action: Literal[
        "proceed",
        "clarify",
        "degrade_or_cancel",
    ] = "proceed"
    reason: str = ""


class MemoryRecallResult(BaseModel):
    """
    Memory Recall Result（记忆召回结果）数据契约。

    功能：
        记录记忆召回是否成功应用、候选数量、语义门槛和最终采用的记忆。
        该对象用于服务内部校验，写入 LangGraph state 前需转换为普通 dict。

    参数：
        status：召回状态，applied 表示已采用，empty 表示无可用记忆，failed 表示召回异常。
        memory_context：可直接注入答案 Prompt（提示词）的记忆文本。
        candidate_count：Chroma 初步语义检索返回的候选数量。
        threshold_passed_count：通过最低语义相关性门槛的候选数量。
        selected_count：去重、排序后最终采用的记忆数量。
        semantic_threshold：本次召回使用的最低语义相关分。
        max_semantic_score：最终采用记忆中的最高语义相关分。
        selected_memory_ids：最终采用的 SQLite 记忆 ID 列表。
        reason：对本次召回结果的中文说明。

    返回值：
        MemoryRecallResult：经 Pydantic（数据校验库）验证的记忆召回结果。
    """

    status: MemoryRecallStatus
    memory_context: str = "暂无用户记忆"
    candidate_count: int = Field(default=0, ge=0)
    threshold_passed_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    semantic_threshold: float = Field(default=0.0, ge=0, le=1)
    max_semantic_score: float | None = Field(default=None, ge=0, le=1)
    selected_memory_ids: list[int] = Field(default_factory=list)
    reason: str


@dataclass
class MemoryRecord:
    """
    MemoryRecord：数据库中的记忆实体。

    作用：
    - 表示 SQLite user_memory 表中的一条记录
    - 后续也可以作为同步到 Chroma 的数据来源
    """

    id: int | None

    user_id: str

    memory_type: MemoryType

    content: str

    confidence: float

    strength: float

    status: MemoryStatus = "active"

    created_at: str | None = None

    last_seen: str | None = None

    source: MemorySource = "conversation"

    importance: float = 0.5

    updated_at: str | None = None

    expires_at: str | None = None
