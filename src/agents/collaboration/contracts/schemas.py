"""
V1.14 多 Agent 协作标准数据结构。

功能：
    规定复杂任务拆解后如何保存计划、步骤、执行结果和最终汇总。当前文件
    只负责数据契约和基础校验，不负责调用 Agent，也不负责修改 LangGraph。

专业名词：
    PlannerAgent：计划智能体，把复杂问题拆成多个步骤。
    Worker：工作智能体，实际执行某一个步骤的 Agent。
    Dependency：依赖关系，表示一个步骤必须等待其他步骤完成。
    Result Aggregator：结果聚合器，把多个步骤结果整理成统一回答。
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


AgentTaskStepStatus = Literal[
    "pending",
    "running",
    "awaiting_input",
    "completed",
    "failed",
    "skipped",
]

AgentTaskPlanStatus = Literal[
    "planned",
    "running",
    "awaiting_input",
    "completed",
    "partial",
    "failed",
    "cancelled",
]

AgentTaskResultStatus = Literal[
    "awaiting_input",
    "completed",
    "failed",
    "skipped",
]

AgentCollaborationStatus = Literal[
    "planned",
    "running",
    "awaiting_input",
    "completed",
    "partial",
    "failed",
    "cancelled",
]


class AgentTaskStep(BaseModel):
    """
    保存复杂任务中的一个执行步骤。

    功能：
        说明这一步要做什么、交给哪个 Agent、需要等待哪些前置步骤，以及
        期望得到什么结果。它描述的是计划，不保存真正的执行输出。

    参数含义：
        step_id:
            当前步骤在一份计划中的唯一编号。
        title:
            方便用户和开发者阅读的步骤名称。
        description:
            当前步骤需要完成的具体工作。
        assigned_agent:
            负责执行该步骤的 Agent 名称。
        depends_on:
            当前步骤必须等待完成的前置步骤编号。
        input_data:
            PlannerAgent 为当前步骤准备的结构化输入。
        expected_output:
            当前步骤预期产出什么内容。
        status:
            当前步骤处于等待、执行、完成或失败等哪种状态。
        allow_failure:
            这一步失败后，后续计划是否仍然允许继续。
        metadata:
            暂时没有固定字段的调试和扩展信息。

    返回值含义：
        AgentTaskStep:
            一项可以序列化并写入 checkpoint 的结构化任务步骤。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(..., min_length=1, description="计划内唯一步骤编号。")
    title: str = Field(..., min_length=1, description="步骤名称。")
    description: str = Field(default="", description="步骤工作说明。")
    assigned_agent: str = Field(
        ...,
        min_length=1,
        description="负责执行该步骤的 Agent 名称。",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="必须先完成的前置步骤编号。",
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="当前步骤的结构化输入。",
    )
    expected_output: str = Field(
        default="",
        description="当前步骤预期产出的内容。",
    )
    status: AgentTaskStepStatus = Field(
        default="pending",
        description="当前步骤执行状态。",
    )
    allow_failure: bool = Field(
        default=False,
        description="步骤失败后是否允许计划继续。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="步骤扩展信息。",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        """
        检查单个步骤声明的依赖是否合理。

        功能：
            阻止步骤依赖自己，也阻止同一个前置步骤被重复填写。其他步骤
            是否真实存在以及是否形成循环，由 AgentTaskPlan 统一检查。

        参数含义：
            self:
                当前已经完成字段校验的任务步骤。

        返回值含义：
            AgentTaskStep:
                依赖合法时返回当前步骤；不合法时抛出 ValueError。
        """

        if self.step_id in self.depends_on:
            raise ValueError("任务步骤不能依赖自己")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("任务步骤的 depends_on 不能包含重复编号")
        return self


class AgentTaskPlan(BaseModel):
    """
    保存 PlannerAgent 为一次复杂任务生成的完整计划。

    功能：
        把用户目标拆成有依赖关系的多个 AgentTaskStep，并检查步骤编号、
        依赖目标和依赖方向是否有效。它回答的是“准备怎么做”。

    参数含义：
        plan_id:
            当前计划的唯一编号。
        objective:
            这份计划最终要完成的用户目标。
        steps:
            按计划保存的全部任务步骤。
        status:
            整份计划当前的执行状态。
        reason:
            PlannerAgent 为什么采用这份拆解方案。
        requires_user_input:
            是否缺少关键信息，需要先向用户提问。
        clarification_prompt:
            需要用户补充信息时展示的问题。
        metadata:
            计划版本、Planner 来源等扩展信息。

    返回值含义：
        AgentTaskPlan:
            依赖完整且不存在循环的结构化任务计划。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    plan_id: str = Field(..., min_length=1, description="任务计划唯一编号。")
    objective: str = Field(..., min_length=1, description="计划最终目标。")
    steps: list[AgentTaskStep] = Field(
        ...,
        min_length=1,
        description="计划包含的任务步骤。",
    )
    status: AgentTaskPlanStatus = Field(
        default="planned",
        description="整份计划的执行状态。",
    )
    reason: str = Field(default="", description="采用当前拆解方案的原因。")
    requires_user_input: bool = Field(
        default=False,
        description="是否需要用户补充关键信息。",
    )
    clarification_prompt: str = Field(
        default="",
        description="需要用户补充信息时展示的问题。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="计划扩展信息。",
    )

    @model_validator(mode="after")
    def validate_plan_structure(self) -> Self:
        """
        检查整份计划的步骤编号和依赖关系。

        功能：
            确认步骤编号不重复、每个前置步骤真实存在、步骤之间没有循环
            依赖，并确保等待用户输入时提供了明确的问题。

        参数含义：
            self:
                当前已经完成字段校验的任务计划。

        返回值含义：
            AgentTaskPlan:
                计划结构合法时返回当前对象；不合法时抛出 ValueError。
        """

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("任务计划中的 step_id 不能重复")

        known_step_ids = set(step_ids)
        missing_dependencies = sorted(
            {
                dependency
                for step in self.steps
                for dependency in step.depends_on
                if dependency not in known_step_ids
            }
        )
        if missing_dependencies:
            raise ValueError(
                "任务计划引用了不存在的前置步骤: "
                f"{missing_dependencies}"
            )

        dependencies_by_step = {
            step.step_id: set(step.depends_on)
            for step in self.steps
        }
        steps_in_current_path: set[str] = set()
        fully_checked_steps: set[str] = set()

        def visit(step_id: str) -> None:
            """
            深度检查一个步骤能否沿依赖关系安全走到起点。

            参数含义：
                step_id:
                    当前正在检查的步骤编号。

            返回值含义：
                None。发现循环依赖时抛出 ValueError。
            """

            if step_id in fully_checked_steps:
                return
            if step_id in steps_in_current_path:
                raise ValueError("任务计划中存在循环依赖")
            steps_in_current_path.add(step_id)
            for dependency in dependencies_by_step[step_id]:
                visit(dependency)
            steps_in_current_path.remove(step_id)
            fully_checked_steps.add(step_id)

        for step_id in step_ids:
            visit(step_id)

        if self.requires_user_input and not self.clarification_prompt:
            raise ValueError("等待用户输入时必须提供 clarification_prompt")
        return self


class AgentTaskResult(BaseModel):
    """
    保存一个 Worker 实际执行任务步骤后的结果。

    功能：
        记录哪一个 Agent 执行了哪一步、是否成功、产出了什么内容以及失败
        原因。它回答的是“这一步实际做得怎样”。

    参数含义：
        step_id:
            对应 AgentTaskStep 的步骤编号。
        assigned_agent:
            实际执行该步骤的 Agent 名称。
        status:
            当前步骤是等待用户输入、完成、失败还是跳过。
        summary:
            方便最终聚合器阅读的结果摘要。
        output:
            Worker 返回的结构化业务数据。
        evidence_ids:
            支撑当前结果的证据编号。
        error_message:
            执行失败时的具体错误原因。
        requires_user_input:
            当前步骤是否需要暂停并等待用户补充信息或确认。
        clarification_prompt:
            当前步骤等待用户时需要展示的问题或确认提示。
        latency_ms:
            当前步骤执行耗时，单位毫秒。
        metadata:
            Worker 调用轨迹等扩展信息。

    返回值含义：
        AgentTaskResult:
            一个步骤的标准终态执行结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(..., min_length=1, description="对应的步骤编号。")
    assigned_agent: str = Field(
        ...,
        min_length=1,
        description="实际执行步骤的 Agent 名称。",
    )
    status: AgentTaskResultStatus = Field(
        ...,
        description="步骤最终执行状态。",
    )
    summary: str = Field(default="", description="步骤结果摘要。")
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="步骤结构化输出。",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="支撑当前结果的证据编号。",
    )
    error_message: str | None = Field(
        default=None,
        description="步骤失败原因。",
    )
    requires_user_input: bool = Field(
        default=False,
        description="当前步骤是否正在等待用户输入。",
    )
    clarification_prompt: str = Field(
        default="",
        description="当前步骤等待用户时展示的问题或确认提示。",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="步骤执行耗时，单位毫秒。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="步骤结果扩展信息。",
    )

    @model_validator(mode="after")
    def validate_result_details(self) -> Self:
        """
        检查失败结果和等待输入结果是否包含必要说明。

        功能：
            failed 必须携带 error_message；awaiting_input 必须明确标记需要
            用户输入并提供提示，避免调度器暂停后不知道该向用户问什么。

        参数含义：
            self:
                当前已经完成字段校验的步骤结果。

        返回值含义：
            AgentTaskResult:
                失败和等待输入信息符合状态要求时返回当前结果。
        """

        if self.status == "failed" and not self.error_message:
            raise ValueError("失败的任务结果必须提供 error_message")
        if self.status == "awaiting_input":
            if not self.requires_user_input:
                raise ValueError(
                    "awaiting_input 结果必须设置 requires_user_input=True"
                )
            if not self.clarification_prompt:
                raise ValueError(
                    "awaiting_input 结果必须提供 clarification_prompt"
                )
        elif self.requires_user_input:
            raise ValueError(
                "只有 awaiting_input 结果才能设置 requires_user_input=True"
            )
        return self


class MultiAgentClarificationField(BaseModel):
    """
    保存某个等待步骤缺少的一项结构化信息。

    功能：
        同时保留机器使用的字段编号和用户可读的中文名称，使调度器既能
        正确分配恢复输入，也能生成不会泄露内部字段细节的展示提示。

    参数含义：
        input_id:
            Skill 使用的稳定字段编号，例如 age。
        name:
            展示给用户的字段名称，例如“年龄”。
        description:
            该信息的用途说明。
        requirement_level:
            字段缺失时是必须补充、允许简化，还是未知来源的普通澄清。

    返回值含义：
        MultiAgentClarificationField:
            一项经过校验的多智能体缺失信息记录。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    input_id: str = Field(..., min_length=1)
    name: str = ""
    description: str = ""
    requirement_level: Literal[
        "hard_required",
        "degradable",
        "optional",
        "unknown",
    ] = "unknown"


class MultiAgentStepClarification(BaseModel):
    """
    保存一个等待步骤需要用户补充的完整信息。

    功能：
        明确记录“哪个步骤由哪个 Worker 执行、缺少哪些字段、原本准备向
        用户询问什么”，避免多个步骤的缺失信息在汇总时失去归属关系。

    参数含义：
        step_id:
            等待输入的步骤编号。
        step_title:
            面向用户和日志的步骤名称。
        assigned_agent:
            执行该步骤的 Worker Agent 名称。
        prompt:
            当前 Worker 返回的原始澄清提示。
        missing_fields:
            当前步骤缺少的结构化字段明细。
        can_run_degraded:
            当前步骤是否允许用户选择简化执行。

    返回值含义：
        MultiAgentStepClarification:
            一个步骤的标准澄清请求。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(..., min_length=1)
    step_title: str = Field(..., min_length=1)
    assigned_agent: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    missing_fields: list[MultiAgentClarificationField] = Field(
        default_factory=list
    )
    can_run_degraded: bool = False


class MultiAgentClarificationBundle(BaseModel):
    """
    保存同一调度批次的完整澄清信息包。

    功能：
        将逐步骤机器记录、字段使用者映射和面向用户的统一提示分开保存。
        字段可以在展示时合并，但每个字段属于哪些步骤的关系不会丢失。

    参数含义：
        step_requests:
            每个等待步骤各自的澄清请求。
        field_consumers:
            字段编号到所有需要该字段的步骤编号列表的映射。
        display_prompt:
            可以直接展示给用户的整批澄清提示。

    返回值含义：
        MultiAgentClarificationBundle:
            可写入任务 metadata、Checkpoint 和可观测系统的澄清包。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    step_requests: list[MultiAgentStepClarification] = Field(
        ...,
        min_length=1,
    )
    field_consumers: dict[str, list[str]] = Field(default_factory=dict)
    display_prompt: str = Field(..., min_length=1)


class MultiAgentClarificationExtractionResult(BaseModel):
    """
    保存一次多智能体澄清回答的字段提取结果。

    功能：
        只回答“用户补充了哪些字段”，不判断这些字段属于哪个步骤。分别
        保存本轮新值、跨轮合并值、缺失字段和歧义字段，供后续确定性分配。

    参数含义：
        requested_field_ids:
            当前等待任务允许提取的字段编号白名单。
        extracted_fields:
            本轮从自然语言中明确识别出的字段和值。
        resolved_fields:
            本轮字段与前几轮已识别字段合并后的结果。
        missing_field_ids:
            合并后仍然没有值的字段编号。
        ambiguous_field_ids:
            同一轮出现多个冲突候选值、暂时不能安全采用的字段编号。
        field_sources:
            每个已识别字段由哪些确定性规则提供。
        field_confidences:
            每个已识别字段的可信度；确定性高精度规则命中时为 1.0。
        coverage_ratio:
            已解决字段数量占申请字段数量的比例，不代表语义可信度。
        llm_fallback_used:
            确定性规则仍有缺失字段时是否调用过 LLM 兜底层。
        llm_reason:
            LLM 对本轮字段提取结果给出的简短原因。
        llm_error_message:
            LLM 调用或输出校验失败时的降级原因；成功时为空。
        rejected_field_ids:
            LLM 返回但不在本轮字段白名单中的字段编号。

    返回值含义：
        MultiAgentClarificationExtractionResult:
            字段提取层的标准输出，不包含步骤分配决策。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    requested_field_ids: list[str] = Field(default_factory=list)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    resolved_fields: dict[str, Any] = Field(default_factory=dict)
    missing_field_ids: list[str] = Field(default_factory=list)
    ambiguous_field_ids: list[str] = Field(default_factory=list)
    field_sources: dict[str, list[str]] = Field(default_factory=dict)
    field_confidences: dict[str, float] = Field(default_factory=dict)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_fallback_used: bool = False
    llm_reason: str = ""
    llm_error_message: str = ""
    rejected_field_ids: list[str] = Field(default_factory=list)


class MultiAgentStepResumeDecision(BaseModel):
    """
    保存一个等待步骤在本轮应该如何继续。

    功能：
        将正常恢复、简化执行和继续等待分开记录，并保留该步骤已经收到的
        输入、允许忽略的字段、决策原因和用户原话，供恢复、审计和观测使用。

    参数含义：
        step_id:
            当前决定所属的多智能体步骤编号。
        action:
            resume 表示输入齐全；degraded 表示用户同意忽略可简化字段；
            keep_waiting 表示仍有字段不能安全跳过。
        provided_inputs:
            已经确定分配给当前步骤的结构化输入。
        ignored_input_ids:
            简化执行时允许 Skill 忽略的字段编号。
        waiting_input_ids:
            当前仍会阻止该步骤执行、需要继续补充或明确处理的字段编号。
        reason:
            产生当前决定的稳定原因说明。
        user_input:
            触发本次决定的用户原始输入，便于后续复盘。

    返回值含义：
        MultiAgentStepResumeDecision:
            一个可写入 DogState 和 Checkpoint 的标准步骤恢复决定。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(..., min_length=1)
    action: Literal["resume", "degraded", "keep_waiting"]
    provided_inputs: dict[str, Any] = Field(default_factory=dict)
    ignored_input_ids: list[str] = Field(default_factory=list)
    waiting_input_ids: list[str] = Field(default_factory=list)
    reason: str = Field(..., min_length=1)
    user_input: str = ""


class MultiAgentTaskResult(BaseModel):
    """
    保存一次多 Agent 协作完成后的统一结果。

    功能：
        把原始任务计划、各步骤实际结果和最终回答放在同一对象中，供主图、
        checkpoint、调试报告和后续响应适配器共同使用。

    参数含义：
        collaboration_id:
            当前协作任务的唯一编号。
        plan:
            PlannerAgent 生成并在执行过程中更新的任务计划。
        status:
            整次协作任务的当前状态。
        task_results:
            已经得到的步骤执行结果。
        final_answer:
            Result Aggregator 整理后的最终回答。
        error_message:
            整次协作失败时的总体错误说明。
        metadata:
            Trace、版本和调试信息。

    返回值含义：
        MultiAgentTaskResult:
            可转换成普通字典的多 Agent 最终协作契约。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    collaboration_id: str = Field(
        ...,
        min_length=1,
        description="协作任务唯一编号。",
    )
    plan: AgentTaskPlan = Field(..., description="当前协作使用的任务计划。")
    status: AgentCollaborationStatus = Field(
        default="planned",
        description="整次协作任务状态。",
    )
    task_results: list[AgentTaskResult] = Field(
        default_factory=list,
        description="已产生的步骤执行结果。",
    )
    final_answer: str = Field(default="", description="协作后的最终回答。")
    error_message: str | None = Field(
        default=None,
        description="协作失败的总体错误说明。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="协作结果扩展信息。",
    )

    @model_validator(mode="after")
    def validate_result_structure(self) -> Self:
        """
        检查步骤结果是否与原任务计划对应。

        功能：
            阻止同一步骤产生重复结果，阻止结果引用计划外步骤，并检查实际
            Worker 是否与计划指定的 Agent 一致。完成状态还必须覆盖全部步骤。

        参数含义：
            self:
                当前已经完成字段校验的协作结果。

        返回值含义：
            MultiAgentTaskResult:
                结果与计划一致时返回当前对象；不一致时抛出 ValueError。
        """

        result_step_ids = [result.step_id for result in self.task_results]
        if len(result_step_ids) != len(set(result_step_ids)):
            raise ValueError("同一个任务步骤不能产生重复结果")

        planned_steps = {
            step.step_id: step
            for step in self.plan.steps
        }
        unknown_step_ids = sorted(set(result_step_ids) - set(planned_steps))
        if unknown_step_ids:
            raise ValueError(
                "协作结果引用了计划外步骤: "
                f"{unknown_step_ids}"
            )

        for result in self.task_results:
            planned_agent = planned_steps[result.step_id].assigned_agent
            if result.assigned_agent != planned_agent:
                raise ValueError(
                    f"步骤 {result.step_id} 的执行 Agent 与计划不一致"
                )

        if self.status == "completed":
            missing_result_ids = sorted(
                set(planned_steps) - set(result_step_ids)
            )
            if missing_result_ids:
                raise ValueError(
                    "已完成的协作仍缺少步骤结果: "
                    f"{missing_result_ids}"
                )
            incomplete_result_ids = sorted(
                result.step_id
                for result in self.task_results
                if result.status != "completed"
            )
            if incomplete_result_ids:
                raise ValueError(
                    "已完成的协作不能包含未完成步骤: "
                    f"{incomplete_result_ids}"
                )
            if not self.final_answer:
                raise ValueError("已完成的协作必须提供 final_answer")

        if self.status == "failed" and not self.error_message:
            raise ValueError("失败的协作必须提供 error_message")
        return self
