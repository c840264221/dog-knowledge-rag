"""Redis 长任务 MVP 的统一数据契约。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


LongTaskStatus = Literal[
    "created",
    "queued",
    "running",
    "awaiting_input",
    "completed",
    "failed",
    "cancelled",
]
LongTaskStepStatus = Literal[
    "pending",
    "ready",
    "running",
    "awaiting_input",
    "completed",
    "failed",
    "skipped",
    "cancelled",
]
LongTaskExecutionMode = Literal["inline", "durable"]
LongTaskProgressionMode = Literal["automatic", "guided"]
LongTaskWaitingReason = Literal[
    "missing_input",
    "confirmation",
    "approval",
]
LongTaskEventActorType = Literal[
    "system",
    "user",
    "worker",
    "agent",
]
LongTaskQueueReason = Literal[
    "submitted",
    "continued",
    "resumed",
    "retry",
    "reconciled",
]
LongTaskBatchStepResultStatus = Literal[
    "completed",
    "awaiting_input",
    "failed",
    "skipped",
]
LongTaskExecutionAction = Literal[
    "advance_task",
    "await_user_input",
    "complete_task",
    "handle_failure",
    "promote_to_durable",
]


def utc_now() -> datetime:
    """
    返回带 UTC 时区的当前时间。

    返回值含义：
        datetime:
            可稳定序列化到数据库、Checkpoint 和 Redis 消息的 UTC 时间。
    """

    return datetime.now(timezone.utc)


class LongTaskContractModel(BaseModel):
    """为全部长任务契约提供严格、不可变的数据模型配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class LongTaskGoal(LongTaskContractModel):
    """
    保存原始用户请求和系统规范化后的执行目标。

    参数含义：
        original_request:
            用户最初的完整输入，用于审计和重新理解需求。
        objective:
            系统解析后的精简目标，供 Planner 和 Worker 执行。
        constraints:
            用户明确要求、禁止事项和业务边界等结构化约束。
        expected_output:
            整个任务最终应该产出什么。
        metadata:
            暂未固定结构的扩展信息。

    返回值含义：
        LongTaskGoal:
            同时保留原始事实和规范化执行目标的任务目标快照。
    """

    original_request: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTaskStep(LongTaskContractModel):
    """
    保存长任务中的一个可独立调度步骤。

    参数含义：
        step_id:
            当前步骤的全局唯一编号。
        task_id:
            当前步骤所属的长任务编号。
        title:
            方便用户和开发者阅读的步骤名称。
        description:
            当前步骤需要完成的具体工作。
        assigned_agent:
            负责执行该步骤的 Agent 或 Worker 类型。
        depends_on:
            当前步骤必须等待完成的前置步骤编号。
        input_data:
            当前步骤的结构化输入。
        expected_output:
            当前步骤预期返回的结果说明。
        status:
            当前步骤所处状态。
        waiting_reason:
            状态为 awaiting_input 时，说明在等待哪类用户输入。
        output_ref:
            大型结果对应的 Artifact 引用，不保存大型内容本身。
        output_summary:
            便于后续步骤快速阅读的结果摘要。
        attempt_count:
            已经开始执行的总次数。
        max_attempts:
            当前步骤允许执行的最大次数。
        version:
            供后续乐观锁使用的版本号。
        claimed_by:
            当前持有执行租约的 Worker 名称；仅运行中的后台步骤使用。
        claim_id:
            当前领取批次的唯一令牌，用于拒绝旧 Worker 的迟到结果。
        lease_expires_at:
            当前领取租约的 UTC 到期时间；到期后其他 Worker 可以接管。

    返回值含义：
        LongTaskStep:
            可以持久化、参与拓扑调度并写入 Checkpoint 的步骤快照。
    """

    step_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = ""
    assigned_agent: str = Field(..., min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    input_data: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""
    status: LongTaskStepStatus = "pending"
    waiting_reason: LongTaskWaitingReason | None = None
    output_ref: str | None = None
    output_summary: str = ""
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=1)
    version: int = Field(default=1, ge=1)
    claimed_by: str | None = Field(default=None, min_length=1)
    claim_id: str | None = Field(default=None, min_length=1)
    lease_expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step_contract(self) -> Self:
        """
        检查依赖、等待原因和执行次数是否符合步骤契约。

        返回值含义：
            LongTaskStep:
                结构合法时返回当前步骤；否则抛出 ValueError。
        """

        if self.step_id in self.depends_on:
            raise ValueError("长任务步骤不能依赖自己")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("depends_on 不能包含重复步骤编号")
        if self.status == "awaiting_input":
            if self.waiting_reason is None:
                raise ValueError(
                    "awaiting_input 步骤必须声明 waiting_reason"
                )
        elif self.waiting_reason is not None:
            raise ValueError(
                "只有 awaiting_input 步骤可以设置 waiting_reason"
            )
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count 不能超过 max_attempts")
        claim_values = (
            self.claimed_by,
            self.claim_id,
            self.lease_expires_at,
        )
        if any(value is not None for value in claim_values) and not all(
            value is not None for value in claim_values
        ):
            raise ValueError("Step 领取信息必须同时包含 Worker、令牌和租约")
        if self.claim_id is not None and self.status != "running":
            raise ValueError("只有 running Step 可以持有执行租约")
        if self.lease_expires_at is not None and (
            self.lease_expires_at.tzinfo is None
            or self.lease_expires_at.utcoffset() is None
        ):
            raise ValueError("lease_expires_at 必须包含时区")
        return self


class LongTaskPendingInteraction(LongTaskContractModel):
    """
    保存长任务暂停后正在等待的结构化用户交互。

    参数含义：
        interaction_id:
            当前等待交互的唯一编号。
        interaction_type:
            缺少输入、普通确认或人工批准。
        source_step_ids:
            导致这次交互产生的已执行或当前步骤编号。
        target_step_ids:
            用户完成交互后准备继续、重试或确认的步骤编号。
        prompt:
            展示给用户的明确问题。
        allowed_actions:
            当前交互允许的标准动作，例如 continue 或 cancel。
        input_contract:
            可选结构化输入要求；缺少业务参数时可声明字段和类型。

    返回值含义：
        LongTaskPendingInteraction:
            可以持久化并交给任务关系门禁处理的等待交互快照。
    """

    interaction_id: str = Field(..., min_length=1)
    interaction_type: LongTaskWaitingReason
    source_step_ids: list[str] = Field(default_factory=list)
    target_step_ids: list[str] = Field(default_factory=list)
    prompt: str = Field(..., min_length=1)
    allowed_actions: list[str] = Field(..., min_length=1)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interaction_contract(self) -> Self:
        """
        检查来源、目标和允许动作没有重复，并要求至少关联一个步骤。

        返回值含义：
            LongTaskPendingInteraction:
                交互结构合法时返回当前对象；否则抛出 ValueError。
        """

        if not self.source_step_ids and not self.target_step_ids:
            raise ValueError("等待交互必须至少关联一个来源或目标步骤")
        for field_name, values in (
            ("source_step_ids", self.source_step_ids),
            ("target_step_ids", self.target_step_ids),
            ("allowed_actions", self.allowed_actions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} 不能包含重复值")
        return self


class LongTask(LongTaskContractModel):
    """
    保存一份可内联、后台或人在回路推进的完整任务快照。

    参数含义：
        task_id:
            整份任务的全局唯一编号。
        user_id:
            创建任务的用户编号，用于权限与数据隔离。
        thread_id:
            创建任务时所属的对话线程编号。
        goal:
            原始请求和规范化目标。
        steps:
            任务包含的全部拓扑步骤。
        status:
            整体任务当前状态。
        execution_mode:
            inline 表示当前请求执行，durable 表示持久化后台执行。
        progression_mode:
            automatic 表示自动推进，guided 表示每步等待用户决定。
        pending_interaction:
            任务处于 awaiting_input 时正在等待的用户交互。
        active_step_ids:
            当前已经进入执行批次、正在运行或步骤内等待输入的步骤编号；
            只等待批准下一批时应为空。
        version:
            供 TaskStore 乐观锁使用的版本号。

    返回值含义：
        LongTask:
            依赖合法且可以安全持久化的完整任务快照。
    """

    task_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    goal: LongTaskGoal
    steps: list[LongTaskStep] = Field(..., min_length=1)
    status: LongTaskStatus = "created"
    execution_mode: LongTaskExecutionMode = "inline"
    progression_mode: LongTaskProgressionMode = "automatic"
    pending_interaction: LongTaskPendingInteraction | None = None
    active_step_ids: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task_contract(self) -> Self:
        """
        检查步骤归属、拓扑依赖、等待交互和活动步骤引用。

        返回值含义：
            LongTask:
                任务结构合法时返回当前任务；否则抛出 ValueError。
        """

        if self.status == "awaiting_input":
            if self.pending_interaction is None:
                raise ValueError(
                    "awaiting_input 任务必须声明 pending_interaction"
                )
        elif self.pending_interaction is not None:
            raise ValueError(
                "只有 awaiting_input 任务可以设置 pending_interaction"
            )

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("长任务中的 step_id 不能重复")

        mismatched_task_steps = [
            step.step_id
            for step in self.steps
            if step.task_id != self.task_id
        ]
        if mismatched_task_steps:
            raise ValueError(
                "步骤 task_id 与所属任务不一致: "
                f"{mismatched_task_steps}"
            )

        known_step_ids = set(step_ids)
        missing_dependencies = sorted(
            {
                dependency_id
                for step in self.steps
                for dependency_id in step.depends_on
                if dependency_id not in known_step_ids
            }
        )
        if missing_dependencies:
            raise ValueError(
                "长任务引用了不存在的依赖步骤: "
                f"{missing_dependencies}"
            )

        if len(self.active_step_ids) != len(
            set(self.active_step_ids)
        ):
            raise ValueError("active_step_ids 不能包含重复编号")
        unknown_active_steps = sorted(
            set(self.active_step_ids) - known_step_ids
        )
        if unknown_active_steps:
            raise ValueError(
                "active_step_ids 引用了不存在的步骤: "
                f"{unknown_active_steps}"
            )
        steps_by_id = {step.step_id: step for step in self.steps}
        inactive_step_ids = [
            step_id
            for step_id in self.active_step_ids
            if steps_by_id[step_id].status
            not in {"ready", "running", "awaiting_input"}
        ]
        if inactive_step_ids:
            raise ValueError(
                "active_step_ids 只能引用 ready、running 或 "
                "awaiting_input 步骤: "
                f"{inactive_step_ids}"
            )
        if self.status in {"completed", "cancelled"} and (
            self.active_step_ids
        ):
            raise ValueError("终态任务不能保留 active_step_ids")

        if self.pending_interaction is not None:
            interaction_step_ids = {
                *self.pending_interaction.source_step_ids,
                *self.pending_interaction.target_step_ids,
            }
            unknown_interaction_steps = sorted(
                interaction_step_ids - known_step_ids
            )
            if unknown_interaction_steps:
                raise ValueError(
                    "pending_interaction 引用了不存在的步骤: "
                    f"{unknown_interaction_steps}"
                )

        self._validate_acyclic_dependencies()
        return self

    def _validate_acyclic_dependencies(self) -> None:
        """
        使用深度优先搜索检查步骤依赖是否构成有向无环图。

        返回值含义：
            None。发现循环依赖时抛出 ValueError。
        """

        dependencies_by_step = {
            step.step_id: set(step.depends_on)
            for step in self.steps
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            """
            深度检查一个步骤的全部前置依赖。

            参数含义：
                step_id:
                    当前需要检查的步骤编号。

            返回值含义：
                None。当前路径重复访问同一步骤时抛出 ValueError。
            """

            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError("长任务步骤存在循环依赖")
            visiting.add(step_id)
            for dependency_id in dependencies_by_step[step_id]:
                visit(dependency_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for candidate_step_id in dependencies_by_step:
            visit(candidate_step_id)


class LongTaskBatchStepResult(LongTaskContractModel):
    """
    保存同一执行批次中一个步骤产生的中立结果事实。

    参数含义：
        step_id:
            当前结果对应的长任务步骤编号。
        status:
            步骤本轮是完成、等待输入、失败还是跳过。
        output_summary:
            供后续策略和步骤快速读取的小型结果摘要。
        output_ref:
            大型结果对应的 Artifact 引用，不在批次结果中保存正文。
        waiting_reason:
            等待用户时的结构化原因。
        user_prompt:
            等待用户时可以展示的明确问题。
        error_message:
            步骤失败时的具体错误说明。
        metadata:
            Agent、Worker、Trace 和耗时等非核心扩展信息。
        claim_id:
            产生当前结果时持有的 Step 领取令牌；后台租约执行时必填。

    返回值含义：
        LongTaskBatchStepResult:
            不依赖单 Agent 或多 Agent 实现的步骤执行结果。
    """

    step_id: str = Field(..., min_length=1)
    status: LongTaskBatchStepResultStatus
    output_summary: str = ""
    output_ref: str | None = None
    waiting_reason: LongTaskWaitingReason | None = None
    user_prompt: str = ""
    error_message: str | None = None
    claim_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_details(self) -> Self:
        """
        检查等待和失败状态携带了对应说明，其他状态没有残留字段。

        返回值含义：
            LongTaskBatchStepResult:
                状态与详细信息一致时返回当前结果；否则抛出 ValueError。
        """

        if self.status == "awaiting_input":
            if self.waiting_reason is None:
                raise ValueError(
                    "awaiting_input 批次步骤结果必须声明 waiting_reason"
                )
            if not self.user_prompt:
                raise ValueError(
                    "awaiting_input 批次步骤结果必须提供 user_prompt"
                )
        elif self.waiting_reason is not None or self.user_prompt:
            raise ValueError(
                "只有 awaiting_input 批次步骤结果可以设置等待信息"
            )

        if self.status == "failed":
            if not self.error_message:
                raise ValueError(
                    "failed 批次步骤结果必须提供 error_message"
                )
        elif self.error_message is not None:
            raise ValueError(
                "只有 failed 批次步骤结果可以设置 error_message"
            )
        return self


class LongTaskBatchResult(LongTaskContractModel):
    """
    保存单 Agent 或多 Agent 完成一个执行批次后的统一事实集合。

    参数含义：
        batch_id:
            当前执行批次的唯一编号。
        task_id:
            当前批次所属的长任务编号。
        step_results:
            本批次每个步骤各自的执行结果；单 Agent 批次只有一项。
        metadata:
            批次级 Trace、运行位置和耗时等扩展信息。

    返回值含义：
        LongTaskBatchResult:
            可交给后续 ExecutionDecision 计算下一动作的中立批次结果。
    """

    batch_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    step_results: list[LongTaskBatchStepResult] = Field(
        ...,
        min_length=1,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_step_results(self) -> Self:
        """
        检查同一批次不会为同一个步骤返回多份相互冲突的结果。

        返回值含义：
            LongTaskBatchResult:
                步骤编号唯一时返回当前批次；否则抛出 ValueError。
        """

        step_ids = [result.step_id for result in self.step_results]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("同一批次不能包含重复的 step_id")
        return self


class LongTaskExecutionContext(LongTaskContractModel):
    """
    保存一次同步执行决策需要的最小运行时计时信息。

    参数含义：
        elapsed_ms:
            当前请求已经用于执行该任务的毫秒数。
        inline_budget_ms:
            当前请求允许同步执行任务的最大毫秒数。

    返回值含义：
        LongTaskExecutionContext:
            可用于计算同步预算是否耗尽的不可变运行时上下文。
    """

    elapsed_ms: float = Field(..., ge=0)
    inline_budget_ms: float = Field(..., gt=0)


class LongTaskProjectedExecutionFacts(LongTaskContractModel):
    """
    保存把当前批次结果投影到完整任务后得到的策略事实。

    参数含义：
        task_id:
            当前事实所属的长任务编号。
        batch_id:
            产生当前事实的执行批次编号。
        execution_mode:
            当前任务仍处于请求内执行还是已经进入后台持久化执行。
        remaining_step_ids:
            投影后尚未成功结束、仍需处理的步骤编号。
        ready_step_ids:
            剩余步骤中依赖已经满足、可以进入下一批的步骤编号。
        completed_step_ids:
            投影后已经完成的步骤编号。
        awaiting_step_ids:
            投影后正在等待用户输入的步骤编号。
        failed_step_ids:
            投影后执行失败、需要治理的步骤编号。
        skipped_step_ids:
            投影后被明确跳过的步骤编号。
        all_steps_successfully_terminal:
            是否所有步骤都以 completed 或 skipped 成功收尾。
        inline_budget_exhausted:
            当前同步执行耗时是否已经达到预算上限。

    返回值含义：
        LongTaskProjectedExecutionFacts:
            不修改真实 Task、可以安全交给 PDP 的预测事实快照。
    """

    task_id: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    execution_mode: LongTaskExecutionMode
    remaining_step_ids: list[str] = Field(default_factory=list)
    ready_step_ids: list[str] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)
    awaiting_step_ids: list[str] = Field(default_factory=list)
    failed_step_ids: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    all_steps_successfully_terminal: bool
    inline_budget_exhausted: bool

    @model_validator(mode="after")
    def validate_projected_facts(self) -> Self:
        """
        检查投影事实中的步骤分类、Ready 子集和完成标记彼此一致。

        返回值含义：
            LongTaskProjectedExecutionFacts:
                事实集合一致时返回当前对象；否则抛出 ValueError。
        """

        status_groups = (
            self.completed_step_ids,
            self.awaiting_step_ids,
            self.failed_step_ids,
            self.skipped_step_ids,
        )
        classified_step_ids = [
            step_id
            for step_ids in status_groups
            for step_id in step_ids
        ]
        if len(classified_step_ids) != len(set(classified_step_ids)):
            raise ValueError("同一步骤不能出现在多个预测状态分类中")
        if not set(self.ready_step_ids).issubset(
            self.remaining_step_ids
        ):
            raise ValueError("ready_step_ids 必须属于 remaining_step_ids")
        if self.all_steps_successfully_terminal:
            if self.remaining_step_ids:
                raise ValueError("全部成功结束时不能保留 remaining_step_ids")
            if self.awaiting_step_ids or self.failed_step_ids:
                raise ValueError("全部成功结束时不能包含等待或失败步骤")
        return self


class LongTaskExecutionDecision(LongTaskContractModel):
    """
    保存批次执行完成后由纯策略计算出的下一动作。

    参数含义：
        task_id:
            当前决策所属的长任务编号。
        batch_id:
            触发当前决策的执行批次编号。
        action:
            外层下一步应该推进任务、等待用户或处理失败。
        reason:
            供日志、调试报告和开发者理解的决策原因。
        completed_step_ids:
            当前批次中已经完成的步骤编号。
        awaiting_step_ids:
            当前批次中正在等待用户的步骤编号。
        failed_step_ids:
            当前批次中执行失败的步骤编号。
        skipped_step_ids:
            当前批次中被跳过的步骤编号。
        remaining_step_ids:
            投影后仍需处理的步骤编号。
        ready_step_ids:
            外层提交当前批次后可以考虑调度的步骤编号。
        inline_budget_exhausted:
            做出当前决策时同步执行预算是否耗尽。

    返回值含义：
        LongTaskExecutionDecision:
            只描述下一动作、不执行状态修改或外部调用的策略结果。
    """

    task_id: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    action: LongTaskExecutionAction
    reason: str = Field(..., min_length=1)
    completed_step_ids: list[str] = Field(default_factory=list)
    awaiting_step_ids: list[str] = Field(default_factory=list)
    failed_step_ids: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    remaining_step_ids: list[str] = Field(default_factory=list)
    ready_step_ids: list[str] = Field(default_factory=list)
    inline_budget_exhausted: bool = False

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """
        检查动作与阻塞步骤一致，并保证每个步骤只属于一种结果状态。

        返回值含义：
            LongTaskExecutionDecision:
                动作与步骤分类一致时返回当前决策；否则抛出 ValueError。
        """

        step_id_groups = (
            self.completed_step_ids,
            self.awaiting_step_ids,
            self.failed_step_ids,
            self.skipped_step_ids,
        )
        all_step_ids = [
            step_id
            for step_ids in step_id_groups
            for step_id in step_ids
        ]
        if len(all_step_ids) != len(set(all_step_ids)):
            raise ValueError("同一步骤不能出现在多个决策结果分类中")

        if self.action == "handle_failure":
            if not self.failed_step_ids:
                raise ValueError(
                    "handle_failure 决策必须包含 failed_step_ids"
                )
        elif self.failed_step_ids:
            raise ValueError(
                "存在失败步骤时 action 必须为 handle_failure"
            )

        if self.action == "await_user_input":
            if not self.awaiting_step_ids:
                raise ValueError(
                    "await_user_input 决策必须包含 awaiting_step_ids"
                )
        elif self.awaiting_step_ids and not self.failed_step_ids:
            raise ValueError(
                "仅存在等待步骤时 action 必须为 await_user_input"
            )

        if self.action == "advance_task" and (
            self.failed_step_ids or self.awaiting_step_ids
        ):
            raise ValueError("advance_task 决策不能包含阻塞步骤")
        if self.action == "complete_task":
            if self.remaining_step_ids:
                raise ValueError("complete_task 决策不能包含剩余步骤")
            if self.failed_step_ids or self.awaiting_step_ids:
                raise ValueError("complete_task 决策不能包含阻塞步骤")
        if self.action == "promote_to_durable":
            if not self.remaining_step_ids:
                raise ValueError(
                    "promote_to_durable 决策必须包含剩余步骤"
                )
            if not self.inline_budget_exhausted:
                raise ValueError(
                    "promote_to_durable 决策要求同步预算已经耗尽"
                )
            if self.failed_step_ids or self.awaiting_step_ids:
                raise ValueError(
                    "promote_to_durable 决策不能绕过阻塞步骤"
                )
        if not set(self.ready_step_ids).issubset(
            self.remaining_step_ids
        ):
            raise ValueError("ready_step_ids 必须属于 remaining_step_ids")
        return self


class LongTaskEvent(LongTaskContractModel):
    """
    保存一条不可变的长任务业务事件。

    参数含义：
        event_id:
            当前事件的唯一编号。
        task_id:
            事件所属任务编号。
        step_id:
            可选的关联步骤编号。
        sequence:
            同一任务内单调递增的事件序号。
        event_type:
            已经发生的业务事实类型，例如 step_completed。
        actor_type:
            产生事件的是系统、用户、Worker 还是 Agent。
        actor_id:
            可选的具体操作者编号。
        payload:
            与当前事件有关的小型结构化数据或 Artifact 引用。

    返回值含义：
        LongTaskEvent:
            可追加到 EventStore 的不可变业务事实。
    """

    event_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    step_id: str | None = None
    sequence: int = Field(..., ge=1)
    event_type: str = Field(..., min_length=1)
    actor_type: LongTaskEventActorType
    actor_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    schema_version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class LongTaskQueueMessage(LongTaskContractModel):
    """
    保存 Redis Streams 中的轻量长任务通知。

    参数含义：
        task_id:
            Worker 收到消息后需要从 TaskStore 加载的任务编号。
        task_version:
            任务入队时的版本号，用于识别陈旧消息。
        reason:
            当前消息因首次提交、继续、恢复、重试或对账而产生。
        ready_step_ids:
            入队时已经满足依赖的步骤编号提示；Worker 仍需查询数据库复核。
        correlation_id:
            可选请求链路编号。

    返回值含义：
        LongTaskQueueMessage:
            体积小、可以编码后写入 Redis Stream 的任务通知。
    """

    task_id: str = Field(..., min_length=1)
    task_version: int = Field(..., ge=1)
    reason: LongTaskQueueReason
    ready_step_ids: list[str] = Field(default_factory=list)
    correlation_id: str | None = None
    enqueued_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_ready_step_ids(self) -> Self:
        """
        检查入队消息中的可执行步骤编号没有重复。

        返回值含义：
            LongTaskQueueMessage:
                消息合法时返回当前对象；否则抛出 ValueError。
        """

        if len(self.ready_step_ids) != len(set(self.ready_step_ids)):
            raise ValueError("ready_step_ids 不能包含重复编号")
        return self
