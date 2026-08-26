from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ChatRequest(BaseModel):
    """
    表示一次新的 Agent 对话请求。

    功能：
        校验用户问题、会话编号和可选链路追踪编号，作为 API 到主图的
        标准输入契约。

    参数含义：
        question:
            用户本轮提出的问题。
        session_id:
            连续对话使用的会话编号，同时作为 LangGraph thread_id。
        trace_id:
            可选链路追踪编号。调用方不提供时由服务端生成。

    返回值含义：
        ChatRequest:
            通过 Pydantic 校验后的新对话请求对象。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=200)
    trace_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("question", "session_id", "trace_id")
    @classmethod
    def validate_non_blank_string(
        cls,
        value: str | None,
    ) -> str | None:
        """
        拒绝只包含空白字符的字符串。

        参数含义：
            value:
                当前准备校验的字符串或 None。

        返回值含义：
            str | None:
                原始非空字符串，或者允许缺省时的 None。
        """

        if value is not None and not value.strip():
            raise ValueError("字段不能只包含空白字符")
        return value


class ResumeRequest(BaseModel):
    """
    表示恢复一条已中断主图的请求。

    功能：
        保存用户补充内容以及中断时使用的 session_id、trace_id，使主图能
        依靠同一个 thread_id 找回检查点并继续执行。

    参数含义：
        resume_value:
            用户对确认问题或补充问题的回答。
        session_id:
            中断请求使用的原会话编号。
        trace_id:
            中断请求使用的原链路追踪编号。

    返回值含义：
        ResumeRequest:
            通过 Pydantic 校验后的恢复请求对象。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    resume_value: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=200)
    trace_id: str = Field(min_length=1, max_length=200)

    @field_validator("resume_value", "session_id", "trace_id")
    @classmethod
    def validate_non_blank_string(cls, value: str) -> str:
        """
        拒绝只包含空白字符的恢复字段。

        参数含义：
            value:
                当前准备校验的字符串。

        返回值含义：
            str:
                原始非空字符串。
        """

        if not value.strip():
            raise ValueError("字段不能只包含空白字符")
        return value


class AgentBusinessError(BaseModel):
    """
    表示 Agent 业务任务失败或取消的结构化原因。

    功能：
        在 HTTP 请求正常结束时，向调用方说明业务任务为什么没有成功，
        避免前端通过分析自然语言 answer 猜测超时、失败或取消原因。

    参数含义：
        code:
            稳定业务错误码。
        message:
            可安全展示的业务结果说明。
        details:
            可选步骤编号、超时秒数和尝试次数等结构化详情。

    返回值含义：
        AgentBusinessError:
            GraphRunResponse 中可选的业务错误对象。
    """

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class LongTaskHandoffResponse(BaseModel):
    """
    表示统一聊天请求已经转入持久化后台执行。

    参数含义：
        task_id:
            前端查询和订阅时使用的长任务编号。
        task_version:
            首次交接成功后的权威快照版本。
        status:
            当前交接完成时的长任务状态。
        execution_mode:
            durable 表示后续由持久化后台 Worker 执行。
        status_url:
            查询最新 LongTask 快照的相对 API 地址。
        events_url:
            订阅 LongTask 状态变化的相对 SSE 地址。

    返回值含义：
        LongTaskHandoffResponse:
            前端无需解析自然语言即可继续查询和订阅的后台任务引用。
    """

    task_id: str = Field(min_length=1)
    task_version: int = Field(ge=1)
    status: Literal["running"]
    execution_mode: Literal["durable"]
    status_url: str = Field(min_length=1)
    events_url: str = Field(min_length=1)


class GraphRunResponse(BaseModel):
    """
    表示主图完成或中断后返回给 API 调用方的统一结果。

    功能：
        把内部 GraphFinalResult / GraphInterruptResult 转换成稳定的 HTTP
        JSON 结构，让前端不必直接依赖 Python dataclass。

    参数含义：
        status:
            completed 表示完成，interrupted 表示等待用户输入。
        business_status:
            Agent 业务结果状态，与 status 表示的 API/主图执行状态相互独立。
        business_error:
            业务失败或取消时的结构化原因。
        long_task:
            请求已经转入后台执行时返回的持久化任务引用。
        answer:
            图正常完成时的最终答案。
        prompt:
            图中断时需要展示给用户的提示。
        session_id:
            当前会话编号。
        thread_id:
            LangGraph 检查点使用的线程编号，当前与 session_id 相同。
        trace_id:
            当前请求的链路追踪编号。
        multi_agent_task_id:
            可用于发送多 Agent 取消请求的任务编号。
        checkpoint_ns:
            LangGraph 检查点命名空间。
        interrupt_type:
            中断业务类型，例如工具确认或用户信息补充。
        metadata:
            主图返回的扩展调试信息。

    返回值含义：
        GraphRunResponse:
            可被 FastAPI 自动序列化为 JSON 的响应对象。
    """

    status: Literal["completed", "interrupted"]
    business_status: Literal[
        "completed",
        "partial",
        "failed",
        "cancelled",
        "awaiting_input",
        "running",
    ]
    business_error: AgentBusinessError | None = None
    long_task: LongTaskHandoffResponse | None = None
    answer: str | None = None
    prompt: str | None = None
    session_id: str
    thread_id: str
    trace_id: str
    multi_agent_task_id: str
    checkpoint_ns: str
    interrupt_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CancellationResponse(BaseModel):
    """
    表示一次多 Agent 取消信号的发送结果。

    功能：
        明确区分“已经找到任务并发送取消信号”和“当前进程没有该运行中任务”。

    参数含义：
        multi_agent_task_id:
            调用方准备取消的任务编号。
        cancellation_requested:
            是否成功找到任务并打开取消令牌。
        message:
            面向调用方的处理说明。

    返回值含义：
        CancellationResponse:
            可被 FastAPI 序列化的取消结果。
    """

    multi_agent_task_id: str
    cancellation_requested: bool
    message: str


class HealthResponse(BaseModel):
    """
    表示 API 存活或就绪检查结果。

    功能：
        为开发环境、容器平台和负载均衡器提供稳定的服务状态 JSON。

    参数含义：
        status:
            当前检查结果，ok 表示进程存活，ready 表示依赖已经启动。
        service:
            当前服务名称。

    返回值含义：
        HealthResponse:
            健康检查响应对象。
    """

    status: Literal["ok", "ready"]
    service: str


class TaskStatusResponse(BaseModel):
    """
    表示一次 API 请求在当前服务进程中的运行状态。

    功能：
        让调用方可以根据 multi_agent_task_id 查询请求是否仍在执行、已经
        完成、等待用户输入、收到取消请求或执行失败。

    参数含义：
        multi_agent_task_id:
            根据 trace_id 构建的任务编号。
        trace_id:
            当前请求链路追踪编号。
        session_id:
            当前连续会话编号。
        status:
            当前 API 请求生命周期状态。
        business_status:
            主图产生结果后的 Agent 业务状态；任务运行中时为 None。
        created_at:
            任务登记时间，使用 UTC ISO 8601 格式。
        updated_at:
            最近一次状态更新时间，使用 UTC ISO 8601 格式。
        error_message:
            执行失败时保存的非敏感错误摘要。

    返回值含义：
        TaskStatusResponse:
            可被 FastAPI 序列化为 JSON 的任务状态快照。
    """

    multi_agent_task_id: str
    trace_id: str
    session_id: str
    status: Literal[
        "running",
        "completed",
        "interrupted",
        "cancel_requested",
        "failed",
    ]
    business_status: Literal[
        "completed",
        "partial",
        "failed",
        "cancelled",
        "awaiting_input",
        "running",
    ] | None = None
    created_at: str
    updated_at: str
    error_message: str | None = None


class LongTaskStepStatusResponse(BaseModel):
    """
    表示一个长任务步骤可安全展示给用户的状态摘要。

    参数含义：
        step_id:
            步骤唯一编号。
        title:
            面向用户的步骤标题。
        status:
            步骤当前生命周期状态。
        depends_on:
            当前步骤依赖的前置步骤编号。
        waiting_reason:
            步骤暂停时等待输入、确认或批准的原因。
        output_summary:
            已完成步骤可公开展示的短结果摘要。
        attempt_count:
            步骤已经开始执行的次数。

    返回值含义：
        LongTaskStepStatusResponse:
            不包含租约、领取令牌和内部输入数据的公开步骤快照。
    """

    step_id: str
    title: str
    status: Literal[
        "pending",
        "ready",
        "running",
        "awaiting_input",
        "completed",
        "failed",
        "skipped",
        "cancelled",
    ]
    depends_on: list[str] = Field(default_factory=list)
    waiting_reason: Literal[
        "missing_input",
        "confirmation",
        "approval",
    ] | None = None
    output_summary: str = ""
    attempt_count: int = Field(ge=0)


class LongTaskInteractionResponse(BaseModel):
    """
    表示长任务当前等待前端处理的一次结构化交互。

    参数含义：
        interaction_id:
            本次交互唯一编号，后续提交响应时用于防止迟到或重复回答。
        interaction_type:
            当前等待的是缺少输入、普通确认还是人工批准。
        prompt:
            前端可以直接展示给用户的问题。
        allowed_actions:
            本次交互允许提交的标准动作。
        input_contract:
            前端构建输入控件或逐步骤回答时使用的结构化要求。

    返回值含义：
        LongTaskInteractionResponse:
            不包含内部诊断 metadata 的公开交互快照。
    """

    interaction_id: str
    interaction_type: Literal[
        "missing_input",
        "confirmation",
        "approval",
    ]
    prompt: str
    allowed_actions: list[str]
    input_contract: dict[str, Any] = Field(default_factory=dict)


class LongTaskStatusResponse(BaseModel):
    """
    表示持久化长任务可安全返回给前端的最新状态。

    参数含义：
        task_id:
            长任务唯一编号。
        thread_id:
            创建任务时所属的对话线程编号。
        objective:
            系统规范化后的任务目标，不返回原始完整用户输入。
        status:
            整份任务当前状态。
        execution_mode:
            inline 为请求内执行，durable 为持久化后台执行。
        progression_mode:
            automatic 为自动推进，guided 为人工确认推进。
        version:
            当前权威任务快照版本。
        total_step_count:
            任务包含的步骤总数。
        completed_step_count:
            已完成或按策略跳过的步骤数量。
        steps:
            全部步骤的公开状态摘要。
        pending_interaction:
            当前需要用户处理的交互；没有等待项时为 None。
        created_at、updated_at:
            任务创建和最近更新时间。

    返回值含义：
        LongTaskStatusResponse:
            不暴露 Worker、租约、领取令牌和内部 metadata 的状态响应。
    """

    task_id: str
    thread_id: str
    objective: str
    status: Literal[
        "created",
        "queued",
        "running",
        "awaiting_input",
        "completed",
        "failed",
        "cancelled",
    ]
    execution_mode: Literal["inline", "durable"]
    progression_mode: Literal["automatic", "guided"]
    version: int = Field(ge=1)
    total_step_count: int = Field(ge=1)
    completed_step_count: int = Field(ge=0)
    steps: list[LongTaskStepStatusResponse]
    pending_interaction: LongTaskInteractionResponse | None = None
    created_at: str
    updated_at: str


class LongTaskInteractionCommandRequest(BaseModel):
    """
    表示前端对长任务缺少输入交互提交的命令。

    参数含义：
        user_id:
            当前调用方声明的用户编号；当前 MVP 用于任务归属校验。
        action:
            submit_input 表示提交逐 Step 回答，cancel 表示取消整份任务。
        answers:
            以等待 Step 编号为键的回答；提交输入时必须非空，取消时必须为空。

    返回值含义：
        LongTaskInteractionCommandRequest:
            已完成动作与回答组合校验的交互命令对象。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=200)
    action: Literal["submit_input", "cancel"]
    answers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action_answers(self) -> "LongTaskInteractionCommandRequest":
        """
        检查提交输入必须携带回答，而取消操作不能混入回答数据。

        返回值含义：
            LongTaskInteractionCommandRequest:
                动作与 answers 组合合法时返回当前请求，否则抛出 ValueError。
        """

        if self.action == "submit_input" and not self.answers:
            raise ValueError("submit_input 必须提供 answers")
        if self.action == "cancel" and self.answers:
            raise ValueError("cancel 不能同时提供 answers")
        if any(not str(step_id).strip() for step_id in self.answers):
            raise ValueError("answers 不能包含空白 step_id")
        return self


class ApiErrorDetail(BaseModel):
    """
    表示 API 错误的稳定业务描述。

    功能：
        使用机器可判断的 code 和面向用户的 message 表达错误，并允许参数
        校验失败时附加不包含敏感输入值的字段详情。

    参数含义：
        code:
            稳定错误编号，前端应根据它决定处理方式。
        message:
            可以安全展示给调用方的错误说明。
        details:
            可选结构化详情，例如错误字段位置和校验类型。

    返回值含义：
        ApiErrorDetail:
            统一错误响应中的 error 对象。
    """

    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ApiErrorResponse(BaseModel):
    """
    表示所有非 SSE HTTP 错误的统一响应。

    功能：
        让参数错误、资源不存在、业务异常和系统异常使用相同 JSON 外壳，
        并携带 trace_id 方便调用方与服务端日志关联。

    参数含义：
        status:
            固定为 error。
        error:
            机器错误码、公开说明和可选详情。
        trace_id:
            当前 HTTP 请求链路编号。

    返回值含义：
        ApiErrorResponse:
            可被 FastAPI 序列化为 JSON 的统一错误响应。
    """

    status: Literal["error"] = "error"
    error: ApiErrorDetail
    trace_id: str
