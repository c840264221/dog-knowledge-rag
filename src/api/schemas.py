from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    ]
    business_error: AgentBusinessError | None = None
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
    ] | None = None
    created_at: str
    updated_at: str
    error_message: str | None = None


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
