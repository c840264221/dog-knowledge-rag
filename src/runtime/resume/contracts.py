from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


DEFAULT_RESUME_CHECKPOINT_NS = "main_graph"


class GraphRunStatus(str, Enum):
    """
    Graph 运行状态枚举。

    功能：
        描述一次 Graph 运行的最终状态，用于替代字符串前缀协议中的隐式状态。

    枚举值含义：
        COMPLETED:
            图运行已经正常完成。
        INTERRUPTED:
            图运行被 interrupt 中断，正在等待用户输入。
        FAILED:
            图运行失败。
    """

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class GraphInterruptType(str, Enum):
    """
    Graph 中断类型枚举。

    功能：
        描述当前 interrupt 的业务来源，方便 UI、API、日志和后续多 Agent 恢复流程判断。

    枚举值含义：
        TOOL_CONFIRMATION:
            工具调用前确认。
        USER_CLARIFICATION:
            需要用户补充或澄清问题。
        HUMAN_REVIEW:
            需要人工审核。
        UNKNOWN:
            未知中断类型，用作默认值。
    """

    TOOL_CONFIRMATION = "tool_confirmation"
    USER_CLARIFICATION = "user_clarification"
    HUMAN_REVIEW = "human_review"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GraphInterruptResult:
    """
    Graph 中断结果契约。

    功能：
        表示一次 Graph 执行因为 interrupt 暂停后，需要返回给 UI / API 的结构化结果。

    参数含义：
        prompt:
            展示给用户的中断提示。
        thread_id:
            LangGraph thread_id，用于后续恢复同一条图执行线程。
        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
        trace_id:
            当前请求链路追踪 ID，可选。
        interrupt_type:
            中断类型，默认 unknown。
        metadata:
            扩展元数据，例如 current_agent、current_node、tool_calls 等。

    返回值含义：
        GraphInterruptResult:
            不可变的结构化中断结果对象。

    输出格式：
        to_dict() 返回适合 UI / API / 日志使用的普通 dict。
    """

    prompt: str
    thread_id: str
    checkpoint_ns: str = DEFAULT_RESUME_CHECKPOINT_NS
    trace_id: str | None = None
    interrupt_type: GraphInterruptType = GraphInterruptType.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)
    status: GraphRunStatus = field(
        default=GraphRunStatus.INTERRUPTED,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        初始化后校验字段。

        功能：
            校验 prompt、thread_id、checkpoint_ns 非空，并复制 metadata，避免外部 dict 修改影响对象内部。

        参数含义：
            无。

        返回值含义：
            None。
        """

        object.__setattr__(
            self,
            "prompt",
            _require_non_empty_string(self.prompt, "prompt"),
        )
        object.__setattr__(
            self,
            "thread_id",
            _require_non_empty_string(self.thread_id, "thread_id"),
        )
        object.__setattr__(
            self,
            "checkpoint_ns",
            _require_non_empty_string(self.checkpoint_ns, "checkpoint_ns"),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        转换为普通字典。

        功能：
            将结构化中断结果转换成 UI / API / 日志可直接使用的 dict。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                当前中断结果的字典表示。
        """

        return {
            "status": self.status.value,
            "prompt": self.prompt,
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "trace_id": self.trace_id,
            "interrupt_type": self.interrupt_type.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphFinalResult:
    """
    Graph 最终完成结果契约。

    功能：
        表示一次 Graph 正常完成后返回给 UI / API 的结构化结果。

    参数含义：
        answer:
            最终答案文本。
        thread_id:
            LangGraph thread_id。
        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
        trace_id:
            当前请求链路追踪 ID，可选。
        metadata:
            扩展元数据。

    返回值含义：
        GraphFinalResult:
            不可变的结构化最终结果对象。
    """

    answer: str
    thread_id: str
    checkpoint_ns: str = DEFAULT_RESUME_CHECKPOINT_NS
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: GraphRunStatus = field(
        default=GraphRunStatus.COMPLETED,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        初始化后校验字段。

        功能：
            校验 answer、thread_id、checkpoint_ns 非空，并复制 metadata。

        参数含义：
            无。

        返回值含义：
            None。
        """

        object.__setattr__(
            self,
            "answer",
            _require_non_empty_string(self.answer, "answer"),
        )
        object.__setattr__(
            self,
            "thread_id",
            _require_non_empty_string(self.thread_id, "thread_id"),
        )
        object.__setattr__(
            self,
            "checkpoint_ns",
            _require_non_empty_string(self.checkpoint_ns, "checkpoint_ns"),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        转换为普通字典。

        功能：
            将最终完成结果转换成 UI / API / 日志可直接使用的 dict。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                当前最终结果的字典表示。
        """

        return {
            "status": self.status.value,
            "answer": self.answer,
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "trace_id": self.trace_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphResumeRequest:
    """
    Graph 恢复请求契约。

    功能：
        表示 UI / API 在用户确认后，发送给后端用于恢复 Graph 执行的结构化请求。

    参数含义：
        resume_value:
            用户确认或补充输入的值，会被后续转换为 Command(resume=resume_value)。
        thread_id:
            LangGraph thread_id，必须和中断时一致。
        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
        trace_id:
            当前请求链路追踪 ID，可选。
        metadata:
            扩展元数据。

    返回值含义：
        GraphResumeRequest:
            不可变的结构化恢复请求对象。
    """

    resume_value: str
    thread_id: str
    checkpoint_ns: str = DEFAULT_RESUME_CHECKPOINT_NS
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        初始化后校验字段。

        功能：
            校验 resume_value、thread_id、checkpoint_ns 非空，并复制 metadata。

        参数含义：
            无。

        返回值含义：
            None。
        """

        object.__setattr__(
            self,
            "resume_value",
            _require_non_empty_string(self.resume_value, "resume_value"),
        )
        object.__setattr__(
            self,
            "thread_id",
            _require_non_empty_string(self.thread_id, "thread_id"),
        )
        object.__setattr__(
            self,
            "checkpoint_ns",
            _require_non_empty_string(self.checkpoint_ns, "checkpoint_ns"),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        转换为普通字典。

        功能：
            将恢复请求转换成 UI / API / 日志可直接使用的 dict。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                当前恢复请求的字典表示。
        """

        return {
            "resume_value": self.resume_value,
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "trace_id": self.trace_id,
            "metadata": dict(self.metadata),
        }


def _require_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """
    校验非空字符串。

    功能：
        将字符串去除前后空白，并在字段为空或类型不正确时抛出 ValueError。

    参数含义：
        value:
            待校验的字段值。
        field_name:
            字段名称，用于错误信息。

    返回值含义：
        str:
            去除前后空白后的非空字符串。
    """

    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是非空字符串。")

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(f"{field_name} 必须是非空字符串。")

    return normalized_value
