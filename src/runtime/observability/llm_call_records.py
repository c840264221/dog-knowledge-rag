"""LLM 调用说明与请求级调用明细契约。"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


LLMCallStatus = Literal["completed", "fallback", "failed"]


class LLMCallPurpose(str, Enum):
    """
    限制系统允许记录的 LLM 调用目的。

    功能：
        使用受控枚举代替任意字符串，避免不同调用方为同一种用途写出不同
        名称，导致后续成本统计无法稳定聚合。

    参数含义：
        无。

    返回值含义：
        LLMCallPurpose：可以写入 LLMCallMetadata 的标准调用目的。
    """

    UNSPECIFIED = "unspecified"
    MEMORY_EXTRACTION = "memory_extraction"
    PET_PROFILE_EXTRACTION = "pet_profile_extraction"
    QUERY_UNDERSTANDING = "query_understanding"
    ROUTING_DECISION = "routing_decision"
    TOOL_PLANNING = "tool_planning"
    TOOL_ANSWER_FORMATTING = "tool_answer_formatting"
    ANSWER_GENERATION = "answer_generation"
    MULTI_AGENT_PLANNING = "multi_agent_planning"
    MULTI_AGENT_AGGREGATION = "multi_agent_aggregation"
    CLARIFICATION_FIELD_EXTRACTION = "clarification_field_extraction"


class LLMCallMetadata(BaseModel):
    """
    描述一次 LLM 逻辑调用的业务身份。

    功能：
        由最了解调用目的的上游组件填写用途、组件、Agent 和多智能体步骤，
        统一传给 LLMProvider，避免不断扩展零散函数参数。

    参数含义：
        call_purpose：本次调用要完成的业务目的，例如 planning。
        component：发起调用的代码组件，例如 planner。
        agent_name：本次调用所属 Agent；不适用或未知时为空。
        step_id：本次调用所属多智能体步骤；不适用或未知时为空。
        extra：不适合提升为固定字段的少量附加信息。

    返回值含义：
        LLMCallMetadata：经过校验且可以安全写入调用记录的说明对象。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
    )

    call_purpose: LLMCallPurpose = LLMCallPurpose.UNSPECIFIED
    component: str = ""
    agent_name: str = ""
    step_id: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


def build_llm_call_metadata(
    *,
    purpose: LLMCallPurpose,
    component: str,
    agent_name: str = "",
    state: Mapping[str, Any] | None = None,
    step_id: str = "",
    extra: Mapping[str, Any] | None = None,
) -> LLMCallMetadata:
    """
    使用显式业务身份和可选 Worker 状态构建 LLM 调用说明。

    功能：
        调用方必须明确提供受控调用目的和组件名称。Agent 名称由调用方
        使用可信常量传入；步骤编号优先使用显式 step_id，否则只从当前
        Worker 的独立 state 中读取 multi_agent_step_id，不从共享运行时
        上下文猜测并行步骤身份。

    参数含义：
        purpose：受 LLMCallPurpose 白名单限制的调用目的。
        component：实际发起调用的代码组件名称。
        agent_name：调用所属 Agent 的可信名称；不适用时为空。
        state：当前调用可访问的图状态；只读取多智能体步骤身份字段。
        step_id：调用方已经明确掌握的步骤编号，优先级高于 state。
        extra：少量不适合提升为固定字段的附加观测信息。

    返回值含义：
        LLMCallMetadata：经过 Pydantic 校验的标准调用身份。
    """

    state_mapping = state if isinstance(state, Mapping) else {}
    resolved_step_id = str(
        step_id
        or state_mapping.get("multi_agent_step_id")
        or ""
    ).strip()
    return LLMCallMetadata(
        call_purpose=purpose,
        component=str(component or "").strip(),
        agent_name=str(agent_name or "").strip(),
        step_id=resolved_step_id,
        extra=dict(extra or {}),
    )


class LLMCallRecord(BaseModel):
    """
    保存一次 safe_ainvoke 逻辑调用的完整观测结果。

    功能：
        把调用身份、模型、内部尝试次数、最终状态、耗时和 Token 用量保存
        成统一结构。主模型重试和备用模型调用只增加 attempt_count，不会
        被错误统计成多个业务调用。

    参数含义：
        call_id：本次逻辑调用的唯一编号。
        trace_id：所属用户请求的追踪编号。
        metadata：调用用途和所属组件等业务身份。
        requested_model：调用方最初请求的模型名称。
        final_model：最终成功返回结果的模型名称；失败时为空。
        attempt_count：主模型与备用模型实际调用次数总和。
        backup_used：是否调用过备用模型。
        status：最终完成、使用兜底文本或彻底失败。
        latency_ms：整次逻辑调用总耗时，单位为毫秒。
        input_tokens：最终成功响应报告的输入 Token 数。
        output_tokens：最终成功响应报告的输出 Token 数。
        total_tokens：最终成功响应报告的 Token 总数。
        error_type：最终一次异常的类型名称；没有异常时为空。

    返回值含义：
        LLMCallRecord：可写入 MetricsScope 的标准调用明细。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = ""
    metadata: LLMCallMetadata = Field(default_factory=LLMCallMetadata)
    requested_model: str = ""
    final_model: str = ""
    attempt_count: int = Field(default=0, ge=0)
    backup_used: bool = False
    status: LLMCallStatus
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    error_type: str = ""
