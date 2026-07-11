"""
ToolAgent Agent 层响应契约。

功能：
    定义 ToolAgent（工具智能体）在编排工具调用流程时使用的标准数据结构。

设计原则：
    1. Agent Schema（智能体数据结构）描述工具调用流程。
    2. Tool Schema（工具数据结构）描述单个工具如何调用和返回。
    3. Agent Schema 组合 ToolCall / ToolResult，而不是重复定义 name、args、content 等字段。
    4. 所有对象都可以通过 model_dump 转换成普通 dict，避免 checkpoint 保存自定义对象时产生兼容风险。

专业名词：
    Schema：数据结构 / 数据模型，用于约束字段和类型。
    ToolCall：工具调用请求，表示工具名称和参数。
    ToolResult：工具执行结果，表示工具是否成功、返回内容和错误信息。
    Orchestration：编排，表示 ToolAgent 如何组织意图、权限、执行和回答。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


ToolAgentPermissionStatus = Literal[
    "not_required",
    "pending",
    "confirmed",
    "rejected",
]

ToolAgentResponseStatus = Literal[
    "no_tool",
    "awaiting_clarification",
    "pending_confirmation",
    "completed",
    "failed",
    "cancelled",
]


class ToolAgentIntent(BaseModel):
    """
    ToolAgent 工具意图契约。

    功能：
        描述用户问题是否需要工具，以及候选工具和判断原因。

    参数：
        need_tool:
            是否需要调用工具。

        candidate_tools:
            候选工具名称列表，例如 weather、date、time。

        reason:
            判断需要或不需要工具的原因。

    返回值：
        ToolAgentIntent:
            Pydantic 数据对象，可通过 model_dump 转成 dict。
    """

    need_tool: bool = Field(
        default=False,
        description="是否需要调用工具",
    )
    candidate_tools: list[str] = Field(
        default_factory=list,
        description="候选工具名称列表",
    )
    reason: str = Field(
        default="",
        description="工具意图判断原因",
    )


class ToolAgentPlannedCall(BaseModel):
    """
    ToolAgent 计划工具调用契约。

    功能：
        描述 ToolAgent 计划执行的一次工具调用，并组合底层 ToolCall。

    参数：
        call_id:
            本次计划调用的唯一 ID，用于把计划调用和执行记录关联起来。

        tool_call:
            ToolCall（工具调用请求），来自底层工具 schema，包含 name 和 args。

        requires_confirmation:
            是否需要用户确认。

        reason:
            为什么计划调用该工具。

    返回值：
        ToolAgentPlannedCall:
            Pydantic 数据对象，可通过 model_dump 转成 dict。
    """

    call_id: str = Field(
        description="计划工具调用 ID",
    )
    tool_call: ToolCall = Field(
        description="底层工具调用请求",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="是否需要用户确认",
    )
    reason: str = Field(
        default="",
        description="计划调用该工具的原因",
    )


class ToolAgentPermissionDecision(BaseModel):
    """
    ToolAgent 工具权限决定契约。

    功能：
        描述一次工具调用是否需要确认，以及用户确认后的状态。

    参数：
        status:
            权限状态，可选 not_required、pending、confirmed、rejected。

        call_ids:
            本次权限决定影响的计划调用 ID 列表。

        prompt:
            需要展示给用户的确认提示。

        reason:
            权限判断原因。

    返回值：
        ToolAgentPermissionDecision:
            Pydantic 数据对象，可通过 model_dump 转成 dict。
    """

    status: ToolAgentPermissionStatus = Field(
        default="not_required",
        description="工具权限状态",
    )
    call_ids: list[str] = Field(
        default_factory=list,
        description="该权限决定影响的计划调用 ID 列表",
    )
    prompt: str = Field(
        default="",
        description="用户确认提示",
    )
    reason: str = Field(
        default="",
        description="权限判断原因",
    )


class ToolAgentExecutionRecord(BaseModel):
    """
    ToolAgent 工具执行记录契约。

    功能：
        描述某个计划工具调用的执行结果，并组合底层 ToolResult。

    参数：
        call_id:
            对应 ToolAgentPlannedCall 的调用 ID。

        tool_result:
            ToolResult（工具执行结果），来自底层工具 schema。

        duration_ms:
            ToolAgent 视角记录的执行耗时，单位毫秒。

        metadata:
            附加调试信息。

    返回值：
        ToolAgentExecutionRecord:
            Pydantic 数据对象，可通过 model_dump 转成 dict。
    """

    call_id: str = Field(
        description="对应计划工具调用 ID",
    )
    tool_result: ToolResult = Field(
        description="底层工具执行结果",
    )
    duration_ms: int | None = Field(
        default=None,
        description="执行耗时，单位毫秒",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="附加调试信息",
    )


class ToolAgentResponse(BaseModel):
    """
    ToolAgent 最终响应契约。

    功能：
        描述 ToolAgent 完成一轮工具任务后的标准输出。

    参数：
        status:
            ToolAgent 当前响应状态，例如 completed、failed、pending_confirmation。

        intent:
            工具意图结果。

        planned_calls:
            计划工具调用列表。

        permission:
            权限确认结果。

        execution_records:
            工具执行记录列表。

        final_answer:
            ToolAgent 生成或交给主图使用的最终回答文本。

        metadata:
            附加调试信息。

    返回值：
        ToolAgentResponse:
            Pydantic 数据对象，可通过 model_dump 转成 dict。
    """

    status: ToolAgentResponseStatus = Field(
        default="no_tool",
        description="ToolAgent 响应状态",
    )
    intent: ToolAgentIntent = Field(
        default_factory=ToolAgentIntent,
        description="工具意图结果",
    )
    planned_calls: list[ToolAgentPlannedCall] = Field(
        default_factory=list,
        description="计划工具调用列表",
    )
    permission: ToolAgentPermissionDecision = Field(
        default_factory=ToolAgentPermissionDecision,
        description="工具权限决定",
    )
    execution_records: list[ToolAgentExecutionRecord] = Field(
        default_factory=list,
        description="工具执行记录列表",
    )
    final_answer: str = Field(
        default="",
        description="最终回答文本",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="附加调试信息",
    )
