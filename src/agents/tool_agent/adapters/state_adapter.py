"""
ToolAgent 旧 state 适配器。

功能：
    将当前主图中已有的工具相关 state 字段，转换成 V1.8 ToolAgentResponse 契约。

设计目标：
    1. 不修改现有主图。
    2. 不执行真实工具。
    3. 不调用 LLM。
    4. 只负责把旧字段适配成新的 ToolAgent Agent 层响应契约。
    5. 输出普通 dict，避免 checkpoint 直接保存自定义对象。

专业名词：
    Adapter：适配器，把旧数据结构转换成新数据结构。
    State：状态，LangGraph 节点之间传递的数据。
    Contract：契约，模块之间约定好的输入输出格式。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from src.agents.tool_agent.contracts.schemas import (
    ToolAgentExecutionRecord,
    ToolAgentIntent,
    ToolAgentPermissionDecision,
    ToolAgentPlannedCall,
    ToolAgentResponse,
    ToolAgentResponseStatus,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.schemas.tool_result_schema import ToolResult


TOOL_AGENT_RESPONSE_STATE_KEY = "tool_agent_response"


def build_tool_agent_response_from_state(
    state: Mapping[str, Any],
) -> ToolAgentResponse:
    """
    从旧 DogState 构建 ToolAgentResponse。

    功能：
        读取旧 state 中的 need_tool、tool_calls、tool_results、tool_confirmed、
        tool_round、final_answer 等字段，并转换成 ToolAgentResponse。

    参数：
        state:
            当前 LangGraph state。
            这里使用 Mapping[str, Any]，方便兼容 dict、DogState 或测试用映射对象。

    返回值：
        ToolAgentResponse:
            ToolAgent Agent 层标准响应对象。
    """

    normalized_tool_calls = normalize_tool_calls(
        state.get(
            "tool_calls",
            [],
        )
    )
    normalized_tool_results = normalize_tool_results(
        state.get(
            "tool_results",
            [],
        )
    )

    planned_calls = build_planned_calls(
        tool_calls=normalized_tool_calls,
        requires_confirmation=should_require_confirmation(
            state=state,
            tool_calls=normalized_tool_calls,
        ),
    )
    execution_records = build_execution_records(
        tool_results=normalized_tool_results,
    )
    intent = build_intent(
        state=state,
        tool_calls=normalized_tool_calls,
        tool_results=normalized_tool_results,
    )
    permission = build_permission_decision(
        state=state,
        planned_calls=planned_calls,
        tool_results=normalized_tool_results,
    )
    status = infer_response_status(
        state=state,
        planned_calls=planned_calls,
        execution_records=execution_records,
        permission=permission,
    )

    return ToolAgentResponse(
        status=status,
        intent=intent,
        planned_calls=planned_calls,
        permission=permission,
        execution_records=execution_records,
        final_answer=str(
            state.get(
                "final_answer",
                "",
            )
            or ""
        ),
        metadata={
            "source": "legacy_tool_state_adapter",
            "tool_round": state.get(
                "tool_round",
                0,
            ),
            "legacy_need_tool": state.get(
                "need_tool",
                False,
            ),
        },
    )


def dump_tool_agent_response_for_state(
    response: ToolAgentResponse,
) -> dict[str, Any]:
    """
    将 ToolAgentResponse 转换成可写回 state 的 dict。

    功能：
        把 ToolAgentResponse 通过 model_dump 转成普通 dict，
        并包装成 {"tool_agent_response": ...} 格式，供 LangGraph 合并 state。

    参数：
        response:
            ToolAgentResponse 标准响应对象。

    返回值：
        dict[str, Any]:
            可写回 DogState 的普通字典。
    """

    return {
        TOOL_AGENT_RESPONSE_STATE_KEY: response.model_dump()
    }


def build_tool_agent_response_state_update(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从旧 state 直接构建 ToolAgentResponse state 更新。

    功能：
        组合 build_tool_agent_response_from_state 和 dump_tool_agent_response_for_state，
        方便后续作为 ToolAgent 适配节点或调试节点复用。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            包含 tool_agent_response 的 state 更新字典。
    """

    response = build_tool_agent_response_from_state(
        state=state,
    )
    return dump_tool_agent_response_for_state(
        response=response,
    )


def normalize_tool_calls(
    raw_tool_calls: Any,
) -> list[ToolCall]:
    """
    归一化旧 state 中的 tool_calls。

    功能：
        将 dict、ToolCall 或 list 形式的旧工具调用转换成 list[ToolCall]。
        无法解析的条目会被跳过，避免适配器影响主链路。

    参数：
        raw_tool_calls:
            旧 state 中的 tool_calls 字段。

    返回值：
        list[ToolCall]:
            归一化后的工具调用列表。
    """

    if raw_tool_calls is None:
        return []

    if isinstance(
        raw_tool_calls,
        ToolCall,
    ):
        return [
            raw_tool_calls
        ]

    if isinstance(
        raw_tool_calls,
        Mapping,
    ):
        raw_items = [
            raw_tool_calls
        ]
    elif isinstance(
        raw_tool_calls,
        list,
    ):
        raw_items = raw_tool_calls
    else:
        return []

    normalized_calls: list[ToolCall] = []

    for item in raw_items:
        parsed_call = parse_tool_call(
            item
        )

        if parsed_call is not None:
            normalized_calls.append(
                parsed_call
            )

    return normalized_calls


def parse_tool_call(
    item: Any,
) -> ToolCall | None:
    """
    解析单个工具调用。

    功能：
        将 ToolCall 或 dict 转换成 ToolCall。
        如果数据缺少 name 或格式非法，则返回 None。

    参数：
        item:
            单个工具调用条目。

    返回值：
        ToolCall | None:
            解析成功时返回 ToolCall，失败时返回 None。
    """

    if isinstance(
        item,
        ToolCall,
    ):
        return item

    if not isinstance(
        item,
        Mapping,
    ):
        return None

    name = item.get(
        "name",
        "",
    )

    if not name:
        return None

    try:
        return ToolCall(
            name=str(
                name
            ),
            args=dict(
                item.get(
                    "args",
                    {},
                )
                or {}
            ),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def normalize_tool_results(
    raw_tool_results: Any,
) -> list[ToolResult]:
    """
    归一化旧 state 中的 tool_results。

    功能：
        将 str、dict、ToolResult 或 list 形式的旧工具结果转换成 list[ToolResult]。

    参数：
        raw_tool_results:
            旧 state 中的 tool_results 字段。

    返回值：
        list[ToolResult]:
            归一化后的工具结果列表。
    """

    if raw_tool_results is None:
        return []

    if isinstance(
        raw_tool_results,
        ToolResult,
    ):
        return [
            raw_tool_results
        ]

    if isinstance(
        raw_tool_results,
        list,
    ):
        raw_items = raw_tool_results
    else:
        raw_items = [
            raw_tool_results
        ]

    normalized_results: list[ToolResult] = []

    for index, item in enumerate(
        raw_items,
        start=1,
    ):
        normalized_results.append(
            parse_tool_result(
                item=item,
                index=index,
            )
        )

    return normalized_results


def parse_tool_result(
    item: Any,
    index: int,
) -> ToolResult:
    """
    解析单个工具结果。

    功能：
        将 ToolResult、dict、str 或其他对象转换成 ToolResult。
        旧字符串结果会作为 legacy_tool_result 的 content 保存。

    参数：
        item:
            单个工具结果条目。

        index:
            当前结果序号，从 1 开始，用于生成兜底工具名。

    返回值：
        ToolResult:
            归一化后的工具结果对象。
    """

    if isinstance(
        item,
        ToolResult,
    ):
        return item

    if isinstance(
        item,
        Mapping,
    ):
        return parse_mapping_tool_result(
            item=item,
            index=index,
        )

    return ToolResult(
        success=True,
        tool_name=f"legacy_tool_result_{index}",
        content=item,
    )


def parse_mapping_tool_result(
    item: Mapping[str, Any],
    index: int,
) -> ToolResult:
    """
    解析 dict 形式的工具结果。

    功能：
        兼容当前 ToolResult 字典，也兼容缺少 success/tool_name 的历史 dict。

    参数：
        item:
            dict 形式的工具结果。

        index:
            当前结果序号，从 1 开始，用于生成兜底工具名。

    返回值：
        ToolResult:
            归一化后的工具结果对象。
    """

    success = bool(
        item.get(
            "success",
            not bool(
                item.get(
                    "error"
                )
            ),
        )
    )

    return ToolResult(
        success=success,
        tool_name=str(
            item.get(
                "tool_name",
                item.get(
                    "name",
                    f"legacy_tool_result_{index}",
                ),
            )
        ),
        content=item.get(
            "content",
            item.get(
                "result",
                item,
            ),
        ),
        error=item.get(
            "error"
        ),
        latency=item.get(
            "latency"
        ),
        retry_count=int(
            item.get(
                "retry_count",
                0,
            )
            or 0
        ),
        metadata=dict(
            item.get(
                "metadata",
                {},
            )
            or {}
        ),
    )


def build_planned_calls(
    tool_calls: list[ToolCall],
    requires_confirmation: bool,
) -> list[ToolAgentPlannedCall]:
    """
    构建计划工具调用列表。

    功能：
        将归一化后的 ToolCall 包装成 ToolAgentPlannedCall。

    参数：
        tool_calls:
            归一化后的工具调用列表。

        requires_confirmation:
            是否需要用户确认。

    返回值：
        list[ToolAgentPlannedCall]:
            ToolAgent 计划工具调用列表。
    """

    return [
        ToolAgentPlannedCall(
            call_id=build_call_id(
                tool_call=tool_call,
                index=index,
            ),
            tool_call=tool_call,
            requires_confirmation=requires_confirmation,
            reason="从旧 state.tool_calls 适配生成。",
        )
        for index, tool_call in enumerate(
            tool_calls,
            start=1,
        )
    ]


def build_execution_records(
    tool_results: list[ToolResult],
) -> list[ToolAgentExecutionRecord]:
    """
    构建工具执行记录列表。

    功能：
        将归一化后的 ToolResult 包装成 ToolAgentExecutionRecord。

    参数：
        tool_results:
            归一化后的工具结果列表。

    返回值：
        list[ToolAgentExecutionRecord]:
            ToolAgent 工具执行记录列表。
    """

    return [
        ToolAgentExecutionRecord(
            call_id=f"executed_{index}_{tool_result.tool_name}",
            tool_result=tool_result,
            duration_ms=convert_latency_to_ms(
                tool_result.latency
            ),
            metadata={
                "source": "legacy_tool_results",
            },
        )
        for index, tool_result in enumerate(
            tool_results,
            start=1,
        )
    ]


def build_intent(
    state: Mapping[str, Any],
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
) -> ToolAgentIntent:
    """
    构建工具意图对象。

    功能：
        根据旧 state 的 need_tool、tool_calls 和 tool_results 推断 ToolAgentIntent。

    参数：
        state:
            当前 LangGraph state。

        tool_calls:
            归一化后的工具调用列表。

        tool_results:
            归一化后的工具结果列表。

    返回值：
        ToolAgentIntent:
            工具意图契约对象。
    """

    candidate_tools = sorted(
        {
            tool_call.name
            for tool_call in tool_calls
        }
        | {
            tool_result.tool_name
            for tool_result in tool_results
        }
    )

    need_tool = bool(
        state.get(
            "need_tool",
            False,
        )
        or tool_calls
        or tool_results
    )

    reason = (
        "从旧工具 state 字段推断需要工具。"
        if need_tool
        else "旧工具 state 中没有工具调用或工具结果。"
    )

    return ToolAgentIntent(
        need_tool=need_tool,
        candidate_tools=candidate_tools,
        reason=reason,
    )


def build_permission_decision(
    state: Mapping[str, Any],
    planned_calls: list[ToolAgentPlannedCall],
    tool_results: list[ToolResult],
) -> ToolAgentPermissionDecision:
    """
    构建工具权限决定。

    功能：
        优先读取新版 tool_agent_permission 标准权限字段；
        当该字段不存在或不符合契约时，再根据旧版 tool_confirmed、
        tool_calls 和 tool_results 字段推断权限状态。

    参数：
        state:
            当前 LangGraph state。

        planned_calls:
            ToolAgent 计划工具调用列表。

        tool_results:
            归一化后的工具结果列表。

    返回值：
        ToolAgentPermissionDecision:
            工具权限决定对象。
    """

    # 新版 ToolAgent 节点已经产出标准权限决定时，优先保留该契约，
    # 避免后续响应适配器再根据旧字段错误反推权限状态。
    raw_permission = state.get(
        "tool_agent_permission"
    )
    if isinstance(
        raw_permission,
        Mapping,
    ):
        try:
            return ToolAgentPermissionDecision.model_validate(
                dict(raw_permission)
            )
        except ValidationError:
            # 兼容旧 checkpoint：旧数据不符合新版契约时继续使用历史字段推断。
            pass

    raw_confirmed = state.get(
        "tool_confirmed",
        "",
    )
    call_ids = [
        planned_call.call_id
        for planned_call in planned_calls
    ]

    if is_rejected_confirmation(
        raw_confirmed
    ) or has_cancelled_tool_result(
        tool_results
    ):
        return ToolAgentPermissionDecision(
            status="rejected",
            call_ids=call_ids,
            reason="用户取消或拒绝了工具调用。",
        )

    if is_confirmed_value(
        raw_confirmed
    ) or tool_results:
        return ToolAgentPermissionDecision(
            status="confirmed",
            call_ids=call_ids,
            reason="旧 state 显示工具已确认或已有工具结果。",
        )

    if planned_calls:
        return ToolAgentPermissionDecision(
            status="pending",
            call_ids=call_ids,
            prompt="是否允许执行计划中的工具调用？",
            reason="存在待执行工具调用，但旧 state 中未记录确认结果。",
        )

    return ToolAgentPermissionDecision(
        status="not_required",
        reason="没有待执行工具调用。",
    )


def infer_response_status(
    state: Mapping[str, Any],
    planned_calls: list[ToolAgentPlannedCall],
    execution_records: list[ToolAgentExecutionRecord],
    permission: ToolAgentPermissionDecision,
) -> ToolAgentResponseStatus:
    """
    推断 ToolAgentResponse 状态。

    功能：
        根据权限状态、执行记录、待执行调用和旧 state 推断最终状态。

    参数：
        state:
            当前 LangGraph state。

        planned_calls:
            计划工具调用列表。

        execution_records:
            工具执行记录列表。

        permission:
            工具权限决定。

    返回值：
        ToolAgentResponseStatus:
            ToolAgent 响应状态。
    """

    # 参数澄清优先于普通 no_tool 判断，避免把“等待补参”误报为“不需要工具”。
    clarification_request = state.get(
        "tool_agent_clarification_request"
    )
    if isinstance(
        clarification_request,
        Mapping,
    ) and clarification_request:
        return "awaiting_clarification"

    if permission.status == "rejected":
        return "cancelled"

    if any(
        not record.tool_result.success
        for record in execution_records
    ):
        return "failed"

    if execution_records:
        return "completed"

    if planned_calls:
        return "pending_confirmation"

    if state.get(
        "need_tool",
        False,
    ):
        return "pending_confirmation"

    return "no_tool"


def should_require_confirmation(
    state: Mapping[str, Any],
    tool_calls: list[ToolCall],
) -> bool:
    """
    判断计划工具调用是否需要确认。

    功能：
        当前旧链路中工具调用默认经过 ask_confirm_tool_node，
        因此只要存在 tool_calls 且没有工具结果，就保守标记为需要确认。

    参数：
        state:
            当前 LangGraph state。

        tool_calls:
            归一化后的工具调用列表。

    返回值：
        bool:
            True 表示需要用户确认。
    """

    return bool(
        tool_calls
        and not state.get(
            "tool_results"
        )
    )


def build_call_id(
    tool_call: ToolCall,
    index: int,
) -> str:
    """
    构建计划调用 ID。

    功能：
        根据工具名和序号生成稳定、可读的调用 ID。

    参数：
        tool_call:
            工具调用对象。

        index:
            当前工具调用序号，从 1 开始。

    返回值：
        str:
            调用 ID。
    """

    return f"planned_{index}_{tool_call.name}"


def convert_latency_to_ms(
    latency: float | None,
) -> int | None:
    """
    将秒级 latency 转换成毫秒。

    功能：
        旧 ToolResult.latency 使用秒为单位，ToolAgentExecutionRecord.duration_ms
        使用毫秒为单位，因此这里做单位转换。

    参数：
        latency:
            秒级耗时，可能为 None。

    返回值：
        int | None:
            毫秒级耗时；latency 为 None 时返回 None。
    """

    if latency is None:
        return None

    return int(
        latency * 1000
    )


def is_confirmed_value(
    value: Any,
) -> bool:
    """
    判断旧确认字段是否表示已确认。

    功能：
        兼容 bool、字符串 y、yes、true、confirmed。

    参数：
        value:
            旧 state 中的 tool_confirmed 字段。

    返回值：
        bool:
            True 表示已确认。
    """

    if value is True:
        return True

    if isinstance(
        value,
        str,
    ):
        return value.strip().lower() in {
            "y",
            "yes",
            "true",
            "confirmed",
        }

    return False


def is_rejected_confirmation(
    value: Any,
) -> bool:
    """
    判断旧确认字段是否表示已拒绝。

    功能：
        兼容 bool False、字符串 n、no、false、rejected、cancelled。

    参数：
        value:
            旧 state 中的 tool_confirmed 字段。

    返回值：
        bool:
            True 表示已拒绝。
    """

    if value is False:
        return True

    if isinstance(
        value,
        str,
    ):
        normalized_value = value.strip().lower()

        if not normalized_value:
            return False

        return normalized_value in {
            "n",
            "no",
            "false",
            "rejected",
            "cancelled",
            "canceled",
        }

    return False


def has_cancelled_tool_result(
    tool_results: list[ToolResult],
) -> bool:
    """
    判断工具结果中是否包含用户取消信息。

    功能：
        兼容旧 ask_confirm_tool_node 返回的“用户取消了工具调用。”字符串结果。

    参数：
        tool_results:
            归一化后的工具结果列表。

    返回值：
        bool:
            True 表示工具结果中包含取消语义。
    """

    for tool_result in tool_results:
        content_text = str(
            tool_result.content
        )

        if "取消" in content_text:
            return True

    return False

