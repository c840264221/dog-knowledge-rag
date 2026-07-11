"""
V1.10 ToolAgent 多轮澄清 Checkpoint 冒烟检查。

功能：
    校验真实主图两轮运行后的 Checkpoint state 和最终结果，
    并生成可供脚本渲染的结构化报告。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.mcp.sqlite.tool_definitions import SQLITE_LIST_TABLES_TOOL_NAME


@dataclass(frozen=True)
class ClarificationCheckpointSmokeResult:
    """
    多轮澄清 Checkpoint 冒烟结果。

    参数：
        passed:
            全部检查是否通过。
        same_thread_id:
            两轮请求是否使用同一 thread_id。
        different_trace_ids:
            两轮请求是否使用不同 trace_id。
        clarification_saved:
            第一轮 Checkpoint 是否保存澄清请求。
        awaiting_clarification:
            第一轮 ToolAgent 响应是否处于等待澄清状态。
        pending_call_saved:
            第一轮 Checkpoint 是否保存待补全工具调用。
        clarification_resumed:
            第二轮是否记录 resumed 恢复动作。
        tool_executed:
            第二轮是否产生成功工具结果。
        clarification_cleared:
            第二轮结束后是否清理澄清字段。
        final_answer_preview:
            第二轮最终回答预览。
        errors:
            未通过检查的中文错误列表。

    返回值：
        ClarificationCheckpointSmokeResult:
            不可变 dataclass 冒烟结果对象。
    """

    passed: bool
    same_thread_id: bool
    different_trace_ids: bool
    clarification_saved: bool
    awaiting_clarification: bool
    pending_call_saved: bool
    clarification_resumed: bool
    tool_executed: bool
    clarification_cleared: bool
    final_answer_preview: str
    errors: list[str]


def validate_clarification_checkpoint_smoke(
    first_state: Mapping[str, Any],
    second_state: Mapping[str, Any],
    first_thread_id: str,
    second_thread_id: str,
    first_trace_id: str,
    second_trace_id: str,
    final_answer: str = "",
    runtime_error: str = "",
) -> ClarificationCheckpointSmokeResult:
    """
    校验真实主图多轮澄清结果。

    功能：
        检查会话 ID、请求追踪 ID、第一轮澄清状态、第二轮恢复动作、
        工具执行结果和澄清字段清理结果。

    参数：
        first_state:
            第一轮结束后从 LangGraph Checkpointer 读取的 state。
        second_state:
            第二轮结束后从 LangGraph Checkpointer 读取的 state。
        first_thread_id:
            第一轮 thread_id。
        second_thread_id:
            第二轮 thread_id。
        first_trace_id:
            第一轮 trace_id。
        second_trace_id:
            第二轮 trace_id。
        final_answer:
            第二轮结构化 Graph 结果中的最终回答。
        runtime_error:
            脚本执行期间捕获的异常文本。

    返回值：
        ClarificationCheckpointSmokeResult:
            动态计算 PASS/FAIL 后的结构化报告。
    """

    errors: list[str] = []
    same_thread_id = bool(
        first_thread_id
        and first_thread_id == second_thread_id
    )
    different_trace_ids = bool(
        first_trace_id
        and second_trace_id
        and first_trace_id != second_trace_id
    )

    clarification_request = first_state.get(
        "tool_agent_clarification_request"
    )
    pending_tool_call = first_state.get(
        "tool_agent_pending_tool_call"
    )
    clarification_saved = bool(
        isinstance(clarification_request, Mapping)
        and clarification_request.get("status") == "pending"
    )
    first_response = first_state.get(
        "tool_agent_response"
    )
    awaiting_clarification = bool(
        isinstance(first_response, Mapping)
        and first_response.get("status") == "awaiting_clarification"
    )
    pending_call_saved = bool(
        isinstance(pending_tool_call, Mapping)
        and pending_tool_call.get("name")
    )

    resolution = second_state.get(
        "tool_agent_clarification_resolution"
    )
    clarification_resumed = bool(
        isinstance(resolution, Mapping)
        and resolution.get("action") == "resumed"
    )
    tool_results = second_state.get(
        "tool_results",
        [],
    )
    tool_executed = bool(
        isinstance(tool_results, list)
        and any(
            isinstance(item, Mapping)
            and item.get("success") is True
            and item.get("tool_name") == SQLITE_LIST_TABLES_TOOL_NAME
            for item in tool_results
        )
    )
    clarification_cleared = (
        second_state.get("tool_agent_clarification_request") is None
        and second_state.get("tool_agent_pending_tool_call") is None
        and not second_state.get("tool_agent_clarification_resume_ready", False)
    )

    if not same_thread_id:
        errors.append("两轮请求没有使用相同 thread_id。")
    if not different_trace_ids:
        errors.append("两轮请求没有使用不同 trace_id。")
    if not clarification_saved:
        errors.append("第一轮 Checkpoint 中没有 pending 澄清请求。")
    if not awaiting_clarification:
        errors.append("第一轮 tool_agent_response 不是 awaiting_clarification。")
    if not pending_call_saved:
        errors.append("第一轮 Checkpoint 中没有待补全工具调用。")
    if not clarification_resumed:
        errors.append("第二轮没有记录 clarification resumed 动作。")
    if not tool_executed:
        errors.append("第二轮没有产生成功的工具执行结果。")
    if not clarification_cleared:
        errors.append("第二轮结束后澄清字段没有完全清理。")
    if not final_answer.strip():
        errors.append("第二轮最终回答为空。")
    if runtime_error:
        errors.append(
            f"真实主图执行异常：{runtime_error}"
        )

    return ClarificationCheckpointSmokeResult(
        passed=not errors,
        same_thread_id=same_thread_id,
        different_trace_ids=different_trace_ids,
        clarification_saved=clarification_saved,
        awaiting_clarification=awaiting_clarification,
        pending_call_saved=pending_call_saved,
        clarification_resumed=clarification_resumed,
        tool_executed=tool_executed,
        clarification_cleared=clarification_cleared,
        final_answer_preview=final_answer[:160],
        errors=errors,
    )


def assert_clarification_checkpoint_smoke(
    result: ClarificationCheckpointSmokeResult,
) -> ClarificationCheckpointSmokeResult:
    """
    断言多轮澄清冒烟结果通过。

    参数：
        result:
            待检查的冒烟结果。

    返回值：
        ClarificationCheckpointSmokeResult:
            通过时原样返回结果；失败时抛出 AssertionError。
    """

    if result.passed:
        return result

    raise AssertionError(
        "V1.10 多轮澄清 Checkpoint 冒烟失败：\n"
        + "\n".join(
            f"- {error}"
            for error in result.errors
        )
    )
