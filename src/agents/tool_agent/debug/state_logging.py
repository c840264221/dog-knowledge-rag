"""
ToolAgent 状态日志工具。

功能：
    为 ToolAgent 各节点提供统一的关键 state 快照日志。

设计原则：
    1. 不直接打印完整 state，避免日志过大。
    2. 只提取工具链路相关关键字段，方便观察真实运行流程。
    3. 使用 json.dumps(indent=4, ensure_ascii=False) 输出，方便控制台阅读。
    4. 对不可 JSON 序列化对象做安全转换，避免日志影响主流程。

专业名词：
    Snapshot：快照，表示当前 state 的关键字段摘要。
    Serialization：序列化，把 Python 对象转换成可打印、可传输的格式。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from src.logger import logger


TOOL_AGENT_LOG_FIELDS = (
    "question",
    "current_agent",
    "next_agent",
    "route_decision",
    "need_tool",
    "tool_calls",
    "tool_results",
    "tool_round",
    "tool_confirmed",
    "tool_confirmation_required",
    "tool_confirmation_mode",
    "tool_confirmation_prompt",
    "tool_agent_allowed_databases",
    "tool_agent_permission",
    "tool_agent_execute_skipped",
    "tool_agent_execute_skip_reason",
    "tool_agent_runtime_execution_records",
    "tool_agent_answer_source",
    "tool_agent_response",
    "final_answer",
)


def build_tool_agent_state_log_snapshot(
    state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    构建 ToolAgent 日志快照。

    功能：
        从完整 state 中提取 ToolAgent 关心的关键字段，
        并把值转换成安全可 JSON 序列化的格式。

    参数：
        state:
            当前 LangGraph state。

        extra:
            额外日志字段，例如 event、route、reason、result_count。

    返回值：
        dict[str, Any]:
            可传给 json.dumps 的日志快照字典。
    """

    snapshot: dict[str, Any] = {}

    for field_name in TOOL_AGENT_LOG_FIELDS:
        if field_name in state:
            snapshot[field_name] = make_json_safe(
                state.get(
                    field_name,
                )
            )

    if extra:
        snapshot.update(
            {
                key: make_json_safe(
                    value,
                )
                for key, value in extra.items()
            }
        )

    return snapshot


def log_tool_agent_state(
    node_name: str,
    event: str,
    state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """
    输出 ToolAgent 状态日志。

    功能：
        构建关键 state 快照，并用缩进 JSON 输出到 info 日志。
        如果日志构建失败，只记录一个简短错误，不影响主流程。

    参数：
        node_name:
            产生日志的节点名称，例如 tool_parse、tool_confirm。

        event:
            当前日志事件名称，例如 tool_parse_start。

        state:
            当前 LangGraph state。

        extra:
            额外日志字段。

    返回值：
        None。
    """

    try:
        payload = build_tool_agent_state_log_snapshot(
            state=state,
            extra={
                "node_name": node_name,
                "event": event,
                **dict(
                    extra or {},
                ),
            },
        )

        formatted_payload = json.dumps(
            payload,
            indent=4,
            ensure_ascii=False,
        )

        logger.opt(
            depth=1
        ).info(
            f"ToolAgent state snapshot [{node_name}]:\n{formatted_payload}"
        )
    except Exception as exc:
        logger.opt(
            depth=1
        ).info(
            f"ToolAgent state 日志构建失败: {exc}"
        )


def make_json_safe(
    value: Any,
) -> Any:
    """
    将任意值转换成 JSON 安全格式。

    功能：
        兼容 Pydantic model、dict、list、tuple、普通标量和不可序列化对象。

    参数：
        value:
            任意 Python 对象。

    返回值：
        Any:
            可被 json.dumps 序列化的值。
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if hasattr(
        value,
        "model_dump",
    ):
        return make_json_safe(
            value.model_dump()
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(
                key
            ): make_json_safe(
                nested_value,
            )
            for key, nested_value in value.items()
        }

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return [
            make_json_safe(
                item,
            )
            for item in value
        ]

    return repr(
        value
    )
