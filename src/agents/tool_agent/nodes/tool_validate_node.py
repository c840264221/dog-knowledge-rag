"""
ToolAgent 工具调用校验节点。

功能：
    在工具解析之后、工具确认之前，根据工具目录校验 tool_calls。

设计原则：
    1. tool_parse_node 只负责生成工具调用计划。
    2. tool_validate_node 只负责校验工具调用计划。
    3. tool_confirm_node 只负责权限确认。
    4. 输出普通 dict，避免 checkpoint 保存自定义对象。

专业名词：
    Validator：
        校验器，负责检查结构化数据是否符合契约。
    ToolCall：
        工具调用请求，表示要调用哪个工具以及传入哪些参数。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.tool_call_validation_adapter import (
    validate_tool_calls_from_state,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.runtime.context import runtime_ctx


ToolValidateNode = Callable[[Mapping[str, Any]], dict[str, Any]]


def build_tool_agent_tool_validate_node(
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolValidateNode:
    """
    构建 ToolAgent 工具调用校验节点。

    功能：
        创建一个 LangGraph 可调用的同步节点。
        节点从 state 中读取 tool_calls 和 tool_agent_tool_catalog，
        调用 validation adapter 完成工具调用校验。

    参数：
        checkpoint_manager:
            检查点管理器。校验完成后按需保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。

    返回值：
        ToolValidateNode:
            同步节点函数，接收 state，返回可合并进 state 的 dict。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    def tool_agent_tool_validate_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行工具调用校验。

        功能：
            1. 写入当前运行时 node 信息。
            2. 读取 state 中的工具目录和 tool_calls。
            3. 校验工具是否存在、args 是否合法。
            4. 将合法 tool_calls 写回 state。
            5. 将校验错误写入 state，方便日志和 debug report 观察。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                可写回 LangGraph state 的校验结果。
        """

        write_tool_validate_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        log_tool_agent_state(
            node_name="tool_validate",
            event="tool_validate_start",
            state=state,
        )

        update = validate_tool_calls_from_state(
            state=state,
        )

        log_tool_agent_state(
            node_name="tool_validate",
            event="tool_validate_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "validation_ok": update.get(
                    "tool_call_validation_ok",
                ),
                "validation_skipped": update.get(
                    "tool_call_validation_skipped",
                ),
                "valid_tool_call_count": len(
                    update.get(
                        "tool_calls",
                        [],
                    )
                    or []
                ),
                "invalid_tool_call_count": len(
                    update.get(
                        "tool_call_validation_invalid_calls",
                        [],
                    )
                    or []
                ),
            },
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return update

    return tool_agent_tool_validate_node


def write_tool_validate_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具校验节点运行时事件。

    功能：
        如果存在 RuntimeContext，则记录当前 node 和 timeline 事件。
        如果不存在，则静默跳过，保证单元测试可独立运行。

    参数：
        runtime_context:
            当前请求 RuntimeContext，可能为 None。

    返回值：
        None。
    """

    if runtime_context is None:
        return

    runtime_context.state().set_node(
        "tool_agent_tool_validate_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_validate_node",
    )
