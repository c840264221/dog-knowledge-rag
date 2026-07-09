"""
ToolAgent 工具执行节点。

功能：
    根据 ToolAgent 当前权限状态执行工具调用，并把执行结果写回 state。

设计原则：
    1. 只复用 ToolAgent runtime_adapter，不直接重复实现 ToolExecutor 调用细节。
    2. pending/rejected 状态不会执行工具，避免绕过用户确认。
    3. confirmed/not_required 状态可以执行工具。
    4. 输出普通 dict，避免 checkpoint 保存自定义对象。

专业名词：
    Execute：执行，真正调用工具并拿到结果。
    Permission：权限，表示工具调用是否允许继续。
    Runtime Adapter：运行时适配器，负责桥接 ToolAgent 和底层 ToolExecutor。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.runtime_adapter import (
    build_tool_agent_runtime_state_update,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    normalize_tool_calls,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.runtime.context import runtime_ctx


ToolExecuteNode = Callable[
    [Mapping[str, Any]],
    Awaitable[dict[str, Any]],
]


def build_tool_agent_tool_execute_node(
    executor: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolExecuteNode:
    """
    构建 ToolAgent 工具执行节点。

    功能：
        创建一个 async 节点。
        节点读取 state.tool_calls 和权限状态，确认允许后调用 runtime_adapter 执行工具。

    参数：
        executor:
            工具执行器。测试时可以传入 fake executor，避免调用真实外部工具。

        checkpoint_manager:
            检查点管理器。工具执行完成后按需保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。

    返回值：
        ToolExecuteNode:
            async 节点函数，接收 state，返回可合并进 state 的 dict。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def tool_agent_tool_execute_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行当前允许执行的工具调用。

        功能：
            1. 写入当前运行时 node 信息。
            2. 读取并归一化 tool_calls。
            3. 根据权限状态判断是否允许执行。
            4. 允许执行时调用 runtime_adapter。
            5. 执行后清空 tool_calls，并刷新 tool_agent_response。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                可写回 LangGraph state 的执行结果。
        """

        write_tool_execute_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        log_tool_agent_state(
            node_name="tool_execute",
            event="tool_execute_start",
            state=state,
        )

        tool_calls = normalize_tool_calls(
            state.get(
                "tool_calls",
                [],
            )
        )

        if not tool_calls:
            update = build_execute_skip_update(
                state=state,
                reason="没有待执行的工具调用。",
            )
            log_tool_agent_state(
                node_name="tool_execute",
                event="tool_execute_skip_no_tool_calls",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "reason": "没有待执行的工具调用。",
                },
            )
            return update

        permission_status = get_permission_status(
            state=state,
        )

        log_tool_agent_state(
            node_name="tool_execute",
            event="tool_execute_permission_checked",
            state=state,
            extra={
                "permission_status": permission_status,
                "tool_call_count": len(
                    tool_calls
                ),
            },
        )

        if permission_status in {
            "pending",
            "rejected",
        }:
            reason = f"当前权限状态为 {permission_status}，不执行工具。"
            update = build_execute_skip_update(
                state=state,
                reason=reason,
            )
            log_tool_agent_state(
                node_name="tool_execute",
                event="tool_execute_skip_permission",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "permission_status": permission_status,
                    "reason": reason,
                },
            )
            return update

        # 权限允许后，通过 runtime_adapter 调用底层 ToolExecutor。
        runtime_update = await build_tool_agent_runtime_state_update(
            tool_calls=tool_calls,
            executor=executor,
        )

        update = build_execute_success_update(
            state=state,
            runtime_update=runtime_update,
        )

        log_tool_agent_state(
            node_name="tool_execute",
            event="tool_execute_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "executed_tool_call_count": len(
                    tool_calls
                ),
                "tool_result_count": len(
                    runtime_update.get(
                        "tool_results",
                        [],
                    )
                ),
            },
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return update

    return tool_agent_tool_execute_node


def write_tool_execute_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具执行节点运行时事件。

    功能：
        如果存在 RuntimeContext，则记录当前 node 和 timeline 事件。

    参数：
        runtime_context:
            当前请求的 RuntimeContext，可能为 None。

    返回值：
        None。
    """

    if runtime_context is None:
        return

    runtime_context.state().set_node(
        "tool_agent_tool_execute_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_execute_node",
    )


def get_permission_status(
    state: Mapping[str, Any],
) -> str:
    """
    获取当前工具权限状态。

    功能：
        优先读取 tool_agent_permission.status，
        如果不存在，则回退读取 tool_confirmed，
        最后默认为 not_required。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        str:
            权限状态字符串，例如 pending、confirmed、rejected、not_required。
    """

    permission = state.get(
        "tool_agent_permission",
        {},
    )

    if isinstance(
        permission,
        Mapping,
    ):
        status = permission.get(
            "status",
            "",
        )
        if status:
            return str(
                status
            )

    tool_confirmed = str(
        state.get(
            "tool_confirmed",
            "",
        )
        or ""
    ).strip()

    if tool_confirmed:
        return tool_confirmed

    return "not_required"


def build_execute_skip_update(
    state: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    """
    构建跳过执行时的 state update。

    功能：
        当没有工具调用、等待确认或用户拒绝时，不执行工具，只刷新响应契约。

    参数：
        state:
            当前 LangGraph state。

        reason:
            跳过执行的原因，写入调试 metadata。

    返回值：
        dict[str, Any]:
            包含 tool_agent_execute_skipped 和 tool_agent_response 的 state update。
    """

    merged_state = dict(
        state
    )
    response = build_tool_agent_response_from_state(
        state=merged_state,
    ).model_dump()

    permission = state.get(
        "tool_agent_permission",
        {},
    )
    if isinstance(
        permission,
        Mapping,
    ) and permission.get(
        "status"
    ):
        response["permission"] = dict(
            permission
        )

    metadata = dict(
        response.get(
            "metadata",
            {},
        )
        or {}
    )
    metadata["execute_skip_reason"] = reason
    response["metadata"] = metadata

    return {
        "tool_agent_execute_skipped": True,
        "tool_agent_execute_skip_reason": reason,
        TOOL_AGENT_RESPONSE_STATE_KEY: response,
    }


def build_execute_success_update(
    state: Mapping[str, Any],
    runtime_update: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构建工具执行成功后的 state update。

    功能：
        合并 runtime_adapter 返回值，清空已执行的 tool_calls，
        更新 need_tool/tool_round，并刷新 tool_agent_response。

    参数：
        state:
            当前 LangGraph state。

        runtime_update:
            runtime_adapter 返回的工具执行结果。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的执行结果。
    """

    update: dict[str, Any] = {
        **dict(
            runtime_update
        ),
        "tool_calls": [],
        "need_tool": False,
        "tool_agent_execute_skipped": False,
        "tool_agent_execute_skip_reason": "",
        "tool_round": int(
            state.get(
                "tool_round",
                0,
            )
            or 0
        )
        + 1,
    }

    merged_state = {
        **dict(
            state
        ),
        **update,
    }
    update[TOOL_AGENT_RESPONSE_STATE_KEY] = (
        build_tool_agent_response_from_state(
            state=merged_state,
        ).model_dump()
    )

    return update
