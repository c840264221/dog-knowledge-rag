"""
ToolAgent 工具确认节点。

功能：
    根据工具注册表中的 require_confirm 配置，为 ToolAgent 生成批量确认计划。

设计原则：
    1. 默认只生成确认状态和确认提示。
    2. 当外部注入 interrupt_func 时，可以触发 LangGraph interrupt。
    2. 多个需要确认的工具合并成一次 batch confirmation（批量确认）。
    3. 低风险工具可以通过 ToolMetadata.require_confirm=False 自动跳过确认。
    4. 输出普通 dict，避免 checkpoint 保存自定义对象。

专业名词：
    Confirmation：确认，执行工具前询问用户是否允许。
    Batch Confirmation：批量确认，把多个工具调用合并成一次用户确认。
    Permission：权限，表示当前工具调用是否允许继续。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.registry_adapter import (
    get_registered_tool_metadata,
    tool_requires_confirmation,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    is_confirmed_value,
    is_rejected_confirmation,
    normalize_tool_calls,
)
from src.agents.tool_agent.contracts.schemas import ToolAgentPermissionDecision
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.runtime.context import runtime_ctx


ToolConfirmNode = Callable[[Mapping[str, Any]], dict[str, Any]]


def build_tool_agent_tool_confirm_node(
    tool_registry: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
    interrupt_func: Callable[[str], Any] | None = None,
) -> ToolConfirmNode:
    """
    构建 ToolAgent 工具确认节点。

    功能：
        创建一个可给 LangGraph 使用的同步节点。
        节点读取 state.tool_calls，并根据工具元数据判断是否需要用户确认。

    参数：
        tool_registry:
            工具注册表对象。默认由 registry_adapter 使用项目全局 registry。

        checkpoint_manager:
            检查点管理器。生成确认计划或确认结果后保存 checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。

        interrupt_func:
            Interrupt 函数。默认 None。
            为 None 时只生成 pending 状态，方便单元测试和独立子图 smoke。
            传入 LangGraph interrupt 时，会真正暂停图并等待用户确认。

    返回值：
        ToolConfirmNode:
            同步节点函数，接收 state，返回可合并进 state 的 dict。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    def tool_agent_tool_confirm_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        判断当前工具调用是否需要确认。

        功能：
            1. 写入当前运行时 node 信息。
            2. 没有 tool_calls 时返回 not_required。
            3. 用户已确认时返回 confirmed。
            4. 用户拒绝时返回 rejected，并清空 tool_calls。
            5. 工具不需要确认时返回 not_required。
            6. 工具需要确认时生成批量确认提示。
            7. 如果注入 interrupt_func，则调用它等待用户输入。
            8. 用户确认时返回 confirmed，用户拒绝时返回 rejected。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                可写回 LangGraph state 的确认结果。
        """

        write_tool_confirm_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        log_tool_agent_state(
            node_name="tool_confirm",
            event="tool_confirm_start",
            state=state,
        )

        tool_calls = normalize_tool_calls(
            state.get(
                "tool_calls",
                [],
            )
        )

        if not tool_calls:
            update = build_confirmation_update(
                state=state,
                tool_calls=[],
                permission=ToolAgentPermissionDecision(
                    status="not_required",
                    reason="没有待确认的工具调用。",
                ),
                tool_confirmed="not_required",
                confirmation_required=False,
                confirmation_prompt="",
            )
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )
            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_no_tool_calls",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "reason": "没有待确认的工具调用。",
                },
            )
            return update

        raw_confirmed = state.get(
            "tool_confirmed",
            "",
        )

        if is_confirmed_value(
            raw_confirmed
        ):
            update = build_confirmed_update(
                state=state,
                tool_calls=tool_calls,
            )
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )
            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_user_already_confirmed",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "raw_confirmed": raw_confirmed,
                },
            )
            return update

        if is_rejected_confirmation(
            raw_confirmed
        ):
            update = build_rejected_update(
                state=state,
                tool_calls=tool_calls,
            )
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )
            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_user_already_rejected",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "raw_confirmed": raw_confirmed,
                },
            )
            return update

        required_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_requires_confirmation(
                tool_name=tool_call.name,
                tool_registry=tool_registry,
            )
        ]

        if not required_tool_calls:
            update = build_confirmation_update(
                state=state,
                tool_calls=tool_calls,
                permission=ToolAgentPermissionDecision(
                    status="not_required",
                    call_ids=build_call_ids(
                        tool_calls=tool_calls,
                    ),
                    reason="当前工具都不需要用户确认。",
                ),
                tool_confirmed="not_required",
                confirmation_required=False,
                confirmation_prompt="",
            )
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )
            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_not_required",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "tool_call_count": len(
                        tool_calls
                    ),
                    "required_tool_call_count": 0,
                },
            )
            return update

        confirmation_prompt = build_batch_confirmation_prompt(
            tool_calls=required_tool_calls,
            tool_registry=tool_registry,
        )

        log_tool_agent_state(
            node_name="tool_confirm",
            event="tool_confirm_required",
            state=state,
            extra={
                "tool_call_count": len(
                    tool_calls
                ),
                "required_tool_call_count": len(
                    required_tool_calls
                ),
                "has_interrupt_func": interrupt_func is not None,
                "confirmation_prompt": confirmation_prompt,
            },
        )

        if interrupt_func is not None:
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )

            # 这里是真正的人机确认入口。主图接入时会传入 LangGraph interrupt，
            # 单元测试中可以传入 fake interrupt，避免真的暂停测试进程。
            user_input = interrupt_func(
                confirmation_prompt,
            )

            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_interrupt_returned",
                state=state,
                extra={
                    "user_input": user_input,
                },
            )

            if is_confirmed_value(
                user_input,
            ):
                update = build_confirmed_update(
                    state=state,
                    tool_calls=tool_calls,
                )
                save_checkpoint_if_needed(
                    checkpoint_manager=checkpoint_manager,
                )
                log_tool_agent_state(
                    node_name="tool_confirm",
                    event="tool_confirm_interrupt_confirmed",
                    state={
                        **dict(
                            state
                        ),
                        **update,
                    },
                )
                return update

            update = build_rejected_update(
                state=state,
                tool_calls=tool_calls,
            )
            save_checkpoint_if_needed(
                checkpoint_manager=checkpoint_manager,
            )
            log_tool_agent_state(
                node_name="tool_confirm",
                event="tool_confirm_interrupt_rejected",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
            )
            return update

        update = build_confirmation_update(
            state=state,
            tool_calls=tool_calls,
            permission=ToolAgentPermissionDecision(
                status="pending",
                call_ids=build_call_ids(
                    tool_calls=required_tool_calls,
                ),
                prompt=confirmation_prompt,
                reason="存在需要用户确认的工具调用。",
            ),
            tool_confirmed="pending",
            confirmation_required=True,
            confirmation_prompt=confirmation_prompt,
        )
        save_checkpoint_if_needed(
            checkpoint_manager=checkpoint_manager,
        )
        log_tool_agent_state(
            node_name="tool_confirm",
            event="tool_confirm_pending",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "reason": "存在需要用户确认的工具调用，但当前未注入 interrupt_func。",
            },
        )
        return update

    return tool_agent_tool_confirm_node


def write_tool_confirm_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具确认节点运行时事件。

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
        "tool_agent_tool_confirm_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_confirm_node",
    )


def build_confirmed_update(
    state: Mapping[str, Any],
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    """
    构建用户已确认的 state update。

    功能：
        标记工具调用已经确认，可以继续进入执行阶段。

    参数：
        state:
            当前 LangGraph state。

        tool_calls:
            当前工具调用列表。

    返回值：
        dict[str, Any]:
            可写回 state 的确认结果。
    """

    return build_confirmation_update(
        state=state,
        tool_calls=tool_calls,
        permission=ToolAgentPermissionDecision(
            status="confirmed",
            call_ids=build_call_ids(
                tool_calls=tool_calls,
            ),
            reason="用户已确认执行工具。",
        ),
        tool_confirmed="confirmed",
        confirmation_required=False,
        confirmation_prompt="",
    )


def build_rejected_update(
    state: Mapping[str, Any],
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    """
    构建用户拒绝确认的 state update。

    功能：
        标记工具调用被拒绝，清空待执行工具，并写入取消结果。

    参数：
        state:
            当前 LangGraph state。

        tool_calls:
            当前工具调用列表。

    返回值：
        dict[str, Any]:
            可写回 state 的拒绝结果。
    """

    update = build_confirmation_update(
        state=state,
        tool_calls=tool_calls,
        permission=ToolAgentPermissionDecision(
            status="rejected",
            call_ids=build_call_ids(
                tool_calls=tool_calls,
            ),
            reason="用户拒绝执行工具。",
        ),
        tool_confirmed="rejected",
        confirmation_required=False,
        confirmation_prompt="",
    )
    update["need_tool"] = False
    update["tool_calls"] = []
    update["tool_results"] = [
        "用户取消了工具调用。"
    ]
    update[TOOL_AGENT_RESPONSE_STATE_KEY] = (
        build_tool_agent_response_from_state(
            state={
                **dict(
                    state
                ),
                **update,
            },
        ).model_dump()
    )
    return update


def build_confirmation_update(
    state: Mapping[str, Any],
    tool_calls: list[ToolCall],
    permission: ToolAgentPermissionDecision,
    tool_confirmed: str,
    confirmation_required: bool,
    confirmation_prompt: str,
) -> dict[str, Any]:
    """
    构建统一的确认 state update。

    功能：
        输出旧工具链路兼容字段和 ToolAgent 权限契约字段。

    参数：
        state:
            当前 LangGraph state。

        tool_calls:
            当前工具调用列表。

        permission:
            ToolAgent 权限决定。

        tool_confirmed:
            写回 state 的确认状态。

        confirmation_required:
            是否需要用户确认。

        confirmation_prompt:
            展示给用户的确认提示。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的普通字典。
    """

    update: dict[str, Any] = {
        "tool_confirmed": tool_confirmed,
        "tool_confirmation_required": confirmation_required,
        "tool_confirmation_mode": (
            "batch"
            if confirmation_required
            else "none"
        ),
        "tool_confirmation_prompt": confirmation_prompt,
        "tool_agent_permission": permission.model_dump(),
    }

    merged_state = {
        **dict(
            state
        ),
        **update,
    }
    response = build_tool_agent_response_from_state(
        state=merged_state,
    ).model_dump()
    response["permission"] = permission.model_dump()
    update[TOOL_AGENT_RESPONSE_STATE_KEY] = response

    return update


def build_batch_confirmation_prompt(
    tool_calls: list[ToolCall],
    tool_registry: Any | None = None,
) -> str:
    """
    构建批量工具确认提示。

    功能：
        将多个工具调用合并成一次用户可读的确认文案。

    参数：
        tool_calls:
            需要确认的工具调用列表。

        tool_registry:
            工具注册表对象，用于读取工具描述。

    返回值：
        str:
            用户可读的批量确认提示文本。
    """

    lines = [
        "我需要调用以下工具来回答你的问题：",
        "",
    ]

    for index, tool_call in enumerate(
        tool_calls,
        start=1,
    ):
        metadata = get_registered_tool_metadata(
            tool_name=tool_call.name,
            tool_registry=tool_registry,
        )
        description = (
            metadata.description
            if metadata is not None
            else "执行工具调用"
        )

        lines.extend(
            [
                f"{index}. {description}",
                f"   工具：{tool_call.name}",
                f"   参数：{format_tool_args(tool_call.args)}",
                "",
            ]
        )

    lines.append(
        "是否允许继续？请输入 y 或 n。"
    )

    return "\n".join(
        lines
    )


def format_tool_args(
    args: Mapping[str, Any],
) -> str:
    """
    格式化工具参数。

    功能：
        将工具参数转换成用户可读文本。

    参数：
        args:
            工具参数字典。

    返回值：
        str:
            用户可读的参数文本。
    """

    if not args:
        return "无"

    return "，".join(
        f"{key}={value}"
        for key, value in args.items()
    )


def build_call_ids(
    tool_calls: list[ToolCall],
) -> list[str]:
    """
    构建确认影响的调用 ID。

    功能：
        根据工具调用顺序和工具名生成稳定 call_id。

    参数：
        tool_calls:
            工具调用列表。

    返回值：
        list[str]:
            调用 ID 列表。
    """

    return [
        f"planned_{index}_{tool_call.name}"
        for index, tool_call in enumerate(
            tool_calls,
            start=1,
        )
    ]


def save_checkpoint_if_needed(
    checkpoint_manager: Any | None,
) -> None:
    """
    按需保存 checkpoint。

    功能：
        checkpoint_manager 存在时调用 save_checkpoint。

    参数：
        checkpoint_manager:
            检查点管理器，可能为 None。

    返回值：
        None。
    """

    if checkpoint_manager is not None:
        checkpoint_manager.save_checkpoint()
