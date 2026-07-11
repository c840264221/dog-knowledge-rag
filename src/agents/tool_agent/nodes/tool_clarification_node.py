"""
ToolAgent 工具参数澄清节点。

功能：
    将结构化澄清请求转换成当前轮次面向用户的回答。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.tool_call_validation_adapter import (
    TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY,
)
from src.agents.tool_agent.debug.state_logging import log_tool_agent_state
from src.runtime.context import runtime_ctx


ToolClarificationNode = Callable[[Mapping[str, Any]], dict[str, Any]]


def build_tool_agent_tool_clarification_node(
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolClarificationNode:
    """
    构建 ToolAgent 参数澄清节点。

    功能：
        创建读取澄清请求并输出用户问题的 LangGraph 同步节点。

    参数：
        runtime_context_getter:
            RuntimeContext（运行时上下文）获取函数。

    返回值：
        ToolClarificationNode:
            接收 state 并返回普通 dict 状态更新的节点函数。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    def tool_agent_tool_clarification_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        生成当前轮次的参数澄清回答。

        功能：
            读取结构化澄清请求，将 question 写入 final_answer，
            并标记当前链路正在等待用户补充输入。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                包含 final_answer、pending_prompt 和等待状态的普通字典。
        """

        runtime_context = runtime_context_getter()
        if runtime_context is not None:
            runtime_context.state().set_node(
                "tool_agent_tool_clarification_node"
            )
            runtime_context.timeline().add_event(
                event_type="node",
                name="tool_agent_tool_clarification_node",
            )

        request = state.get(
            TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY,
            {},
        )
        question = (
            str(request.get("question", ""))
            if isinstance(request, Mapping)
            else ""
        )
        question = question or "请补充工具调用所需的必填参数。"
        update = {
            "final_answer": question,
            "pending_prompt": question,
            "waiting_user_input": True,
            "has_asked_user": True,
            "tool_agent_answer_source": "tool_clarification",
        }

        log_tool_agent_state(
            node_name="tool_clarification",
            event="tool_clarification_created",
            state={
                **dict(state),
                **update,
            },
            extra={
                "missing_fields": request.get("missing_fields", [])
                if isinstance(request, Mapping)
                else [],
            },
        )
        return update

    return tool_agent_tool_clarification_node
