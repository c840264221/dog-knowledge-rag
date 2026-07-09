"""
ToolAgent 响应适配节点。

功能：
    将当前 LangGraph state 中的旧工具字段转换成 tool_agent_response。

当前阶段：
    V1.8 已接入新版 ToolAgent 主图分支。
    本节点用于统一生成 tool_agent_response，方便调试和后续响应契约收敛。

专业名词：
    Node：节点，LangGraph 中接收 state 并返回 state 更新的执行单元。
    Adapter：适配器，把旧字段转换成新契约。
    State Update：状态更新，节点返回给 LangGraph 合并进 state 的 dict。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from src.agents.tool_agent.adapters.state_adapter import (
    build_tool_agent_response_state_update,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)


ToolAgentResponseAdapterNode = Callable[
    [Mapping[str, Any]],
    dict[str, Any],
]


def build_tool_agent_response_adapter_node() -> ToolAgentResponseAdapterNode:
    """
    构建 ToolAgent 响应适配节点。

    功能：
        返回一个 LangGraph 可调用节点。
        该节点读取旧工具 state 字段，并返回包含 tool_agent_response 的 state update。

    参数：
        无。

    返回值：
        ToolAgentResponseAdapterNode:
            接收 Mapping[str, Any]，返回 dict[str, Any] 的同步节点函数。
    """

    def tool_agent_response_adapter_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行 ToolAgent 响应适配。

        功能：
            将旧 state 中的 need_tool、tool_calls、tool_results、tool_confirmed、
            tool_round、final_answer 等字段转换成 tool_agent_response。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                包含 tool_agent_response 的 state update。
        """

        update = build_tool_agent_response_state_update(
            state=state,
        )

        tool_agent_response = update.get(
            "tool_agent_response",
            {},
        )

        log_tool_agent_state(
            node_name="response_adapter",
            event="response_adapter_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "response_status": (
                    tool_agent_response.get(
                        "status",
                    )
                    if isinstance(
                        tool_agent_response,
                        Mapping,
                    )
                    else ""
                ),
            },
        )

        return update

    return tool_agent_response_adapter_node

