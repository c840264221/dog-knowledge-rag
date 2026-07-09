"""
ToolAgent 统一入口。

功能：
    组装新版 ToolAgent（工具智能体）的最小主链路。

当前阶段：
    V1.8 ToolAgent Graph Assembly MVP。
    这里只把 parse -> confirm -> execute -> answer -> response_adapter 五个节点串起来，
    暂时不接入主图。

专业名词：
    Agent：智能体，负责一类完整业务能力的执行入口。
    Node：节点，接收 state 并返回 state update 的执行单元。
    State Update：状态更新，节点返回后合并进当前 state 的 dict。
    Assembly：组装，把已有节点按业务顺序串成一个可调用入口。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.tool_agent.nodes.response_adapter_node import (
    build_tool_agent_response_adapter_node,
)
from src.agents.tool_agent.nodes.tool_answer_node import (
    build_tool_agent_tool_answer_node,
)
from src.agents.tool_agent.nodes.tool_confirm_node import (
    build_tool_agent_tool_confirm_node,
)
from src.agents.tool_agent.nodes.tool_execute_node import (
    build_tool_agent_tool_execute_node,
)
from src.agents.tool_agent.nodes.tool_parse_node import (
    build_tool_agent_tool_parse_node,
)


ToolAgentNode = Callable[
    [Mapping[str, Any]],
    Awaitable[dict[str, Any]],
]


def build_tool_agent(
    parser: Any | None = None,
    llm_provider: Any | None = None,
    tool_registry: Any | None = None,
    executor: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolAgentNode:
    """
    构建 ToolAgent 统一入口节点。

    功能：
        将 ToolAgent 内部的工具解析、工具确认、工具执行、工具答案、响应适配五个节点串成一个 async 节点。
        当前负责生成标准工具调用计划、确认状态，并在权限允许时执行工具和生成回答。

    参数：
        parser:
            工具解析器。可传入测试 fake parser，也可以不传。

        llm_provider:
            LLM Provider（大语言模型服务提供者）。
            当 parser=None 时，工具解析节点会用它构建 LLM 解析链。

        tool_registry:
            工具注册表。工具确认节点会用它判断某个工具是否需要用户确认。

        executor:
            工具执行器。工具执行节点会把它传给 runtime_adapter。
            测试时可以传入 fake executor，避免调用真实外部工具。

        checkpoint_manager:
            检查点管理器。内部节点按需保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。内部节点用它写入当前 node 和 timeline。

    返回值：
        ToolAgentNode:
            async ToolAgent 入口节点，接收 state，返回合并后的完整 state。
    """

    parse_node = build_tool_agent_tool_parse_node(
        parser=parser,
        llm_provider=llm_provider,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )
    confirm_node = build_tool_agent_tool_confirm_node(
        tool_registry=tool_registry,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )
    execute_node = build_tool_agent_tool_execute_node(
        executor=executor,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )
    answer_node = build_tool_agent_tool_answer_node(
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )
    response_adapter_node = build_tool_agent_response_adapter_node()

    async def tool_agent_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行 ToolAgent 最小链路。

        功能：
            1. 调用工具解析节点，生成 need_tool 和 tool_calls。
            2. 调用工具确认节点，生成 permission 和确认提示。
            3. 调用工具执行节点，在权限允许时执行工具。
            4. 调用工具答案节点，将 tool_results 转成 final_answer。
            5. 调用响应适配节点，刷新 tool_agent_response。
            6. 返回合并后的完整 state，方便后续主图直接继续使用。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                合并 ToolAgent 内部节点更新后的完整 state。
        """

        working_state = dict(
            state
        )

        # 先解析用户问题，判断是否需要调用工具，并生成 tool_calls。
        parse_update = await parse_node(
            working_state
        )
        working_state = merge_state_update(
            state=working_state,
            update=parse_update,
        )

        # 再根据工具注册表判断是否需要用户确认。
        confirm_update = confirm_node(
            working_state
        )
        working_state = merge_state_update(
            state=working_state,
            update=confirm_update,
        )

        # 然后在权限允许时执行工具；pending/rejected 会自动跳过。
        execute_update = await execute_node(
            working_state
        )
        working_state = merge_state_update(
            state=working_state,
            update=execute_update,
        )

        # 接着把结构化工具结果格式化成用户可读 final_answer。
        answer_update = answer_node(
            working_state
        )
        working_state = merge_state_update(
            state=working_state,
            update=answer_update,
        )

        # 最后统一刷新 ToolAgent 响应契约，保证输出字段稳定。
        response_update = response_adapter_node(
            working_state
        )
        working_state = merge_state_update(
            state=working_state,
            update=response_update,
        )

        return working_state

    return tool_agent_node


def merge_state_update(
    state: Mapping[str, Any],
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    合并 state 和节点返回的 update。

    功能：
        模拟 LangGraph 的 state 合并效果。
        当前 ToolAgent 统一入口还没有真正编译成子图，所以用这个函数在入口内部完成小步合并。

    参数：
        state:
            当前已有 state。

        update:
            节点返回的 state update。可以为 None 或空 dict。

    返回值：
        dict[str, Any]:
            合并后的新 state，不会原地修改传入的 state。
    """

    merged_state = dict(
        state
    )

    if update:
        merged_state.update(
            dict(
                update
            )
        )

    return merged_state
