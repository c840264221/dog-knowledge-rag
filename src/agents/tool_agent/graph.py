"""
ToolAgent LangGraph 子图构建模块。

功能：
    使用 LangGraph 的 add_node、add_edge、add_conditional_edges 和 compile
    构建新版 ToolAgent（工具智能体）子图。

当前阶段：
    V1.8 ToolAgent Subgraph MVP。
    当前子图已经开始接入主图。
    interrupt/resume 仍采用可注入方式，后续再接真实 UI 恢复链路。

专业名词：
    Graph：图，由节点和边组成的执行流程。
    Subgraph：子图，可以作为主图中的一个 Agent 节点使用。
    Conditional Edge：条件边，根据 state 决定下一步走向。
    Compile：编译，把图定义转换成可执行 LangGraph 对象。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
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
    get_permission_status,
)
from src.agents.tool_agent.nodes.tool_parse_node import (
    build_tool_agent_tool_parse_node,
)
from src.graph.states.dog_state import DogState

from src.logger import logger


ToolConfirmRoute = Literal[
    "pending_confirmation",
    "rejected",
    "allowed",
]


def build_tool_agent_graph(
    parser: Any | None = None,
    llm_provider: Any | None = None,
    tool_registry: Any | None = None,
    executor: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
    interrupt_func: Callable[[str], Any] | None = None,
):
    """
    构建 ToolAgent LangGraph 子图。

    功能：
        将 ToolAgent 内部节点注册成 LangGraph 子图。
        当前流程为：
        tool_parse -> tool_confirm -> 条件路由 ->
        tool_execute / tool_answer / response_adapter。

    参数：
        parser:
            工具解析器。测试时可传入 fake parser。

        llm_provider:
            LLM Provider（大语言模型服务提供者）。
            当 parser=None 时，工具解析节点会用它构建 LLM 解析链。

        tool_registry:
            工具注册表。确认节点用它判断工具是否需要用户确认。

        executor:
            工具执行器。执行节点会把它传给 runtime_adapter。

        checkpoint_manager:
            检查点管理器。内部节点按需保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。内部节点用它写入 node 和 timeline。

        interrupt_func:
            Interrupt 函数。传入 LangGraph interrupt 时，
            tool_confirm 节点会真正暂停子图并等待用户确认。

    返回值：
        CompiledStateGraph:
            编译后的 ToolAgent LangGraph 子图。
    """

    logger.info("构建 tool_agent 中...")

    graph = StateGraph(
        DogState
    )

    # 先注册工具解析节点，负责从用户问题生成 tool_calls。
    graph.add_node(
        "tool_parse",
        build_tool_agent_tool_parse_node(
            parser=parser,
            llm_provider=llm_provider,
            checkpoint_manager=checkpoint_manager,
            runtime_context_getter=runtime_context_getter,
        ),
    )

    # 再注册工具确认节点，负责生成权限状态和确认提示。
    graph.add_node(
        "tool_confirm",
        build_tool_agent_tool_confirm_node(
            tool_registry=tool_registry,
            checkpoint_manager=checkpoint_manager,
            runtime_context_getter=runtime_context_getter,
            interrupt_func=interrupt_func,
        ),
    )

    # 注册工具执行节点，只在权限允许时调用底层 ToolExecutor。
    graph.add_node(
        "tool_execute",
        build_tool_agent_tool_execute_node(
            executor=executor,
            checkpoint_manager=checkpoint_manager,
            runtime_context_getter=runtime_context_getter,
        ),
    )

    # 注册工具答案节点，把 tool_results 转换成 final_answer。
    graph.add_node(
        "tool_answer",
        build_tool_agent_tool_answer_node(
            checkpoint_manager=checkpoint_manager,
            runtime_context_getter=runtime_context_getter,
        ),
    )

    # 注册响应适配节点，统一生成 tool_agent_response。
    graph.add_node(
        "response_adapter",
        build_tool_agent_response_adapter_node(),
    )

    graph.set_entry_point(
        "tool_parse"
    )

    graph.add_edge(
        "tool_parse",
        "tool_confirm",
    )

    # 确认节点之后根据权限状态决定是否执行工具。
    graph.add_conditional_edges(
        "tool_confirm",
        route_after_tool_confirm,
        {
            "pending_confirmation": "response_adapter",
            "rejected": "tool_answer",
            "allowed": "tool_execute",
        },
    )

    graph.add_edge(
        "tool_execute",
        "tool_answer",
    )
    graph.add_edge(
        "tool_answer",
        "response_adapter",
    )
    graph.add_edge(
        "response_adapter",
        END,
    )

    logger.info("✅ tool_agent 构建完成")

    return graph.compile()


def route_after_tool_confirm(
    state: Mapping[str, Any],
) -> ToolConfirmRoute:
    """
    根据工具确认结果决定下一步路由。

    功能：
        读取 state 中的工具权限状态。
        pending 表示等待用户确认，直接结束当前 ToolAgent 子图。
        rejected 表示用户拒绝，进入答案节点生成取消说明。
        confirmed/not_required 表示允许执行，进入工具执行节点。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        ToolConfirmRoute:
            pending_confirmation、rejected 或 allowed。
    """

    permission_status = get_permission_status(
        state=state,
    )

    if permission_status == "pending":
        log_tool_agent_state(
            node_name="route_after_tool_confirm",
            event="route_after_tool_confirm",
            state=state,
            extra={
                "permission_status": permission_status,
                "route": "pending_confirmation",
            },
        )
        return "pending_confirmation"

    if permission_status == "rejected":
        log_tool_agent_state(
            node_name="route_after_tool_confirm",
            event="route_after_tool_confirm",
            state=state,
            extra={
                "permission_status": permission_status,
                "route": "rejected",
            },
        )
        return "rejected"

    log_tool_agent_state(
        node_name="route_after_tool_confirm",
        event="route_after_tool_confirm",
        state=state,
        extra={
            "permission_status": permission_status,
            "route": "allowed",
        },
    )

    return "allowed"
