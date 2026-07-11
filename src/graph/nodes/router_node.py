from __future__ import annotations

from typing import (
    Any,
)

from src.agents.root_agent.supervisor import (
    root_supervisor_node,
)
from src.agents.tool_agent.adapters.clarification_resume_adapter import (
    resolve_tool_clarification_input,
)
from src.graph.states.dog_state import (
    DogState,
)


async def semantic_router_node(
        state: DogState,
) -> dict[str, Any]:
    """
    Main Graph 语义路由兼容节点。

    功能：
        作为旧主图节点名 semantic_router 的兼容入口。
        V1.7 起，真实路由逻辑已经迁移到：

            src.agents.root_agent.supervisor.root_supervisor_node

        当前函数只负责转调新版 Root Supervisor，避免大改 GraphRuntimeService：
        1. 主图节点名仍然保留 semantic_router。
        2. 旧 checkpoint / timeline / graph edge 不需要立刻迁移。
        3. 新版路由逻辑集中维护在 root_agent 目录。

    参数：
        state:
            DogState，LangGraph 当前状态。

    返回值：
        dict[str, Any]:
            root_supervisor_node 返回的局部状态。

    专业名词：
        Adapter：
            适配器。保留旧入口，但内部调用新版实现。

        Backward Compatibility：
            向后兼容。避免旧主图节点名、checkpoint、日志链路立刻失效。
    """

    clarification_resolution = resolve_tool_clarification_input(
        state=state,
    )
    clarification_update = clarification_resolution.get(
        "state_update",
        {},
    )
    resolved_state = {
        **dict(state),
        **dict(clarification_update),
    }
    root_update = await root_supervisor_node(
        resolved_state,
    )

    return {
        **dict(clarification_update),
        **root_update,
    }
