from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
)

from src.agents.root_agent.supervisor import (
    root_supervisor_node,
)
from src.agents.tool_agent.adapters.clarification_resume_adapter import (
    resolve_tool_clarification_input,
)
from src.agents.collaboration.adapters import (
    MultiAgentClarificationFieldResolver,
    resolve_multi_agent_resume_input,
)
from src.graph.states.dog_state import (
    DogState,
)
from src.runtime.resume import resolve_pending_task_relation
from src.runtime.resume.pending_tasks import PendingTaskError
from src.runtime.resume.state_adapter import transition_pending_task_kind


async def semantic_router_node(
        state: DogState,
        *,
        multi_agent_field_resolver: (
            MultiAgentClarificationFieldResolver | None
        ) = None,
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

    # 新主图已经在独立门卫节点完成分类；旧调用方直接调用本节点时保留兼容。
    if bool(state.get("task_relation_guard_processed")):
        task_relation_update: dict[str, Any] = {}
        raw_decision = state.get("task_relation_decision")
        relation_action = (
            str(raw_decision.get("relation") or "none").strip()
            if isinstance(raw_decision, Mapping)
            else "none"
        )
    else:
        task_relation_resolution = resolve_pending_task_relation(state)
        task_relation_update = dict(
            task_relation_resolution.get("state_update") or {}
        )
        relation_action = str(
            task_relation_resolution.get("action") or "none"
        ).strip()
    resolved_state = {
        **dict(state),
        **dict(task_relation_update),
    }

    # 只有确认继续旧任务时，才允许各业务恢复适配器解析补充内容。
    should_run_business_resume = relation_action in {
        "none",
        "resume",
    }
    selected_pending_kind = str(
        resolved_state.get("task_relation_pending_kind") or ""
    ).strip()
    should_resume_tool = (
        should_run_business_resume
        and selected_pending_kind in {"", "tool"}
    )
    should_resume_multi_agent = (
        should_run_business_resume
        and selected_pending_kind in {"", "multi_agent"}
    )
    clarification_resolution = (
        resolve_tool_clarification_input(state=resolved_state)
        if should_resume_tool
        else {"action": "none", "state_update": {}}
    )
    clarification_update = clarification_resolution.get(
        "state_update",
        {},
    )
    resolved_state = {**resolved_state, **dict(clarification_update)}
    multi_agent_resolution = (
        await resolve_multi_agent_resume_input(
            resolved_state,
            field_resolver=multi_agent_field_resolver,
        )
        if should_resume_multi_agent
        else {"action": "none", "state_update": {}}
    )
    multi_agent_update = multi_agent_resolution.get(
        "state_update",
        {},
    )
    resolved_state = {
        **resolved_state,
        **dict(multi_agent_update),
    }
    pending_task_update: dict[str, Any] = {}
    try:
        raw_pending_tasks = resolved_state.get("pending_tasks")
        if clarification_resolution.get("action") == "resumed":
            raw_pending_tasks = transition_pending_task_kind(
                raw_tasks=(
                    raw_pending_tasks
                    if isinstance(raw_pending_tasks, Mapping)
                    else None
                ),
                task_kind="tool",
                target_status="running",
            )
        if multi_agent_resolution.get("action") in {"resume", "replan"}:
            raw_pending_tasks = transition_pending_task_kind(
                raw_tasks=(
                    raw_pending_tasks
                    if isinstance(raw_pending_tasks, Mapping)
                    else None
                ),
                task_kind="multi_agent",
                target_status="running",
            )
        if isinstance(raw_pending_tasks, Mapping):
            pending_task_update = {
                "pending_tasks": dict(raw_pending_tasks),
            }
    except (PendingTaskError, TypeError, ValueError) as exc:
        return {
            **dict(task_relation_update),
            **dict(clarification_update),
            **dict(multi_agent_update),
            "route_decision": {
                "route": "FINISH",
                "query_type": "finish",
                "confidence": 1.0,
                "reason": "等待任务状态迁移失败，本轮业务未执行。",
                "requires_rag": False,
                "requires_tool": False,
                "requires_memory": False,
                "source": "pending_task_state_guard",
                "hints": {
                    "error_type": type(exc).__name__,
                },
            },
            "next_agent": "FINISH",
            "current_agent": "root_agent",
            "tool_calls": [],
            "need_tool": False,
            "tool_agent_clarification_resume_ready": False,
            "multi_agent_resume_ready": False,
            "final_answer": (
                "任务状态已经发生变化，本轮没有继续执行。"
                "请重新查看当前等待任务后再操作。"
            ),
        }
    resolved_state = {
        **resolved_state,
        **pending_task_update,
    }
    root_update = await root_supervisor_node(
        resolved_state,
    )

    return {
        **dict(task_relation_update),
        **dict(clarification_update),
        **dict(multi_agent_update),
        **pending_task_update,
        **root_update,
    }


def build_semantic_router_node(
    *,
    multi_agent_field_resolver: (
        MultiAgentClarificationFieldResolver | None
    ) = None,
):
    """
    构建注入多智能体字段解析器的语义路由节点。

    功能：
        生产主图可以注入带 LLM 兜底的字段解析器；旧测试和兼容调用仍可
        直接使用 semantic_router_node，并自动退化为确定性规则解析。

    参数含义：
        multi_agent_field_resolver:
            可选的多智能体澄清字段解析器。

    返回值含义：
        Callable:
            可注册到 LangGraph 的异步语义路由节点。
    """

    async def _semantic_router_node(state: DogState) -> dict[str, Any]:
        return await semantic_router_node(
            state,
            multi_agent_field_resolver=multi_agent_field_resolver,
        )

    return _semantic_router_node
