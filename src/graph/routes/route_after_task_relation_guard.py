"""任务关系门卫完成后的主图路由。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


TaskRelationGuardRoute = Literal[
    "continue",
    "finish",
]


def route_after_task_relation_guard(
    state: Mapping[str, Any],
) -> TaskRelationGuardRoute:
    """
    根据任务关系判断决定是否继续调用 Memory 和业务路由。

    功能：
        取消旧任务或无法区分新旧任务时结束本轮；普通输入、恢复输入和
        新任务继续进入记忆抽取节点。

    参数含义：
        state:
            任务关系门卫执行后的主图状态。

    返回值含义：
        TaskRelationGuardRoute:
            continue 表示继续，finish 表示直接结束本轮。
    """

    raw_decision = state.get("task_relation_decision")
    relation = (
        str(raw_decision.get("relation") or "").strip()
        if isinstance(raw_decision, Mapping)
        else ""
    )
    if relation in {"cancel", "ambiguous"}:
        return "finish"
    return "continue"


def build_task_relation_guard_route_map(
    end_node: Any,
) -> dict[str, Any]:
    """
    构建任务关系逻辑路由到真实主图节点的映射。

    功能：
        把 continue 映射到 memory_extract，把 finish 映射到 LangGraph END。

    参数含义：
        end_node:
            LangGraph 的 END 结束节点。

    返回值含义：
        dict[str, Any]:
            add_conditional_edges 使用的路由映射。
    """

    return {
        "continue": "memory_extract",
        "finish": end_node,
    }
