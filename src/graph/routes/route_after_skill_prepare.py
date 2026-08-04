"""主图 Skill（技能）准备节点的后置路由。"""

from __future__ import annotations

from typing import Any, Literal, Mapping


SkillPrepareRoute = Literal[
    "dog_knowledge_agent",
    "general_agent",
    "awaiting_input",
]


def route_after_skill_prepare(
    state: Mapping[str, Any],
) -> SkillPrepareRoute:
    """
    根据 Skill 准备结果决定继续执行还是等待用户输入。

    功能：
        awaiting_input 时结束本轮主图并等待用户补充；no_skill 或 ready 时
        继续进入 RootAgent 首轮确定的目标 Agent。

    参数含义：
        state:
            Skill 准备节点执行后的主图状态。

    返回值含义：
        SkillPrepareRoute:
            awaiting_input、dog_knowledge_agent 或 general_agent。
    """

    if str(state.get("skill_status") or "").strip() == "awaiting_input":
        return "awaiting_input"

    target_agent = str(state.get("skill_target_agent") or "").strip()
    if target_agent == "dog_knowledge_agent":
        return "dog_knowledge_agent"
    return "general_agent"


def build_skill_prepare_route_map(end_node: Any) -> dict[str, Any]:
    """
    构建 Skill 后置路由到真实主图节点的映射。

    功能：
        把逻辑目标转换成 StateGraph 节点名，并让等待输入状态走到 END。

    参数含义：
        end_node:
            LangGraph END 节点。

    返回值含义：
        dict[str, Any]:
            add_conditional_edges 使用的路由映射表。
    """

    return {
        "dog_knowledge_agent": "dog_knowledge_agent",
        "general_agent": "general",
        "awaiting_input": end_node,
    }
