"""回答生成前的宠物档案字段访问规划节点。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from src.memory.pet_profile_access_policy import (
    resolve_pet_profile_field_access,
)


AnswerProfileAccessPlanNode = Callable[
    [Mapping[str, Any]],
    dict[str, Any],
]


def build_answer_profile_access_plan_node(
    *,
    agent_name: str = "dog_knowledge_agent",
) -> AnswerProfileAccessPlanNode:
    """
    构建回答生成阶段的宠物档案访问规划节点。

    参数含义：
        agent_name：申请读取宠物档案的 Agent（智能体）名称。

    返回值含义：
        AnswerProfileAccessPlanNode：把查询理解建议转换为最小权限读取决策的节点。
    """

    def answer_profile_access_plan_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        计算本次回答真正允许读取的宠物档案字段。

        参数含义：
            state：包含查询理解建议字段的当前 DogState（狗狗主图状态）。

        返回值含义：
            dict[str, Any]：只包含 answer_profile_access_decision 的局部状态更新。
        """

        raw_suggested_attributes = state.get(
            "pet_profile_suggested_attributes"
        )
        suggested_attributes = (
            list(raw_suggested_attributes)
            if isinstance(raw_suggested_attributes, (list, tuple, set))
            else []
        )
        decision = resolve_pet_profile_field_access(
            purpose="answer_context",
            agent_name=agent_name,
            suggested_attributes=suggested_attributes,
        )
        return {
            "answer_profile_access_decision": decision.model_dump(
                mode="python"
            ),
        }

    return answer_profile_access_plan_node
