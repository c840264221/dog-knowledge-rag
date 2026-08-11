"""回答阶段宠物档案访问规划节点单元测试。"""

from src.agents.dog_knowledge_agent.nodes.answer_profile_access_plan_node import (
    build_answer_profile_access_plan_node,
)


def test_answer_profile_plan_should_ignore_skill_required_fields() -> None:
    """验证回答读取计划只使用查询理解建议，不混入 Skill 补参字段。"""

    node = build_answer_profile_access_plan_node()
    update = node(
        {
            "pet_profile_suggested_attributes": ["weight_kg"],
            "skill_required_pet_profile_attributes": ["breed", "age_years"],
        }
    )

    decision = update["answer_profile_access_decision"]
    assert decision["purpose"] == "answer_context"
    assert decision["allowed_attributes"] == ["weight_kg"]
    assert decision["skill_required_attributes"] == []
