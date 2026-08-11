"""宠物档案字段最小权限策略单元测试。"""

import src.memory.pet_profile_access_policy as access_policy

from src.memory.pet_profile_access_policy import (
    resolve_pet_profile_field_access,
)


def test_access_policy_should_union_requests_then_apply_allowlist() -> None:
    """验证建议字段与 Skill 字段先取并集，再经过 Agent 白名单。"""

    decision = resolve_pet_profile_field_access(
        purpose="answer_context",
        agent_name="dog_knowledge_agent",
        skill_required_attributes=["breed", "age_years"],
        suggested_attributes=["weight_kg", "breed", "unknown_field"],
    )

    assert decision.requested_attributes == [
        "weight_kg",
        "breed",
        "unknown_field",
        "age_years",
    ]
    assert decision.allowed_attributes == [
        "weight_kg",
        "breed",
        "age_years",
    ]
    assert decision.invalid_suggested_attributes == ["unknown_field"]
    assert decision.denied_skill_required_attributes == []


def test_access_policy_should_not_trust_skill_above_agent_allowlist() -> None:
    """验证 Skill 必需字段不能绕过未登记 Agent 的数据库读取权限。"""

    decision = resolve_pet_profile_field_access(
        purpose="skill_prefill",
        agent_name="unknown_agent",
        skill_required_attributes=["breed", "training_goal"],
    )

    assert decision.allowed_attributes == []
    assert decision.denied_skill_required_attributes == [
        "breed",
        "training_goal",
    ]


def test_access_policy_should_return_empty_without_field_request() -> None:
    """验证没有明确字段申请时不会默认读取完整宠物档案。"""

    decision = resolve_pet_profile_field_access(
        purpose="answer_context",
        agent_name="dog_knowledge_agent"
    )

    assert decision.requested_attributes == []
    assert decision.allowed_attributes == []


def test_access_policy_should_apply_read_and_processing_allowlists(
    monkeypatch,
) -> None:
    """验证合法字段仍必须同时通过数据库读取权限和 Agent 处理权限。"""

    monkeypatch.setitem(
        access_policy.PET_PROFILE_READ_ALLOWLISTS,
        "limited_agent",
        frozenset({"breed", "age_years", "health_condition"}),
    )
    monkeypatch.setitem(
        access_policy.PET_PROFILE_PROCESSING_ALLOWLISTS,
        "limited_agent",
        frozenset({"breed", "age_years"}),
    )

    decision = resolve_pet_profile_field_access(
        purpose="skill_prefill",
        agent_name="limited_agent",
        skill_required_attributes=["age_years", "training_goal"],
        suggested_attributes=[
            "breed",
            "health_condition",
            "allergy",
            "unknown_field",
        ],
    )

    assert decision.requested_attributes == [
        "breed",
        "health_condition",
        "allergy",
        "unknown_field",
        "age_years",
        "training_goal",
    ]
    assert decision.allowed_attributes == ["breed", "age_years"]
    assert decision.invalid_suggested_attributes == ["unknown_field"]
    assert decision.denied_skill_required_attributes == ["training_goal"]
    assert "health_condition" in decision.processing_denied_attributes
    assert decision.blocked_skill_attributes == ["training_goal"]
    assert decision.skill_resolution_action == "degrade_or_cancel"


def test_access_policy_should_clarify_without_revealing_read_policy(
    monkeypatch,
) -> None:
    """数据库不可读但 Agent 可处理时，应由普通业务澄清补充字段。"""

    monkeypatch.setitem(
        access_policy.PET_PROFILE_READ_ALLOWLISTS,
        "training_agent",
        frozenset({"breed"}),
    )
    monkeypatch.setitem(
        access_policy.PET_PROFILE_PROCESSING_ALLOWLISTS,
        "training_agent",
        frozenset({"breed", "training_goal"}),
    )

    decision = resolve_pet_profile_field_access(
        purpose="skill_prefill",
        agent_name="training_agent",
        skill_required_attributes=["breed", "training_goal"],
    )

    assert decision.allowed_attributes == ["breed"]
    assert decision.user_suppliable_skill_attributes == ["training_goal"]
    assert decision.blocked_skill_attributes == []
    assert decision.skill_resolution_action == "clarify"
