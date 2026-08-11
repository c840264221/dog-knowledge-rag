"""宠物档案字段最小权限读取策略。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import get_args

from src.memory.memory_schema import (
    PetProfileAttribute,
    PetProfileFieldAccessDecision,
)
from src.settings import settings


# 兼容现有测试和调用方的模块级视图；真实策略来源统一放在 settings 中。
PET_PROFILE_READ_ALLOWLISTS: dict[str, frozenset[str]] = {
    agent_name: policy.database_read_attributes
    for agent_name, policy in settings.pet_profile_access.agent_policies.items()
}
PET_PROFILE_PROCESSING_ALLOWLISTS: dict[str, frozenset[str]] = {
    agent_name: policy.processing_attributes
    for agent_name, policy in settings.pet_profile_access.agent_policies.items()
}


def resolve_pet_profile_field_access(
    *,
    purpose: str,
    agent_name: str,
    skill_required_attributes: Iterable[str] = (),
    suggested_attributes: Iterable[str] = (),
) -> PetProfileFieldAccessDecision:
    """
    计算本轮允许从数据库读取的宠物档案字段。

    功能：
        先合并上游建议字段与 Skill 必需字段，再使用宠物档案字段契约和
        Agent 数据库读取白名单进行裁剪。Skill 必需字段不会自动越权。

    参数含义：
        purpose：读取用途，只允许 skill_prefill 或 answer_context。
        agent_name：申请读取档案的 Agent 名称。
        skill_required_attributes：当前 Skill 运行所需的档案字段。
        suggested_attributes：上游查询理解阶段建议读取的档案字段。

    返回值含义：
        PetProfileFieldAccessDecision：字段申请、允许和拒绝结果。
    """

    normalized_purpose = str(purpose or "").strip()
    if normalized_purpose not in {"skill_prefill", "answer_context"}:
        raise ValueError(f"不支持的宠物档案读取用途: {normalized_purpose}")

    normalized_agent_name = str(agent_name or "").strip()

    # Skill 声明和上游建议都可能来自配置或模型输出，先统一清理空值和重复值。
    normalized_skill_attributes = _unique_non_empty_strings(
        skill_required_attributes
    )
    normalized_suggested_attributes = _unique_non_empty_strings(
        suggested_attributes
    )
    requested_attributes = _unique_non_empty_strings(
        [*normalized_suggested_attributes, *normalized_skill_attributes]
    )

    valid_attributes = frozenset(get_args(PetProfileAttribute))
    agent_allowlist = PET_PROFILE_READ_ALLOWLISTS.get(
        normalized_agent_name,
        frozenset(),
    )
    processing_allowlist = PET_PROFILE_PROCESSING_ALLOWLISTS.get(
        normalized_agent_name,
        frozenset(),
    )
    allowed_attributes = [
        attribute
        for attribute in requested_attributes
        if attribute in valid_attributes
        and attribute in agent_allowlist
        and attribute in processing_allowlist
    ]
    denied_skill_required_attributes = [
        attribute
        for attribute in normalized_skill_attributes
        if attribute not in allowed_attributes
    ]
    invalid_suggested_attributes = [
        attribute
        for attribute in normalized_suggested_attributes
        if attribute not in valid_attributes
    ]
    processing_denied_attributes = [
        attribute
        for attribute in requested_attributes
        if attribute in valid_attributes
        and attribute not in processing_allowlist
    ]
    # 数据库不能主动读取、但 Agent 可以处理的字段，可以通过不暴露权限细节的
    # 普通业务澄清由用户补充。
    user_suppliable_skill_attributes = [
        attribute
        for attribute in normalized_skill_attributes
        if attribute in valid_attributes
        and attribute in processing_allowlist
        and attribute not in agent_allowlist
    ]
    # 不属于数据契约或 Agent 无权处理的 Skill 字段，即使用户提供也不能使用。
    blocked_skill_attributes = [
        attribute
        for attribute in normalized_skill_attributes
        if attribute not in valid_attributes
        or attribute not in processing_allowlist
    ]
    if blocked_skill_attributes:
        skill_resolution_action = "degrade_or_cancel"
    elif user_suppliable_skill_attributes:
        skill_resolution_action = "clarify"
    else:
        skill_resolution_action = "proceed"

    return PetProfileFieldAccessDecision(
        purpose=normalized_purpose,
        agent_name=normalized_agent_name,
        skill_required_attributes=normalized_skill_attributes,
        suggested_attributes=normalized_suggested_attributes,
        requested_attributes=requested_attributes,
        allowed_attributes=allowed_attributes,
        denied_skill_required_attributes=denied_skill_required_attributes,
        user_suppliable_skill_attributes=user_suppliable_skill_attributes,
        blocked_skill_attributes=blocked_skill_attributes,
        processing_denied_attributes=processing_denied_attributes,
        invalid_suggested_attributes=invalid_suggested_attributes,
        skill_resolution_action=skill_resolution_action,
        reason=(
            f"申请读取 {len(requested_attributes)} 个字段，"
            f"Agent 数据库白名单允许 {len(allowed_attributes)} 个字段，"
            f"需要用户补充 {len(user_suppliable_skill_attributes)} 个字段，"
            f"禁止处理 {len(blocked_skill_attributes)} 个 Skill 必需字段。"
        ),
    )


def _unique_non_empty_strings(values: Iterable[str]) -> list[str]:
    """
    按原顺序清理并去重字段名称。

    参数含义：
        values：可能包含空白和重复项的字段名称序列。

    返回值含义：
        list[str]：只包含非空唯一字段名称的列表。
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        result.append(normalized_value)
    return result
