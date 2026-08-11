"""Skill（技能）执行前的宠物档案预填公共能力。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from src.logger import logger
from src.memory.memory_schema import PetProfileRecallResult
from src.memory.pet_profile_access_policy import (
    resolve_pet_profile_field_access,
)
from src.skills.runtime import SkillRuntime


class PetProfileRecallService(Protocol):
    """声明 Skill 预填只需要宠物档案服务提供的最小调用能力。"""

    def recall_profile(self, **kwargs: Any) -> PetProfileRecallResult:
        """
        召回指定用户当前宠物的档案字段。

        参数含义：
            **kwargs：用户标识、当前宠物标识和允许读取的字段。

        返回值含义：
            PetProfileRecallResult：结构化宠物档案召回结果。
        """


@dataclass(frozen=True)
class SkillPetProfilePrefillResult:
    """保存一次 Skill 宠物档案预填产生的数据和状态更新。"""

    # 交给 SkillRuntime 的外部默认输入，例如 pet_profile 下的年龄和犬种。
    available_input_sources: dict[str, Mapping[str, Any]]

    # 交给主图或 Worker 保存的权限决策、召回结果和当前宠物标识。
    state_update: dict[str, Any]


def prepare_skill_pet_profile_prefill(
    *,
    skill_runtime: SkillRuntime,
    selected_skill_id: str | None,
    provided_inputs: Mapping[str, Any],
    ignored_input_ids: list[str],
    agent_name: str,
    pet_profile_service: PetProfileRecallService | None,
    user_id: str,
    active_pet_key: Any = None,
    active_pet_name: Any = None,
) -> SkillPetProfilePrefillResult:
    """
    为已经选中的 Skill 准备允许从宠物档案补全的默认输入。

    功能：
        先计算 Skill 当前仍缺少哪些宠物档案字段，再经过 Agent 字段权限
        校验，只查询允许读取的字段。查询成功后把档案事实作为 Skill 默认
        输入，同时返回可写入状态的权限和召回记录。

    参数含义：
        skill_runtime：提供 Skill 输入契约和缺失字段计算能力的运行器。
        selected_skill_id：当前已经选中的技能编号；为空表示没有命中技能。
        provided_inputs：用户本轮和历史轮次已经提供的技能输入。
        ignored_input_ids：简化执行时允许忽略的技能输入编号。
        agent_name：本次执行 Skill 的 Agent 名称，用于字段权限校验。
        pet_profile_service：可选宠物档案服务；为空时不查询数据库。
        user_id：当前用户标识，用于隔离不同用户的宠物档案。
        active_pet_key：上游已经确认的当前宠物稳定标识。
        active_pet_name：上游已经确认的当前宠物展示名称。

    返回值含义：
        SkillPetProfilePrefillResult：包含 Skill 外部默认输入和状态更新。
    """

    # 没有命中 Skill 时不申请任何字段，但仍生成一份明确的权限决策记录。
    required_profile_attributes: list[str] = []
    if selected_skill_id is not None:
        required_profile_attributes = (
            skill_runtime.get_missing_source_required_fields(
                skill_id=selected_skill_id,
                source_name="pet_profile",
                provided_inputs=provided_inputs,
                ignored_input_ids=ignored_input_ids,
            )
        )

    access_decision = resolve_pet_profile_field_access(
        purpose="skill_prefill",
        agent_name=agent_name,
        skill_required_attributes=required_profile_attributes,
    )
    state_update: dict[str, Any] = {
        "skill_required_pet_profile_attributes": (
            required_profile_attributes
        ),
        "skill_profile_access_decision": access_decision.model_dump(
            mode="python"
        ),
    }
    available_input_sources: dict[str, Mapping[str, Any]] = {}

    # 只有确实命中 Skill、存在服务且权限允许字段时才访问宠物档案数据库。
    if (
        selected_skill_id is None
        or pet_profile_service is None
        or not access_decision.allowed_attributes
    ):
        return SkillPetProfilePrefillResult(
            available_input_sources=available_input_sources,
            state_update=state_update,
        )

    try:
        profile_result = pet_profile_service.recall_profile(
            user_id=str(user_id or "").strip() or "default_user",
            active_pet_key=active_pet_key,
            active_pet_name=active_pet_name,
            selected_attributes=access_decision.allowed_attributes,
        )
        profile_data = profile_result.model_dump(mode="python")
        state_update["skill_profile_recall_result"] = profile_data
        if profile_data.get("status") == "applied":
            raw_facts = profile_data.get("facts")
            if isinstance(raw_facts, Mapping):
                available_input_sources["pet_profile"] = dict(raw_facts)
            state_update.update(
                {
                    "active_pet_key": str(
                        profile_data.get("pet_key") or ""
                    ),
                    "active_pet_name": str(
                        profile_data.get("pet_name") or ""
                    ),
                }
            )
    except Exception as profile_error:
        logger.warning(
            "Skill 准备阶段召回宠物档案失败，已继续使用用户输入: %s",
            profile_error,
        )
        state_update["skill_profile_recall_result"] = {
            "status": "failed",
            "reason": (
                "Skill 准备阶段宠物档案召回失败，"
                f"已回退到用户输入：{profile_error}"
            ),
        }

    return SkillPetProfilePrefillResult(
        available_input_sources=available_input_sources,
        state_update=state_update,
    )
