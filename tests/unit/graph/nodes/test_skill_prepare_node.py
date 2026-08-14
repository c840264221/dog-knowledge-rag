"""主图 Skill 准备节点单元测试。"""

from __future__ import annotations

from src.graph.nodes.skill_prepare_node import build_skill_prepare_node
from src.memory.memory_schema import PetProfileRecallResult
from src.runtime.resume import resolve_pending_task_relation


class FakePetProfileService:
    """返回预设宠物档案并记录调用次数的测试服务。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def recall_profile(self, **kwargs) -> PetProfileRecallResult:
        """
        返回测试使用的单宠物档案。

        参数含义：**kwargs 为节点传入的用户和当前宠物信息。
        返回值含义：PetProfileRecallResult，已经成功应用的宠物档案。
        """

        self.calls.append(dict(kwargs))
        return PetProfileRecallResult(
            status="applied",
            pet_key="pet_doudou",
            pet_name="豆豆",
            selection_source="single_pet_fallback",
            facts={
                "breed": "金毛",
                "age_years": "6岁",
            },
            selected_attributes=["breed", "age_years"],
            reason="测试档案召回成功。",
        )


def test_skill_prepare_node_should_save_waiting_state() -> None:
    """
    测试首次命中 Skill 但资料不足时保存恢复所需状态。

    功能：
        验证节点会保存技能编号、已提取输入和用户提示，同时不提前生成完整
        Skill 上下文。

    参数含义：
        无。

    返回值含义：
        None。
    """

    node = build_skill_prepare_node()

    update = node(
        {
            "question": "帮我为6岁的金毛制定训练计划。",
            "route_decision": {
                "route": "dog_knowledge_agent",
            },
        }
    )

    assert update["skill_status"] == "awaiting_input"
    assert update["skill_selected_id"] == "dog-training-plan"
    assert update["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }
    assert update["waiting_user_input"] is True
    assert "当前行为基础" in update["skill_pending_prompt"]
    assert update["skill_context"] == ""
    assert update["skill_original_question"] == "帮我为6岁的金毛制定训练计划。"
    assert update["skill_target_agent"] == "dog_knowledge_agent"


def test_skill_prepare_node_should_resume_saved_skill() -> None:
    """
    测试节点使用 checkpoint 中的技能编号和历史输入恢复准备。

    功能：
        本轮简短回答没有训练计划触发词，节点仍应继续上一轮 Skill，并把新旧
        输入合并成 ready 状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    node = build_skill_prepare_node()

    state = {
        "question": "它目前会坐下，希望学习等待和召回。",
        "user_id": "test_user",
        "session_id": "test_session",
        "skill_status": "awaiting_input",
        "skill_selected_id": "dog-training-plan",
        "skill_inputs": {
            "breed": "Golden Retriever",
            "age": "6岁",
        },
        "skill_pending_prompt": "请补充当前行为和训练目标。",
        "skill_original_question": "帮我为6岁的金毛制定训练计划。",
        "skill_target_agent": "dog_knowledge_agent",
    }
    relation_update = resolve_pending_task_relation(state)["state_update"]

    update = node({**state, **relation_update})

    assert update["skill_status"] == "ready"
    assert update["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }
    assert update["skill_pending_prompt"] == ""
    assert update["waiting_user_input"] is False
    assert "技能：狗狗训练计划" in update["skill_context"]
    pending_task = update["pending_tasks"][
        "skill:dog-training-plan"
    ]
    assert pending_task["status"] == "running"
    assert pending_task["version"] == 2
    assert "用户原始任务" in update["question"]
    assert "帮我为6岁的金毛制定训练计划。" in update["question"]
    assert "用户本轮补充信息" in update["question"]
    assert "已经校验通过的 Skill 输入" in update["question"]
    assert "技能：狗狗训练计划" in update["question"]
    assert "技能：狗狗训练计划" not in update["retrieval_question"]
    assert "用户原始任务" in update["retrieval_question"]
    assert update["memory_retrieval_text"] == update["retrieval_question"]
    assert "技能：狗狗训练计划" not in update["memory_retrieval_text"]
    assert "技能：狗狗训练计划" in update["skill_context"]


def test_skill_prepare_node_should_continue_in_degraded_mode() -> None:
    """
    验证用户选择简化执行后继续同一个 Skill，并记录答案限制。

    功能：
        breed 和 current_behavior 已被系统确认允许忽略，因此节点不再重复
        澄清，直接使用年龄和训练目标准备简化版技能上下文。

    参数含义：无。
    返回值含义：None，pytest 根据节点状态更新判断是否通过。
    """

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node(
        {
            "question": "简化执行",
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_inputs": {
                "age": "6岁",
                "training_goal": "学习等待和召回",
            },
            "skill_original_question": "帮我制定训练计划。",
            "skill_target_agent": "dog_knowledge_agent",
            "skill_execution_mode": "degraded",
            "skill_ignored_input_ids": ["breed", "current_behavior"],
            "skill_degradation_reason": (
                "user_selected_degraded_execution"
            ),
            "skill_degradation_user_input": "简化执行",
        }
    )

    assert update["skill_status"] == "ready"
    assert update["skill_execution_mode"] == "degraded"
    assert update["skill_ignored_input_ids"] == [
        "breed",
        "current_behavior",
    ]
    assert update["skill_pending_prompt"] == ""
    assert update["waiting_user_input"] is False
    assert "本次 Skill 使用简化执行模式" in update["question"]
    assert "breed、current_behavior" in update["question"]
    assert "不得编造缺失信息" in update["question"]
    assert "简化执行" not in update["retrieval_question"]
    assert "用户原始任务" in update["question"]
    assert "帮我制定训练计划。" in update["question"]
    assert "已经校验通过的 Skill 输入" in update["question"]
    assert "技能：狗狗训练计划" in update["question"]
    assert "技能：狗狗训练计划" not in update["retrieval_question"]
    assert "用户原始任务" in update["retrieval_question"]
    assert update["memory_retrieval_text"] == update["retrieval_question"]
    assert "技能：狗狗训练计划" not in update["memory_retrieval_text"]
    assert "技能：狗狗训练计划" in update["skill_context"]
    assert profile_service.calls == []


def test_skill_prepare_node_should_leave_unmatched_question_unselected() -> None:
    """
    测试普通问题不会产生虚假的 Skill 恢复状态。

    功能：
        未命中 Skill 时返回 no_skill，并保持技能编号、输入和上下文为空。

    参数含义：
        无。

    返回值含义：
        None。
    """

    node = build_skill_prepare_node()

    update = node({"question": "成都今天的天气怎么样"})

    assert update["skill_status"] == "no_skill"
    assert update["skill_selected_id"] == ""
    assert update["skill_inputs"] == {}
    assert update["skill_context"] == ""
    assert "waiting_user_input" not in update


def test_skill_prepare_node_should_fill_inputs_from_pet_profile() -> None:
    """验证 Skill 在澄清前会先使用当前宠物档案补全必需输入。"""

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node(
        {
            "user_id": "user_001",
            "question": (
                "帮我制定训练计划，它目前会坐下，希望学习等待和召回。"
            ),
            "route_decision": {"route": "dog_knowledge_agent"},
        }
    )

    assert update["skill_status"] == "ready"
    assert update["skill_inputs"] == {
        "breed": "金毛",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }
    assert update["active_pet_key"] == "pet_doudou"
    assert update["skill_profile_recall_result"]["status"] == "applied"
    assert len(profile_service.calls) == 1
    assert profile_service.calls[0]["selected_attributes"] == [
        "breed",
        "age_years",
    ]
    assert update["skill_required_pet_profile_attributes"] == [
        "breed",
        "age_years",
    ]
    assert update["skill_profile_access_decision"]["allowed_attributes"] == [
        "breed",
        "age_years",
    ]


def test_complete_user_inputs_should_not_query_pet_profile() -> None:
    """验证用户已经提供全部 Skill 参数时不会额外查询宠物档案。"""

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node(
        {
            "user_id": "user_001",
            "question": (
                "我家的狗狗是一只6岁的金毛，现在会坐下，"
                "期望学会等待和召回，帮我安排训练计划"
            ),
            "route_decision": {"route": "dog_knowledge_agent"},
        }
    )

    assert update["skill_status"] == "ready"
    assert update["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学会等待和召回",
    }
    assert update["skill_required_pet_profile_attributes"] == []
    assert update["skill_profile_access_decision"]["allowed_attributes"] == []
    assert profile_service.calls == []


def test_missing_profile_inputs_should_query_only_breed_and_age() -> None:
    """验证通用训练请求只从档案补全犬种和年龄，再澄清任务参数。"""

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node(
        {
            "user_id": "user_001",
            "question": "帮我家狗狗安排训练计划",
            "route_decision": {"route": "dog_knowledge_agent"},
        }
    )

    assert len(profile_service.calls) == 1
    assert profile_service.calls[0]["selected_attributes"] == [
        "breed",
        "age_years",
    ]
    assert update["skill_required_pet_profile_attributes"] == [
        "breed",
        "age_years",
    ]
    assert update["skill_profile_access_decision"]["allowed_attributes"] == [
        "breed",
        "age_years",
    ]
    assert update["skill_inputs"] == {
        "breed": "金毛",
        "age": "6岁",
    }
    assert update["skill_status"] == "awaiting_input"
    assert "当前行为基础" in update["skill_pending_prompt"]
    assert "训练目标" in update["skill_pending_prompt"]


def test_skill_required_fields_should_not_bypass_agent_profile_permission() -> None:
    """验证 Skill 必需字段不会绕过 general_agent 的档案读取白名单。"""

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node(
        {
            "user_id": "user_001",
            "question": "制定训练计划，它会坐下，希望学习等待。",
            "route_decision": {"route": "general_agent"},
        }
    )

    assert profile_service.calls == []
    assert update["skill_status"] == "awaiting_input"
    assert update["skill_profile_access_decision"][
        "denied_skill_required_attributes"
    ] == ["breed", "age_years"]


def test_unmatched_question_should_not_query_pet_profile() -> None:
    """验证没有命中 Skill 的普通问题不会产生额外宠物档案查询。"""

    profile_service = FakePetProfileService()
    node = build_skill_prepare_node(
        pet_profile_service=profile_service
    )

    update = node({"user_id": "user_001", "question": "成都天气如何"})

    assert update["skill_status"] == "no_skill"
    assert profile_service.calls == []
