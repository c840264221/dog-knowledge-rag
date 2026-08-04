"""主图 Skill 准备节点单元测试。"""

from __future__ import annotations

from src.graph.nodes.skill_prepare_node import build_skill_prepare_node


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

    update = node(
        {
            "question": "它目前会坐下，希望学习等待和召回。",
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_inputs": {
                "breed": "Golden Retriever",
                "age": "6岁",
            },
            "skill_original_question": "帮我为6岁的金毛制定训练计划。",
            "skill_target_agent": "dog_knowledge_agent",
        }
    )

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
    assert "用户原始任务" in update["question"]
    assert "帮我为6岁的金毛制定训练计划。" in update["question"]
    assert "用户本轮补充信息" in update["question"]
    assert "已经校验通过的 Skill 输入" in update["question"]
    assert "技能：狗狗训练计划" in update["question"]
    assert "技能：狗狗训练计划" not in update["retrieval_question"]
    assert "用户原始任务" in update["retrieval_question"]
    assert "技能：狗狗训练计划" in update["skill_context"]


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
