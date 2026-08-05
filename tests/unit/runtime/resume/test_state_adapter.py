"""等待任务关系主图状态适配器测试。"""

from src.runtime.resume import resolve_pending_task_relation


def _build_pending_multi_agent_state(question: str) -> dict:
    """
    构建一个正在等待用户补充信息的多智能体状态。

    参数含义：
        question:
            本轮用户输入。

    返回值含义：
        dict:
            可交给任务关系状态适配器的最小状态。
    """

    return {
        "question": question,
        "multi_agent_task_result": {
            "status": "awaiting_input",
        },
        "multi_agent_pending_prompt": "请补充狗狗年龄。",
    }


def test_complete_request_should_clear_old_pending_task() -> None:
    """
    测试完整的新请求会清理旧多智能体等待状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_pending_multi_agent_state(
            "请使用多个智能体制定新的健康方案。"
        )
    )
    update = result["state_update"]

    assert result["action"] == "new_task"
    assert update["multi_agent_task_result"] == {}
    assert update["multi_agent_resume_action"] == "new_question"
    assert update["question"] == "请使用多个智能体制定新的健康方案。"
    assert update["task_relation_pending_kind"] == "multi_agent"


def test_answer_shaped_input_should_preserve_pending_task() -> None:
    """
    测试资料型回答只记录继续判断，不提前清理旧任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    state = _build_pending_multi_agent_state("6岁，体重30公斤")
    result = resolve_pending_task_relation(state)
    update = result["state_update"]

    assert result["action"] == "resume"
    assert "multi_agent_task_result" not in update
    assert update["task_relation_decision"]["relation"] == "resume"


def test_ambiguous_input_should_request_explicit_choice() -> None:
    """
    测试无法判断的输入会保留旧任务并要求用户明确选择。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_pending_multi_agent_state("成都天气")
    )
    update = result["state_update"]

    assert result["action"] == "ambiguous"
    assert "multi_agent_task_result" not in update
    assert update["task_relation_requires_confirmation"] is True
    assert update["waiting_user_input"] is True
    assert "继续任务" in update["pending_prompt"]
    assert "新问题" in update["pending_prompt"]


def test_cancel_should_clear_all_pending_task_types() -> None:
    """
    测试明确取消会统一清理 Tool、Multi-Agent 和 Skill 等待字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    state = _build_pending_multi_agent_state("取消")
    state.update(
        {
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
        }
    )
    result = resolve_pending_task_relation(state)
    update = result["state_update"]

    # 多个等待状态虽然异常，但明确“取消”仍应安全地清理全部状态。
    assert result["action"] == "cancel"
    assert update["multi_agent_task_result"] == {}
    assert update["skill_status"] == "no_skill"
    assert update["task_relation_requires_confirmation"] is False
