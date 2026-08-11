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


def _build_pending_skill_state(question: str) -> dict:
    """
    构建正在等待训练 Skill 补充行为和目标的状态。

    参数含义：
        question:
            本轮用户补充的自然语言。

    返回值含义：
        dict:
            包含已选技能、历史输入和等待提示的最小状态。
    """

    return {
        "question": question,
        "skill_status": "awaiting_input",
        "skill_selected_id": "dog-training-plan",
        "skill_inputs": {
            "breed": "金毛",
            "age": "6岁",
        },
        "skill_pending_prompt": "请补充当前行为和训练目标。",
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


def test_multi_agent_degraded_choice_should_reach_resume_adapter() -> None:
    """
    测试多智能体等待期间的简化指令不会被外层任务关系门卫拦截。

    功能：
        门卫只确认这条输入是在继续旧任务；具体步骤能否简化仍交给后续
        多智能体恢复适配器依据结构化字段要求判断。

    参数含义：
        无。

    返回值含义：
        None，pytest 根据任务关系结果判断是否通过。
    """

    result = resolve_pending_task_relation(
        _build_pending_multi_agent_state("简化执行")
    )
    update = result["state_update"]

    assert result["action"] == "resume"
    assert update["task_relation_decision"]["relation"] == "resume"
    assert update["task_relation_requires_confirmation"] is False


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


def test_partial_skill_input_should_resume_pending_skill() -> None:
    """
    测试只补充一个当前缺失字段也应继续等待中的 Skill。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_pending_skill_state("当前会坐下")
    )

    assert result["action"] == "resume"
    assert result["state_update"]["task_relation_pending_kind"] == "skill"
    assert "current_behavior" in (
        result["state_update"]["task_relation_decision"]["reason"]
    )


def test_complete_skill_input_should_resume_pending_skill() -> None:
    """
    测试一次补充多个缺失字段可以继续等待中的 Skill。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_pending_skill_state("已经掌握坐下，期望学会等待")
    )

    assert result["action"] == "resume"
    reason = result["state_update"]["task_relation_decision"]["reason"]
    assert "current_behavior" in reason
    assert "training_goal" in reason


def test_unrelated_input_should_remain_ambiguous_for_pending_skill() -> None:
    """
    测试没有补充任何技能字段的输入仍需用户明确新旧任务关系。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_pending_skill_state("成都天气怎么样")
    )

    assert result["action"] == "ambiguous"
    assert result["state_update"]["task_relation_requires_confirmation"] is True


def test_degraded_choice_should_use_saved_skill_check_result() -> None:
    """
    验证用户选择简化执行时只采用系统确认过的可简化缺失字段。

    参数含义：无。
    返回值含义：None，pytest 根据状态更新判断是否通过。
    """

    state = _build_pending_skill_state("简化执行")
    state["skill_inputs"]["training_goal"] = "学习等待"
    state["skill_runtime_result"] = {
        "input_check": {
            "can_run_degraded": True,
            "missing_degradable_input_ids": ["current_behavior"],
        }
    }

    result = resolve_pending_task_relation(state)
    update = result["state_update"]

    assert result["action"] == "resume"
    assert update["skill_execution_mode"] == "degraded"
    assert update["skill_ignored_input_ids"] == ["current_behavior"]
    assert update["skill_degradation_reason"] == (
        "user_selected_degraded_execution"
    )
    assert update["skill_degradation_user_input"] == "简化执行"
