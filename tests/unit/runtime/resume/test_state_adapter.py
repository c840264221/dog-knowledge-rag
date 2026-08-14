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


def _build_multiple_pending_state(question: str) -> dict:
    """
    构建工具与多智能体同时等待输入的测试状态。

    参数含义：
        question:
            本轮用户输入。

    返回值含义：
        dict:
            包含两个独立等待模块及其确定性契约的最小状态。
    """

    state = _build_pending_multi_agent_state(question)
    state.update(
        {
            "tool_agent_clarification_request": {
                "status": "pending",
                "tool_name": "sqlite_list_tables",
                "missing_fields": ["database_name"],
                "options": {"database_name": ["memory", "rag"]},
                "question": "请选择数据库别名。",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_list_tables",
                "args": {},
            },
        }
    )
    return state


def test_new_task_should_suspend_instead_of_cancel_pending_task() -> None:
    """
    测试完整的新请求只隔离旧业务字段，并保留统一等待任务快照。

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
    pending_task = update["pending_tasks"]["multi_agent:pending"]
    assert pending_task["status"] == "awaiting_input"
    assert pending_task["payload"]["resume_state"][
        "multi_agent_task_result"
    ] == {"status": "awaiting_input"}


def test_suspended_tool_task_should_resume_from_collection_payload() -> None:
    """
    测试新问题隔离旧字段后，下一轮仍能从任务 Payload 恢复 Tool。

    参数含义：
        无。

    返回值含义：
        None。
    """

    pending_state = {
        "question": "请介绍金毛犬。",
        "user_id": "test_user",
        "session_id": "test_thread",
        "tool_agent_clarification_request": {
            "status": "pending",
            "tool_name": "sqlite_list_tables",
            "missing_fields": ["database_name"],
            "options": {"database_name": ["memory", "rag"]},
            "question": "请选择数据库别名。",
        },
        "tool_agent_pending_tool_call": {
            "name": "sqlite_list_tables",
            "args": {},
        },
    }

    new_task_result = resolve_pending_task_relation(pending_state)
    suspended_tasks = new_task_result["state_update"]["pending_tasks"]
    resume_result = resolve_pending_task_relation(
        {
            "question": "继续任务：memory",
            "user_id": "test_user",
            "session_id": "test_thread",
            "pending_tasks": suspended_tasks,
        }
    )
    resume_update = resume_result["state_update"]

    assert resume_result["action"] == "resume"
    assert resume_update["question"] == "memory"
    assert resume_update["task_relation_decision"][
        "selected_task_id"
    ] == "tool:sqlite_list_tables"
    assert resume_update["tool_agent_pending_tool_call"] == {
        "name": "sqlite_list_tables",
        "args": {},
    }
    assert resume_update["tool_agent_clarification_request"][
        "missing_fields"
    ] == ["database_name"]


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


def test_cancel_with_multiple_tasks_should_require_target_selection() -> None:
    """
    测试多个等待任务下只说取消时不会猜测取消目标。

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

    assert result["action"] == "ambiguous"
    assert "multi_agent_task_result" not in update
    assert "skill_status" not in update
    assert update["task_relation_requires_confirmation"] is True
    assert update["task_relation_selection_action"] == "cancel"
    assert "请选择要取消的任务" in update["pending_prompt"]
    assert "全部取消" in update["pending_prompt"]


def test_cancel_all_should_clear_every_pending_task() -> None:
    """
    测试明确“全部取消”会统一清理所有等待任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    state = _build_pending_multi_agent_state("全部取消")
    state.update(
        {
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
        }
    )

    result = resolve_pending_task_relation(state)
    update = result["state_update"]

    assert result["action"] == "cancel"
    assert result["state_update"]["pending_tasks"] == {}
    assert update["multi_agent_task_result"] == {}
    assert update["skill_status"] == "no_skill"
    assert update["final_answer"] == "已取消全部等待任务。"


def test_cancel_selection_should_only_clear_selected_task() -> None:
    """
    测试跨轮选择取消目标后只清理目标任务并保留其他等待任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    first_result = resolve_pending_task_relation(
        _build_multiple_pending_state("取消")
    )
    second_state = _build_multiple_pending_state("1")
    second_state.update(
        {
            "task_relation_candidates": first_result["state_update"][
                "task_relation_candidates"
            ],
            "task_relation_unassigned_input": "取消",
            "task_relation_selection_action": "cancel",
        }
    )

    second_result = resolve_pending_task_relation(second_state)
    update = second_result["state_update"]

    assert second_result["action"] == "cancel"
    assert update["tool_agent_pending_tool_call"] is None
    assert "multi_agent_task_result" not in update
    assert update["task_relation_pending_kind"] == "tool"
    assert update["task_relation_decision"]["selected_task_id"] == (
        "tool:sqlite_list_tables"
    )
    assert "其他等待任务仍保留" in update["final_answer"]


def test_direct_cancel_selection_should_cancel_numbered_task() -> None:
    """
    测试“取消任务：编号”可以在一轮内定向取消候选任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_multiple_pending_state("取消任务：2")
    )
    update = result["state_update"]

    assert result["action"] == "cancel"
    assert update["multi_agent_task_result"] == {}
    assert "tool_agent_pending_tool_call" not in update
    assert update["task_relation_pending_kind"] == "multi_agent"


def test_remaining_task_should_resume_after_targeted_cancel() -> None:
    """
    测试定向取消一个任务后，另一个等待任务下一轮仍然可以恢复。

    参数含义：
        无。

    返回值含义：
        None。
    """

    original_state = _build_multiple_pending_state("取消任务：1")
    cancel_result = resolve_pending_task_relation(original_state)
    remaining_state = {
        **original_state,
        **cancel_result["state_update"],
        "question": "简化执行",
    }

    resume_result = resolve_pending_task_relation(remaining_state)
    update = resume_result["state_update"]

    assert resume_result["action"] == "resume"
    assert update["task_relation_pending_kind"] == "multi_agent"
    assert update["task_relation_decision"]["relation"] == "resume"


def test_multiple_pending_tasks_should_return_numbered_candidates() -> None:
    """
    测试多个契约都无法唯一命中时会暂存输入并要求用户选择任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_multiple_pending_state("500")
    )
    update = result["state_update"]

    assert result["action"] == "ambiguous"
    assert update["task_relation_pending_kind"] == "multiple"
    assert update["task_relation_unassigned_input"] == "500"
    assert len(update["task_relation_candidates"]) == 2
    assert set(update["pending_tasks"]) == {
        "tool:sqlite_list_tables",
        "multi_agent:pending",
    }
    assert all(
        task["status"] == "awaiting_input"
        and task["version"] == 1
        for task in update["pending_tasks"].values()
    )
    assert update["task_relation_decision"][
        "requires_task_selection"
    ] is True
    assert "1. 补充工具" in update["pending_prompt"]
    assert "2. 补充多智能体" in update["pending_prompt"]


def test_task_number_selection_should_rebind_saved_input() -> None:
    """
    测试用户选择候选编号后会把上一轮原始输入绑定到目标任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    first_result = resolve_pending_task_relation(
        _build_multiple_pending_state("500")
    )
    second_state = _build_multiple_pending_state("2")
    second_state.update(
        {
            "task_relation_candidates": first_result["state_update"][
                "task_relation_candidates"
            ],
            "task_relation_unassigned_input": "500",
        }
    )

    second_result = resolve_pending_task_relation(second_state)
    update = second_result["state_update"]

    assert second_result["action"] == "resume"
    assert update["question"] == "500"
    assert update["task_relation_pending_kind"] == "multi_agent"
    assert update["task_relation_candidates"] == []
    assert update["task_relation_unassigned_input"] == ""
    assert update["task_relation_decision"]["selected_task_id"] == (
        "multi_agent:pending"
    )


def test_invalid_task_number_should_preserve_unassigned_input() -> None:
    """
    测试无效候选编号不会覆盖上一轮尚未绑定的业务输入。

    参数含义：
        无。

    返回值含义：
        None。
    """

    first_result = resolve_pending_task_relation(
        _build_multiple_pending_state("500")
    )
    second_state = _build_multiple_pending_state("9")
    second_state.update(
        {
            "task_relation_candidates": first_result["state_update"][
                "task_relation_candidates"
            ],
            "task_relation_unassigned_input": "500",
        }
    )

    second_result = resolve_pending_task_relation(second_state)
    update = second_result["state_update"]

    assert second_result["action"] == "ambiguous"
    assert update["task_relation_unassigned_input"] == "500"
    assert "你输入的“500”" in update["pending_prompt"]


def test_unique_contract_match_should_select_only_tool_task() -> None:
    """
    测试确定性契约只命中一个任务时可以直接定向恢复。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = resolve_pending_task_relation(
        _build_multiple_pending_state("memory")
    )
    update = result["state_update"]

    assert result["action"] == "resume"
    assert update["task_relation_pending_kind"] == "tool"
    assert update["task_relation_decision"]["selected_task_id"] == (
        "tool:sqlite_list_tables"
    )
    assert update["task_relation_decision"]["candidate_task_ids"] == [
        "tool:sqlite_list_tables",
        "multi_agent:pending",
    ]


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
