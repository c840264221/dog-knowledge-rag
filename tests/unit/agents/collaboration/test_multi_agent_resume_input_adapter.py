"""
多 Agent 主图恢复输入适配器测试。

功能：
    验证单步骤恢复、取消、新问题切换和多步骤 JSON 回答能生成正确状态更新。
"""

from __future__ import annotations

import json

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
    resolve_multi_agent_resume_input,
)


def build_paused_task_result(
    awaiting_step_ids: list[str],
) -> MultiAgentTaskResult:
    """
    构建包含指定等待步骤的暂停任务结果。

    参数含义：
        awaiting_step_ids:
            需要返回 awaiting_input 的步骤编号。

    返回值含义：
        MultiAgentTaskResult:
            可以写入 DogState 和 Checkpoint 的暂停任务结果。
    """

    steps = [
        AgentTaskStep(
            step_id=step_id,
            title=f"等待步骤 {step_id}",
            assigned_agent=f"agent_{index}",
            status="awaiting_input",
        )
        for index, step_id in enumerate(awaiting_step_ids)
    ]
    plan = AgentTaskPlan(
        plan_id="resume_adapter_plan",
        objective="验证多 Agent 跨轮恢复输入",
        steps=steps,
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="请补充信息。",
    )
    results = [
        AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt=f"请回答 {step.step_id}",
        )
        for step in steps
    ]
    return MultiAgentTaskResult(
        collaboration_id="resume_adapter_task",
        plan=plan,
        status="awaiting_input",
        task_results=results,
    )


def build_planner_waiting_task_result() -> MultiAgentTaskResult:
    """
    构建 Planner 缺少信息、尚未执行 Worker 的暂停任务。

    功能：
        模拟 Planner 已经提出澄清问题，但计划步骤还没有进入 Worker，
        因此 task_results 中不存在 awaiting_input 步骤结果。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskResult:
            可以用于验证 replan（重新规划）分支的暂停任务结果。
    """

    plan = AgentTaskPlan(
        plan_id="planner_waiting_plan",
        objective="为狗狗制定健康和训练综合方案",
        steps=[
            AgentTaskStep(
                step_id="build_plan",
                title="生成执行计划",
                assigned_agent="general_agent",
            )
        ],
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="请提供狗狗的年龄和体重。",
    )
    return MultiAgentTaskResult(
        collaboration_id="planner_waiting_task",
        plan=plan,
        status="awaiting_input",
    )


def test_single_waiting_step_should_use_question_as_resume_input() -> None:
    """
    检查单个等待步骤是否直接接收本轮用户输入。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = resolve_multi_agent_resume_input(
        {
            "question": "允许读取",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "resume"
    assert resolution["state_update"]["multi_agent_resume_ready"] is True
    assert resolution["state_update"]["multi_agent_resume_inputs"] == {
        "load_profile": "允许读取"
    }


def test_planner_waiting_should_prepare_replan_with_user_input() -> None:
    """
    检查 Planner 等待时是否把用户回答整理成重新规划上下文。

    功能：
        Planner 暂停不对应某个 Worker 步骤，因此用户回答应触发 replan，
        并表示本轮输入已经准备好交给多 Agent 入口继续处理。

    参数含义：
        无。

    返回值含义：
        None。
    """

    paused_result = build_planner_waiting_task_result()

    resolution = resolve_multi_agent_resume_input(
        {
            "question": "3 岁，体重 20 公斤",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "replan"
    state_update = resolution["state_update"]
    assert state_update["multi_agent_resume_action"] == "replan"
    assert state_update["multi_agent_resume_inputs"] == {
        "planner_clarification": "3 岁，体重 20 公斤"
    }
    assert state_update["multi_agent_resume_ready"] is True
    assert state_update["waiting_user_input"] is False


def test_cancel_input_should_clear_paused_task() -> None:
    """
    检查用户取消时是否清理暂停任务。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = resolve_multi_agent_resume_input(
        {
            "question": "取消",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "cancelled"
    assert resolution["state_update"]["multi_agent_task_result"] == {}
    assert resolution["state_update"]["multi_agent_resume_ready"] is False


def test_new_question_prefix_should_clear_task_and_keep_new_question() -> None:
    """
    检查明确切换新问题时是否清理旧任务并移除业务前缀。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = resolve_multi_agent_resume_input(
        {
            "question": "新问题：金毛每天需要运动多久？",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "new_question"
    assert resolution["state_update"]["multi_agent_task_result"] == {}
    assert resolution["state_update"]["question"] == (
        "金毛每天需要运动多久？"
    )


def test_multiple_waiting_steps_should_require_step_id_json() -> None:
    """
    检查多个等待步骤是否要求按步骤编号提交完整 JSON 回答。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )
    valid_answer = json.dumps(
        {
            "health_confirm": "允许查询健康资料",
            "training_confirm": "允许查询训练资料",
        },
        ensure_ascii=False,
    )

    resolution = resolve_multi_agent_resume_input(
        {
            "question": valid_answer,
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "resume"
    assert set(
        resolution["state_update"]["multi_agent_resume_inputs"]
    ) == {"health_confirm", "training_confirm"}


def test_incomplete_multiple_step_input_should_request_clarification() -> None:
    """
    检查多个等待步骤回答不完整时是否继续等待用户。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )

    resolution = resolve_multi_agent_resume_input(
        {
            "question": '{"health_confirm": "允许"}',
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "needs_clarification"
    assert resolution["state_update"]["waiting_user_input"] is True
    assert "training_confirm" in resolution["state_update"][
        "multi_agent_pending_prompt"
    ]
