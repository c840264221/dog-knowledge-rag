"""
多 Agent 主图恢复输入适配器测试。

功能：
    验证单步骤恢复、取消、新问题切换和多步骤 JSON 回答能生成正确状态更新。
"""

from __future__ import annotations

import json

import pytest

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentClarificationExtractionResult,
    MultiAgentTaskResult,
    resolve_multi_agent_resume_input,
)


pytestmark = pytest.mark.asyncio


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


async def test_single_waiting_step_should_use_question_as_resume_input() -> None:
    """
    检查单个等待步骤是否直接接收本轮用户输入。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = await resolve_multi_agent_resume_input(
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


async def test_planner_waiting_should_prepare_replan_with_user_input() -> None:
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

    resolution = await resolve_multi_agent_resume_input(
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


async def test_cancel_input_should_clear_paused_task() -> None:
    """
    检查用户取消时是否清理暂停任务。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = await resolve_multi_agent_resume_input(
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


async def test_new_question_prefix_should_clear_task_and_keep_new_question() -> None:
    """
    检查明确切换新问题时是否清理旧任务并移除业务前缀。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(["load_profile"])

    resolution = await resolve_multi_agent_resume_input(
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


async def test_multiple_waiting_steps_should_require_step_id_json() -> None:
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

    resolution = await resolve_multi_agent_resume_input(
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


async def test_incomplete_multiple_step_input_should_request_clarification() -> None:
    """
    检查多个等待步骤回答不完整时是否继续等待用户。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )

    resolution = await resolve_multi_agent_resume_input(
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


async def test_multiple_step_prompt_should_show_titles_and_missing_fields() -> None:
    """
    检查多步骤恢复提示会展示步骤名称和缺失字段中文名。

    参数含义：
        无。

    返回值含义：
        None。
    """

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [
                {
                    "step_id": "health_confirm",
                    "step_title": "生成健康建议",
                    "assigned_agent": "health_agent",
                    "prompt": "请补充年龄。",
                    "missing_fields": [
                        {"input_id": "age", "name": "年龄"}
                    ],
                    "can_run_degraded": False,
                },
                {
                    "step_id": "training_confirm",
                    "step_title": "生成训练计划",
                    "assigned_agent": "training_agent",
                    "prompt": "请补充训练目标。",
                    "missing_fields": [
                        {
                            "input_id": "training_goal",
                            "name": "训练目标",
                        }
                    ],
                    "can_run_degraded": True,
                },
            ],
            "field_consumers": {
                "age": ["health_confirm"],
                "training_goal": ["training_confirm"],
            },
            "display_prompt": "请补充年龄和训练目标。",
        }
    }

    resolution = await resolve_multi_agent_resume_input(
        {
            "question": "信息还不完整",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )
    prompt = resolution["state_update"]["multi_agent_pending_prompt"]

    assert "生成健康建议（health_confirm），需要补充：年龄" in prompt
    assert (
        "生成训练计划（training_confirm），需要补充：训练目标"
        in prompt
    )
    assert "请直接使用自然语言回答" in prompt


async def test_natural_answer_should_be_allocated_by_field_consumers() -> None:
    """验证一条自然语言回答会按字段使用关系分配给多个步骤。"""

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [],
            "field_consumers": {
                "age": ["health_confirm", "training_confirm"],
                "training_goal": ["training_confirm"],
            },
            "display_prompt": "请补充年龄和训练目标。",
        }
    }

    resolution = await resolve_multi_agent_resume_input(
        {
            "question": "它6岁，希望学习等待和召回。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "resume"
    assert resolution["state_update"]["multi_agent_resume_inputs"] == {
        "health_confirm": {"age": "6岁"},
        "training_confirm": {
            "age": "6岁",
            "training_goal": "学习等待和召回",
        },
    }


async def test_resume_should_support_one_step_degraded_and_one_step_ready() -> None:
    """验证同批步骤可以分别选择正常恢复和简化执行。"""

    paused_result = build_paused_task_result(
        ["health_step", "training_step"]
    )
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [
                {
                    "step_id": "health_step",
                    "step_title": "生成健康建议",
                    "assigned_agent": "health_agent",
                    "prompt": "请补充年龄。",
                    "missing_fields": [
                        {
                            "input_id": "age",
                            "name": "年龄",
                            "requirement_level": "hard_required",
                        }
                    ],
                    "can_run_degraded": False,
                },
                {
                    "step_id": "training_step",
                    "step_title": "生成训练计划",
                    "assigned_agent": "training_agent",
                    "prompt": "请补充训练目标。",
                    "missing_fields": [
                        {
                            "input_id": "training_goal",
                            "name": "训练目标",
                            "requirement_level": "degradable",
                        }
                    ],
                    "can_run_degraded": True,
                },
            ],
            "field_consumers": {
                "age": ["health_step"],
                "training_goal": ["training_step"],
            },
            "display_prompt": "请补充年龄和训练目标。",
        }
    }

    resolution = await resolve_multi_agent_resume_input(
        {
            "question": "它6岁，生成训练计划这个步骤简化执行。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "resume"
    decisions = resolution["state_update"][
        "multi_agent_step_resume_decisions"
    ]
    assert decisions["health_step"]["action"] == "resume"
    assert decisions["training_step"]["action"] == "degraded"
    assert decisions["training_step"]["ignored_input_ids"] == [
        "training_goal"
    ]
    assert resolution["state_update"]["multi_agent_resume_inputs"] == {
        "health_step": {"age": "6岁"},
        "training_step": {},
    }


async def test_degraded_control_should_not_use_llm_field_fallback() -> None:
    """
    验证简化执行只运行确定性字段规则，不调用 LLM 字段兜底。

    参数含义：无。
    返回值含义：None，pytest 根据恢复决定和分层提取调用次数判断是否通过。
    """

    class RecordingResolver:
        """记录确定性提取与分层 LLM 提取调用次数的测试替身。"""

        def __init__(self) -> None:
            self.rule_call_count = 0
            self.layered_call_count = 0

        def extract(self, *, user_text, requested_field_ids, existing_fields):
            """返回未补充字段的确定性提取结果。"""

            _ = user_text
            self.rule_call_count += 1
            resolved_fields = dict(existing_fields or {})
            return MultiAgentClarificationExtractionResult(
                requested_field_ids=requested_field_ids,
                resolved_fields=resolved_fields,
                missing_field_ids=[
                    field_id
                    for field_id in requested_field_ids
                    if field_id not in resolved_fields
                ],
            )

        async def extract_layered(self, **kwargs):
            """记录错误进入 LLM 兜底层并让测试失败。"""

            _ = kwargs
            self.layered_call_count += 1
            raise AssertionError("简化执行不应调用 LLM 字段兜底")

    paused_result = build_paused_task_result(["training_step"])
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [
                {
                    "step_id": "training_step",
                    "step_title": "生成训练计划",
                    "assigned_agent": "training_agent",
                    "prompt": "请补充犬种和当前行为。",
                    "missing_fields": [
                        {
                            "input_id": "breed",
                            "name": "犬种",
                            "requirement_level": "degradable",
                        },
                        {
                            "input_id": "current_behavior",
                            "name": "当前行为基础",
                            "requirement_level": "degradable",
                        },
                    ],
                    "can_run_degraded": True,
                }
            ],
            "field_consumers": {
                "breed": ["training_step"],
                "current_behavior": ["training_step"],
            },
            "display_prompt": "请补充犬种和当前行为。",
        }
    }
    resolver = RecordingResolver()

    resolution = await resolve_multi_agent_resume_input(
        {
            "question": "简化执行",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        },
        field_resolver=resolver,
    )

    assert resolution["action"] == "resume"
    assert resolver.rule_call_count == 1
    assert resolver.layered_call_count == 0
    decision = resolution["state_update"][
        "multi_agent_step_resume_decisions"
    ]["training_step"]
    assert decision["action"] == "degraded"
    assert decision["ignored_input_ids"] == [
        "breed",
        "current_behavior",
    ]


async def test_field_consumers_must_cover_every_waiting_step() -> None:
    """验证字段关系未覆盖整批等待步骤时不会错误恢复任务。"""

    paused_result = build_paused_task_result(
        ["health_confirm", "legacy_confirm"]
    )
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [],
            "field_consumers": {
                "age": ["health_confirm"],
            },
            "display_prompt": "请补充等待步骤需要的信息。",
        }
    }

    resolution = await resolve_multi_agent_resume_input(
        {
            "question": "它6岁。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert resolution["action"] == "needs_clarification"
    assert resolution["state_update"]["multi_agent_resume_ready"] is False


async def test_partial_natural_answers_should_merge_across_turns() -> None:
    """验证用户分两轮补充年龄和训练目标时不会丢失第一轮结果。"""

    paused_result = build_paused_task_result(
        ["health_confirm", "training_confirm"]
    )
    paused_result.metadata = {
        "clarification_bundle": {
            "step_requests": [
                {
                    "step_id": "health_confirm",
                    "step_title": "生成健康建议",
                    "missing_fields": [
                        {"input_id": "age", "name": "年龄"}
                    ],
                },
                {
                    "step_id": "training_confirm",
                    "step_title": "生成训练计划",
                    "missing_fields": [
                        {"input_id": "age", "name": "年龄"},
                        {
                            "input_id": "training_goal",
                            "name": "训练目标",
                        },
                    ],
                },
            ],
            "field_consumers": {
                "age": ["health_confirm", "training_confirm"],
                "training_goal": ["training_confirm"],
            },
            "display_prompt": "请补充年龄和训练目标。",
        }
    }

    first_resolution = await resolve_multi_agent_resume_input(
        {
            "question": "它6岁。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert first_resolution["action"] == "needs_clarification"
    partial_inputs = first_resolution["state_update"][
        "multi_agent_resume_inputs"
    ]
    assert partial_inputs == {
        "health_confirm": {"age": "6岁"},
        "training_confirm": {"age": "6岁"},
    }
    assert "训练目标" in first_resolution["state_update"][
        "multi_agent_pending_prompt"
    ]

    second_resolution = await resolve_multi_agent_resume_input(
        {
            "question": "希望学习等待和召回。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
            "multi_agent_resume_inputs": partial_inputs,
        }
    )

    assert second_resolution["action"] == "resume"
    assert second_resolution["state_update"][
        "multi_agent_resume_inputs"
    ]["training_confirm"] == {
        "age": "6岁",
        "training_goal": "学习等待和召回",
    }
