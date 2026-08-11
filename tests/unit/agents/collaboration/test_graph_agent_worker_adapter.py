"""
Graph Agent Worker Adapter 测试。

功能：
    验证步骤输入、前置结果、普通回答和等待用户状态能在现有 Agent state
    与多 Agent Worker 结果之间正确转换。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskResult,
    AgentTaskStep,
    GraphAgentWorkerAdapter,
    build_default_agent_state,
)
from src.memory.memory_schema import PetProfileRecallResult
from src.skills import build_default_skill_runtime


class FakeWorkerPetProfileService:
    """为 Worker Preflight 返回固定档案并记录查询参数。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def recall_profile(self, **kwargs: Any) -> PetProfileRecallResult:
        """
        返回包含犬种和年龄的单宠物档案。

        参数含义：
            **kwargs：Worker 传入的用户、宠物和允许读取字段。

        返回值含义：
            PetProfileRecallResult：测试使用的成功召回结果。
        """

        self.calls.append(dict(kwargs))
        return PetProfileRecallResult(
            status="applied",
            pet_key="pet_doudou",
            pet_name="豆豆",
            selection_source="active_pet",
            facts={
                "breed": "金毛",
                "age_years": "6岁",
            },
            selected_attributes=["breed", "age_years"],
            reason="Worker 测试档案召回成功。",
        )


def test_graph_worker_adapter_should_convert_completed_state() -> None:
    """
    检查普通 Agent 最终 state 是否转换成 completed Worker 结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录输入 state 并返回健康知识回答。"""

        received_states.append(state)
        return {
            **state,
            "final_answer": "幼犬应按计划免疫。",
            "evidence_ids": ["health_chunk_001"],
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="health_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_health",
        title="查询健康知识",
        description="根据幼犬资料查询健康知识",
        assigned_agent="health_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "completed"
    assert result.summary == "幼犬应按计划免疫。"
    assert result.output["final_answer"] == "幼犬应按计划免疫。"
    assert result.evidence_ids == ["health_chunk_001"]
    assert received_states[0]["question"] == "根据幼犬资料查询健康知识"


def test_graph_worker_adapter_should_include_dependency_results() -> None:
    """
    检查前置 Worker 结果是否同时进入结构化 state 和当前问题文本。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录同步调用输入并返回训练建议。"""

        received_states.append(state)
        return {
            "final_answer": "每天进行短时正向训练。",
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="training_agent",
        runner=runner,
    )
    dependency_result = AgentTaskResult(
        step_id="load_profile",
        assigned_agent="profile_agent",
        status="completed",
        summary="已读取三个月大金毛资料。",
        output={"age_months": 3},
    )
    step = AgentTaskStep(
        step_id="query_training",
        title="查询训练知识",
        assigned_agent="training_agent",
        depends_on=["load_profile"],
    )

    asyncio.run(
        adapter(
            step,
            {"load_profile": dependency_result},
        )
    )

    input_state = received_states[0]
    assert input_state["multi_agent_dependency_results"][
        "load_profile"
    ]["output"] == {"age_months": 3}
    assert "已读取三个月大金毛资料" in input_state["question"]


def test_graph_worker_adapter_should_convert_waiting_state() -> None:
    """
    检查 ToolAgent 等待确认状态是否转换成 awaiting_input 结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回等待用户确认的 ToolAgent state。"""

        return {
            **state,
            "waiting_user_input": True,
            "tool_confirmation_prompt": "是否允许查询健康数据库？",
            "tool_agent_response": {
                "status": "awaiting_confirmation",
            },
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_database",
        title="查询健康数据库",
        assigned_agent="tool_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "awaiting_input"
    assert result.requires_user_input is True
    assert result.clarification_prompt == "是否允许查询健康数据库？"


def test_graph_worker_adapter_should_reject_wrong_agent() -> None:
    """
    检查步骤指定的 Agent 与适配器不一致时是否停止执行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回不会在本测试中真正使用的 state。"""

        return state

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_health",
        title="查询健康知识",
        assigned_agent="health_agent",
    )

    with pytest.raises(ValueError, match="只负责 tool_agent"):
        asyncio.run(adapter(step, {}))


def test_graph_worker_adapter_should_require_waiting_prompt() -> None:
    """
    检查 Agent 声明等待用户却没有提示时是否拒绝生成无效暂停结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """返回缺少用户提示的等待状态。"""

        return {
            **state,
            "waiting_user_input": True,
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="tool_agent",
        runner=runner,
    )
    step = AgentTaskStep(
        step_id="query_database",
        title="查询数据库",
        assigned_agent="tool_agent",
    )

    with pytest.raises(ValueError, match="没有提供等待提示"):
        asyncio.run(adapter(step, {}))


def test_default_state_builder_should_include_resume_context() -> None:
    """
    检查恢复步骤是否把上次输出和用户回答追加到 Agent 问题中。

    参数含义：
        无。

    返回值含义：
        None。
    """

    step = AgentTaskStep(
        step_id="confirm_profile",
        title="确认读取资料",
        assigned_agent="profile_agent",
        input_data={
            "multi_agent_is_resuming": True,
            "multi_agent_resume_input": "允许读取",
            "multi_agent_previous_worker_output": {
                "pending_action": "读取宠物档案"
            },
        },
    )

    state = build_default_agent_state(step, {})

    assert "当前步骤正在从等待用户输入的状态恢复" in state["question"]
    assert "读取宠物档案" in state["question"]
    assert "允许读取" in state["question"]


def test_default_state_builder_should_keep_identity_and_clear_control_fields(
) -> None:
    """
    检查 Worker State 是否保留可信用户身份并清除 Planner 内部路由字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    step = AgentTaskStep(
        step_id="query_health",
        title="查询健康知识",
        assigned_agent="dog_knowledge_agent",
        input_data={
            "question": "查询金毛健康知识",
            "user_id": "user_001",
            "session_id": "session_001",
            "trace_id": "trace_001",
            "intent": "dog_recommendation",
            "route_decision": {"route": "recommendation_agent"},
            "answer_strategy": {"task_type": "recommendation"},
        },
    )

    state = build_default_agent_state(step, {})

    assert state["user_id"] == "user_001"
    assert state["session_id"] == "session_001"
    assert state["trace_id"] == "trace_001"
    assert state["question"] == "查询金毛健康知识"
    assert state["retrieval_question"] == "查询金毛健康知识"
    assert state["memory_retrieval_text"] == "查询金毛健康知识"
    assert "intent" not in state
    assert "route_decision" not in state
    assert "answer_strategy" not in state


def test_graph_worker_adapter_should_pause_for_missing_skill_inputs() -> None:
    """
    检查步骤级 Skill 缺少输入时在调用子 Agent 前暂停。

    功能：
        训练计划步骤只包含犬种和年龄时，应返回 awaiting_input，并把 Skill
        恢复数据保存在该步骤 output，runner 不应提前执行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runner_call_count = 0

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录不应发生的子 Agent 调用。"""

        nonlocal runner_call_count
        runner_call_count += 1
        return state

    adapter = GraphAgentWorkerAdapter(
        agent_name="dog_knowledge_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
    )
    step = AgentTaskStep(
        step_id="build_training_plan",
        title="制定训练计划",
        description="为6岁的金毛制定训练计划",
        assigned_agent="dog_knowledge_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "awaiting_input"
    assert runner_call_count == 0
    assert result.output["skill_selected_id"] == "dog-training-plan"
    assert result.output["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }
    assert result.metadata["skill_runtime"]["status"] == "awaiting_input"


def test_graph_worker_preflight_should_not_call_agent_runner() -> None:
    """验证执行前检查发现 Skill 缺字段时不会启动子 Agent。"""

    runner_call_count = 0

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录不应发生的子 Agent 调用。"""

        nonlocal runner_call_count
        runner_call_count += 1
        return state

    adapter = GraphAgentWorkerAdapter(
        agent_name="dog_knowledge_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
    )
    step = AgentTaskStep(
        step_id="training_preflight",
        title="制定训练计划",
        description="为6岁的金毛制定训练计划",
        assigned_agent="dog_knowledge_agent",
    )

    result = adapter.preflight(step, {})

    assert result is not None
    assert result.status == "awaiting_input"
    assert runner_call_count == 0


def test_graph_worker_preflight_should_prefill_skill_from_pet_profile() -> None:
    """验证多智能体 Worker 会在澄清前先使用宠物档案补全 Skill。"""

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """本用例只检查 Preflight，不应调用该执行函数。"""

        return state

    profile_service = FakeWorkerPetProfileService()
    adapter = GraphAgentWorkerAdapter(
        agent_name="dog_knowledge_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
        pet_profile_service=profile_service,
    )
    step = AgentTaskStep(
        step_id="training_profile_prefill",
        title="制定训练计划",
        description="帮我家狗狗安排训练计划",
        assigned_agent="dog_knowledge_agent",
        input_data={
            "user_id": "user_001",
            "active_pet_key": "pet_doudou",
            "active_pet_name": "豆豆",
        },
    )

    result = adapter.preflight(step, {})

    assert result is not None
    assert result.status == "awaiting_input"
    assert result.output["skill_inputs"] == {
        "breed": "金毛",
        "age": "6岁",
    }
    assert result.output["active_pet_key"] == "pet_doudou"
    assert result.output["skill_profile_recall_result"]["status"] == (
        "applied"
    )
    assert profile_service.calls == [
        {
            "user_id": "user_001",
            "active_pet_key": "pet_doudou",
            "active_pet_name": "豆豆",
            "selected_attributes": ["breed", "age_years"],
        }
    ]
    assert "犬种" not in result.clarification_prompt
    assert "年龄" not in result.clarification_prompt
    assert "当前行为基础" in result.clarification_prompt
    assert "训练目标" in result.clarification_prompt


def test_graph_worker_adapter_should_run_unmatched_step_normally() -> None:
    """
    检查启用 SkillRuntime 后未命中技能的步骤保持旧执行行为。

    功能：
        普通资料整理步骤不适用训练计划 Skill，Adapter 应继续调用 runner，
        不能因为生产环境配置了 SkillRuntime 就拦截所有多智能体步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """保存普通步骤输入并返回整理结果。"""

        received_states.append(state)
        return {"final_answer": "资料整理完成。"}

    adapter = GraphAgentWorkerAdapter(
        agent_name="general_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
    )
    step = AgentTaskStep(
        step_id="organize_materials",
        title="整理已有资料",
        assigned_agent="general_agent",
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "completed"
    assert result.summary == "资料整理完成。"
    assert len(received_states) == 1
    assert "skill_context" not in received_states[0]
    assert "skill_runtime" not in result.metadata


def test_graph_worker_adapter_should_resume_step_skill_before_runner() -> None:
    """
    检查 Scheduler 恢复步骤后由 Worker 恢复该步骤内部 Skill。

    功能：
        从 previous_worker_output 读取历史 Skill 输入，与用户补充内容合并，
        准备完成后才调用 runner，并将 Skill 上下文放入独立状态字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    received_states: list[Mapping[str, Any]] = []

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """保存恢复后的 Worker state 并返回步骤答案。"""

        received_states.append(state)
        return {
            **state,
            "final_answer": "已完成分阶段训练计划。",
        }

    adapter = GraphAgentWorkerAdapter(
        agent_name="dog_knowledge_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
    )
    step = AgentTaskStep(
        step_id="build_training_plan",
        title="制定训练计划",
        assigned_agent="dog_knowledge_agent",
        input_data={
            "multi_agent_is_resuming": True,
            "multi_agent_resume_input": (
                "它目前会坐下，希望学习等待和召回。"
            ),
            "multi_agent_previous_worker_output": {
                "skill_selected_id": "dog-training-plan",
                "skill_status": "awaiting_input",
                "skill_inputs": {
                    "breed": "Golden Retriever",
                    "age": "6岁",
                },
            },
        },
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "completed"
    assert len(received_states) == 1
    received_state = received_states[0]
    assert received_state["skill_status"] == "ready"
    assert received_state["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }
    assert "已经校验通过的 Skill 输入" in received_state["question"]
    assert "技能：狗狗训练计划" in received_state["question"]
    assert "技能：狗狗训练计划" in received_state["skill_context"]
    assert received_state["retrieval_question"] == "制定训练计划"


def test_graph_worker_adapter_should_merge_structured_resume_fields() -> None:
    """验证 Worker 会直接合并调度器分配的结构化恢复字段。"""

    received_states: list[Mapping[str, Any]] = []

    async def runner(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """记录准备完成的状态并返回固定答案。"""

        received_states.append(state)
        return {**state, "final_answer": "训练计划已生成。"}

    adapter = GraphAgentWorkerAdapter(
        agent_name="dog_knowledge_agent",
        runner=runner,
        skill_runtime=build_default_skill_runtime(),
    )
    step = AgentTaskStep(
        step_id="build_training_plan",
        title="制定训练计划",
        assigned_agent="dog_knowledge_agent",
        input_data={
            "multi_agent_is_resuming": True,
            "multi_agent_resume_input": {
                "current_behavior": "会坐下",
                "training_goal": "学习等待和召回",
            },
            "multi_agent_previous_worker_output": {
                "skill_selected_id": "dog-training-plan",
                "skill_status": "awaiting_input",
                "skill_inputs": {
                    "breed": "Golden Retriever",
                    "age": "6岁",
                },
            },
        },
    )

    result = asyncio.run(adapter(step, {}))

    assert result.status == "completed"
    assert received_states[0]["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "会坐下",
        "training_goal": "学习等待和召回",
    }
