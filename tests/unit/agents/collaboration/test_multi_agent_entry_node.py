"""多 Agent 主图入口节点测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskCancellationRegistry,
    MultiAgentTaskCancellationToken,
    MultiAgentTaskResult,
    build_multi_agent_entry_node,
)
from src.runtime.long_tasks import LongTask


class FakeMultiAgentOrchestrator:
    """记录主图入口选择 run 还是 resume 的测试编排器。"""

    def __init__(self, result: MultiAgentTaskResult) -> None:
        self.result = result
        self.run_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []

    async def run(
        self,
        objective: str,
        **kwargs: Any,
    ) -> MultiAgentTaskResult:
        """记录新任务或重新规划调用并返回固定结果。"""

        self.run_calls.append({"objective": objective, **kwargs})
        return self.result

    async def resume(
        self,
        task_result: MultiAgentTaskResult,
        *,
        user_inputs: dict[str, Any],
        cancellation_token: MultiAgentTaskCancellationToken | None = None,
    ) -> MultiAgentTaskResult:
        """记录恢复调用并返回固定结果。"""

        self.resume_calls.append(
            {
                "task_result": task_result,
                "user_inputs": user_inputs,
                "cancellation_token": cancellation_token,
            }
        )
        return self.result


class FakeLongTaskApplicationService:
    """记录主图入口交给长任务应用服务的持久化快照。"""

    def __init__(self, *, fail: bool = False) -> None:
        """
        初始化交接记录并配置可选失败行为。

        参数含义：
            fail:
                是否模拟 Store 或 Redis Stream 交接失败。

        返回值含义：
            None。
        """

        self.fail = fail
        self.tasks: list[LongTask] = []

    async def handoff_paused_collaboration_task(
        self,
        task: LongTask,
    ) -> LongTask:
        """
        记录准备持久化的任务并返回原快照。

        参数含义：
            task:
                入口节点适配出的 durable LongTask。

        返回值含义：
            LongTask:
                模拟 Store 创建成功后的任务；失败模式抛出 RuntimeError。
        """

        self.tasks.append(task)
        if self.fail:
            raise RuntimeError("模拟长任务交接失败")
        return task

def build_entry_task_result(
    *,
    status: str,
) -> MultiAgentTaskResult:
    """
    构建主图入口测试需要的完成或暂停任务结果。

    参数含义：
        status:
            completed 或 awaiting_input。

    返回值含义：
        MultiAgentTaskResult:
            与指定状态一致的测试任务结果。
    """

    is_waiting = status == "awaiting_input"
    step = AgentTaskStep(
        step_id="profile",
        title="读取资料",
        assigned_agent="dog_knowledge_agent",
        status=("awaiting_input" if is_waiting else "completed"),
    )
    plan = AgentTaskPlan(
        plan_id="entry_plan",
        objective="生成综合方案",
        steps=[step],
        status=("awaiting_input" if is_waiting else "completed"),
        requires_user_input=is_waiting,
        clarification_prompt=("是否继续？" if is_waiting else ""),
    )
    result = AgentTaskResult(
        step_id=step.step_id,
        assigned_agent=step.assigned_agent,
        status=("awaiting_input" if is_waiting else "completed"),
        requires_user_input=is_waiting,
        clarification_prompt=("是否继续？" if is_waiting else ""),
    )
    return MultiAgentTaskResult(
        collaboration_id="entry_task",
        plan=plan,
        status=("awaiting_input" if is_waiting else "completed"),
        task_results=[result],
        final_answer=("综合方案已生成。" if not is_waiting else ""),
    )


def build_budget_paused_entry_result() -> MultiAgentTaskResult:
    """
    构建第一批完成、第二步等待后台继续的预算暂停结果。

    返回值含义：
        MultiAgentTaskResult:
            可由协作适配器安全转换为 durable LongTask 的运行中结果。
    """

    plan = AgentTaskPlan(
        plan_id="entry_paused_plan",
        objective="读取档案并生成照护建议",
        status="running",
        steps=[
            AgentTaskStep(
                step_id="profile",
                title="读取资料",
                assigned_agent="dog_knowledge_agent",
                status="completed",
            ),
            AgentTaskStep(
                step_id="answer",
                title="生成建议",
                assigned_agent="general_agent",
                depends_on=["profile"],
                status="pending",
            ),
        ],
    )
    return MultiAgentTaskResult(
        collaboration_id="entry_budget_task",
        plan=plan,
        status="running",
        task_results=[
            AgentTaskResult(
                step_id="profile",
                assigned_agent="dog_knowledge_agent",
                status="completed",
                summary="档案读取完成。",
                metadata={"scheduler_attempt_count": 1},
            )
        ],
        metadata={
            "execution_paused": {
                "reason": "inline_budget_exhausted",
                "elapsed_seconds": 2.0,
                "budget_seconds": 1.0,
                "remaining_step_ids": ["answer"],
            },
            "awaiting_result_aggregation": False,
        },
    )


def test_entry_node_should_handoff_budget_paused_result() -> None:
    """
    验证入口使用用户会话信息转换、保存并返回后台任务引用。

    返回值含义：
        None。
    """

    orchestrator = FakeMultiAgentOrchestrator(
        build_budget_paused_entry_result()
    )
    application_service = FakeLongTaskApplicationService()
    node = build_multi_agent_entry_node(
        orchestrator=orchestrator,
        long_task_application_service=application_service,
    )

    update = asyncio.run(
        node(
            {
                "question": "根据档案生成照护建议",
                "user_id": "user_001",
                "session_id": "thread_001",
                "trace_id": "trace_001",
                "multi_agent_resume_action": "none",
            }
        )
    )

    assert len(application_service.tasks) == 1
    task = application_service.tasks[0]
    assert task.task_id == "entry_budget_task"
    assert task.user_id == "user_001"
    assert task.thread_id == "thread_001"
    assert task.goal.original_request == "根据档案生成照护建议"
    assert task.execution_mode == "durable"
    assert [step.status for step in task.steps] == [
        "completed",
        "ready",
    ]
    handoff = update["multi_agent_task_result"]["metadata"][
        "durable_handoff"
    ]
    assert handoff == {
        "task_id": "entry_budget_task",
        "task_version": 1,
        "status": "running",
        "execution_mode": "durable",
        "owner_user_id": "user_001",
    }
    assert update["waiting_user_input"] is False
    assert "entry_budget_task" in update["final_answer"]


def test_entry_node_should_require_handoff_service_for_budget_pause() -> None:
    """
    验证预算暂停时没有交接服务不会被伪装成成功响应。

    返回值含义：
        None。
    """

    node = build_multi_agent_entry_node(
        orchestrator=FakeMultiAgentOrchestrator(
            build_budget_paused_entry_result()
        )
    )

    with pytest.raises(RuntimeError, match="交接服务尚未配置"):
        asyncio.run(
            node(
                {
                    "question": "根据档案生成照护建议",
                    "user_id": "user_001",
                    "session_id": "thread_001",
                    "multi_agent_resume_action": "none",
                }
            )
        )


def test_entry_node_should_propagate_handoff_failure() -> None:
    """
    验证持久化或发布失败时入口不会返回虚假的后台任务编号。

    返回值含义：
        None。
    """

    application_service = FakeLongTaskApplicationService(fail=True)
    node = build_multi_agent_entry_node(
        orchestrator=FakeMultiAgentOrchestrator(
            build_budget_paused_entry_result()
        ),
        long_task_application_service=application_service,
    )

    with pytest.raises(RuntimeError, match="模拟长任务交接失败"):
        asyncio.run(
            node(
                {
                    "question": "根据档案生成照护建议",
                    "user_id": "user_001",
                    "session_id": "thread_001",
                    "multi_agent_resume_action": "none",
                }
            )
        )


def test_entry_node_should_not_handoff_completed_result() -> None:
    """
    验证普通完成结果不会调用可选长任务应用服务。

    返回值含义：
        None。
    """

    application_service = FakeLongTaskApplicationService()
    node = build_multi_agent_entry_node(
        orchestrator=FakeMultiAgentOrchestrator(
            build_entry_task_result(status="completed")
        ),
        long_task_application_service=application_service,
    )

    update = asyncio.run(
        node(
            {
                "question": "生成综合方案",
                "multi_agent_resume_action": "none",
            }
        )
    )

    assert application_service.tasks == []
    assert update["final_answer"] == "综合方案已生成。"


def test_resumed_budget_handoff_should_preserve_plan_objective() -> None:
    """
    验证恢复回答触发后台交接时不会覆盖任务的完整原始目标。

    返回值含义：
        None。
    """

    application_service = FakeLongTaskApplicationService()
    node = build_multi_agent_entry_node(
        orchestrator=FakeMultiAgentOrchestrator(
            build_budget_paused_entry_result()
        ),
        long_task_application_service=application_service,
    )
    previous_waiting_result = build_entry_task_result(
        status="awaiting_input"
    )

    asyncio.run(
        node(
            {
                "question": "允许继续",
                "user_id": "user_001",
                "session_id": "thread_001",
                "multi_agent_resume_action": "resume",
                "multi_agent_task_result": (
                    previous_waiting_result.model_dump(mode="python")
                ),
                "multi_agent_resume_inputs": {
                    "profile": "允许继续",
                },
            }
        )
    )

    assert application_service.tasks[0].goal.original_request == (
        "读取档案并生成照护建议"
    )


def test_entry_node_should_run_new_task() -> None:
    """
    检查普通复杂目标是否调用 orchestrator.run。

    参数含义：无。
    返回值含义：None。
    """

    orchestrator = FakeMultiAgentOrchestrator(
        build_entry_task_result(status="completed")
    )
    node = build_multi_agent_entry_node(orchestrator=orchestrator)

    update = asyncio.run(
        node(
            {
                "question": "生成健康和训练综合方案",
                "multi_agent_resume_action": "none",
            }
        )
    )

    assert orchestrator.run_calls[0]["objective"] == (
        "生成健康和训练综合方案"
    )
    assert orchestrator.run_calls[0]["multi_agent_task_id"].startswith(
        "multi_agent_task_"
    )
    assert orchestrator.run_calls[0]["cancellation_token"] is None
    assert orchestrator.resume_calls == []
    assert update["final_answer"] == "综合方案已生成。"


def test_entry_node_should_resume_paused_task() -> None:
    """
    检查恢复状态是否调用 orchestrator.resume 并传入结构化回答。

    参数含义：无。
    返回值含义：None。
    """

    paused_result = build_entry_task_result(status="awaiting_input")
    orchestrator = FakeMultiAgentOrchestrator(
        build_entry_task_result(status="completed")
    )
    node = build_multi_agent_entry_node(orchestrator=orchestrator)

    update = asyncio.run(
        node(
            {
                "question": "允许继续",
                "multi_agent_resume_action": "resume",
                "multi_agent_task_result": paused_result.model_dump(
                    mode="python"
                ),
                "multi_agent_resume_inputs": {
                    "profile": "允许继续"
                },
            }
        )
    )

    assert orchestrator.run_calls == []
    assert orchestrator.resume_calls[0]["user_inputs"] == {
        "profile": "允许继续"
    }
    assert update["waiting_user_input"] is False


def test_entry_node_should_replan_with_planner_clarification() -> None:
    """
    检查 Planner 澄清回答是否携带原目标和上下文重新调用 run。

    功能：
        验证 replan 不会把“3 岁，20 公斤”误当成一个新目标，而是继续使用
        暂停计划的 objective，并将新回答和上一次问题写入 context。

    参数含义：
        无。

    返回值含义：
        None。
    """

    paused_result = build_entry_task_result(status="awaiting_input")
    orchestrator = FakeMultiAgentOrchestrator(
        build_entry_task_result(status="completed")
    )
    node = build_multi_agent_entry_node(orchestrator=orchestrator)

    update = asyncio.run(
        node(
            {
                "question": "3 岁，20 公斤",
                "memory_context": "用户养的是一只金毛。",
                "user_id": "user_001",
                "session_id": "session_001",
                "trace_id": "trace_001",
                "active_pet_key": "pet_doudou",
                "active_pet_name": "豆豆",
                "multi_agent_resume_action": "replan",
                "multi_agent_task_result": paused_result.model_dump(
                    mode="python"
                ),
                "multi_agent_resume_inputs": {
                    "planner_clarification": "3 岁，20 公斤"
                },
            }
        )
    )

    assert orchestrator.resume_calls == []
    run_call = orchestrator.run_calls[0]
    assert run_call["objective"] == paused_result.plan.objective
    assert run_call["context"] == {
        "user_clarification": "3 岁，20 公斤",
        "previous_clarification_prompt": "是否继续？",
        "memory_context": "用户养的是一只金毛。",
        "user_id": "user_001",
        "session_id": "session_001",
        "trace_id": "trace_001",
    }
    assert run_call["worker_runtime_context"] == {
        "user_id": "user_001",
        "session_id": "session_001",
        "trace_id": "trace_001",
        "active_pet_key": "pet_doudou",
        "active_pet_name": "豆豆",
    }
    assert update["final_answer"] == "综合方案已生成。"


def test_entry_node_should_save_awaiting_result_for_checkpoint() -> None:
    """
    检查暂停结果是否转换成可写入 Checkpoint 的主图字段。

    参数含义：无。
    返回值含义：None。
    """

    orchestrator = FakeMultiAgentOrchestrator(
        build_entry_task_result(status="awaiting_input")
    )
    node = build_multi_agent_entry_node(orchestrator=orchestrator)

    update = asyncio.run(
        node(
            {
                "question": "生成综合方案",
                "multi_agent_resume_action": "none",
            }
        )
    )

    assert update["multi_agent_task_result"]["status"] == "awaiting_input"
    assert update["multi_agent_pending_prompt"] == "是否继续？"
    assert update["waiting_user_input"] is True
    assert update["final_answer"] == "是否继续？"


def test_entry_node_should_prefer_batch_clarification_prompt() -> None:
    """
    检查主图会优先展示调度器整理的整批澄清提示。

    功能：
        即使步骤结果仍保留旧版单步骤提示，也应把整批提示写入 DogState，
        避免用户只能看到第一个 Worker 的问题。

    参数含义：
        无。

    返回值含义：
        None。
    """

    waiting_result = build_entry_task_result(status="awaiting_input")
    waiting_result.metadata = {
        "clarification_prompt": "整批统一提示。",
        "clarification_bundle": {
            "step_requests": [],
            "field_consumers": {},
            "display_prompt": "健康分析需要年龄；训练计划需要训练目标。",
        },
    }
    orchestrator = FakeMultiAgentOrchestrator(waiting_result)
    node = build_multi_agent_entry_node(orchestrator=orchestrator)

    update = asyncio.run(
        node(
            {
                "question": "生成综合方案",
                "multi_agent_resume_action": "none",
            }
        )
    )

    assert update["multi_agent_pending_prompt"] == (
        "健康分析需要年龄；训练计划需要训练目标。"
    )
    assert update["final_answer"] == (
        "健康分析需要年龄；训练计划需要训练目标。"
    )


def test_entry_node_should_register_and_cleanup_cancellation_token() -> None:
    """
    检查多 Agent 入口会在调用期间登记令牌并在结束后清理。

    功能：
        使用固定 trace_id 验证任务编号可预测、编排器收到共享令牌，并且
        正常返回后登记表不再保留已经结束的任务。

    参数含义：无。
    返回值含义：None。
    """

    registry = MultiAgentTaskCancellationRegistry()
    orchestrator = FakeMultiAgentOrchestrator(
        build_entry_task_result(status="completed")
    )
    node = build_multi_agent_entry_node(
        orchestrator=orchestrator,
        cancellation_registry=registry,
    )

    update = asyncio.run(
        node(
            {
                "question": "生成健康和训练综合方案",
                "trace_id": "trace_cancel_001",
                "multi_agent_resume_action": "none",
            }
        )
    )

    run_call = orchestrator.run_calls[0]
    task_id = "multi_agent_task_trace_cancel_001"
    assert run_call["multi_agent_task_id"] == task_id
    assert isinstance(
        run_call["cancellation_token"],
        MultiAgentTaskCancellationToken,
    )
    assert registry.contains(task_id) is False
    assert update["final_answer"] == "综合方案已生成。"


def test_entry_node_should_cleanup_registry_after_orchestration_error() -> None:
    """
    检查编排器抛出异常时运行中任务登记也会被清理。

    功能：
        验证入口使用 finally 清理令牌，避免失败任务永久占用任务编号。

    参数含义：无。
    返回值含义：None。
    """

    class FailingOrchestrator(FakeMultiAgentOrchestrator):
        """模拟执行期间抛出异常的多 Agent 编排器。"""

        async def run(
            self,
            objective: str,
            **kwargs: Any,
        ) -> MultiAgentTaskResult:
            """记录调用后抛出固定异常。"""

            self.run_calls.append({"objective": objective, **kwargs})
            raise RuntimeError("模拟编排失败")

    registry = MultiAgentTaskCancellationRegistry()
    orchestrator = FailingOrchestrator(
        build_entry_task_result(status="completed")
    )
    node = build_multi_agent_entry_node(
        orchestrator=orchestrator,
        cancellation_registry=registry,
    )
    task_id = "multi_agent_task_trace_error_001"

    with pytest.raises(RuntimeError, match="模拟编排失败"):
        asyncio.run(
            node(
                {
                    "question": "生成综合方案",
                    "trace_id": "trace_error_001",
                    "multi_agent_resume_action": "none",
                }
            )
        )

    assert registry.contains(task_id) is False
