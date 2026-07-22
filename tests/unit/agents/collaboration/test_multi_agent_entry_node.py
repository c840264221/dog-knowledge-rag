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
