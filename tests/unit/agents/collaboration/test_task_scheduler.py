"""
多 Agent 任务调度器测试。

功能：
    使用确定性 Worker 函数验证依赖批次、并发调度、失败阻断、允许失败和
    等待用户输入，不调用真实 Agent、LLM 或外部服务。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

import src.agents.collaboration.scheduler.scheduler as scheduler_module
from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskScheduler,
)


def build_scheduler_plan(
    *,
    profile_allow_failure: bool = False,
) -> AgentTaskPlan:
    """
    构建包含串行和并行关系的调度测试计划。

    参数含义：
        profile_allow_failure:
            读取资料失败后是否仍允许后续步骤继续。

    返回值含义：
        AgentTaskPlan:
            读取资料后并行查询健康、训练知识，最后生成方案的测试计划。
    """

    return AgentTaskPlan(
        plan_id="scheduler_plan_001",
        objective="生成狗狗健康和训练方案",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
                allow_failure=profile_allow_failure,
            ),
            AgentTaskStep(
                step_id="query_health",
                title="查询健康知识",
                assigned_agent="health_agent",
                depends_on=["load_profile"],
            ),
            AgentTaskStep(
                step_id="query_training",
                title="查询训练知识",
                assigned_agent="training_agent",
                depends_on=["load_profile"],
            ),
            AgentTaskStep(
                step_id="build_plan",
                title="生成方案",
                assigned_agent="general_agent",
                depends_on=["query_health", "query_training"],
            ),
        ],
    )


def build_success_worker(
    calls: list[tuple[str, list[str]]],
):
    """
    创建记录调用顺序和依赖结果的成功 Worker。

    参数含义：
        calls:
            用于保存步骤编号及其收到的依赖结果编号。

    返回值含义：
        Callable:
            返回 completed AgentTaskResult 的异步 Worker。
    """

    async def worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        记录调用并返回当前步骤成功结果。

        参数含义：
            step:
                当前执行步骤。
            dependency_results:
                当前步骤收到的前置步骤结果。

        返回值含义：
            AgentTaskResult:
                当前步骤的成功结果。
        """

        calls.append((step.step_id, list(dependency_results)))
        await asyncio.sleep(0)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary=f"{step.title}完成",
            output={"step": step.step_id},
        )

    return worker


def test_scheduler_should_execute_steps_by_dependency_batches() -> None:
    """
    检查调度器是否按依赖批次执行并把前置结果传给后续 Worker。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    worker = build_success_worker(calls)
    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": worker,
            "health_agent": worker,
            "training_agent": worker,
            "general_agent": worker,
        }
    )

    result = asyncio.run(
        scheduler.execute(
            build_scheduler_plan(),
            collaboration_id="task_001",
        )
    )

    assert result.plan.status == "completed"
    assert result.status == "running"
    assert result.metadata["ready_batches"] == [
        ["load_profile"],
        ["query_health", "query_training"],
        ["build_plan"],
    ]
    assert calls == [
        ("load_profile", []),
        ("query_health", ["load_profile"]),
        ("query_training", ["load_profile"]),
        ("build_plan", ["query_health", "query_training"]),
    ]
    assert result.metadata["awaiting_result_aggregation"] is True


def test_scheduler_should_skip_steps_blocked_by_failure() -> None:
    """
    检查不允许失败的前置步骤出错后是否跳过其全部后续步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    success_worker = build_success_worker(calls)

    async def failing_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """模拟读取资料时抛出异常。"""

        _ = step, dependency_results
        raise RuntimeError("宠物档案服务不可用")

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": failing_profile_worker,
            "health_agent": success_worker,
            "training_agent": success_worker,
            "general_agent": success_worker,
        }
    )

    result = asyncio.run(scheduler.execute(build_scheduler_plan()))

    statuses = {
        item.step_id: item.status
        for item in result.task_results
    }
    assert statuses == {
        "load_profile": "failed",
        "query_health": "skipped",
        "query_training": "skipped",
        "build_plan": "skipped",
    }
    assert result.status == "failed"
    assert calls == []


def test_scheduler_should_continue_after_allowed_failure() -> None:
    """
    检查允许失败的前置步骤出错后是否继续执行后续步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    success_worker = build_success_worker(calls)

    def failing_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回一条允许继续执行的失败结果。"""

        _ = dependency_results
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="failed",
            error_message="没有找到宠物档案",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": failing_profile_worker,
            "health_agent": success_worker,
            "training_agent": success_worker,
            "general_agent": success_worker,
        }
    )

    result = asyncio.run(
        scheduler.execute(
            build_scheduler_plan(profile_allow_failure=True)
        )
    )

    assert result.status == "partial"
    assert result.plan.status == "partial"
    assert [call[0] for call in calls] == [
        "query_health",
        "query_training",
        "build_plan",
    ]


def test_scheduler_should_not_run_plan_waiting_for_user() -> None:
    """
    检查计划等待用户澄清时是否完全不调用 Worker。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    worker = build_success_worker(calls)
    plan_data = build_scheduler_plan().model_dump(mode="python")
    plan_data.update(
        {
            "status": "awaiting_input",
            "requires_user_input": True,
            "clarification_prompt": "请补充狗狗年龄。",
        }
    )
    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": worker,
        }
    )

    result = asyncio.run(
        scheduler.execute(AgentTaskPlan.model_validate(plan_data))
    )

    assert result.status == "awaiting_input"
    assert result.task_results == []
    assert calls == []


def test_scheduler_should_convert_wrong_worker_result_to_failure() -> None:
    """
    检查 Worker 返回错误步骤编号时是否转换成标准失败结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def wrong_result_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """故意返回与当前步骤不一致的结果。"""

        _ = step, dependency_results
        return AgentTaskResult(
            step_id="wrong_step",
            assigned_agent="profile_agent",
            status="completed",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": wrong_result_worker,
        }
    )
    plan = AgentTaskPlan(
        plan_id="wrong_result_plan",
        objective="测试错误 Worker 结果",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
            )
        ],
    )

    result = asyncio.run(scheduler.execute(plan))

    assert result.status == "failed"
    assert result.task_results[0].status == "failed"
    assert "错误的 step_id" in result.task_results[0].error_message


def test_scheduler_log_should_use_indented_json(monkeypatch) -> None:
    """
    检查调度器关键日志是否使用四空格缩进的 JSON 格式。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用来接收日志而不写入真实日志文件。

    返回值含义：
        None。
    """

    messages: list[str] = []
    recording_logger = SimpleNamespace(
        info=messages.append,
        warning=messages.append,
        error=messages.append,
    )
    monkeypatch.setattr(scheduler_module, "logger", recording_logger)

    scheduler_module._log_scheduler_event(
        level="info",
        event="test_event",
        payload={
            "plan_id": "plan_001",
            "step_ids": ["load_profile"],
        },
    )

    assert len(messages) == 1
    assert '\n    "event": "test_event"' in messages[0]
    assert '\n    "plan_id": "plan_001"' in messages[0]
