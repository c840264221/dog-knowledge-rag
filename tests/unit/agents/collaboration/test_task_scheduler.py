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

import pytest

import src.agents.collaboration.scheduler.scheduler as scheduler_module
from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskCancellationRegistry,
    MultiAgentTaskCancellationToken,
    MultiAgentTaskResult,
    MultiAgentTaskScheduler,
)


def test_cancellation_registry_should_manage_running_task_token() -> None:
    """
    检查取消登记表能注册、取消并移除运行中任务。

    参数含义：无。
    返回值含义：None。
    """

    registry = MultiAgentTaskCancellationRegistry()
    token = registry.register("multi_agent_task_registry_001")

    assert registry.contains("multi_agent_task_registry_001") is True
    assert registry.cancel("multi_agent_task_registry_001") is True
    assert token.is_cancelled is True
    first_cancel_requested_at = token.cancel_requested_at
    assert isinstance(first_cancel_requested_at, float)
    token.cancel()
    assert token.cancel_requested_at == first_cancel_requested_at
    assert registry.unregister(
        "multi_agent_task_registry_001",
        token,
    ) is True
    assert registry.contains("multi_agent_task_registry_001") is False
    assert registry.cancel("multi_agent_task_registry_001") is False


def test_cancellation_registry_should_reject_duplicate_running_task() -> None:
    """
    检查同一任务编号不能同时登记两次。

    参数含义：无。
    返回值含义：None。
    """

    registry = MultiAgentTaskCancellationRegistry()
    registry.register("multi_agent_task_duplicate")

    with pytest.raises(ValueError, match="已经在运行"):
        registry.register("multi_agent_task_duplicate")


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
    assert [
        {
            "step_id": item["step_id"],
            "batch_numbers": item["batch_numbers"],
            "status": item["status"],
            "attempt_count": item["attempt_count"],
        }
        for item in result.metadata["worker_step_trace"]
    ] == [
        {
            "step_id": "load_profile",
            "batch_numbers": [1],
            "status": "completed",
            "attempt_count": 1,
        },
        {
            "step_id": "query_health",
            "batch_numbers": [2],
            "status": "completed",
            "attempt_count": 1,
        },
        {
            "step_id": "query_training",
            "batch_numbers": [2],
            "status": "completed",
            "attempt_count": 1,
        },
        {
            "step_id": "build_plan",
            "batch_numbers": [3],
            "status": "completed",
            "attempt_count": 1,
        },
    ]


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
    failure_attempts: list[str] = []
    success_worker = build_success_worker(calls)

    def failing_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回一条允许继续执行的失败结果。"""

        _ = dependency_results
        failure_attempts.append(step.step_id)
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
        },
        maximum_step_attempts=3,
    )

    result = asyncio.run(
        scheduler.execute(
            build_scheduler_plan(profile_allow_failure=True)
        )
    )

    assert result.status == "partial"
    assert result.plan.status == "partial"
    assert failure_attempts == ["load_profile"]
    assert [call[0] for call in calls] == [
        "query_health",
        "query_training",
        "build_plan",
    ]


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_scheduler_should_reject_non_positive_step_timeout(
    timeout_seconds: float,
) -> None:
    """
    检查单步骤超时秒数为零或负数时是否拒绝创建调度器。

    参数含义：
        timeout_seconds:
            pytest 依次传入的非法超时秒数。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="step_timeout_seconds"):
        MultiAgentTaskScheduler(
            workers={"profile_agent": build_success_worker([])},
            step_timeout_seconds=timeout_seconds,
        )


def test_scheduler_should_reject_non_positive_step_attempts() -> None:
    """
    检查单步骤最大尝试次数小于一时是否拒绝创建调度器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="maximum_step_attempts"):
        MultiAgentTaskScheduler(
            workers={"profile_agent": build_success_worker([])},
            maximum_step_attempts=0,
        )


def test_scheduler_should_retry_worker_exception_until_success() -> None:
    """
    检查 Worker 临时抛出异常后是否在最大次数内重试并成功返回。

    参数含义：
        无。

    返回值含义：
        None。
    """

    attempt_numbers: list[int] = []

    async def unstable_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """第一次模拟临时故障，第二次返回成功结果。"""

        _ = dependency_results
        attempt_numbers.append(len(attempt_numbers) + 1)
        if len(attempt_numbers) == 1:
            raise ConnectionError("临时网络错误")
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary="第二次执行成功。",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={"profile_agent": unstable_worker},
        maximum_step_attempts=2,
    )
    plan = AgentTaskPlan(
        plan_id="worker_retry_plan",
        objective="测试 Worker 异常重试",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
            )
        ],
    )

    result = asyncio.run(scheduler.execute(plan))

    assert result.status == "running"
    assert attempt_numbers == [1, 2]
    assert result.task_results[0].status == "completed"
    assert result.task_results[0].metadata["scheduler_attempt_count"] == 2
    assert result.metadata["maximum_step_attempts"] == 2


def test_scheduler_should_convert_worker_timeout_to_failure() -> None:
    """
    检查异步 Worker 超时后是否转换成包含超时信息的标准失败结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    attempt_numbers: list[int] = []

    async def slow_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """模拟长时间没有返回结果的异步 Worker。"""

        _ = dependency_results
        attempt_numbers.append(len(attempt_numbers) + 1)
        await asyncio.sleep(60)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={"profile_agent": slow_worker},
        step_timeout_seconds=0.01,
        maximum_step_attempts=2,
    )
    plan = AgentTaskPlan(
        plan_id="worker_timeout_plan",
        objective="测试 Worker 超时保护",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
            )
        ],
    )

    result = asyncio.run(scheduler.execute(plan))
    timeout_result = result.task_results[0]

    assert result.status == "failed"
    assert timeout_result.status == "failed"
    assert "执行超过 0.01 秒" in str(timeout_result.error_message)
    assert timeout_result.metadata["scheduler_generated_failure"] is True
    assert timeout_result.metadata["timed_out"] is True
    assert timeout_result.metadata["timeout_seconds"] == 0.01
    assert timeout_result.metadata["scheduler_attempt_count"] == 2
    assert attempt_numbers == [1, 2]
    assert result.metadata["step_timeout_seconds"] == 0.01


def test_scheduler_should_continue_after_allowed_timeout() -> None:
    """
    检查允许失败的步骤超时后是否继续执行依赖它的后续步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    success_worker = build_success_worker(calls)

    async def slow_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """模拟允许失败但执行超时的资料读取 Worker。"""

        _ = dependency_results
        await asyncio.sleep(60)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": slow_profile_worker,
            "health_agent": success_worker,
            "training_agent": success_worker,
            "general_agent": success_worker,
        },
        step_timeout_seconds=0.05,
    )

    result = asyncio.run(
        scheduler.execute(
            build_scheduler_plan(profile_allow_failure=True)
        )
    )

    assert result.status == "partial"
    assert result.plan.status == "partial"
    assert result.task_results[0].metadata["timed_out"] is True
    assert [call[0] for call in calls] == [
        "query_health",
        "query_training",
        "build_plan",
    ]


def test_scheduler_should_cancel_before_starting_workers() -> None:
    """
    检查执行前已经收到取消请求时是否跳过全部步骤且不调用 Worker。

    参数含义：
        无。

    返回值含义：
        None。
    """

    calls: list[tuple[str, list[str]]] = []
    cancellation_token = MultiAgentTaskCancellationToken()
    cancellation_token.cancel()
    scheduler = MultiAgentTaskScheduler(
        workers={"profile_agent": build_success_worker(calls)}
    )

    result = asyncio.run(
        scheduler.execute(
            build_scheduler_plan(),
            cancellation_token=cancellation_token,
        )
    )

    assert result.status == "cancelled"
    assert result.plan.status == "cancelled"
    assert result.plan.requires_user_input is False
    assert result.final_answer == "多 Agent 任务已取消。"
    assert result.metadata["cancellation_requested"] is True
    assert all(
        task_result.status == "skipped"
        for task_result in result.task_results
    )
    assert calls == []


def test_scheduler_should_cancel_running_worker_without_retry() -> None:
    """
    检查执行中取消时是否终止异步 Worker，并且不把取消当成异常重试。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def run_cancellation_scenario() -> tuple[
        MultiAgentTaskResult,
        list[str],
    ]:
        """
        启动一个持续等待的 Worker，在确认启动后发出取消请求。

        返回值含义：
            tuple[MultiAgentTaskResult, list[str]]:
                取消后的任务结果和 Worker 实际调用记录。
        """

        calls: list[str] = []
        worker_started = asyncio.Event()
        cancellation_token = MultiAgentTaskCancellationToken()

        async def waiting_worker(
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult:
            """模拟已经开始但长时间没有完成的异步 Worker。"""

            _ = dependency_results
            calls.append(step.step_id)
            worker_started.set()
            await asyncio.sleep(60)
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
            )

        scheduler = MultiAgentTaskScheduler(
            workers={"profile_agent": waiting_worker},
            maximum_step_attempts=3,
        )
        plan = AgentTaskPlan(
            plan_id="running_cancellation_plan",
            objective="测试执行中取消",
            steps=[
                AgentTaskStep(
                    step_id="load_profile",
                    title="读取资料",
                    assigned_agent="profile_agent",
                )
            ],
        )
        execution_task = asyncio.create_task(
            scheduler.execute(
                plan,
                cancellation_token=cancellation_token,
            )
        )
        await worker_started.wait()
        cancellation_token.cancel()
        result = await asyncio.wait_for(execution_task, timeout=1)
        return result, calls

    result, calls = asyncio.run(run_cancellation_scenario())

    assert result.status == "cancelled"
    assert result.plan.status == "cancelled"
    assert result.task_results[0].status == "skipped"
    assert result.task_results[0].metadata["cancelled"] is True
    assert result.task_results[0].metadata[
        "scheduler_attempt_count"
    ] == 1
    assert result.metadata["cancellation_response_latency_ms"] >= 0
    assert result.metadata["worker_step_trace"] == [
        {
            "step_id": "load_profile",
            "step_title": "读取资料",
            "assigned_agent": "profile_agent",
            "depends_on": [],
            "batch_numbers": [1],
            "status": "skipped",
            "attempt_count": 1,
            "latency_ms": result.task_results[0].latency_ms,
            "timed_out": False,
            "cancelled": True,
        }
    ]
    assert calls == ["load_profile"]


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


def test_scheduler_logs_should_include_step_title_and_agent(
    monkeypatch,
) -> None:
    """
    检查任务、批次和步骤完成日志是否包含可读步骤信息。

    功能：
        保留稳定 step_id 的同时，验证日志还会展示 title 和 assigned_agent，
        方便并发执行时区分每条日志属于哪个业务步骤和 Worker Agent。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用来接收日志而不写入真实控制台。

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

    async def health_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回健康分析步骤的固定成功结果。"""

        _ = dependency_results
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary="健康分析完成。",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={"dog_knowledge_agent": health_worker}
    )
    plan = AgentTaskPlan(
        plan_id="readable_log_plan",
        objective="生成健康方案",
        steps=[
            AgentTaskStep(
                step_id="step1",
                title="分析健康注意事项",
                assigned_agent="dog_knowledge_agent",
            )
        ],
    )

    asyncio.run(scheduler.execute(plan))

    joined_messages = "\n".join(messages)
    assert '"step_id": "step1"' in joined_messages
    assert '"title": "分析健康注意事项"' in joined_messages
    assert '"step_title": "分析健康注意事项"' in joined_messages
    assert '"assigned_agent": "dog_knowledge_agent"' in joined_messages


def test_scheduler_should_pause_when_worker_awaits_input() -> None:
    """
    检查 Worker 等待用户确认时是否暂停计划且不执行后续步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    downstream_calls: list[tuple[str, list[str]]] = []

    async def waiting_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """返回一条等待用户确认的标准步骤结果。"""

        _ = dependency_results
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="是否允许读取宠物档案？",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": waiting_worker,
            "health_agent": build_success_worker(downstream_calls),
        }
    )
    plan = AgentTaskPlan(
        plan_id="worker_waiting_plan",
        objective="查询狗狗健康知识",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
            ),
            AgentTaskStep(
                step_id="query_health",
                title="查询健康知识",
                assigned_agent="health_agent",
                depends_on=["load_profile"],
            ),
        ],
    )

    result = asyncio.run(scheduler.execute(plan))

    assert result.status == "awaiting_input"
    assert result.plan.status == "awaiting_input"
    assert result.plan.requires_user_input is True
    assert result.plan.steps[0].status == "awaiting_input"
    assert result.plan.steps[1].status == "pending"
    assert result.metadata["awaiting_step_ids"] == ["load_profile"]
    assert downstream_calls == []


def test_scheduler_should_collect_all_batch_clarification_requests() -> None:
    """
    检查同批多个 Worker 等待输入时不会丢失步骤和字段对应关系。

    功能：
        两个并行步骤都缺少年龄，其中训练步骤还缺少训练目标。调度器应
        保留两份步骤明细，并记录 age 同时被两个步骤使用。

    参数含义：
        无。

    返回值含义：
        None。
    """

    async def waiting_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """根据步骤返回包含固定 Skill 缺失字段的等待结果。"""

        _ = dependency_results
        requirements = [
            {
                "input_id": "age",
                "name": "年龄",
                "description": "狗狗年龄。",
                "requirement_level": "hard_required",
                "source_mappings": {"pet_profile": "age_years"},
            }
        ]
        if step.step_id == "step_training":
            requirements.append(
                {
                    "input_id": "training_goal",
                    "name": "训练目标",
                    "description": "希望狗狗学会的行为。",
                    "requirement_level": "degradable",
                    "source_mappings": {},
                }
            )
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt=(
                "请补充年龄和训练目标。"
                if step.step_id == "step_training"
                else "请补充年龄。"
            ),
            metadata={
                "skill_runtime": {
                    "input_check": {
                        "missing_input_requirements": requirements,
                        "can_run_degraded": (
                            step.step_id == "step_training"
                        ),
                    }
                }
            },
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "health_agent": waiting_worker,
            "training_agent": waiting_worker,
        },
        maximum_parallel_steps=2,
    )
    plan = AgentTaskPlan(
        plan_id="batch_clarification_plan",
        objective="生成健康和训练方案",
        steps=[
            AgentTaskStep(
                step_id="step_health",
                title="生成健康建议",
                assigned_agent="health_agent",
            ),
            AgentTaskStep(
                step_id="step_training",
                title="生成训练计划",
                assigned_agent="training_agent",
            ),
        ],
    )

    result = asyncio.run(scheduler.execute(plan))
    bundle = result.metadata["clarification_bundle"]

    assert result.status == "awaiting_input"
    assert result.metadata["awaiting_step_ids"] == [
        "step_health",
        "step_training",
    ]
    assert [
        request["step_id"] for request in bundle["step_requests"]
    ] == ["step_health", "step_training"]
    assert bundle["field_consumers"] == {
        "age": ["step_health", "step_training"],
        "training_goal": ["step_training"],
    }
    assert "生成健康建议（step_health）" in bundle["display_prompt"]
    assert "生成训练计划（step_training）" in bundle["display_prompt"]
    assert result.plan.clarification_prompt == bundle["display_prompt"]


def test_scheduler_preflight_should_block_entire_parallel_batch() -> None:
    """验证一个步骤预检查等待时，同批所有 Worker 都不会真正执行。"""

    class PreflightWorker:
        """提供可配置执行前检查结果的测试 Worker。"""

        def __init__(self, *, should_wait: bool) -> None:
            self.should_wait = should_wait
            self.call_count = 0

        def preflight(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult | None:
            """需要等待时返回标准暂停结果，否则允许执行。"""

            _ = dependency_results
            if not self.should_wait:
                return None
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="请补充训练目标。",
            )

        async def __call__(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult:
            """记录真正执行次数并返回完成结果。"""

            _ = dependency_results
            self.call_count += 1
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
            )

    waiting_worker = PreflightWorker(should_wait=True)
    ready_worker = PreflightWorker(should_wait=False)
    scheduler = MultiAgentTaskScheduler(
        workers={
            "training_agent": waiting_worker,
            "health_agent": ready_worker,
        },
        maximum_parallel_steps=2,
    )
    plan = AgentTaskPlan(
        plan_id="strict_preflight_plan",
        objective="并行生成健康和训练建议",
        steps=[
            AgentTaskStep(
                step_id="training_step",
                title="生成训练计划",
                assigned_agent="training_agent",
            ),
            AgentTaskStep(
                step_id="health_step",
                title="生成健康建议",
                assigned_agent="health_agent",
            ),
        ],
    )

    result = asyncio.run(scheduler.execute(plan))

    assert result.status == "awaiting_input"
    assert waiting_worker.call_count == 0
    assert ready_worker.call_count == 0
    assert result.metadata["ready_batches"] == []
    assert result.metadata["preflight_batches"] == [
        ["training_step", "health_step"]
    ]
    assert result.plan.steps[0].status == "awaiting_input"
    assert result.plan.steps[1].status == "pending"


def test_scheduler_resume_should_apply_degraded_mode_to_one_step_only() -> None:
    """验证简化执行控制字段只写入用户选择的等待步骤。"""

    class DegradableWorker:
        """首次预检查等待，收到简化决定后允许执行的测试 Worker。"""

        def __init__(self) -> None:
            self.received_inputs: list[dict[str, object]] = []

        def preflight(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult | None:
            """没有简化标记时返回等待结果。"""

            _ = dependency_results
            if (
                step.input_data.get("multi_agent_skill_execution_mode")
                == "degraded"
            ):
                return None
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                output={
                    "skill_status": "awaiting_input",
                    "skill_selected_id": "training-skill",
                    "skill_inputs": {},
                },
                requires_user_input=True,
                clarification_prompt="请补充训练目标或选择简化执行。",
            )

        async def __call__(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult:
            """保存步骤私有输入并返回完成结果。"""

            _ = dependency_results
            self.received_inputs.append(dict(step.input_data))
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
            )

    class ReadyWorker:
        """始终通过预检查的普通测试 Worker。"""

        def __init__(self) -> None:
            self.received_inputs: list[dict[str, object]] = []

        def preflight(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> None:
            """允许步骤直接执行。"""

            _ = step, dependency_results
            return None

        async def __call__(
            self,
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult:
            """保存步骤输入并返回完成结果。"""

            _ = dependency_results
            self.received_inputs.append(dict(step.input_data))
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
            )

    training_worker = DegradableWorker()
    health_worker = ReadyWorker()
    scheduler = MultiAgentTaskScheduler(
        workers={
            "training_agent": training_worker,
            "health_agent": health_worker,
        },
        maximum_parallel_steps=2,
    )
    plan = AgentTaskPlan(
        plan_id="degraded_resume_plan",
        objective="生成训练和健康建议",
        steps=[
            AgentTaskStep(
                step_id="training_step",
                title="生成训练计划",
                assigned_agent="training_agent",
            ),
            AgentTaskStep(
                step_id="health_step",
                title="生成健康建议",
                assigned_agent="health_agent",
            ),
        ],
    )

    paused_result = asyncio.run(scheduler.execute(plan))
    resumed_result = asyncio.run(
        scheduler.resume(
            paused_result,
            user_inputs={"training_step": {}},
            step_resume_decisions={
                "training_step": {
                    "step_id": "training_step",
                    "action": "degraded",
                    "provided_inputs": {},
                    "ignored_input_ids": ["training_goal"],
                    "waiting_input_ids": [],
                    "reason": "用户选择简化执行。",
                    "user_input": "简化执行",
                }
            },
        )
    )

    assert resumed_result.status == "running"
    assert len(training_worker.received_inputs) == 1
    assert len(health_worker.received_inputs) == 1
    training_input = training_worker.received_inputs[0]
    health_input = health_worker.received_inputs[0]
    assert training_input["multi_agent_skill_execution_mode"] == "degraded"
    assert training_input["multi_agent_skill_ignored_input_ids"] == [
        "training_goal"
    ]
    assert "multi_agent_skill_execution_mode" not in health_input


def test_scheduler_should_resume_without_repeating_completed_steps() -> None:
    """
    检查用户回答后是否只重跑等待步骤并继续执行后续步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    profile_calls: list[dict[str, object]] = []
    downstream_calls: list[tuple[str, list[str]]] = []

    async def resumable_profile_worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """第一次等待确认，第二次读取恢复输入后完成步骤。"""

        _ = dependency_results
        profile_calls.append(dict(step.input_data))
        if not step.input_data.get("multi_agent_is_resuming"):
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                output={"pending_action": "读取宠物档案"},
                requires_user_input=True,
                clarification_prompt="是否允许读取宠物档案？",
            )
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary="用户已授权，资料读取完成。",
        )

    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": resumable_profile_worker,
            "health_agent": build_success_worker(downstream_calls),
        }
    )
    plan = AgentTaskPlan(
        plan_id="worker_resume_plan",
        objective="读取资料后查询健康知识",
        steps=[
            AgentTaskStep(
                step_id="load_profile",
                title="读取资料",
                assigned_agent="profile_agent",
            ),
            AgentTaskStep(
                step_id="query_health",
                title="查询健康知识",
                assigned_agent="health_agent",
                depends_on=["load_profile"],
            ),
        ],
    )

    paused_result = asyncio.run(scheduler.execute(plan))
    resumed_result = asyncio.run(
        scheduler.resume(
            paused_result,
            user_inputs={"load_profile": "允许读取"},
        )
    )

    assert resumed_result.status == "running"
    assert resumed_result.plan.status == "completed"
    assert len(profile_calls) == 2
    assert profile_calls[1]["multi_agent_resume_input"] == "允许读取"
    assert profile_calls[1]["multi_agent_previous_worker_output"] == {
        "pending_action": "读取宠物档案"
    }
    assert downstream_calls == [("query_health", ["load_profile"])]
    assert resumed_result.metadata["resume_count"] == 1


def test_scheduler_resume_should_require_all_waiting_inputs() -> None:
    """
    检查恢复时遗漏等待步骤回答是否被明确拒绝。

    参数含义：
        无。

    返回值含义：
        None。
    """

    waiting_result = AgentTaskResult(
        step_id="load_profile",
        assigned_agent="profile_agent",
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="是否允许读取宠物档案？",
    )
    plan_data = build_scheduler_plan().model_dump(mode="python")
    plan_data["status"] = "awaiting_input"
    plan_data["requires_user_input"] = True
    plan_data["clarification_prompt"] = "是否允许读取宠物档案？"
    plan_data["steps"][0]["status"] = "awaiting_input"
    paused_result = MultiAgentTaskResult(
        collaboration_id="resume_missing_input_task",
        plan=AgentTaskPlan.model_validate(plan_data),
        status="awaiting_input",
        task_results=[waiting_result],
    )
    scheduler = MultiAgentTaskScheduler(
        workers={"profile_agent": build_success_worker([])}
    )

    with pytest.raises(ValueError, match="仍缺少等待步骤"):
        asyncio.run(scheduler.resume(paused_result, user_inputs={}))
