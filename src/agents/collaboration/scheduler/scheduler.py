"""
多 Agent 任务调度器。

功能：
    根据 AgentTaskPlan 中的 depends_on 查找当前可执行步骤，把同一批步骤
    并发交给已注册 Worker，并将执行结果整理成 MultiAgentTaskResult。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from threading import Lock
from typing import Any
from uuid import uuid4

from src.agents.collaboration.contracts import (
    AgentTaskPlan,
    AgentTaskPlanStatus,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)
from src.logger import logger


# WorkerHandler 不是一个真正的 Worker，而是对“Worker 函数长什么样”的说明：
# 第一个参数是当前要执行的完整步骤，第二个参数是它依赖的步骤结果；
# 返回值可以是同步结果，也可以是需要 await 的异步结果。
WorkerHandler = Callable[
    [AgentTaskStep, Mapping[str, AgentTaskResult]],
    AgentTaskResult | Awaitable[AgentTaskResult],
]


def build_multi_agent_task_id(trace_id: str | None = None) -> str:
    """
    根据链路追踪编号构建整次多 Agent 任务编号。

    功能：
        有 trace_id 时生成可由 UI 和主图共同推导的稳定编号；没有时使用
        随机编号，避免不同任务互相覆盖。

    参数含义：
        trace_id:
            当前请求的链路追踪编号。

    返回值含义：
        str:
            以 multi_agent_task_ 开头的整次任务编号。
    """

    normalized_trace_id = str(trace_id or "").strip()
    task_suffix = normalized_trace_id or uuid4().hex
    return f"multi_agent_task_{task_suffix}"


class _WorkerStepTimeoutError(TimeoutError):
    """
    表示调度器等待异步 Worker 时超过了单步骤时间限制。

    参数含义：
        step_id:
            发生超时的任务步骤编号。
        timeout_seconds:
            当前调度器配置的单步骤超时秒数。

    返回值含义：
        _WorkerStepTimeoutError:
            供调度器内部区分超时和普通 Worker 异常的异常对象。
    """

    def __init__(self, step_id: str, timeout_seconds: float) -> None:
        self.step_id = step_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"步骤 {step_id} 执行超过 {timeout_seconds:g} 秒"
        )


class _WorkerStepCancelledError(RuntimeError):
    """表示异步 Worker 因整次多 Agent 任务取消而停止执行。"""


class MultiAgentTaskCancellationToken:
    """
    保存一次多 Agent 任务的协作式取消信号。

    功能：
        调用方通过 cancel 发出取消请求；Scheduler 和正在等待的异步 Worker
        通过 is_cancelled 或 wait 感知请求。令牌只管理信号，不保存任务结果。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskCancellationToken:
            可以在 Scheduler 与外部控制端之间共享的取消令牌。
    """

    def __init__(self) -> None:
        self._cancelled_event = asyncio.Event()
        self._cancelled = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._lock = Lock()

    @property
    def is_cancelled(self) -> bool:
        """
        返回当前任务是否已经收到取消请求。

        返回值含义：
            bool:
                cancel 已被调用时返回 True，否则返回 False。
        """

        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        """
        发出取消请求并唤醒正在等待该信号的 Scheduler。

        返回值含义：
            None。
        """

        with self._lock:
            self._cancelled = True
            event_loop = self._event_loop

        # UI 和 Worker 可能由不同线程中的事件循环驱动。已经有等待者时，
        # 必须通过目标事件循环的线程安全入口唤醒它。
        if event_loop is not None and event_loop.is_running():
            event_loop.call_soon_threadsafe(self._cancelled_event.set)
        else:
            self._cancelled_event.set()

    async def wait(self) -> None:
        """
        异步等待取消请求到达。

        返回值含义：
            None:
                cancel 被调用后结束等待。
        """

        event_loop = asyncio.get_running_loop()
        with self._lock:
            self._event_loop = event_loop
            already_cancelled = self._cancelled
        if already_cancelled:
            self._cancelled_event.set()
        await self._cancelled_event.wait()


class MultiAgentTaskCancellationRegistry:
    """
    保存当前进程中正在运行的多 Agent 任务取消令牌。

    功能：
        使用 multi_agent_task_id 注册、查找和移除取消令牌，让 UI、API
        等外部入口能够找到正在运行的任务并发出取消请求。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskCancellationRegistry:
            可以由 GraphRuntimeService 长期持有的运行中任务登记表。
    """

    def __init__(self) -> None:
        self._tokens: dict[str, MultiAgentTaskCancellationToken] = {}
        self._lock = Lock()

    def register(
        self,
        multi_agent_task_id: str,
    ) -> MultiAgentTaskCancellationToken:
        """
        为一份正在启动的任务创建并登记取消令牌。

        参数含义：
            multi_agent_task_id:
                整次多 Agent 任务的唯一编号。

        返回值含义：
            MultiAgentTaskCancellationToken:
                Scheduler 与外部取消入口共享的取消令牌。
        """

        normalized_task_id = str(multi_agent_task_id or "").strip()
        if not normalized_task_id:
            raise ValueError("multi_agent_task_id 不能为空")
        with self._lock:
            if normalized_task_id in self._tokens:
                raise ValueError(
                    "多 Agent 任务已经在运行: "
                    f"{normalized_task_id}"
                )
            token = MultiAgentTaskCancellationToken()
            self._tokens[normalized_task_id] = token
        return token

    def cancel(self, multi_agent_task_id: str) -> bool:
        """
        向指定的运行中任务发出取消请求。

        参数含义：
            multi_agent_task_id:
                需要取消的整次多 Agent 任务编号。

        返回值含义：
            bool:
                找到运行中任务并发出信号时返回 True，否则返回 False。
        """

        normalized_task_id = str(multi_agent_task_id or "").strip()
        with self._lock:
            token = self._tokens.get(normalized_task_id)
        if token is None:
            return False
        token.cancel()
        return True

    def unregister(
        self,
        multi_agent_task_id: str,
        token: MultiAgentTaskCancellationToken,
    ) -> bool:
        """
        移除已经结束的任务令牌。

        参数含义：
            multi_agent_task_id:
                已经结束的整次多 Agent 任务编号。
            token:
                启动该任务时得到的令牌，用于避免误删同编号的新任务。

        返回值含义：
            bool:
                成功移除当前令牌时返回 True，否则返回 False。
        """

        normalized_task_id = str(multi_agent_task_id or "").strip()
        with self._lock:
            if self._tokens.get(normalized_task_id) is not token:
                return False
            del self._tokens[normalized_task_id]
        return True

    def contains(self, multi_agent_task_id: str) -> bool:
        """
        检查指定任务当前是否仍登记为运行中。

        参数含义：
            multi_agent_task_id:
                需要检查的整次多 Agent 任务编号。

        返回值含义：
            bool:
                登记表中存在该任务时返回 True，否则返回 False。
        """

        normalized_task_id = str(multi_agent_task_id or "").strip()
        with self._lock:
            return normalized_task_id in self._tokens


class MultiAgentTaskScheduler:
    """
    按任务依赖关系调用 Worker Agent。

    功能：
        每轮寻找依赖结果已经齐全的步骤，并发调用对应 Worker。Worker 必须
        返回 AgentTaskResult；调用异常会在限制次数内重试，最终仍失败时
        转换成失败结果，不会让调度器直接崩溃。

    参数含义：
        workers:
            Agent 名称到 Worker 调用函数的映射。名称必须与步骤中的
            assigned_agent 一致。
        maximum_parallel_steps:
            同一批最多并发执行多少个步骤，避免一次启动过多外部调用。
        step_timeout_seconds:
            单个异步 Worker 最长允许执行多少秒；None 表示不启用超时限制。
        maximum_step_attempts:
            单个 Worker 最多尝试执行多少次，包含第一次执行；默认 1 表示
            不重试。

    返回值含义：
        MultiAgentTaskScheduler:
            可以通过 execute 异步执行一份 AgentTaskPlan 的调度器。
    """

    def __init__(
        self,
        *,
        workers: Mapping[str, WorkerHandler],
        maximum_parallel_steps: int = 4,
        step_timeout_seconds: float | None = None,
        maximum_step_attempts: int = 1,
    ) -> None:
        if not workers:
            raise ValueError("任务调度器必须至少注册一个 Worker")
        if maximum_parallel_steps < 1:
            raise ValueError("maximum_parallel_steps 必须大于 0")
        if step_timeout_seconds is not None and step_timeout_seconds <= 0:
            raise ValueError("step_timeout_seconds 必须大于 0 或为 None")
        if maximum_step_attempts < 1:
            raise ValueError("maximum_step_attempts 必须大于 0")

        self.workers = {
            str(name).strip(): worker
            for name, worker in workers.items()
            if str(name).strip()
        }
        if not self.workers:
            raise ValueError("Worker 名称不能为空")
        if not all(callable(worker) for worker in self.workers.values()):
            raise ValueError("每个 Worker 都必须是可调用对象")
        self.maximum_parallel_steps = maximum_parallel_steps
        self.step_timeout_seconds = (
            float(step_timeout_seconds)
            if step_timeout_seconds is not None
            else None
        )
        self.maximum_step_attempts = maximum_step_attempts

    async def execute(
        self,
        plan: AgentTaskPlan,
        *,
        collaboration_id: str | None = None,
        cancellation_token: MultiAgentTaskCancellationToken | None = None,
    ) -> MultiAgentTaskResult:
        """
        按依赖关系执行一份多 Agent 任务计划。

        功能：
            逐批选择可执行步骤，并将同一批步骤并发交给 Worker。失败步骤会
            根据 allow_failure 决定后续步骤继续执行还是标记为 skipped。

        参数含义：
            plan:
                PlannerAgent 生成且已经通过依赖校验的任务计划。
            collaboration_id:
                可选的多 Agent 任务编号；未提供时由调度器生成。
            cancellation_token:
                可选任务取消令牌；收到取消请求后停止未完成步骤。

        返回值含义：
            MultiAgentTaskResult:
                包含更新后计划、全部步骤结果和调度状态的结构化结果。
                本阶段不生成 final_answer，成功执行后等待结果聚合器继续处理。
        """

        resolved_id = str(
            collaboration_id or f"multi_agent_task_{uuid4().hex}"
        ).strip()
        if not resolved_id:
            raise ValueError("collaboration_id 不能为空")

        _log_scheduler_event(
            level="info",
            event="multi_agent_task_started",
            payload={
                "multi_agent_task_id": resolved_id,
                "plan_id": plan.plan_id,
                "plan_status": plan.status,
                "step_count": len(plan.steps),
                "step_timeout_seconds": self.step_timeout_seconds,
                "maximum_step_attempts": self.maximum_step_attempts,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "title": step.title,
                        "assigned_agent": step.assigned_agent,
                    }
                    for step in plan.steps
                ],
            },
        )

        if cancellation_token is not None and cancellation_token.is_cancelled:
            return _build_cancelled_task_result(
                plan=plan,
                multi_agent_task_id=resolved_id,
                results_by_step_id={},
                ready_batches=[],
                base_metadata={
                    "step_timeout_seconds": self.step_timeout_seconds,
                    "maximum_step_attempts": self.maximum_step_attempts,
                },
            )

        if plan.status == "awaiting_input" or plan.requires_user_input:
            awaiting_plan = _build_updated_plan(
                plan=plan,
                results_by_step_id={},
                status="awaiting_input",
            )
            waiting_result = MultiAgentTaskResult(
                collaboration_id=resolved_id,
                plan=awaiting_plan,
                status="awaiting_input",
                metadata={
                    "scheduler": type(self).__name__,
                    "ready_batches": [],
                    "step_timeout_seconds": self.step_timeout_seconds,
                    "maximum_step_attempts": self.maximum_step_attempts,
                },
            )
            _log_scheduler_event(
                level="info",
                event="multi_agent_task_waiting_for_user",
                payload={
                    "multi_agent_task_id": resolved_id,
                    "plan_id": plan.plan_id,
                    "clarification_prompt": plan.clarification_prompt,
                },
            )
            return waiting_result

        if plan.status != "planned":
            _log_scheduler_event(
                level="warning",
                event="multi_agent_task_rejected",
                payload={
                    "multi_agent_task_id": resolved_id,
                    "plan_id": plan.plan_id,
                    "reason": "plan_status_is_not_planned",
                    "actual_status": plan.status,
                },
            )
            raise ValueError(
                "任务调度器只接受 status=planned 的新计划，"
                f"实际为 {plan.status}"
            )
        non_pending_step_ids = [
            step.step_id
            for step in plan.steps
            if step.status != "pending"
        ]
        if non_pending_step_ids:
            _log_scheduler_event(
                level="warning",
                event="multi_agent_task_rejected",
                payload={
                    "multi_agent_task_id": resolved_id,
                    "plan_id": plan.plan_id,
                    "reason": "steps_are_not_pending",
                    "step_ids": non_pending_step_ids,
                },
            )
            raise ValueError(
                "新计划中的步骤必须全部是 pending: "
                f"{non_pending_step_ids}"
            )

        return await self._execute_remaining_steps(
            plan=plan,
            multi_agent_task_id=resolved_id,
            results_by_step_id={},
            ready_batches=[],
            base_metadata={
                "step_timeout_seconds": self.step_timeout_seconds,
                "maximum_step_attempts": self.maximum_step_attempts,
            },
            cancellation_token=cancellation_token,
        )

    async def resume(
        self,
        task_result: MultiAgentTaskResult,
        *,
        user_inputs: Mapping[str, Any],
        cancellation_token: MultiAgentTaskCancellationToken | None = None,
    ) -> MultiAgentTaskResult:
        """
        根据用户补充内容恢复一份暂停的多 Agent 任务。

        功能：
            校验任务确实处于 awaiting_input，为每个等待步骤绑定对应用户
            输入，保留已经完成的结果，并从等待步骤继续执行剩余计划。

        参数含义：
            task_result:
                Scheduler 之前返回的 awaiting_input 暂停结果。
            user_inputs:
                等待步骤编号到用户回答的映射。键必须覆盖全部等待步骤。
            cancellation_token:
                可选任务取消令牌；收到取消请求后停止未完成步骤。

        返回值含义：
            MultiAgentTaskResult:
                继续执行后的最新结果；如果 Worker 再次追问，仍可能返回
                awaiting_input，否则返回等待结果聚合或失败的状态。
        """

        awaiting_results = _validate_resume_request(
            task_result=task_result,
            user_inputs=user_inputs,
        )
        resumable_plan = _build_resumable_plan(
            task_result=task_result,
            user_inputs=user_inputs,
            awaiting_results=awaiting_results,
        )
        results_by_step_id = {
            result.step_id: result
            for result in task_result.task_results
            if result.status != "awaiting_input"
        }
        ready_batches = [
            list(batch)
            for batch in task_result.metadata.get("ready_batches", [])
            if isinstance(batch, list)
        ]

        _log_scheduler_event(
            level="info",
            event="multi_agent_task_resumed",
            payload={
                "multi_agent_task_id": task_result.collaboration_id,
                "plan_id": task_result.plan.plan_id,
                "resumed_step_ids": list(awaiting_results),
            },
        )
        return await self._execute_remaining_steps(
            plan=resumable_plan,
            multi_agent_task_id=task_result.collaboration_id,
            results_by_step_id=results_by_step_id,
            ready_batches=ready_batches,
            base_metadata={
                **task_result.metadata,
                "step_timeout_seconds": self.step_timeout_seconds,
                "maximum_step_attempts": self.maximum_step_attempts,
                "resume_count": int(
                    task_result.metadata.get("resume_count", 0)
                ) + 1,
            },
            cancellation_token=cancellation_token,
        )

    async def _execute_remaining_steps(
        self,
        *,
        plan: AgentTaskPlan,
        multi_agent_task_id: str,
        results_by_step_id: dict[str, AgentTaskResult],
        ready_batches: list[list[str]],
        base_metadata: Mapping[str, Any],
        cancellation_token: MultiAgentTaskCancellationToken | None,
    ) -> MultiAgentTaskResult:
        """
        执行一份计划中尚未产生结果的步骤。

        功能：
            同时服务首次执行和恢复执行。已有结果不会再次调用 Worker，
            只把没有结果的步骤放入待执行集合并继续依赖调度。

        参数含义：
            plan:
                当前需要继续执行的计划。
            multi_agent_task_id:
                整次多 Agent 任务编号。
            results_by_step_id:
                已经得到且需要保留的步骤结果。
            ready_batches:
                之前已经启动过的执行批次，会继续追加新批次。
            base_metadata:
                首次执行或暂停结果中需要继续保留的任务扩展信息。
            cancellation_token:
                可选任务取消令牌，用于执行前和执行中停止剩余步骤。

        返回值含义：
            MultiAgentTaskResult:
                剩余步骤执行后的最新多 Agent 任务结果。
        """

        steps_by_id = {
            step.step_id: step
            for step in plan.steps
        }
        pending_step_ids = {
            step.step_id
            for step in plan.steps
            if step.step_id not in results_by_step_id
        }

        while pending_step_ids:
            if (
                cancellation_token is not None
                and cancellation_token.is_cancelled
            ):
                return _build_cancelled_task_result(
                    plan=plan,
                    multi_agent_task_id=multi_agent_task_id,
                    results_by_step_id=results_by_step_id,
                    ready_batches=ready_batches,
                    base_metadata=base_metadata,
                )

            skipped_step_ids = _skip_steps_with_blocking_dependencies(
                plan=plan,
                multi_agent_task_id=multi_agent_task_id,
                pending_step_ids=pending_step_ids,
                steps_by_id=steps_by_id,
                results_by_step_id=results_by_step_id,
            )
            if skipped_step_ids:
                continue

            ready_steps = [
                step
                for step in plan.steps
                if (
                    step.step_id in pending_step_ids
                    and all(
                        dependency_id in results_by_step_id
                        for dependency_id in step.depends_on
                    )
                )
            ]
            if not ready_steps:
                raise RuntimeError("没有可执行步骤，任务计划无法继续")

            current_batch = ready_steps[: self.maximum_parallel_steps]
            current_batch_ids = [
                step.step_id
                for step in current_batch
            ]
            ready_batches.append(current_batch_ids)
            _log_scheduler_event(
                level="info",
                event="multi_agent_batch_started",
                payload={
                    "multi_agent_task_id": multi_agent_task_id,
                    "plan_id": plan.plan_id,
                    "batch_number": len(ready_batches),
                    "step_ids": current_batch_ids,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "title": step.title,
                            "assigned_agent": step.assigned_agent,
                        }
                        for step in current_batch
                    ],
                },
            )
            batch_results = await asyncio.gather(
                *[
                    self._execute_step(
                        step=step,
                        multi_agent_task_id=multi_agent_task_id,
                        cancellation_token=cancellation_token,
                        dependency_results={
                            dependency_id: results_by_step_id[dependency_id]
                            for dependency_id in step.depends_on
                        },
                    )
                    for step in current_batch
                ]
            )
            for result in batch_results:
                results_by_step_id[result.step_id] = result
                pending_step_ids.remove(result.step_id)
                completed_step = steps_by_id[result.step_id]
                _log_scheduler_event(
                    level=(
                        "warning"
                        if result.status != "completed"
                        else "info"
                    ),
                    event="multi_agent_step_finished",
                    payload={
                        "multi_agent_task_id": multi_agent_task_id,
                        "plan_id": plan.plan_id,
                        "step_id": result.step_id,
                        "step_title": completed_step.title,
                        "assigned_agent": result.assigned_agent,
                        "status": result.status,
                        "latency_ms": result.latency_ms,
                    },
                )

            if (
                cancellation_token is not None
                and cancellation_token.is_cancelled
            ):
                return _build_cancelled_task_result(
                    plan=plan,
                    multi_agent_task_id=multi_agent_task_id,
                    results_by_step_id=results_by_step_id,
                    ready_batches=ready_batches,
                    base_metadata=base_metadata,
                )

            awaiting_results = [
                result
                for result in batch_results
                if result.status == "awaiting_input"
            ]
            if awaiting_results:
                return _build_awaiting_input_result(
                    plan=plan,
                    multi_agent_task_id=multi_agent_task_id,
                    results_by_step_id=results_by_step_id,
                    ready_batches=ready_batches,
                    awaiting_results=awaiting_results,
                    base_metadata=base_metadata,
                )

        ordered_results = [
            results_by_step_id[step.step_id]
            for step in plan.steps
        ]
        blocking_failure_ids = [
            result.step_id
            for result in ordered_results
            if (
                result.status in {"failed", "skipped"}
                and not steps_by_id[result.step_id].allow_failure
            )
        ]
        allowed_failure_ids = [
            result.step_id
            for result in ordered_results
            if (
                result.status in {"failed", "skipped"}
                and steps_by_id[result.step_id].allow_failure
            )
        ]

        if blocking_failure_ids:
            plan_status = "failed"
            task_status = "failed"
            error_message = (
                "以下步骤失败或被跳过，任务无法完整完成: "
                f"{blocking_failure_ids}"
            )
        elif allowed_failure_ids:
            plan_status = "partial"
            task_status = "partial"
            error_message = None
        else:
            plan_status = "completed"
            task_status = "running"
            error_message = None

        updated_plan = _build_updated_plan(
            plan=plan,
            results_by_step_id=results_by_step_id,
            status=plan_status,
        )
        final_result = MultiAgentTaskResult(
            collaboration_id=multi_agent_task_id,
            plan=updated_plan,
            status=task_status,
            task_results=ordered_results,
            error_message=error_message,
            metadata={
                **base_metadata,
                "scheduler": type(self).__name__,
                "ready_batches": ready_batches,
                "awaiting_result_aggregation": task_status == "running",
            },
        )
        final_log_level = (
            "error"
            if task_status == "failed"
            else "warning"
            if task_status == "partial"
            else "info"
        )
        _log_scheduler_event(
            level=final_log_level,
            event="multi_agent_scheduling_finished",
            payload={
                "multi_agent_task_id": multi_agent_task_id,
                "plan_id": plan.plan_id,
                "plan_status": plan_status,
                "task_status": task_status,
                "failed_or_skipped_step_ids": (
                    blocking_failure_ids + allowed_failure_ids
                ),
                "ready_batches": ready_batches,
            },
        )
        return final_result

    async def _execute_step(
        self,
        *,
        step: AgentTaskStep,
        multi_agent_task_id: str,
        cancellation_token: MultiAgentTaskCancellationToken | None,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        调用一个步骤指定的 Worker，有限重试后把异常转换成失败结果。

        参数含义：
            step:
                当前需要执行的完整任务步骤。
            multi_agent_task_id:
                当前步骤所属的整次多 Agent 任务编号，用于关联日志。
            cancellation_token:
                可选任务取消令牌，用于终止正在等待的异步 Worker。
            dependency_results:
                当前步骤依赖的前置步骤结果，键为前置 step_id。

        返回值含义：
            AgentTaskResult:
                Worker 的合法结果，或由调度器生成的失败结果。
        """

        started_at = time.perf_counter()
        attempt_count = 0
        try:
            worker = self.workers.get(step.assigned_agent)
            if worker is None:
                raise ValueError(
                    f"没有注册 Worker: {step.assigned_agent}"
                )

            # maximum_step_attempts 包含第一次执行。例如值为 3，表示首次
            # 执行失败后最多再重试两次。
            for attempt_number in range(1, self.maximum_step_attempts + 1):
                if (
                    cancellation_token is not None
                    and cancellation_token.is_cancelled
                ):
                    raise _WorkerStepCancelledError(
                        "多 Agent 任务收到取消请求"
                    )
                attempt_count = attempt_number
                try:
                    raw_result = worker(step, dependency_results)
                    if inspect.isawaitable(raw_result):
                        # 超时只包住异步 Worker；同步函数在返回前会占用当前
                        # 线程，无法由 asyncio 安全地中断。
                        if self.step_timeout_seconds is None:
                            raw_result = await self._await_worker_result(
                                raw_result=raw_result,
                                cancellation_token=cancellation_token,
                            )
                        else:
                            try:
                                raw_result = await self._await_worker_result(
                                    raw_result=raw_result,
                                    cancellation_token=cancellation_token,
                                )
                            except TimeoutError as exc:
                                raise _WorkerStepTimeoutError(
                                    step.step_id,
                                    self.step_timeout_seconds,
                                ) from exc
                    break
                except _WorkerStepCancelledError:
                    raise
                except Exception as exc:
                    if attempt_number >= self.maximum_step_attempts:
                        raise
                    _log_scheduler_event(
                        level="warning",
                        event="multi_agent_step_retrying",
                        payload={
                            "multi_agent_task_id": multi_agent_task_id,
                            "step_id": step.step_id,
                            "step_title": step.title,
                            "assigned_agent": step.assigned_agent,
                            "failed_attempt_number": attempt_number,
                            "next_attempt_number": attempt_number + 1,
                            "maximum_step_attempts": (
                                self.maximum_step_attempts
                            ),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )

            # Worker 已经正常返回后再检查契约；类型或编号错误属于代码问题，
            # 重复调用通常不会修好，因此不进入上面的重试循环。
            if not isinstance(raw_result, AgentTaskResult):
                raise TypeError("Worker 必须返回 AgentTaskResult")
            if raw_result.step_id != step.step_id:
                raise ValueError("Worker 返回了错误的 step_id")
            if raw_result.assigned_agent != step.assigned_agent:
                raise ValueError("Worker 返回了错误的 assigned_agent")

            result_data = raw_result.model_dump(mode="python")
            if result_data["latency_ms"] is None:
                result_data["latency_ms"] = _elapsed_ms(started_at)
            result_data["metadata"] = {
                **result_data["metadata"],
                "scheduler_attempt_count": attempt_count,
            }
            return AgentTaskResult.model_validate(result_data)
        except _WorkerStepCancelledError:
            _log_scheduler_event(
                level="info",
                event="multi_agent_step_cancelled",
                payload={
                    "multi_agent_task_id": multi_agent_task_id,
                    "step_id": step.step_id,
                    "step_title": step.title,
                    "assigned_agent": step.assigned_agent,
                    "attempt_count": attempt_count,
                },
            )
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="skipped",
                summary="多 Agent 任务已取消，当前步骤停止执行。",
                latency_ms=_elapsed_ms(started_at),
                metadata={
                    "cancelled": True,
                    "scheduler_attempt_count": attempt_count,
                },
            )
        except _WorkerStepTimeoutError as exc:
            timeout_seconds = exc.timeout_seconds
            error_message = (
                f"步骤 {step.step_id} 执行超过 "
                f"{timeout_seconds:g} 秒，已由调度器终止等待"
            )
            _log_scheduler_event(
                level="warning",
                event="multi_agent_step_timed_out",
                payload={
                    "multi_agent_task_id": multi_agent_task_id,
                    "step_id": step.step_id,
                    "step_title": step.title,
                    "assigned_agent": step.assigned_agent,
                    "timeout_seconds": timeout_seconds,
                    "attempt_count": attempt_count,
                },
            )
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="failed",
                error_message=error_message,
                latency_ms=_elapsed_ms(started_at),
                metadata={
                    "scheduler_generated_failure": True,
                    "timed_out": True,
                    "timeout_seconds": timeout_seconds,
                    "scheduler_attempt_count": attempt_count,
                },
            )
        except Exception as exc:
            _log_scheduler_event(
                level="warning",
                event="multi_agent_step_failed",
                payload={
                    "multi_agent_task_id": multi_agent_task_id,
                    "step_id": step.step_id,
                    "step_title": step.title,
                    "assigned_agent": step.assigned_agent,
                    "attempt_count": attempt_count,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="failed",
                error_message=str(exc),
                latency_ms=_elapsed_ms(started_at),
                metadata={
                    "scheduler_generated_failure": True,
                    "scheduler_attempt_count": attempt_count,
                },
            )

    async def _await_worker_result(
        self,
        *,
        raw_result: Awaitable[AgentTaskResult],
        cancellation_token: MultiAgentTaskCancellationToken | None,
    ) -> AgentTaskResult:
        """
        同时等待异步 Worker 完成、任务取消或步骤超时。

        功能：
            Worker 与取消信号竞速；取消先到时撤销 Worker，Worker 先完成时
            返回结果，两者都未完成且超过限制时抛出 TimeoutError。

        参数含义：
            raw_result:
                Worker 返回的异步结果。
            cancellation_token:
                可选任务取消令牌。

        返回值含义：
            AgentTaskResult:
                Worker 正常完成后返回的原始步骤结果。
        """

        if cancellation_token is None:
            if self.step_timeout_seconds is None:
                return await raw_result
            return await asyncio.wait_for(
                raw_result,
                timeout=self.step_timeout_seconds,
            )

        worker_future = asyncio.ensure_future(raw_result)
        cancellation_future = asyncio.create_task(
            cancellation_token.wait()
        )
        watched_futures = [worker_future, cancellation_future]
        try:
            done, _ = await asyncio.wait(
                watched_futures,
                timeout=self.step_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_future in done:
                raise _WorkerStepCancelledError(
                    "多 Agent 任务收到取消请求"
                )
            if worker_future in done:
                return worker_future.result()
            raise TimeoutError("异步 Worker 执行超时")
        finally:
            unfinished_futures = [
                future
                for future in watched_futures
                if not future.done()
            ]
            for future in unfinished_futures:
                future.cancel()
            # 同时回收已完成和被取消的 Future，避免竞速时遗留
            # “Task exception was never retrieved” 警告。
            await asyncio.gather(
                *watched_futures,
                return_exceptions=True,
            )


def _build_cancelled_task_result(
    *,
    plan: AgentTaskPlan,
    multi_agent_task_id: str,
    results_by_step_id: Mapping[str, AgentTaskResult],
    ready_batches: list[list[str]],
    base_metadata: Mapping[str, Any],
) -> MultiAgentTaskResult:
    """
    构建整次多 Agent 任务取消后的标准结果。

    功能：
        保留取消前已经进入终态的步骤结果，把正在等待输入和其余未执行
        步骤统一标记为 skipped，并把计划及整次任务状态更新为 cancelled。

    参数含义：
        plan:
            当前正在执行的任务计划。
        multi_agent_task_id:
            整次多 Agent 任务编号。
        results_by_step_id:
            取消发生前已经产生的步骤结果。
        ready_batches:
            取消前实际启动过的步骤批次。
        base_metadata:
            需要继续保留的任务扩展信息。

    返回值含义：
        MultiAgentTaskResult:
            状态为 cancelled 且覆盖全部计划步骤的取消结果。
    """

    cancelled_results = {
        step_id: result
        for step_id, result in results_by_step_id.items()
        if result.status != "awaiting_input"
    }
    for step in plan.steps:
        if step.step_id in cancelled_results:
            continue
        cancelled_results[step.step_id] = AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="skipped",
            summary="多 Agent 任务已取消，当前步骤未执行。",
            metadata={
                "cancelled": True,
            },
        )

    updated_plan = _build_updated_plan(
        plan=plan,
        results_by_step_id=cancelled_results,
        status="cancelled",
    )
    ordered_results = [
        cancelled_results[step.step_id]
        for step in plan.steps
    ]
    _log_scheduler_event(
        level="info",
        event="multi_agent_task_cancelled",
        payload={
            "multi_agent_task_id": multi_agent_task_id,
            "plan_id": plan.plan_id,
            "completed_step_ids": [
                result.step_id
                for result in ordered_results
                if result.status == "completed"
            ],
            "skipped_step_ids": [
                result.step_id
                for result in ordered_results
                if result.status == "skipped"
            ],
        },
    )
    return MultiAgentTaskResult(
        collaboration_id=multi_agent_task_id,
        plan=updated_plan,
        status="cancelled",
        task_results=ordered_results,
        final_answer="多 Agent 任务已取消。",
        metadata={
            **base_metadata,
            "scheduler": "MultiAgentTaskScheduler",
            "ready_batches": ready_batches,
            "cancellation_requested": True,
            "awaiting_result_aggregation": False,
        },
    )


def _skip_steps_with_blocking_dependencies(
    *,
    plan: AgentTaskPlan,
    multi_agent_task_id: str,
    pending_step_ids: set[str],
    steps_by_id: Mapping[str, AgentTaskStep],
    results_by_step_id: dict[str, AgentTaskResult],
) -> list[str]:
    """
    把被非允许失败步骤阻断的后续步骤标记为 skipped。

    功能：
        如果某个前置步骤已经失败或跳过，并且该前置步骤不允许失败，当前
        步骤就不能获得可靠输入，因此不再调用 Worker，直接生成跳过结果。

    参数含义：
        plan:
            当前正在执行的完整任务计划。
        multi_agent_task_id:
            当前步骤所属的整次多 Agent 任务编号，用于关联日志。
        pending_step_ids:
            尚未产生结果的步骤编号集合，会移除本轮跳过的步骤。
        steps_by_id:
            step_id 到完整步骤对象的索引。
        results_by_step_id:
            已经得到的步骤结果，会写入本轮生成的跳过结果。

    返回值含义：
        list[str]:
            本轮被标记为 skipped 的步骤编号。
    """

    skipped_step_ids: list[str] = []
    for step in plan.steps:
        if step.step_id not in pending_step_ids:
            continue
        blocking_dependency_ids = [
            dependency_id
            for dependency_id in step.depends_on
            if (
                dependency_id in results_by_step_id
                and results_by_step_id[dependency_id].status
                in {"failed", "skipped"}
                and not steps_by_id[dependency_id].allow_failure
            )
        ]
        if not blocking_dependency_ids:
            continue

        results_by_step_id[step.step_id] = AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="skipped",
            summary="前置步骤失败，当前步骤未执行。",
            metadata={
                "blocking_dependency_ids": blocking_dependency_ids,
            },
        )
        pending_step_ids.remove(step.step_id)
        skipped_step_ids.append(step.step_id)
        _log_scheduler_event(
            level="warning",
            event="multi_agent_step_skipped",
            payload={
                "multi_agent_task_id": multi_agent_task_id,
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "step_title": step.title,
                "assigned_agent": step.assigned_agent,
                "blocking_dependency_ids": blocking_dependency_ids,
            },
        )
    return skipped_step_ids


def _validate_resume_request(
    *,
    task_result: MultiAgentTaskResult,
    user_inputs: Mapping[str, Any],
) -> dict[str, AgentTaskResult]:
    """
    检查暂停结果和用户回答是否满足恢复执行条件。

    功能：
        只允许恢复 Worker 执行期间产生的 awaiting_input 结果，并要求用户
        回答完整覆盖全部等待步骤，避免把一个回答错误交给另一个 Agent。

    参数含义：
        task_result:
            准备恢复的多 Agent 暂停结果。
        user_inputs:
            等待步骤编号到用户回答的映射。

    返回值含义：
        dict[str, AgentTaskResult]:
            等待步骤编号到原等待结果的索引；校验失败时抛出 ValueError。
    """

    if task_result.status != "awaiting_input":
        raise ValueError(
            "只能恢复 status=awaiting_input 的多 Agent 任务，"
            f"实际为 {task_result.status}"
        )
    awaiting_results = {
        result.step_id: result
        for result in task_result.task_results
        if result.status == "awaiting_input"
    }
    if not awaiting_results:
        raise ValueError(
            "当前任务是 Planner 等待补充信息，尚未产生等待中的 Worker；"
            "请补充规划上下文后重新生成计划"
        )

    expected_step_ids = set(awaiting_results)
    provided_step_ids = set(user_inputs)
    missing_step_ids = sorted(expected_step_ids - provided_step_ids)
    unknown_step_ids = sorted(provided_step_ids - expected_step_ids)
    if missing_step_ids:
        raise ValueError(
            "恢复任务仍缺少等待步骤的用户回答: "
            f"{missing_step_ids}"
        )
    if unknown_step_ids:
        raise ValueError(
            "恢复任务包含并未等待输入的步骤编号: "
            f"{unknown_step_ids}"
        )
    empty_input_step_ids = sorted(
        step_id
        for step_id in expected_step_ids
        if not str(user_inputs[step_id] or "").strip()
    )
    if empty_input_step_ids:
        raise ValueError(
            "等待步骤的用户回答不能为空: "
            f"{empty_input_step_ids}"
        )
    return awaiting_results


def _build_resumable_plan(
    *,
    task_result: MultiAgentTaskResult,
    user_inputs: Mapping[str, Any],
    awaiting_results: Mapping[str, AgentTaskResult],
) -> AgentTaskPlan:
    """
    把暂停计划转换成可以继续调度的新计划对象。

    功能：
        清除整份计划的等待标记，把等待步骤重新设为 pending，并把用户回答
        和该 Worker 上一次输出写入 input_data；其他步骤及其状态保持不变。

    参数含义：
        task_result:
            Scheduler 之前返回的暂停结果。
        user_inputs:
            等待步骤编号到用户回答的映射。
        awaiting_results:
            等待步骤编号到上一次 Worker 结果的索引。

    返回值含义：
        AgentTaskPlan:
            已加入恢复上下文、可以继续执行剩余步骤的新计划。
    """

    plan_data = task_result.plan.model_dump(mode="python")
    plan_data["status"] = "running"
    plan_data["requires_user_input"] = False
    plan_data["clarification_prompt"] = ""

    for step_data in plan_data["steps"]:
        step_id = step_data["step_id"]
        if step_id not in awaiting_results:
            continue

        # 等待步骤需要重新进入 Worker；已经完成的其他步骤不会重新执行。
        step_data["status"] = "pending"
        step_data["input_data"] = {
            **step_data["input_data"],
            "multi_agent_resume_input": user_inputs[step_id],
            "multi_agent_previous_worker_output": (
                awaiting_results[step_id].output
            ),
            "multi_agent_is_resuming": True,
        }
    return AgentTaskPlan.model_validate(plan_data)


def _build_awaiting_input_result(
    *,
    plan: AgentTaskPlan,
    multi_agent_task_id: str,
    results_by_step_id: Mapping[str, AgentTaskResult],
    ready_batches: list[list[str]],
    awaiting_results: list[AgentTaskResult],
    base_metadata: Mapping[str, Any],
) -> MultiAgentTaskResult:
    """
    构建 Worker 执行中等待用户输入时的暂停结果。

    功能：
        保留已经完成和正在等待的步骤结果，把整份计划改成 awaiting_input，
        其余未执行步骤继续保持 pending，等待用户输入后由后续恢复流程处理。

    参数含义：
        plan:
            Scheduler 当前执行的原始任务计划。
        multi_agent_task_id:
            当前整次多 Agent 任务编号。
        results_by_step_id:
            暂停前已经产生的步骤结果。
        ready_batches:
            暂停前实际启动过的步骤批次。
        awaiting_results:
            本轮返回 awaiting_input 的 Worker 结果。
        base_metadata:
            恢复执行前已经存在且需要继续保留的任务扩展信息。

    返回值含义：
        MultiAgentTaskResult:
            状态为 awaiting_input、可以展示澄清提示的暂停任务结果。
    """

    first_awaiting_result = awaiting_results[0]
    updated_plan = _build_updated_plan(
        plan=plan,
        results_by_step_id=results_by_step_id,
        status="awaiting_input",
        clarification_prompt=first_awaiting_result.clarification_prompt,
    )
    ordered_results = [
        results_by_step_id[step.step_id]
        for step in plan.steps
        if step.step_id in results_by_step_id
    ]
    awaiting_step_ids = [
        result.step_id
        for result in awaiting_results
    ]
    _log_scheduler_event(
        level="info",
        event="multi_agent_task_waiting_for_worker_input",
        payload={
            "multi_agent_task_id": multi_agent_task_id,
            "plan_id": plan.plan_id,
            "awaiting_step_ids": awaiting_step_ids,
        },
    )
    return MultiAgentTaskResult(
        collaboration_id=multi_agent_task_id,
        plan=updated_plan,
        status="awaiting_input",
        task_results=ordered_results,
        metadata={
            **base_metadata,
            "scheduler": "MultiAgentTaskScheduler",
            "ready_batches": ready_batches,
            "awaiting_step_ids": awaiting_step_ids,
            "clarification_prompt": (
                first_awaiting_result.clarification_prompt
            ),
        },
    )


def _build_updated_plan(
    *,
    plan: AgentTaskPlan,
    results_by_step_id: Mapping[str, AgentTaskResult],
    status: AgentTaskPlanStatus,
    clarification_prompt: str = "",
) -> AgentTaskPlan:
    """
    根据调度结果创建带有最新步骤状态的新计划对象。

    功能：
        不直接修改 PlannerAgent 返回的原对象，而是复制普通字典、更新计划
        和步骤状态，再次通过 AgentTaskPlan 校验后返回。

    参数含义：
        plan:
            PlannerAgent 生成的原始计划。
        results_by_step_id:
            已经得到的步骤执行结果。
        status:
            调度完成后需要写入的计划状态。
        clarification_prompt:
            计划等待用户输入时需要展示的问题；其他状态保持为空。

    返回值含义：
        AgentTaskPlan:
            状态已经更新且重新通过 Pydantic 校验的新计划。
    """

    plan_data = plan.model_dump(mode="python")
    plan_data["status"] = status
    if status == "awaiting_input" and clarification_prompt:
        plan_data["requires_user_input"] = True
        plan_data["clarification_prompt"] = clarification_prompt
    elif status != "awaiting_input":
        # 任务完成、失败或取消后，不应继续携带上一轮等待用户的标记。
        plan_data["requires_user_input"] = False
        plan_data["clarification_prompt"] = ""
    for step_data in plan_data["steps"]:
        result = results_by_step_id.get(step_data["step_id"])
        if result is not None:
            step_data["status"] = result.status
    return AgentTaskPlan.model_validate(plan_data)


def _elapsed_ms(started_at: float) -> float:
    """
    计算步骤执行耗时。

    参数含义：
        started_at:
            time.perf_counter 返回的高精度开始时间。

    返回值含义：
        float:
            非负的毫秒耗时。
    """

    return max(0.0, (time.perf_counter() - started_at) * 1000)


def _log_scheduler_event(
    *,
    level: str,
    event: str,
    payload: Mapping[str, Any],
) -> None:
    """
    用缩进 JSON 输出一条简短的调度器关键事件日志。

    功能：
        给日志补充 event 字段，再使用四空格缩进输出。日志格式化失败时只
        记录一条简短提示，不能因为日志问题打断多 Agent 任务执行。

    参数含义：
        level:
            日志级别，只使用 info、warning 或 error。
        event:
            事件名称，例如任务开始、步骤失败或调度结束。
        payload:
            当前事件需要记录的少量关键字段，不应放入完整 Worker 输出。

    返回值含义：
        None:
            只负责输出日志，不返回业务数据。
    """

    try:
        formatted_payload = json.dumps(
            {
                "event": event,
                **dict(payload),
            },
            indent=4,
            ensure_ascii=False,
            default=str,
        )
        log_method = getattr(logger, level, logger.info)
        log_method(
            "MultiAgentTaskScheduler event:\n"
            f"{formatted_payload}"
        )
    except Exception as exc:
        logger.info(f"调度器日志构建失败: {exc}")
