"""复用现有多 Agent Worker 执行 LongTaskStep 的桥接适配器。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from src.agents.collaboration.contracts import (
    AgentTaskResult,
    AgentTaskStep,
)
from src.agents.collaboration.scheduler.scheduler import WorkerHandler
from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchStepResult,
    LongTaskStep,
    LongTaskWaitingReason,
)


class LongTaskStepExecutionAdapterError(ValueError):
    """表示 LongTask Step 无法安全交给现有多 Agent Worker 执行。"""


class LongTaskStepExecutorAdapter:
    """
    把 LongTaskStep 适配为现有协作 Worker 输入并转换执行结果。

    功能：
        根据 assigned_agent 选择已注册 Worker，重建已完成依赖的最小结果，
        调用现有 Worker 后转换为 RuntimeDriver 使用的批次步骤结果。

    参数含义：
        workers:
            Agent 名称到现有 WorkerHandler 的映射，例如
            dog_knowledge_agent 到 GraphAgentWorkerAdapter。

    返回值含义：
        LongTaskStepExecutorAdapter:
            符合 LongTaskStepExecutor 调用签名的异步适配器对象。
    """

    def __init__(self, workers: Mapping[str, WorkerHandler]) -> None:
        normalized_workers = {
            str(agent_name or "").strip(): worker
            for agent_name, worker in workers.items()
            if str(agent_name or "").strip()
        }
        if not normalized_workers:
            raise ValueError("LongTask Step Executor 至少需要一个 Worker")
        if any(not callable(worker) for worker in normalized_workers.values()):
            raise ValueError("LongTask Step Executor 的 Worker 必须可调用")
        self._workers = normalized_workers

    async def __call__(
        self,
        task: LongTask,
        step: LongTaskStep,
        claim_id: str,
    ) -> LongTaskBatchStepResult:
        """
        使用 assigned_agent 对应的现有 Worker 执行一个已领取 Step。

        参数含义：
            task:
                包含目标、拓扑和最新步骤状态的 LongTask 快照。
            step:
                当前 Worker 已经获得租约的 running 步骤。
            claim_id:
                本次 Step 执行权令牌，必须与步骤中的当前令牌一致。

        返回值含义：
            LongTaskBatchStepResult:
                保留结果状态、摘要、Artifact 引用和 claim_id 的统一事实。
        """

        _validate_claimed_step(
            task=task,
            step=step,
            claim_id=claim_id,
        )
        worker = self._workers.get(step.assigned_agent)
        if worker is None:
            raise LongTaskStepExecutionAdapterError(
                "LongTask Step 没有注册对应 Worker: "
                f"{step.assigned_agent}"
            )

        worker_step = _build_worker_step(task=task, step=step)
        dependency_results = _build_dependency_results(
            task=task,
            step=step,
        )
        raw_result = worker(worker_step, dependency_results)
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result
        if not isinstance(raw_result, AgentTaskResult):
            raise LongTaskStepExecutionAdapterError(
                "协作 Worker 必须返回 AgentTaskResult"
            )
        _validate_worker_result(step=step, result=raw_result)
        return _adapt_worker_result(
            result=raw_result,
            claim_id=claim_id,
        )


def _validate_claimed_step(
    *,
    task: LongTask,
    step: LongTaskStep,
    claim_id: str,
) -> None:
    """
    检查当前步骤确实属于任务并持有本次执行令牌。

    参数含义：
        task：当前权威任务快照。
        step：准备交给协作 Worker 的步骤。
        claim_id：Handler 本次领取得到的执行令牌。

    返回值含义：
        None：归属、状态和领取令牌均正确；否则抛出适配异常。
    """

    if step.task_id != task.task_id:
        raise LongTaskStepExecutionAdapterError("Step 不属于当前 LongTask")
    if step.status != "running":
        raise LongTaskStepExecutionAdapterError(
            "只有 running Step 可以交给后台 Worker 执行"
        )
    if step.claim_id != claim_id:
        raise LongTaskStepExecutionAdapterError(
            "Step 当前 claim_id 与执行令牌不一致"
        )


def _build_worker_step(
    *,
    task: LongTask,
    step: LongTaskStep,
) -> AgentTaskStep:
    """
    把已领取 LongTaskStep 转换成现有协作 Worker 使用的步骤契约。

    参数含义：
        task：提供任务目标与约束的最新 LongTask。
        step：需要转换的已领取步骤。

    返回值含义：
        AgentTaskStep：保留业务输入、依赖和 Agent 分工的 running 步骤。
    """

    input_data = dict(step.input_data)
    input_data["long_task_goal"] = task.goal.model_dump(mode="python")
    return AgentTaskStep(
        step_id=step.step_id,
        title=step.title,
        description=step.description,
        assigned_agent=step.assigned_agent,
        depends_on=list(step.depends_on),
        input_data=input_data,
        expected_output=step.expected_output,
        status="running",
        allow_failure=bool(step.metadata.get("allow_failure", False)),
        metadata={
            **step.metadata,
            "source_long_task_id": task.task_id,
            "source_long_task_step_version": step.version,
        },
    )


def _build_dependency_results(
    *,
    task: LongTask,
    step: LongTaskStep,
) -> dict[str, AgentTaskResult]:
    """
    从 LongTask 中重建当前 Worker 所需的最小前置步骤结果。

    参数含义：
        task：保存所有步骤最新状态、摘要和 Artifact 引用的任务快照。
        step：声明 depends_on 的当前执行步骤。

    返回值含义：
        dict[str, AgentTaskResult]：按依赖 Step 编号索引的最小结果集合。
    """

    steps_by_id = {candidate.step_id: candidate for candidate in task.steps}
    dependency_results: dict[str, AgentTaskResult] = {}
    for dependency_id in step.depends_on:
        dependency = steps_by_id.get(dependency_id)
        if dependency is None:
            raise LongTaskStepExecutionAdapterError(
                f"LongTask 缺少依赖 Step: {dependency_id}"
            )
        if dependency.status not in {"completed", "skipped"}:
            raise LongTaskStepExecutionAdapterError(
                "依赖 Step 尚未成功终结: "
                f"step_id={dependency_id}, status={dependency.status}"
            )
        dependency_output = (
            {"output_ref": dependency.output_ref}
            if dependency.output_ref is not None
            else {}
        )
        dependency_results[dependency_id] = AgentTaskResult(
            step_id=dependency.step_id,
            assigned_agent=dependency.assigned_agent,
            status=dependency.status,
            summary=dependency.output_summary,
            output=dependency_output,
            metadata={"source": "long_task_snapshot"},
        )
    return dependency_results


def _validate_worker_result(
    *,
    step: LongTaskStep,
    result: AgentTaskResult,
) -> None:
    """
    检查协作 Worker 没有返回其他步骤或其他 Agent 的结果。

    参数含义：
        step：本次交给 Worker 的 LongTaskStep。
        result：Worker 实际返回的 AgentTaskResult。

    返回值含义：
        None：Step 与 Agent 归属一致；否则抛出适配异常。
    """

    if result.step_id != step.step_id:
        raise LongTaskStepExecutionAdapterError(
            "协作 Worker 返回了错误 step_id"
        )
    if result.assigned_agent != step.assigned_agent:
        raise LongTaskStepExecutionAdapterError(
            "协作 Worker 返回了错误 assigned_agent"
        )


def _adapt_worker_result(
    *,
    result: AgentTaskResult,
    claim_id: str,
) -> LongTaskBatchStepResult:
    """
    把现有协作 Worker 结果转换成 LongTask 批次步骤结果。

    参数含义：
        result：已经通过归属检查的 AgentTaskResult。
        claim_id：本次 Step 执行权令牌。

    返回值含义：
        LongTaskBatchStepResult：可交给 RuntimeDriver 的中立执行事实。
    """

    output_ref = _extract_output_ref(result)
    metadata = {
        "assigned_agent": result.assigned_agent,
        "evidence_ids": list(result.evidence_ids),
        "latency_ms": result.latency_ms,
        "agent_result_metadata": dict(result.metadata),
    }
    if result.status == "awaiting_input":
        return LongTaskBatchStepResult(
            step_id=result.step_id,
            status="awaiting_input",
            output_summary=result.summary,
            output_ref=output_ref,
            waiting_reason=_resolve_waiting_reason(result),
            user_prompt=result.clarification_prompt,
            claim_id=claim_id,
            metadata=metadata,
        )
    if result.status == "failed":
        return LongTaskBatchStepResult(
            step_id=result.step_id,
            status="failed",
            output_summary=result.summary,
            output_ref=output_ref,
            error_message=result.error_message,
            claim_id=claim_id,
            metadata=metadata,
        )
    return LongTaskBatchStepResult(
        step_id=result.step_id,
        status=result.status,
        output_summary=result.summary,
        output_ref=output_ref,
        claim_id=claim_id,
        metadata=metadata,
    )


def _extract_output_ref(result: AgentTaskResult) -> str | None:
    """
    从 Worker 结果中读取已经持久化的大型输出引用。

    参数含义：
        result：可能在 output 或 metadata 中携带 output_ref 的结果。

    返回值含义：
        str | None：规范化的 Artifact 引用；未提供时返回 None。
    """

    raw_output_ref = result.output.get("output_ref")
    if raw_output_ref is None:
        raw_output_ref = result.metadata.get("output_ref")
    if raw_output_ref is None:
        return None
    normalized_output_ref = str(raw_output_ref).strip()
    return normalized_output_ref or None


def _resolve_waiting_reason(
    result: AgentTaskResult,
) -> LongTaskWaitingReason:
    """
    把 Worker 可选等待原因规范化为 LongTask 等待原因。

    参数含义：
        result：状态为 awaiting_input 的协作 Worker 结果。

    返回值含义：
        LongTaskWaitingReason：合法显式原因，缺省时使用 missing_input。
    """

    raw_reason = result.metadata.get("waiting_reason")
    if raw_reason == "confirmation":
        return "confirmation"
    if raw_reason == "approval":
        return "approval"
    return "missing_input"
