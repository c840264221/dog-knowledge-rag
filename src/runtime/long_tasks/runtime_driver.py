"""长任务批次结果的最小 Runtime Driver / PEP。"""

from __future__ import annotations

from typing import Any

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskBatchStepResult,
    LongTaskExecutionContext,
    LongTaskExecutionDecision,
    LongTaskQueueMessage,
    LongTaskStep,
    LongTaskWaitingReason,
)
from src.runtime.long_tasks.commit_service import (
    build_batch_result_fingerprint_view,
    build_commit_request,
    calculate_request_fingerprint,
    resolve_batch_actor,
    supports_reliable_commit,
)
from src.runtime.long_tasks.claim_service import (
    LongTaskStepClaimConflictError,
)
from src.runtime.long_tasks.execution_policy import (
    decide_long_task_execution,
    project_long_task_execution_facts,
)
from src.runtime.long_tasks.interaction_service import (
    LongTaskInteractionService,
)
from src.runtime.long_tasks.state_machine import (
    promote_long_task_to_durable,
    revise_long_task_runtime_state,
    transition_long_task,
    transition_long_task_step,
)
from src.runtime.long_tasks.store import (
    LongTaskCommitFingerprintConflictError,
    LongTaskNotFoundError,
    LongTaskStore,
)


class UnsupportedLongTaskDriverActionError(RuntimeError):
    """表示当前 Driver 尚未接入某项决策要求的专用业务处理器。"""


class LongTaskRuntimeDriver:
    """
    加载任务、运行 PDP、执行已支持动作并以乐观锁保存新快照。

    参数含义：
        store:
            提供最新 LongTask 快照和 expected_version 保存能力的 Store。

    返回值含义：
        LongTaskRuntimeDriver:
            可以消费一个 BatchResult 的最小 PEP 执行入口。
    """

    def __init__(
        self,
        store: LongTaskStore,
        *,
        interaction_service: LongTaskInteractionService | None = None,
    ) -> None:
        self._store = store
        self._interaction_service = (
            interaction_service or LongTaskInteractionService()
        )

    async def handle_batch_result(
        self,
        *,
        task_id: str,
        batch_result: LongTaskBatchResult,
        execution_context: LongTaskExecutionContext,
        publish_ready_message: bool = False,
        correlation_id: str | None = None,
    ) -> LongTask:
        """
        对最新任务执行事实投影、策略决策、状态提交和乐观锁保存。

        参数含义：
            task_id:
                需要提交当前批次结果的长任务编号。
            batch_result:
                单 Agent 或多 Agent 返回的统一步骤结果。
            execution_context:
                当前请求已用同步时间和允许的时间预算。
            publish_ready_message:
                可靠 Store 是否需要在同一提交中发布后续 Ready Step 通知。
            correlation_id:
                可选的上游队列或请求链路编号。

        返回值含义：
            LongTask:
                已执行受支持 Decision 并成功保存的最新任务快照。
        """

        fingerprint_payload = {
            "action": "handle_batch_result",
            "task_id": task_id,
            "batch_result": build_batch_result_fingerprint_view(
                batch_result
            ),
        }
        commit_id = f"batch:{task_id}:{batch_result.batch_id}"
        committed_task = await self._load_idempotent_result_if_committed(
            task_id=task_id,
            commit_id=commit_id,
            fingerprint_payload=fingerprint_payload,
        )
        if committed_task is not None:
            return committed_task

        current_task = await self._require_running_task(task_id)
        facts = project_long_task_execution_facts(
            task=current_task,
            batch_result=batch_result,
            execution_context=execution_context,
        )
        decision = decide_long_task_execution(facts)
        self._require_supported_action(decision)
        next_task = self._apply_supported_decision(
            task=current_task,
            batch_result=batch_result,
            decision=decision,
        )
        if supports_reliable_commit(self._store):
            ready_step_ids = [
                step.step_id
                for step in next_task.steps
                if step.status == "ready"
            ]
            queue_message = (
                LongTaskQueueMessage(
                    task_id=next_task.task_id,
                    task_version=next_task.version,
                    reason="continued",
                    ready_step_ids=ready_step_ids,
                    correlation_id=correlation_id or task_id,
                )
                if publish_ready_message
                and next_task.execution_mode == "durable"
                and next_task.status == "running"
                and ready_step_ids
                else None
            )
            actor_type, actor_id = resolve_batch_actor(batch_result)
            request = build_commit_request(
                previous_task=current_task,
                next_task=next_task,
                commit_id=commit_id,
                fingerprint_payload=fingerprint_payload,
                actor_type=actor_type,
                actor_id=actor_id,
                correlation_id=correlation_id or batch_result.batch_id,
                queue_message=queue_message,
            )
            await self._store.commit(request)  # type: ignore[attr-defined]
            authoritative_task = await self._store.load(task_id)
            if authoritative_task is None:
                raise LongTaskNotFoundError(
                    "批次提交回执已经产生但权威任务不存在: "
                    f"{task_id}"
                )
            return authoritative_task
        return await self._store.save(
            next_task,
            expected_version=current_task.version,
        )

    async def _load_idempotent_result_if_committed(
        self,
        *,
        task_id: str,
        commit_id: str,
        fingerprint_payload: dict[str, Any],
    ) -> LongTask | None:
        """
        在重新验证 running 状态前查询批次提交是否已经成功。

        参数含义：
            task_id/commit_id：批次所属任务和稳定逻辑提交编号。
            fingerprint_payload：本次批次的规范化关键输入。

        返回值含义：
            LongTask | None：重复成功时返回最新任务，否则返回 None。
        """

        load_receipt = getattr(self._store, "load_receipt", None)
        if not callable(load_receipt):
            return None
        receipt = await load_receipt(
            task_id=task_id,
            commit_id=commit_id,
        )
        if receipt is None:
            return None
        fingerprint = calculate_request_fingerprint(
            fingerprint_payload
        )
        if receipt.request_fingerprint != fingerprint:
            raise LongTaskCommitFingerprintConflictError(
                "batch commit_id 已用于不同业务动作: "
                f"task_id={task_id}, commit_id={commit_id}"
            )
        task = await self._store.load(task_id)
        if task is None:
            raise LongTaskNotFoundError(
                f"批次回执存在但长任务不存在: {task_id}"
            )
        return task

    async def _require_running_task(self, task_id: str) -> LongTask:
        """
        加载必须存在且当前处于 running 的任务快照。

        参数含义：
            task_id:
                准备提交批次结果的任务编号。

        返回值含义：
            LongTask:
                Store 中最新的运行中任务；不存在或状态错误时抛出异常。
        """

        task = await self._store.load(task_id)
        if task is None:
            raise LongTaskNotFoundError(f"长任务不存在: {task_id}")
        if task.status != "running":
            raise ValueError(
                "Runtime Driver 只能提交 running 任务的批次结果: "
                f"{task.status}"
            )
        return task

    @staticmethod
    def _require_supported_action(
        decision: LongTaskExecutionDecision,
    ) -> None:
        """
        拒绝尚未接入交互聚合器或失败治理器的决策动作。

        参数含义：
            decision:
                PDP 根据当前预测事实返回的结构化决策。

        返回值含义：
            None。动作已支持时正常返回，否则抛出明确边界异常。
        """

        if decision.action == "handle_failure":
            raise UnsupportedLongTaskDriverActionError(
                "handle_failure 尚未接入重试与降级策略"
            )

    def _apply_supported_decision(
        self,
        *,
        task: LongTask,
        batch_result: LongTaskBatchResult,
        decision: LongTaskExecutionDecision,
    ) -> LongTask:
        """
        应用步骤结果，并执行推进、完成或 durable 升级动作。

        参数含义：
            task:
                本次决策使用的旧任务快照。
            batch_result:
                需要正式提交的批次步骤结果。
            decision:
                已通过支持范围检查的 PDP 决策。

        返回值含义：
            LongTask:
                版本恰好递增 1、可以交给 Store 乐观锁保存的新快照。
        """

        updated_steps = _apply_batch_step_results(
            task=task,
            batch_result=batch_result,
            ready_step_ids=set(decision.ready_step_ids),
        )

        if decision.action == "advance_task":
            return revise_long_task_runtime_state(
                task,
                steps=updated_steps,
                active_step_ids=[],
            )

        transition_base = _build_transition_base(
            task=task,
            steps=updated_steps,
        )
        if decision.action == "complete_task":
            return transition_long_task(
                transition_base,
                target_status="completed",
                active_step_ids=[],
            )
        if decision.action == "promote_to_durable":
            return promote_long_task_to_durable(transition_base)
        if decision.action == "await_user_input":
            interaction_data = _build_waiting_interaction_data(
                task=transition_base,
                batch_result=batch_result,
                decision=decision,
            )
            return self._interaction_service.wait_for_step_input(
                transition_base,
                waiting_step_ids=decision.awaiting_step_ids,
                interaction_type=interaction_data["interaction_type"],
                prompt=interaction_data["prompt"],
                input_contract=interaction_data["input_contract"],
                metadata=interaction_data["metadata"],
            )
        raise UnsupportedLongTaskDriverActionError(
            f"Runtime Driver 不认识决策动作: {decision.action}"
        )


def _build_transition_base(
    *,
    task: LongTask,
    steps: list[LongTaskStep],
) -> LongTask:
    """
    构建包含最新步骤、但尚未递增 Task 版本的复合迁移草稿。

    参数含义：
        task:
            本次批次提交前的权威任务快照。
        steps:
            已通过步骤状态机迁移的新步骤集合。

    返回值含义：
        LongTask:
            仅供后续 Task 状态机使用、不会直接写入 Store 的同版本草稿。
    """

    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "steps": steps,
            "active_step_ids": [],
        }
    )
    return LongTask.model_validate(task_data)


def _build_waiting_interaction_data(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
    decision: LongTaskExecutionDecision,
) -> dict[str, Any]:
    """
    聚合同一批次中一个或多个等待 Step 的前端交互数据。

    参数含义：
        task:
            已应用 Step 结果、可读取步骤标题的同版本任务草稿。
        batch_result:
            保存每个等待 Step 原始 user_prompt 和 waiting_reason 的批次结果。
        decision:
            PDP 已确认需要等待用户输入的结构化决策。

    返回值含义：
        dict[str, Any]:
            包含聚合 interaction_type、公开 prompt、input_contract 和诊断
            metadata 的临时数据；该对象本身不会持久化。
    """

    waiting_ids = set(decision.awaiting_step_ids)
    waiting_results = [
        result
        for result in batch_result.step_results
        if result.step_id in waiting_ids
        and result.status == "awaiting_input"
    ]
    if len(waiting_results) != len(waiting_ids):
        raise ValueError("等待决策与批次 awaiting_input 结果不一致")

    steps_by_id = {step.step_id: step for step in task.steps}
    items = [
        {
            "step_id": result.step_id,
            "title": steps_by_id[result.step_id].title,
            "waiting_reason": result.waiting_reason,
            "prompt": result.user_prompt,
        }
        for result in waiting_results
    ]
    waiting_reasons = {
        result.waiting_reason for result in waiting_results
    }
    interaction_type: LongTaskWaitingReason = (
        next(iter(waiting_reasons))
        if len(waiting_reasons) == 1
        else "missing_input"
    )
    if len(items) == 1:
        prompt = str(items[0]["prompt"])
    else:
        prompt_lines = ["继续任务前，请处理以下信息："]
        prompt_lines.extend(
            f"{index}. {item['title']}：{item['prompt']}"
            for index, item in enumerate(items, start=1)
        )
        prompt = "\n".join(prompt_lines)

    return {
        "interaction_type": interaction_type,
        "prompt": prompt,
        "input_contract": {"items": items},
        "metadata": {
            "batch_id": batch_result.batch_id,
            "waiting_step_count": len(items),
        },
    }


def _apply_batch_step_results(
    *,
    task: LongTask,
    batch_result: LongTaskBatchResult,
    ready_step_ids: set[str],
) -> list[LongTaskStep]:
    """
    使用步骤状态机应用批次结果，并把新满足依赖的 pending 步骤改为 ready。

    参数含义：
        task:
            批次提交前的任务快照。
        batch_result:
            当前批次返回的逐步骤事实。
        ready_step_ids:
            PDP 预测出的下一批可执行步骤编号。

    返回值含义：
        list[LongTaskStep]:
            结果状态、摘要、引用和 Ready 状态均已更新的新步骤列表。
    """

    result_by_step_id = {
        result.step_id: result
        for result in batch_result.step_results
    }
    updated_steps: list[LongTaskStep] = []
    for step in task.steps:
        result = result_by_step_id.get(step.step_id)
        updated_step = (
            _apply_single_step_result(
                step,
                result,
                trace_id=batch_result.trace_id,
            )
            if result is not None
            else step
        )
        if (
            updated_step.step_id in ready_step_ids
            and updated_step.status == "pending"
        ):
            updated_step = transition_long_task_step(
                updated_step,
                target_status="ready",
            )
        updated_steps.append(updated_step)
    return updated_steps


def _apply_single_step_result(
    step: LongTaskStep,
    result: LongTaskBatchStepResult,
    *,
    trace_id: str | None,
) -> LongTaskStep:
    """
    通过步骤状态机提交结果，并保存业务事实与小型诊断引用。

    参数含义：
        step:
            批次执行前的步骤快照。
        result:
            当前 Step 对应的统一批次结果。
        trace_id:
            当前批次的可选诊断链路编号。

    返回值含义：
        LongTaskStep:
            状态和输出信息均已更新、版本递增 1 的步骤快照。
    """

    if step.claim_id != result.claim_id:
        raise LongTaskStepClaimConflictError(
            "批次结果 claim_id 与当前 Step 领取令牌不一致: "
            f"step_id={step.step_id}"
        )

    transitioned_step = transition_long_task_step(
        step,
        target_status=result.status,
        waiting_reason=result.waiting_reason,
    )
    step_data = transitioned_step.model_dump(mode="python")
    stable_metadata = dict(transitioned_step.metadata)
    stable_metadata.pop("last_batch_result", None)
    step_data.update(
        {
            "output_summary": result.output_summary,
            "output_ref": result.output_ref,
            "last_error_code": (
                result.error_code if result.status == "failed" else None
            ),
            "last_error_message": (
                result.error_message
                if result.status == "failed"
                else None
            ),
            "last_trace_id": trace_id,
            "last_span_id": result.span_id,
            "metadata": stable_metadata,
        }
    )
    return LongTaskStep.model_validate(step_data)
