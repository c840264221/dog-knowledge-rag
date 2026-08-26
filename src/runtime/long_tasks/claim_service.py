"""LongTask Step 的最小领取、租约与乐观锁提交服务。"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskStep,
    utc_now,
)
from src.runtime.long_tasks.state_machine import (
    revise_long_task_runtime_state,
    transition_long_task_step,
)
from src.runtime.long_tasks.store import (
    LongTaskNotFoundError,
    LongTaskStore,
)


class LongTaskStepClaimError(RuntimeError):
    """表示 Step 当前不满足后台领取或接管条件。"""


class LongTaskStepClaimConflictError(LongTaskStepClaimError):
    """表示 Step 已被其他有效租约占用，或结果携带了旧领取令牌。"""


class LongTaskStepAttemptsExhaustedError(LongTaskStepClaimError):
    """表示 Step 已经达到最大执行次数，不能再次领取。"""


class LongTaskStepClaimService:
    """
    使用权威 LongTask 快照和乐观锁领取一个可执行 Step。

    参数含义：
        store:
            提供最新 LongTask 加载和 expected_version 保存能力的 Store。

    返回值含义：
        LongTaskStepClaimService:
            不执行实际业务，只负责 Step 领取事实持久化的应用服务。
    """

    def __init__(self, store: LongTaskStore) -> None:
        self._store = store

    async def claim_step(
        self,
        *,
        task_id: str,
        step_id: str,
        worker_name: str,
        claim_id: str,
        lease_duration_ms: int,
        now: datetime | None = None,
    ) -> LongTask:
        """
        领取 Ready Step，或在旧租约过期后接管 Running Step。

        参数含义：
            task_id:
                准备修改的权威长任务编号。
            step_id:
                当前 Worker 准备执行的步骤编号。
            worker_name:
                当前后台 Worker 的稳定名称。
            claim_id:
                本次执行唯一令牌；重复请求使用相同令牌可获得幂等结果。
            lease_duration_ms:
                从 now 开始计算的租约有效毫秒数。
            now:
                可选 UTC 当前时间，主要供确定性测试使用。

        返回值含义：
            LongTask:
                领取事实已通过乐观锁保存的最新任务；相同有效领取重复提交
                时直接返回当前任务，不重复递增版本和执行次数。
        """

        normalized_task_id = _require_text(task_id, "task_id")
        normalized_step_id = _require_text(step_id, "step_id")
        normalized_worker = _require_text(worker_name, "worker_name")
        normalized_claim_id = _require_text(claim_id, "claim_id")
        if lease_duration_ms < 1:
            raise ValueError("lease_duration_ms 必须大于等于 1")
        resolved_now = now or utc_now()
        if resolved_now.tzinfo is None or resolved_now.utcoffset() is None:
            raise ValueError("now 必须包含时区")

        task = await self._store.load(normalized_task_id)
        if task is None:
            raise LongTaskNotFoundError(
                f"长任务不存在: {normalized_task_id}"
            )
        self._require_durable_running_task(task)
        step = _find_step(task, normalized_step_id)

        if _is_same_active_claim(
            step,
            worker_name=normalized_worker,
            claim_id=normalized_claim_id,
            now=resolved_now,
        ):
            return task

        claimed_step = _build_claimed_step(
            step,
            worker_name=normalized_worker,
            claim_id=normalized_claim_id,
            lease_expires_at=(
                resolved_now + timedelta(milliseconds=lease_duration_ms)
            ),
            now=resolved_now,
        )
        updated_steps = [
            claimed_step if candidate.step_id == step.step_id else candidate
            for candidate in task.steps
        ]
        updated_task = revise_long_task_runtime_state(
            task,
            steps=updated_steps,
            active_step_ids=[
                *task.active_step_ids,
                *(
                    []
                    if step.step_id in task.active_step_ids
                    else [step.step_id]
                ),
            ],
        )
        return await self._store.save(
            updated_task,
            expected_version=task.version,
        )

    @staticmethod
    def _require_durable_running_task(task: LongTask) -> None:
        """
        校验当前任务允许被后台 Worker 领取。

        参数含义：
            task:
                Store 中加载到的最新权威任务。

        返回值含义：
            None：任务为 durable + running 时正常返回，否则抛出异常。
        """

        if task.status != "running" or task.execution_mode != "durable":
            raise LongTaskStepClaimError(
                "只有 durable 且 running 的任务可以领取后台 Step"
            )


def _find_step(task: LongTask, step_id: str) -> LongTaskStep:
    """
    从任务快照中查找指定 Step。

    参数含义：
        task:
            包含全部步骤的权威任务快照。
        step_id:
            准备查找的步骤编号。

    返回值含义：
        LongTaskStep：找到的步骤；不存在时抛出领取异常。
    """

    for step in task.steps:
        if step.step_id == step_id:
            return step
    raise LongTaskStepClaimError(f"长任务中不存在 Step: {step_id}")


def _is_same_active_claim(
    step: LongTaskStep,
    *,
    worker_name: str,
    claim_id: str,
    now: datetime,
) -> bool:
    """
    判断请求是否为相同 Worker 对同一有效领取的幂等重放。

    参数含义：
        step:
            Store 中最新的步骤快照。
        worker_name:
            当前请求的 Worker 名称。
        claim_id:
            当前请求的领取令牌。
        now:
            本次判断使用的 UTC 时间。

    返回值含义：
        bool：领取身份一致且租约仍有效时返回 True。
    """

    return bool(
        step.status == "running"
        and step.claimed_by == worker_name
        and step.claim_id == claim_id
        and step.lease_expires_at is not None
        and step.lease_expires_at > now
    )


def _build_claimed_step(
    step: LongTaskStep,
    *,
    worker_name: str,
    claim_id: str,
    lease_expires_at: datetime,
    now: datetime,
) -> LongTaskStep:
    """
    构建首次领取或租约过期后重新领取的 Step 快照。

    参数含义：
        step:
            当前 Ready 或已经租约过期的 Running Step。
        worker_name:
            新租约的 Worker 名称。
        claim_id:
            新租约的唯一领取令牌。
        lease_expires_at:
            新租约到期时间。
        now:
            本次领取发生时间。

    返回值含义：
        LongTaskStep：版本和执行次数均已正确递增的新步骤快照。
    """

    if step.attempt_count >= step.max_attempts:
        raise LongTaskStepAttemptsExhaustedError(
            "Step 已达到最大执行次数，不能再次领取"
        )

    if step.status == "ready":
        base_step = transition_long_task_step(
            step,
            target_status="running",
        )
    elif step.status == "running":
        if step.lease_expires_at is None:
            raise LongTaskStepClaimConflictError(
                "Running Step 没有后台租约，不能直接接管"
            )
        if step.lease_expires_at > now:
            raise LongTaskStepClaimConflictError(
                "Step 当前仍由有效租约占用"
            )
        step_data = step.model_dump(mode="python")
        step_data.update(
            {
                "attempt_count": step.attempt_count + 1,
                "version": step.version + 1,
                "updated_at": now,
            }
        )
        base_step = LongTaskStep.model_validate(step_data)
    else:
        raise LongTaskStepClaimError(
            f"只有 ready 或租约过期的 running Step 可以领取: {step.status}"
        )

    claimed_data = base_step.model_dump(mode="python")
    claimed_data.update(
        {
            "claimed_by": worker_name,
            "claim_id": claim_id,
            "lease_expires_at": lease_expires_at,
            "updated_at": now,
        }
    )
    return LongTaskStep.model_validate(claimed_data)


def _require_text(value: str, field_name: str) -> str:
    """
    规范并校验领取服务的必填文本。

    参数含义：
        value:
            准备去除首尾空白的原始文本。
        field_name:
            校验失败时显示的字段名称。

    返回值含义：
        str：规范化后的非空文本。
    """

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized
