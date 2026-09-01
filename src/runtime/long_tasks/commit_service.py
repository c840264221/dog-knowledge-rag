"""长任务可靠业务提交的草稿构建与兼容调用工具。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskBatchResult,
    LongTaskCommitRequest,
    LongTaskEventActorType,
    LongTaskEventDraft,
    LongTaskQueueMessage,
)
from src.runtime.long_tasks.store import LongTaskStore


def supports_reliable_commit(store: LongTaskStore) -> bool:
    """
    判断 Store 是否公开可靠业务提交能力。

    参数含义：
        store：准备保存 LongTask 的 Store 实例。

    返回值含义：
        bool：存在可调用 commit 方法时为 True，否则为 False。
    """

    return callable(getattr(store, "commit", None))


def calculate_request_fingerprint(payload: Mapping[str, Any]) -> str:
    """
    为稳定业务动作输入计算规范化 SHA-256 请求指纹。

    参数含义：
        payload：不包含随机编号和当前时间的逻辑提交关键字段。

    返回值含义：
        str：带 sha256 前缀的稳定十六进制业务动作指纹。
    """

    return _calculate_fingerprint(payload)


def calculate_submission_fingerprint(payload: Mapping[str, Any]) -> str:
    """
    为最终持久化内容的稳定语义视图计算 SHA-256 提交指纹。

    参数含义：
        payload：包含快照、事件草稿、消息和版本条件，但已排除框架自动
        时间等易变字段的提交语义视图。

    返回值含义：
        str：带 sha256 前缀的稳定十六进制提交内容指纹。
    """

    return _calculate_fingerprint(payload)


def build_long_task_fingerprint_view(task: LongTask) -> dict[str, Any]:
    """
    构建排除框架易变时间字段的 LongTask 指纹视图。

    参数含义：
        task：准备参与请求指纹或提交内容指纹计算的任务快照。

    返回值含义：
        dict[str, Any]：保留业务状态、版本和扩展数据，但不包含框架自动
        生成的任务时间、步骤时间、租约到期时间和等待交互创建时间。
    """

    task_view = task.model_dump(mode="json")
    task_view.pop("created_at", None)
    task_view.pop("updated_at", None)
    for step_view in task_view.get("steps", []):
        step_view.pop("created_at", None)
        step_view.pop("updated_at", None)
        step_view.pop("lease_expires_at", None)
        step_view.pop("last_trace_id", None)
        step_view.pop("last_span_id", None)
    interaction_view = task_view.get("pending_interaction")
    if isinstance(interaction_view, dict):
        interaction_view.pop("created_at", None)
    return task_view


def build_queue_message_fingerprint_view(
    message: LongTaskQueueMessage | None,
) -> dict[str, Any] | None:
    """
    构建排除框架入队时间的队列消息指纹视图。

    参数含义：
        message：准备随快照原子提交的可选队列通知。

    返回值含义：
        dict[str, Any] | None：消息不存在时返回 None；存在时返回保留调度
        语义但不包含自动生成 enqueued_at 的消息数据。
    """

    if message is None:
        return None
    message_view = message.model_dump(mode="json")
    message_view.pop("enqueued_at", None)
    return message_view


def resolve_batch_actor(
    batch_result: LongTaskBatchResult,
) -> tuple[LongTaskEventActorType, str | None]:
    """
    解析批次稳定提交主体，并兼容旧 metadata.worker_name 格式。

    参数含义：
        batch_result：准备生成请求指纹和业务事件的完整批次结果。

    返回值含义：
        tuple[LongTaskEventActorType, str | None]：归一化主体类型和逻辑
        身份；旧数据没有任何身份时返回 worker 与 None。
    """

    if batch_result.actor_type is not None:
        return batch_result.actor_type, batch_result.actor_id
    legacy_worker_name = batch_result.metadata.get("worker_name")
    return (
        "worker",
        str(legacy_worker_name)
        if legacy_worker_name is not None
        else None,
    )


def build_batch_result_fingerprint_view(
    batch_result: LongTaskBatchResult,
) -> dict[str, Any]:
    """
    构建只包含稳定业务结果字段的批次结果指纹视图。

    参数含义：
        batch_result：Worker 准备提交的完整批次结果，允许携带诊断信息。

    返回值含义：
        dict[str, Any]：保留批次身份、步骤状态、产物、交互、错误和领取
        令牌，排除批次级与步骤级 metadata，并按 step_id 稳定排序。
    """

    actor_type, actor_id = resolve_batch_actor(batch_result)
    ordered_results = sorted(
        batch_result.step_results,
        key=lambda result: result.step_id,
    )
    return {
        "batch_id": batch_result.batch_id,
        "task_id": batch_result.task_id,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "step_results": [
            result.model_dump(
                mode="json",
                exclude={"metadata", "span_id"},
            )
            for result in ordered_results
        ],
    }


def _calculate_fingerprint(payload: Mapping[str, Any]) -> str:
    """
    使用统一 JSON 规范化规则计算 SHA-256 指纹。

    参数含义：
        payload：准备规范化并计算摘要的键值映射。

    返回值含义：
        str：带 sha256 前缀的稳定十六进制指纹。
    """

    normalized_json = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_transition_event_drafts(
    *,
    previous_task: LongTask | None,
    next_task: LongTask,
    actor_type: LongTaskEventActorType,
    actor_id: str | None,
    correlation_id: str | None,
    create_event_type: str = "task_submitted",
) -> list[LongTaskEventDraft]:
    """
    根据前后快照生成本次提交已经接受的最小业务事件草稿。

    参数含义：
        previous_task：更新前快照；创建任务时传 None。
        next_task：准备成为权威状态的新快照。
        actor_type/actor_id：本次变化的直接业务触发者。
        correlation_id：请求、批次或人工交互链路编号。
        create_event_type：首次创建使用的业务事件类型。

    返回值含义：
        list[LongTaskEventDraft]：按步骤变化、任务变化顺序排列的事件草稿。
    """

    if previous_task is None:
        return [
            LongTaskEventDraft(
                event_type=create_event_type,
                actor_type=actor_type,
                actor_id=actor_id,
                correlation_id=correlation_id,
                payload={
                    "status": next_task.status,
                    "execution_mode": next_task.execution_mode,
                    "ready_step_ids": [
                        step.step_id
                        for step in next_task.steps
                        if step.status == "ready"
                    ],
                },
            )
        ]

    previous_steps = {
        step.step_id: step for step in previous_task.steps
    }
    drafts: list[LongTaskEventDraft] = []
    for next_step in next_task.steps:
        previous_step = previous_steps.get(next_step.step_id)
        if previous_step is None or previous_step.status == next_step.status:
            continue
        event_type = _step_event_type(next_step.status)
        payload: dict[str, Any] = {
            "previous_status": previous_step.status,
            "status": next_step.status,
        }
        if next_step.output_summary:
            payload["output_summary"] = next_step.output_summary
        if next_step.output_ref is not None:
            payload["output_ref"] = next_step.output_ref
        drafts.append(
            LongTaskEventDraft(
                event_type=event_type,
                actor_type=actor_type,
                actor_id=actor_id,
                step_id=next_step.step_id,
                payload=payload,
                correlation_id=correlation_id,
            )
        )

    if previous_task.status != next_task.status:
        drafts.append(
            LongTaskEventDraft(
                event_type=_task_event_type(
                    previous_task.status,
                    next_task.status,
                ),
                actor_type=actor_type,
                actor_id=actor_id,
                payload={
                    "previous_status": previous_task.status,
                    "status": next_task.status,
                },
                correlation_id=correlation_id,
            )
        )
    elif previous_task.execution_mode != next_task.execution_mode:
        drafts.append(
            LongTaskEventDraft(
                event_type="task_promoted_to_durable",
                actor_type=actor_type,
                actor_id=actor_id,
                payload={
                    "previous_execution_mode": previous_task.execution_mode,
                    "execution_mode": next_task.execution_mode,
                },
                correlation_id=correlation_id,
            )
        )

    if not drafts:
        drafts.append(
            LongTaskEventDraft(
                event_type="task_updated",
                actor_type=actor_type,
                actor_id=actor_id,
                payload={"status": next_task.status},
                correlation_id=correlation_id,
            )
        )
    return drafts


def build_commit_request(
    *,
    previous_task: LongTask | None,
    next_task: LongTask,
    commit_id: str,
    fingerprint_payload: Mapping[str, Any],
    actor_type: LongTaskEventActorType,
    actor_id: str | None,
    correlation_id: str | None,
    queue_message: LongTaskQueueMessage | None = None,
    create_event_type: str = "task_submitted",
) -> LongTaskCommitRequest:
    """
    构建一份经过版本、事件和消息一致性校验的可靠提交请求。

    参数含义：
        previous_task/next_task：业务变化前后的任务快照。
        commit_id：同一次逻辑提交重试时保持稳定的编号。
        fingerprint_payload：计算幂等指纹的稳定业务输入。
        actor_type/actor_id：直接触发业务变化的可信主体。
        correlation_id：本次提交的链路编号。
        queue_message：需要与快照一起入队的可选通知。
        create_event_type：创建任务时使用的事件类型。

    返回值含义：
        LongTaskCommitRequest：可交给 Redis Store 原子执行的完整请求。
    """

    operation = "create" if previous_task is None else "update"
    expected_version = (
        previous_task.version if previous_task is not None else None
    )
    event_drafts = build_transition_event_drafts(
        previous_task=previous_task,
        next_task=next_task,
        actor_type=actor_type,
        actor_id=actor_id,
        correlation_id=correlation_id,
        create_event_type=create_event_type,
    )
    submission_payload = {
        "operation": operation,
        "expected_version": expected_version,
        "task": build_long_task_fingerprint_view(next_task),
        "event_drafts": [
            draft.model_dump(mode="json") for draft in event_drafts
        ],
        "queue_message": build_queue_message_fingerprint_view(
            queue_message
        ),
    }
    return LongTaskCommitRequest(
        operation=operation,
        task=next_task,
        expected_version=expected_version,
        commit_id=commit_id,
        request_fingerprint=calculate_request_fingerprint(
            fingerprint_payload
        ),
        submission_fingerprint=calculate_submission_fingerprint(
            submission_payload
        ),
        event_drafts=event_drafts,
        queue_message=queue_message,
    )


def _step_event_type(status: str) -> str:
    """把已接受的步骤状态映射为过去式业务事件类型。"""

    return {
        "ready": "step_became_ready",
        "running": "step_started",
        "awaiting_input": "step_awaiting_input",
        "completed": "step_completed",
        "failed": "step_failed",
        "skipped": "step_skipped",
        "cancelled": "step_cancelled",
    }.get(status, "step_updated")


def _task_event_type(previous_status: str, next_status: str) -> str:
    """把已接受的任务状态迁移映射为过去式业务事件类型。"""

    if next_status == "running" and previous_status == "awaiting_input":
        return "task_resumed"
    return {
        "queued": "task_queued",
        "running": "task_started",
        "awaiting_input": "task_awaiting_input",
        "completed": "task_completed",
        "failed": "task_failed",
        "cancelled": "task_cancelled",
    }.get(next_status, "task_updated")
