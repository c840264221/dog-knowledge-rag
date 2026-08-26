"""
多 Agent 协作主图入口节点。

功能：
    根据 DogState 中的恢复动作选择新建、重新规划或恢复多 Agent 任务，
    再把标准任务结果转换成可以写入主图和 Checkpoint 的普通字典字段。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Awaitable

from src.agents.collaboration.adapters import (
    adapt_paused_collaboration_result_to_long_task,
)
from src.agents.collaboration.contracts import MultiAgentTaskResult
from src.agents.collaboration.scheduler import (
    MultiAgentTaskCancellationRegistry,
    MultiAgentTaskCancellationToken,
    build_multi_agent_task_id,
)
from src.graph.states.dog_state import DogState


MultiAgentEntryNode = Callable[
    [DogState],
    Awaitable[dict[str, Any]],
]


def build_multi_agent_entry_node(
    *,
    orchestrator: Any,
    cancellation_registry: MultiAgentTaskCancellationRegistry | None = None,
    long_task_application_service: Any | None = None,
) -> MultiAgentEntryNode:
    """
    构建注入 MultiAgentOrchestrator 的主图异步节点。

    功能：
        新复杂目标调用 run；Worker 等待后的回答调用 resume；Planner 等待
        后的回答携带补充上下文重新规划；澄清未完成时只返回等待提示。

    参数含义：
        orchestrator:
            提供 run 和 resume 方法的多 Agent 总编排器。
        cancellation_registry:
            可选的运行中任务取消登记表；提供后会为每次执行登记取消令牌。
        long_task_application_service:
            可选长任务应用服务；预算暂停结果需要通过它创建 durable 快照
            并发布后台执行通知。普通请求不会使用该依赖。

    返回值含义：
        MultiAgentEntryNode:
            可以注册到 StateGraph 的异步多 Agent 入口节点。
    """

    if orchestrator is None:
        raise ValueError("多 Agent 主图入口必须提供 orchestrator")
    if not callable(getattr(orchestrator, "run", None)):
        raise ValueError("orchestrator 缺少 run 方法")
    if not callable(getattr(orchestrator, "resume", None)):
        raise ValueError("orchestrator 缺少 resume 方法")
    if long_task_application_service is not None and not callable(
        getattr(
            long_task_application_service,
            "handoff_paused_collaboration_task",
            None,
        )
    ):
        raise ValueError(
            "long_task_application_service 缺少协作交接方法"
        )

    async def multi_agent_entry_node(
        state: DogState,
    ) -> dict[str, Any]:
        """
        执行或恢复一次多 Agent 协作任务。

        参数含义：
            state:
                RootAgent 已完成路由和恢复输入整理的当前 DogState。

        返回值含义：
            dict[str, Any]:
                多 Agent 结果、最终回答和下一轮恢复字段组成的局部状态。
        """

        action = str(
            state.get("multi_agent_resume_action") or "none"
        )
        if action == "needs_clarification":
            prompt = str(
                state.get("multi_agent_pending_prompt")
                or "请补充多 Agent 任务所需信息。"
            )
            return {
                "current_agent": "multi_agent",
                "final_answer": prompt,
                "pending_prompt": prompt,
                "waiting_user_input": True,
            }

        if action == "resume":
            task_result = _load_pending_task_result(state)
            user_inputs = _load_resume_inputs(state)
            step_resume_decisions = _load_step_resume_decisions(state)
            result = await _run_with_registered_cancellation(
                cancellation_registry=cancellation_registry,
                multi_agent_task_id=task_result.collaboration_id,
                operation=lambda token: orchestrator.resume(
                    task_result,
                    user_inputs=user_inputs,
                    cancellation_token=token,
                    **(
                        {
                            "step_resume_decisions": (
                                step_resume_decisions
                            )
                        }
                        if step_resume_decisions
                        else {}
                    ),
                ),
            )
        elif action == "replan":
            task_result = _load_pending_task_result(state)
            resume_inputs = _load_resume_inputs(state)
            result = await _run_with_registered_cancellation(
                cancellation_registry=cancellation_registry,
                multi_agent_task_id=task_result.collaboration_id,
                operation=lambda token: orchestrator.run(
                    task_result.plan.objective,
                    multi_agent_task_id=task_result.collaboration_id,
                    cancellation_token=token,
                    context={
                        "user_clarification": resume_inputs.get(
                            "planner_clarification",
                            "",
                        ),
                        "previous_clarification_prompt": (
                            task_result.plan.clarification_prompt
                        ),
                        "memory_context": state.get("memory_context", ""),
                        "user_id": state.get("user_id", ""),
                        "session_id": state.get("session_id", ""),
                        "trace_id": state.get("trace_id", ""),
                    },
                    worker_runtime_context=(
                        _build_worker_runtime_context(state)
                    ),
                ),
            )
        else:
            multi_agent_task_id = build_multi_agent_task_id(
                str(state.get("trace_id") or "")
            )
            result = await _run_with_registered_cancellation(
                cancellation_registry=cancellation_registry,
                multi_agent_task_id=multi_agent_task_id,
                operation=lambda token: orchestrator.run(
                    str(state.get("question") or "").strip(),
                    multi_agent_task_id=multi_agent_task_id,
                    cancellation_token=token,
                    context={
                        "memory_context": state.get("memory_context", ""),
                        "user_id": state.get("user_id", ""),
                        "session_id": state.get("session_id", ""),
                        "trace_id": state.get("trace_id", ""),
                    },
                    worker_runtime_context=(
                        _build_worker_runtime_context(state)
                    ),
                ),
            )

        if _is_inline_budget_paused(result):
            return await _handoff_paused_collaboration_result(
                task_result=result,
                state=state,
                resume_action=action,
                long_task_application_service=(
                    long_task_application_service
                ),
            )
        return build_multi_agent_state_update(result)

    return multi_agent_entry_node


def _is_inline_budget_paused(
    task_result: MultiAgentTaskResult,
) -> bool:
    """
    判断协作结果是否位于请求内时间预算耗尽的批次边界。

    参数含义：
        task_result:
            Orchestrator 返回的最新多智能体任务结果。

    返回值含义：
        bool:
            结果仍为 running 且暂停原因为 inline_budget_exhausted 时为 True。
    """

    pause_facts = task_result.metadata.get("execution_paused")
    return (
        task_result.status == "running"
        and isinstance(pause_facts, Mapping)
        and pause_facts.get("reason") == "inline_budget_exhausted"
    )


async def _handoff_paused_collaboration_result(
    *,
    task_result: MultiAgentTaskResult,
    state: Mapping[str, Any],
    resume_action: str,
    long_task_application_service: Any | None,
) -> dict[str, Any]:
    """
    把预算暂停结果转换为 LongTask、完成内部交接并构建主图更新。

    参数含义：
        task_result:
            已在完整批次边界暂停的多智能体结果。
        state:
            提供用户、会话和原始问题的当前 DogState。
        resume_action:
            本轮是新任务、恢复还是重新规划，用于选择正确的原始请求。
        long_task_application_service:
            提供 durable 快照创建和 Redis Stream 发布能力的应用服务。

    返回值含义：
        dict[str, Any]:
            包含后台任务引用和用户可见提示的主图局部状态。
    """

    if long_task_application_service is None:
        raise RuntimeError("多智能体长任务交接服务尚未配置")
    original_request = _resolve_handoff_original_request(
        state=state,
        task_result=task_result,
        resume_action=resume_action,
    )
    long_task = adapt_paused_collaboration_result_to_long_task(
        task_result=task_result,
        user_id=str(state.get("user_id") or "").strip(),
        thread_id=str(state.get("session_id") or "").strip(),
        original_request=original_request,
    )
    saved_task = await (
        long_task_application_service.handoff_paused_collaboration_task(
            long_task
        )
    )
    return _build_durable_handoff_state_update(
        task_result=task_result,
        saved_task=saved_task,
    )


def _resolve_handoff_original_request(
    *,
    state: Mapping[str, Any],
    task_result: MultiAgentTaskResult,
    resume_action: str,
) -> str:
    """
    选择交接快照中应长期保留的原始用户目标。

    参数含义：
        state:
            当前主图状态，其中 question 可能是原始问题或恢复回答。
        task_result:
            包含规范化 Plan 目标的预算暂停结果。
        resume_action:
            当前是否由 resume 或 replan 路径产生暂停。

    返回值含义：
        str:
            新请求优先使用本轮 question；恢复路径使用 Plan objective，避免
            把“允许继续”等简短回答误存成完整任务目标。
    """

    if resume_action in {"resume", "replan"}:
        return task_result.plan.objective
    return str(state.get("question") or "").strip()


def _build_durable_handoff_state_update(
    *,
    task_result: MultiAgentTaskResult,
    saved_task: Any,
) -> dict[str, Any]:
    """
    构建已转入后台执行后的主图局部状态。

    参数含义：
        task_result:
            请求内暂停的多智能体结果，用于保留计划和已完成历史。
        saved_task:
            应用服务已经写入 Store 的 LongTask 快照。

    返回值含义：
        dict[str, Any]:
            清空人工恢复字段、携带 durable_handoff 引用和用户提示的状态。
    """

    result_data = task_result.model_dump(mode="python")
    metadata = dict(result_data.get("metadata") or {})
    metadata["durable_handoff"] = {
        "task_id": saved_task.task_id,
        "task_version": saved_task.version,
        "status": saved_task.status,
        "execution_mode": saved_task.execution_mode,
        "owner_user_id": saved_task.user_id,
    }
    result_data["metadata"] = metadata
    return {
        "multi_agent_task_result": result_data,
        "multi_agent_resume_action": "none",
        "multi_agent_resume_inputs": {},
        "multi_agent_step_resume_decisions": {},
        "multi_agent_resume_ready": False,
        "multi_agent_clarification_extraction": {},
        "multi_agent_pending_prompt": "",
        "pending_prompt": "",
        "waiting_user_input": False,
        "current_agent": "multi_agent",
        "final_answer": (
            "任务已转入后台执行，可使用任务编号 "
            f"{saved_task.task_id} 查询状态。"
        ),
    }


def _build_worker_runtime_context(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从主图状态提取只允许程序传给 Worker 的可信运行数据。

    功能：
        将用户、会话、追踪和当前宠物标识从 Planner 提示上下文中分离，
        供总编排器在计划校验完成后写入每个步骤。

    参数含义：
        state：当前主图状态。

    返回值含义：
        dict[str, Any]：可以安全写入 Worker input_data 的运行字段。
    """

    return {
        field_name: state.get(field_name, "")
        for field_name in (
            "user_id",
            "session_id",
            "trace_id",
            "active_pet_key",
            "active_pet_name",
        )
    }


async def _run_with_registered_cancellation(
    *,
    cancellation_registry: MultiAgentTaskCancellationRegistry | None,
    multi_agent_task_id: str,
    operation: Callable[
        [MultiAgentTaskCancellationToken | None],
        Awaitable[MultiAgentTaskResult],
    ],
) -> MultiAgentTaskResult:
    """
    在运行中任务登记表的保护下执行一次编排操作。

    功能：
        有登记表时创建共享取消令牌并传给编排器；操作结束后无论成功或
        抛出异常都移除登记，避免已结束任务继续占用任务编号。

    参数含义：
        cancellation_registry:
            可选的运行中任务取消登记表。
        multi_agent_task_id:
            当前整次多 Agent 任务编号。
        operation:
            接收取消令牌并返回任务结果的异步编排调用。

    返回值含义：
        MultiAgentTaskResult:
            编排器生成的最新多 Agent 任务结果。
    """

    if cancellation_registry is None:
        return await operation(None)

    token = cancellation_registry.register(multi_agent_task_id)
    try:
        return await operation(token)
    finally:
        cancellation_registry.unregister(multi_agent_task_id, token)


def _load_pending_task_result(
    state: Mapping[str, Any],
) -> MultiAgentTaskResult:
    """
    从主图状态读取并校验暂停任务结果。

    参数含义：
        state:
            包含 multi_agent_task_result 的当前主图状态。

    返回值含义：
        MultiAgentTaskResult:
            通过 Schema 校验的暂停任务结果。
    """

    raw_result = state.get("multi_agent_task_result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("主图状态缺少可恢复的 multi_agent_task_result")
    return MultiAgentTaskResult.model_validate(raw_result)


def _load_resume_inputs(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从主图状态读取已经整理好的恢复输入。

    参数含义：
        state:
            包含 multi_agent_resume_inputs 的当前主图状态。

    返回值含义：
        dict[str, Any]:
            等待步骤编号到用户回答的独立字典。
    """

    raw_inputs = state.get("multi_agent_resume_inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise ValueError("主图状态缺少 multi_agent_resume_inputs")
    return dict(raw_inputs)


def _load_step_resume_decisions(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从主图状态读取每个等待步骤已经确定的恢复方式。

    参数含义：
        state:
            包含 multi_agent_step_resume_decisions 的当前主图状态。

    返回值含义：
        dict[str, Any]:
            步骤编号到正常恢复、简化执行或继续等待决定的映射。
    """

    raw_decisions = state.get("multi_agent_step_resume_decisions")
    if not isinstance(raw_decisions, Mapping):
        return {}
    return dict(raw_decisions)


def build_multi_agent_state_update(
    task_result: MultiAgentTaskResult,
) -> dict[str, Any]:
    """
    把多 Agent 标准结果转换成主图局部状态。

    功能：
        始终保存可序列化任务结果；等待输入时写入提示和等待标记，任务结束
        时写入最终回答并清空恢复字段。

    参数含义：
        task_result:
            Orchestrator 返回的最新多 Agent 任务结果。

    返回值含义：
        dict[str, Any]:
            可以由 LangGraph 合并并自动写入 Checkpoint 的普通字典。
    """

    result_data = task_result.model_dump(mode="python")
    if task_result.status == "awaiting_input":
        prompt = _extract_waiting_prompt(task_result)
        return {
            "multi_agent_task_result": result_data,
            "multi_agent_resume_action": "none",
            "multi_agent_resume_inputs": {},
            "multi_agent_step_resume_decisions": {},
            "multi_agent_resume_ready": False,
            "multi_agent_clarification_extraction": {},
            "multi_agent_pending_prompt": prompt,
            "pending_prompt": prompt,
            "waiting_user_input": True,
            "current_agent": "multi_agent",
            "final_answer": prompt,
        }

    final_answer = str(
        task_result.final_answer
        or task_result.error_message
        or "多 Agent 任务已经结束，但没有生成可展示的回答。"
    )
    return {
        "multi_agent_task_result": result_data,
        "multi_agent_resume_action": "none",
        "multi_agent_resume_inputs": {},
        "multi_agent_step_resume_decisions": {},
        "multi_agent_resume_ready": False,
        "multi_agent_clarification_extraction": {},
        "multi_agent_pending_prompt": "",
        "pending_prompt": "",
        "waiting_user_input": False,
        "current_agent": "multi_agent",
        "final_answer": final_answer,
    }


def _extract_waiting_prompt(
    task_result: MultiAgentTaskResult,
) -> str:
    """
    从暂停任务中提取优先展示给用户的问题。

    参数含义：
        task_result:
            状态为 awaiting_input 的多 Agent 任务结果。

    返回值含义：
        str:
            优先返回调度器生成的整批提示；旧任务没有整批数据时，再返回
            metadata、Planner 或 Worker 提供的第一个非空等待提示。
    """

    # 新版调度器会把同批次全部等待步骤整理成一个澄清包。
    clarification_bundle = task_result.metadata.get(
        "clarification_bundle"
    )
    if isinstance(clarification_bundle, Mapping):
        bundle_prompt = str(
            clarification_bundle.get("display_prompt") or ""
        ).strip()
        if bundle_prompt:
            return bundle_prompt

    metadata_prompt = str(
        task_result.metadata.get("clarification_prompt") or ""
    ).strip()
    if metadata_prompt:
        return metadata_prompt
    plan_prompt = str(task_result.plan.clarification_prompt or "").strip()
    if plan_prompt:
        return plan_prompt

    # 兼容尚未保存整批澄清数据的旧 Checkpoint。
    for result in task_result.task_results:
        if result.status == "awaiting_input":
            prompt = str(result.clarification_prompt or "").strip()
            if prompt:
                return prompt
    raise ValueError("awaiting_input 多 Agent 任务缺少等待提示")
