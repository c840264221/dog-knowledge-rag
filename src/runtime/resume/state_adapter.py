"""等待任务与本轮输入关系的主图状态适配器。"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Literal

from src.agents.tool_agent.adapters.clarification_resume_adapter import (
    build_clarification_cleanup_update,
    match_clarification_candidate,
)
from src.agents.collaboration.adapters.resume_input_adapter import (
    MULTI_AGENT_DEGRADED_INPUTS,
)
from src.runtime.resume.task_relation import (
    TaskRelationDecision,
    classify_pending_task_relation,
)
from src.runtime.resume.pending_tasks import (
    MultiAgentPendingPayload,
    PendingInputContract,
    PendingTaskCollection,
    PendingTaskSnapshot,
    PendingTaskStatus,
    PendingTaskType,
    SkillPendingPayload,
    ToolPendingPayload,
)
from src.skills import SkillRuntime, build_default_skill_runtime


PendingTaskKind = Literal[
    "tool",
    "multi_agent",
    "skill",
    "multiple",
]


_SKILL_DEGRADED_EXECUTION_INPUTS = {
    "简化执行",
    "简化运行",
    "按现有信息继续",
    "使用现有信息继续",
}

_TASK_SELECTION_PATTERN = re.compile(
    r"^(?:(?:选择|继续)任务\s*[:：]?\s*)?(\d+)$"
)
_DIRECT_CANCEL_SELECTION_PATTERN = re.compile(
    r"^取消任务\s*[:：]?\s*(\d+)$"
)
_CANCEL_ALL_INPUTS = {
    "全部取消",
    "取消全部",
    "全部停止",
    "cancel all",
}

_TOOL_RESUME_STATE_KEYS = (
    "tool_agent_clarification_request",
    "tool_agent_pending_tool_call",
    "tool_agent_pending_original_question",
    "tool_agent_pending_created_at",
)
_MULTI_AGENT_RESUME_STATE_KEYS = (
    "multi_agent_task_result",
    "multi_agent_resume_action",
    "multi_agent_resume_inputs",
    "multi_agent_step_resume_decisions",
    "multi_agent_resume_ready",
    "multi_agent_clarification_extraction",
    "multi_agent_pending_prompt",
)
_SKILL_RESUME_STATE_KEYS = (
    "skill_runtime_result",
    "skill_selected_id",
    "skill_inputs",
    "skill_status",
    "skill_pending_prompt",
    "skill_context",
    "skill_original_question",
    "skill_target_agent",
    "skill_execution_mode",
    "skill_ignored_input_ids",
    "skill_degradation_reason",
    "skill_degradation_user_input",
    "retrieval_question",
    "memory_retrieval_text",
    "pet_profile_recall_result",
    "pet_profile_suggested_attributes",
    "skill_required_pet_profile_attributes",
    "skill_profile_recall_result",
    "skill_profile_access_decision",
    "answer_profile_access_decision",
    "dog_query_understanding_result",
)


def resolve_pending_task_relation(
    state: Mapping[str, Any],
    *,
    skill_runtime: SkillRuntime | None = None,
) -> dict[str, Any]:
    """
    在各业务恢复适配器运行前判断本轮输入和等待任务的关系。

    功能：
        找出 Checkpoint 恢复出的 Tool、Multi-Agent 或 Skill 等待状态，
        再把本轮输入分类为继续、开始新任务、取消或无法判断。多个任务
        无法唯一匹配时要求用户选择；定向取消只清理目标任务，明确全部
        取消时才统一清理所有等待状态。

    参数含义：
        state:
            已恢复等待任务白名单字段的本轮主图状态。
        skill_runtime:
            可选的 Skill 运行器，用于判断本轮输入是否补充了等待技能所缺字段。

    返回值含义：
        dict[str, Any]:
            action 表示任务关系；state_update 是可以合并回 DogState 的
            局部状态。没有等待任务时 action 为 none。
    """

    # 先恢复统一集合，再把尚未登记的旧业务等待字段增量同步进去。这样即使
    # 新问题执行时隔离了旧字段，挂起任务仍能从自己的 Payload 中被发现。
    legacy_pending_kinds = _find_pending_task_kinds(state)
    legacy_candidates = _build_pending_task_candidates(
        state=state,
        pending_kinds=legacy_pending_kinds,
    )
    pending_task_collection = _build_pending_task_collection(
        state=state,
        pending_kinds=legacy_pending_kinds,
        candidates=legacy_candidates,
    )
    pending_candidates = _build_collection_task_candidates(
        pending_task_collection
    )
    if not pending_candidates:
        return {
            "action": "none",
            "state_update": {},
        }

    user_input = str(state.get("question") or "").strip()
    pending_kinds = [
        candidate["task_kind"]
        for candidate in pending_candidates
    ]
    direct_cancel_candidate = _resolve_direct_cancel_selection(
        current_candidates=pending_candidates,
        user_input=user_input,
    )
    selected_candidate = (
        direct_cancel_candidate
        or _resolve_saved_task_selection(
            state=state,
            current_candidates=pending_candidates,
            user_input=user_input,
        )
    )
    saved_unassigned_input = str(
        state.get("task_relation_unassigned_input") or ""
    ).strip()
    saved_selection_action = str(
        state.get("task_relation_selection_action") or "resume"
    ).strip()
    cancel_all = user_input.casefold() in _CANCEL_ALL_INPUTS
    if selected_candidate is not None:
        pending_kind = selected_candidate["task_kind"]
        selected_action = (
            "cancel"
            if direct_cancel_candidate is not None
            or saved_selection_action == "cancel"
            else "resume"
        )
        selected_input = (
            saved_unassigned_input
            if selected_action == "resume"
            else "取消"
        )
        decision = TaskRelationDecision(
            relation=selected_action,
            normalized_input=selected_input,
            confidence=1.0,
            reason=(
                "用户已明确选择要取消的等待任务。"
                if selected_action == "cancel"
                else "用户已明确选择要恢复的等待任务。"
            ),
            source="explicit",
            selected_task_id=selected_candidate["task_id"],
            candidate_task_ids=[
                candidate["task_id"]
                for candidate in pending_candidates
            ],
        )
    else:
        initial_decision = classify_pending_task_relation(user_input)
        if (
            saved_unassigned_input
            and initial_decision.relation not in {"cancel", "new_task"}
        ):
            pending_kind = "multiple"
            decision = TaskRelationDecision(
                relation="ambiguous",
                normalized_input=saved_unassigned_input,
                confidence=0.0,
                reason="用户尚未选择有效的等待任务编号。",
                source="fallback",
                candidate_task_ids=[
                    candidate["task_id"]
                    for candidate in pending_candidates
                ],
                requires_task_selection=True,
            )
        elif (
            len(pending_kinds) > 1
            and initial_decision.relation == "cancel"
            and not cancel_all
        ):
            pending_kind = "multiple"
            decision = TaskRelationDecision(
                relation="ambiguous",
                normalized_input="取消",
                confidence=0.0,
                reason="存在多个等待任务，需要用户明确选择取消目标。",
                source="fallback",
                candidate_task_ids=[
                    candidate["task_id"]
                    for candidate in pending_candidates
                ],
                requires_task_selection=True,
            )
        elif (
            len(pending_kinds) > 1
            and initial_decision.relation not in {"cancel", "new_task"}
        ):
            matched_candidates = _find_contextually_matched_candidates(
                state=state,
                candidates=pending_candidates,
                collection=pending_task_collection,
                initial_decision=initial_decision,
                skill_runtime=skill_runtime,
            )
            if len(matched_candidates) == 1:
                selected_candidate = matched_candidates[0]
                pending_kind = selected_candidate["task_kind"]
                decision = TaskRelationDecision(
                    relation="resume",
                    normalized_input=user_input,
                    confidence=1.0,
                    reason="输入契约只匹配一个等待任务，允许定向恢复。",
                    source="rule",
                    selected_task_id=selected_candidate["task_id"],
                    candidate_task_ids=[
                        candidate["task_id"]
                        for candidate in pending_candidates
                    ],
                )
            else:
                pending_kind = "multiple"
                decision = TaskRelationDecision(
                    relation="ambiguous",
                    normalized_input=user_input,
                    confidence=0.0,
                    reason="多个等待任务无法通过确定性契约唯一匹配。",
                    source="fallback",
                    candidate_task_ids=[
                        candidate["task_id"]
                        for candidate in pending_candidates
                    ],
                    requires_task_selection=True,
                )
        else:
            pending_kind = pending_kinds[0]
            single_candidate = pending_candidates[0]
            decision = _resolve_contextual_relation(
                state=_build_task_resume_state(
                    state=state,
                    task=pending_task_collection.require(
                        single_candidate["task_id"]
                    ),
                ),
                pending_kind=pending_kind,
                initial_decision=initial_decision,
                skill_runtime=skill_runtime,
            )

    if len(pending_kinds) == 1 and selected_candidate is None:
        pending_kind = pending_kinds[0]
        if decision.relation in {"resume", "cancel"}:
            selected_candidate = pending_candidates[0]
            decision = decision.model_copy(
                update={
                    "selected_task_id": selected_candidate["task_id"],
                    "candidate_task_ids": [selected_candidate["task_id"]],
                }
            )

    selected_resume_update: dict[str, Any] = {}
    if decision.relation == "resume" and selected_candidate is not None:
        restored_selected_state = _build_task_resume_state(
            state={},
            task=pending_task_collection.require(
                selected_candidate["task_id"]
            ),
        )
        selected_resume_update = {
            key: value
            for key, value in restored_selected_state.items()
            if state.get(key) != value
        }

    common_update = {
        **selected_resume_update,
        "question": decision.normalized_input,
        "task_relation_decision": decision.model_dump(mode="python"),
        "task_relation_pending_kind": pending_kind,
        "task_relation_requires_confirmation": False,
        "task_relation_candidates": (
            pending_candidates
            if decision.requires_task_selection
            else []
        ),
        "task_relation_unassigned_input": "",
        "task_relation_selection_action": "",
        "pending_tasks": pending_task_collection.to_state(),
    }
    if pending_kind == "skill" and decision.relation == "resume":
        resolved_raw_user_input = (
            decision.normalized_input
            if selected_candidate is not None
            else user_input
        )
        common_update.update(
            _build_skill_degraded_execution_update(
                state={**dict(state), **selected_resume_update},
                normalized_input=decision.normalized_input,
                raw_user_input=resolved_raw_user_input,
            )
        )

    if decision.relation == "resume":
        return {
            "action": "resume",
            "state_update": common_update,
        }

    if decision.relation == "new_task":
        return {
            "action": "new_task",
            "state_update": {
                **_build_all_pending_cleanup_update("new_question"),
                **common_update,
                "pending_tasks": pending_task_collection.to_state(),
            },
        }

    if decision.relation == "cancel":
        targeted_cancel = (
            not cancel_all
            and pending_kind != "multiple"
        )
        cleanup_update = (
            _build_pending_task_cleanup_update(
                pending_kind=pending_kind,
                action="cancelled",
            )
            if targeted_cancel
            else _build_all_pending_cleanup_update("cancelled")
        )
        if targeted_cancel:
            selected_task_id = str(
                decision.selected_task_id
                or (
                    selected_candidate["task_id"]
                    if selected_candidate is not None
                    else ""
                )
            ).strip()
            selected_task = pending_task_collection.require(
                selected_task_id
            )
            pending_task_collection.transition(
                task_id=selected_task.task_id,
                target_status="cancelled",
                expected_version=selected_task.version,
            )
        else:
            _transition_all_pending_tasks(
                collection=pending_task_collection,
                target_status="cancelled",
            )
        cancelled_title = (
            selected_candidate["title"]
            if selected_candidate is not None
            else "上一条等待中的任务"
        )
        return {
            "action": "cancel",
            "state_update": {
                **cleanup_update,
                **common_update,
                "pending_tasks": pending_task_collection.to_state(),
                "final_answer": (
                    "已取消全部等待任务。"
                    if not targeted_cancel
                    else (
                        f"已取消{cancelled_title}，其他等待任务仍保留。"
                        if len(pending_kinds) > 1
                        else "已取消上一条等待中的任务。"
                    )
                ),
            },
        }

    if decision.requires_task_selection:
        confirmation_prompt = _build_task_selection_prompt(
            user_input=decision.normalized_input,
            candidates=pending_candidates,
            selection_action=(
                "cancel"
                if decision.normalized_input == "取消"
                else "resume"
            ),
        )
    else:
        selected_pending_prompt = (
            selected_candidate["pending_prompt"]
            if selected_candidate is not None
            else (
                pending_candidates[0]["pending_prompt"]
                if len(pending_candidates) == 1
                else _find_pending_prompt(state, pending_kind)
            )
        )
        confirmation_prompt = _build_relation_confirmation_prompt(
            pending_kind=pending_kind,
            pending_prompt=selected_pending_prompt,
        )
    return {
        "action": "ambiguous",
        "state_update": {
            **common_update,
            "task_relation_requires_confirmation": True,
            "task_relation_candidates": pending_candidates,
            "task_relation_unassigned_input": (
                decision.normalized_input
                if decision.requires_task_selection
                else ""
            ),
            "task_relation_selection_action": (
                "cancel"
                if decision.normalized_input == "取消"
                else "resume"
            ),
            "pending_prompt": confirmation_prompt,
            "waiting_user_input": True,
            "final_answer": confirmation_prompt,
        },
    }


def _build_pending_task_candidates(
    *,
    state: Mapping[str, Any],
    pending_kinds: list[PendingTaskKind],
) -> list[dict[str, str]]:
    """
    把现有模块等待字段投影成统一任务候选。

    参数含义：
        state:
            当前主图状态。
        pending_kinds:
            已确认正在等待输入的模块列表。

    返回值含义：
        list[dict[str, str]]:
            按稳定模块顺序排列的任务编号、类型、标题和提示。
    """

    candidates: list[dict[str, str]] = []
    for pending_kind in pending_kinds:
        task_id, title = _resolve_pending_task_identity(
            state=state,
            pending_kind=pending_kind,
        )
        candidates.append(
            {
                "task_id": task_id,
                "task_kind": pending_kind,
                "title": title,
                "pending_prompt": _find_pending_prompt(
                    state,
                    pending_kind,
                ),
            }
        )
    return candidates


def _build_pending_task_collection(
    *,
    state: Mapping[str, Any],
    pending_kinds: list[PendingTaskKind],
    candidates: list[dict[str, str]],
) -> PendingTaskCollection:
    """
    把现有业务等待字段同步为统一活动任务集合。

    功能：
        优先恢复 DogState 中已经存在的统一任务快照，并保留暂时没有投影到
        全局业务字段的挂起任务；尚未登记的旧 Tool、Skill 或 Multi-Agent
        等待状态会注册进集合。已登记任务仍在等待时，使用最新旧业务字段
        刷新它自己的恢复 Payload。

    参数含义：
        state:
            当前主图状态，包含旧业务等待字段和可选 pending_tasks。
        pending_kinds:
            当前确实处于等待输入状态的业务模块列表。
        candidates:
            根据旧业务字段生成的稳定任务编号和展示信息。

    返回值含义：
        PendingTaskCollection:
            与当前旧业务等待状态一致的活动任务集合。
    """

    raw_tasks = state.get("pending_tasks")
    collection = PendingTaskCollection.from_state(
        raw_tasks if isinstance(raw_tasks, Mapping) else None
    )
    candidates_by_kind = {
        candidate["task_kind"]: candidate
        for candidate in candidates
    }

    for pending_kind in pending_kinds:
        candidate = candidates_by_kind[pending_kind]
        latest_task = _build_pending_task_snapshot(
            state=state,
            candidate=candidate,
        )
        existing_task = collection.get(candidate["task_id"])
        if existing_task is None:
            collection.register(latest_task)
            continue
        if existing_task.status != "awaiting_input":
            continue
        if (
            existing_task.pending_prompt == latest_task.pending_prompt
            and existing_task.input_contracts == latest_task.input_contracts
            and existing_task.payload == latest_task.payload
        ):
            continue
        collection.refresh_waiting_task(
            latest_task,
            expected_version=existing_task.version,
        )
    return collection


def _build_collection_task_candidates(
    collection: PendingTaskCollection,
) -> list[dict[str, str]]:
    """
    从统一集合构建等待用户选择的活动任务候选。

    参数含义：
        collection:
            已恢复并同步旧业务状态的活动任务集合。

    返回值含义：
        list[dict[str, str]]:
            只包含 awaiting_input 任务的稳定候选列表。
    """

    return [
        {
            "task_id": task.task_id,
            "task_kind": task.task_kind,
            "title": task.title,
            "pending_prompt": task.pending_prompt,
        }
        for task in collection.list_tasks(status="awaiting_input")
    ]


def _build_pending_task_snapshot(
    *,
    state: Mapping[str, Any],
    candidate: Mapping[str, str],
) -> PendingTaskSnapshot:
    """
    从一个旧业务等待候选构建统一任务快照。

    参数含义：
        state:
            包含 Tool、Skill 或 Multi-Agent 旧等待字段的当前状态。
        candidate:
            已解析出的任务编号、类型、标题和等待提示。

    返回值含义：
        PendingTaskSnapshot:
            可以注册进 PendingTaskCollection 的 awaiting_input 快照。
    """

    task_kind = str(candidate["task_kind"])
    task_id = str(candidate["task_id"])
    prompt = str(candidate.get("pending_prompt") or "请补充任务所需信息。")
    user_id = str(state.get("user_id") or "legacy_user").strip()
    thread_id = str(state.get("session_id") or "legacy_thread").strip()

    if task_kind == "tool":
        raw_call = state.get("tool_agent_pending_tool_call")
        call = dict(raw_call) if isinstance(raw_call, Mapping) else {}
        raw_request = state.get("tool_agent_clarification_request")
        request = (
            dict(raw_request)
            if isinstance(raw_request, Mapping)
            else {}
        )
        missing_fields = [
            str(field_id)
            for field_id in request.get("missing_fields", [])
            if str(field_id).strip()
        ]
        raw_options = request.get("options")
        options = (
            dict(raw_options)
            if isinstance(raw_options, Mapping)
            else {}
        )
        contracts = [
            PendingInputContract(
                field_id=field_id,
                value_type="string",
                description=f"工具缺失参数 {field_id}",
                enum_values=[
                    str(option)
                    for option in options.get(field_id, [])
                ],
            )
            for field_id in missing_fields
        ]
        payload = ToolPendingPayload(
            tool_name=str(call.get("name") or "unknown_tool"),
            arguments=(
                dict(call.get("args"))
                if isinstance(call.get("args"), Mapping)
                else {}
            ),
            missing_fields=missing_fields,
            call_id=str(call.get("call_id") or "").strip() or None,
            resume_state=_capture_pending_resume_state(
                state=state,
                task_kind="tool",
            ),
        )
    elif task_kind == "multi_agent":
        raw_result = state.get("multi_agent_task_result")
        result = (
            dict(raw_result)
            if isinstance(raw_result, Mapping)
            else {}
        )
        contracts = [_build_generic_pending_input_contract(prompt)]
        payload = MultiAgentPendingPayload(
            collaboration_id=str(
                result.get("collaboration_id") or "pending"
            ),
            waiting_step_ids=_find_multi_agent_waiting_step_ids(result),
            resume_state=_capture_pending_resume_state(
                state=state,
                task_kind="multi_agent",
            ),
        )
    else:
        contracts = [_build_generic_pending_input_contract(prompt)]
        payload = SkillPendingPayload(
            skill_id=str(state.get("skill_selected_id") or "pending"),
            inputs=(
                dict(state.get("skill_inputs"))
                if isinstance(state.get("skill_inputs"), Mapping)
                else {}
            ),
            target_agent=str(
                state.get("skill_target_agent")
                or "dog_knowledge_agent"
            ),
            resume_state=_capture_pending_resume_state(
                state=state,
                task_kind="skill",
            ),
        )

    return PendingTaskSnapshot(
        task_id=task_id,
        task_kind=task_kind,
        user_id=user_id,
        thread_id=thread_id,
        title=str(candidate["title"]),
        pending_prompt=prompt,
        input_contracts=contracts,
        payload=payload,
    )


def _capture_pending_resume_state(
    *,
    state: Mapping[str, Any],
    task_kind: PendingTaskType,
) -> dict[str, Any]:
    """
    按业务类型白名单捕获等待任务恢复所需的 DogState 字段。

    参数含义：
        state:
            当前包含旧业务等待信息的主图状态。
        task_kind:
            需要捕获的 tool、skill 或 multi_agent 类型。

    返回值含义：
        dict[str, Any]:
            只包含对应业务恢复白名单字段的普通字典。
    """

    return {
        key: _copy_resume_value(state[key])
        for key in _get_resume_state_keys(task_kind)
        if key in state
    }


def _get_resume_state_keys(
    task_kind: PendingTaskType,
) -> tuple[str, ...]:
    """
    返回指定任务类型允许持久化和恢复的 DogState 字段白名单。

    参数含义：
        task_kind:
            tool、skill 或 multi_agent 任务类型。

    返回值含义：
        tuple[str, ...]:
            该业务恢复链路允许使用的固定字段名称。
    """

    return {
        "tool": _TOOL_RESUME_STATE_KEYS,
        "multi_agent": _MULTI_AGENT_RESUME_STATE_KEYS,
        "skill": _SKILL_RESUME_STATE_KEYS,
    }[task_kind]


def _copy_resume_value(value: Any) -> Any:
    """
    复制可写入 Checkpoint 的常用容器，避免复用可变状态引用。

    参数含义：
        value:
            DogState 中待保存的标量、字典或列表。

    返回值含义：
        Any:
            字典和列表的递归副本；其他不可变值原样返回。
    """

    if isinstance(value, Mapping):
        return {
            str(key): _copy_resume_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_resume_value(item) for item in value]
    return value


def _build_task_resume_state(
    *,
    state: Mapping[str, Any],
    task: PendingTaskSnapshot,
) -> dict[str, Any]:
    """
    把选中任务的类型化 Payload 恢复为本轮业务状态。

    参数含义：
        state:
            当前主图状态，作为恢复结果的基础。
        task:
            用户明确选择的 awaiting_input 任务。

    返回值含义：
        dict[str, Any]:
            合并目标任务恢复白名单字段后的新状态字典。
    """

    raw_resume_state = task.payload.resume_state
    allowed_keys = frozenset(_get_resume_state_keys(task.task_kind))
    return {
        **dict(state),
        **{
            key: _copy_resume_value(value)
            for key, value in raw_resume_state.items()
            if key in allowed_keys
        },
    }


def _build_generic_pending_input_contract(
    prompt: str,
) -> PendingInputContract:
    """
    为尚未统一字段契约的旧模块创建最小自然语言输入契约。

    参数含义：
        prompt:
            旧业务模块向用户展示的补充问题。

    返回值含义：
        PendingInputContract:
            要求补充字符串输入的最小契约。
    """

    return PendingInputContract(
        field_id="user_input",
        value_type="string",
        description=prompt,
    )


def _find_multi_agent_waiting_step_ids(
    raw_result: Mapping[str, Any],
) -> list[str]:
    """
    从旧多智能体结果中提取等待输入的步骤编号。

    参数含义：
        raw_result:
            MultiAgentTaskResult 的普通字典形式。

    返回值含义：
        list[str]:
            状态为 awaiting_input 的步骤编号列表。
    """

    raw_step_results = raw_result.get("step_results")
    if not isinstance(raw_step_results, list):
        return []
    return [
        str(step.get("step_id"))
        for step in raw_step_results
        if isinstance(step, Mapping)
        and str(step.get("status") or "").strip() == "awaiting_input"
        and str(step.get("step_id") or "").strip()
    ]


def transition_pending_task_kind(
    *,
    raw_tasks: Mapping[str, Any] | None,
    task_kind: PendingTaskType,
    target_status: PendingTaskStatus,
    task_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    迁移指定业务类型的唯一活动任务并返回可写入 DogState 的快照。

    参数含义：
        raw_tasks:
            DogState 中的统一活动任务普通字典。
        task_kind:
            需要迁移的 tool、skill 或 multi_agent 类型。
        target_status:
            状态机允许的目标状态。
        task_id:
            已由任务关系门禁明确选中的任务编号；为空时仅兼容同类型唯一任务。

    返回值含义：
        dict[str, dict[str, Any]]:
            状态迁移后的活动任务集合普通字典。
    """

    collection = PendingTaskCollection.from_state(raw_tasks)
    normalized_task_id = str(task_id or "").strip()
    if normalized_task_id:
        selected_task = collection.require(normalized_task_id)
        if selected_task.task_kind != task_kind:
            raise ValueError("选中任务的 task_kind 与业务恢复类型不一致")
        matched_tasks = [selected_task]
    else:
        matched_tasks = [
            task
            for task in collection.list_tasks()
            if task.task_kind == task_kind
        ]
    if not matched_tasks:
        return collection.to_state()
    if len(matched_tasks) != 1:
        raise ValueError(
            f"当前旧业务链路无法映射多个同类型等待任务: {task_kind}"
        )
    task = matched_tasks[0]
    collection.transition(
        task_id=task.task_id,
        target_status=target_status,
        expected_version=task.version,
    )
    return collection.to_state()


def _transition_all_pending_tasks(
    *,
    collection: PendingTaskCollection,
    target_status: PendingTaskStatus,
) -> None:
    """
    把集合中的全部活动任务迁移到同一个目标状态。

    参数含义：
        collection:
            已与旧等待字段同步的活动任务集合。
        target_status:
            状态机允许的目标状态。

    返回值含义：
        None:
            成功迁移后原地更新集合；非法迁移时抛出领域异常。
    """

    for task in collection.list_tasks():
        collection.transition(
            task_id=task.task_id,
            target_status=target_status,
            expected_version=task.version,
        )


def _resolve_pending_task_identity(
    *,
    state: Mapping[str, Any],
    pending_kind: PendingTaskKind,
) -> tuple[str, str]:
    """
    为旧模块等待状态生成稳定任务编号和用户可读标题。

    参数含义：
        state:
            当前主图状态。
        pending_kind:
            等待任务所属模块。

    返回值含义：
        tuple[str, str]:
            第一项是稳定任务编号，第二项是展示标题。
    """

    if pending_kind == "tool":
        raw_call = state.get("tool_agent_pending_tool_call")
        tool_name = (
            str(raw_call.get("name") or "tool").strip()
            if isinstance(raw_call, Mapping)
            else "tool"
        )
        return f"tool:{tool_name}", f"补充工具 {tool_name} 的参数"
    if pending_kind == "multi_agent":
        raw_result = state.get("multi_agent_task_result")
        collaboration_id = (
            str(raw_result.get("collaboration_id") or "pending").strip()
            if isinstance(raw_result, Mapping)
            else "pending"
        )
        return (
            f"multi_agent:{collaboration_id}",
            "补充多智能体协作任务信息",
        )
    selected_skill_id = str(
        state.get("skill_selected_id") or "pending"
    ).strip()
    return (
        f"skill:{selected_skill_id}",
        f"补充 Skill 技能 {selected_skill_id} 的输入",
    )


def _resolve_saved_task_selection(
    *,
    state: Mapping[str, Any],
    current_candidates: list[dict[str, str]],
    user_input: str,
) -> dict[str, str] | None:
    """
    解析用户对上一轮多任务候选列表作出的选择。

    参数含义：
        state:
            包含上一轮候选列表和未绑定输入的当前状态。
        current_candidates:
            根据仍有效的业务等待字段重新生成的候选列表。
        user_input:
            用户本轮用于选择任务的原始文本。

    返回值含义：
        dict[str, str] | None:
            选择合法且候选仍有效时返回目标任务，否则返回 None。
    """

    unassigned_input = str(
        state.get("task_relation_unassigned_input") or ""
    ).strip()
    raw_saved_candidates = state.get("task_relation_candidates")
    if not unassigned_input or not isinstance(raw_saved_candidates, list):
        return None

    match = _TASK_SELECTION_PATTERN.fullmatch(user_input)
    if match is None:
        return None
    selected_index = int(match.group(1)) - 1
    if selected_index < 0 or selected_index >= len(raw_saved_candidates):
        return None

    raw_selected = raw_saved_candidates[selected_index]
    if not isinstance(raw_selected, Mapping):
        return None
    selected_task_id = str(raw_selected.get("task_id") or "").strip()
    return next(
        (
            candidate
            for candidate in current_candidates
            if candidate["task_id"] == selected_task_id
        ),
        None,
    )


def _resolve_direct_cancel_selection(
    *,
    current_candidates: list[dict[str, str]],
    user_input: str,
) -> dict[str, str] | None:
    """
    解析“取消任务：编号”形式的单轮定向取消指令。

    参数含义：
        current_candidates:
            当前仍然有效的等待任务候选。
        user_input:
            用户本轮原始输入。

    返回值含义：
        dict[str, str] | None:
            编号合法时返回目标候选；格式或编号无效时返回 None。
    """

    match = _DIRECT_CANCEL_SELECTION_PATTERN.fullmatch(user_input)
    if match is None:
        return None
    selected_index = int(match.group(1)) - 1
    if selected_index < 0 or selected_index >= len(current_candidates):
        return None
    return current_candidates[selected_index]


def _find_contextually_matched_candidates(
    *,
    state: Mapping[str, Any],
    candidates: list[dict[str, str]],
    collection: PendingTaskCollection,
    initial_decision: TaskRelationDecision,
    skill_runtime: SkillRuntime | None,
) -> list[dict[str, str]]:
    """
    使用各业务现有确定性契约筛选可以安全恢复的候选任务。

    参数含义：
        state:
            当前主图状态。
        candidates:
            当前所有等待任务候选。
        collection:
            保存每个候选独立恢复 Payload 的统一活动任务集合。
        initial_decision:
            通用任务关系分类结果。
        skill_runtime:
            可选的技能运行器。

    返回值含义：
        list[dict[str, str]]:
            契约明确判定为 resume 的任务候选；多个命中时仍不得自动执行。
    """

    matched_candidates: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_state = _build_task_resume_state(
            state=state,
            task=collection.require(candidate["task_id"]),
        )
        contextual_decision = _resolve_contextual_relation(
            state=candidate_state,
            pending_kind=candidate["task_kind"],
            initial_decision=initial_decision,
            skill_runtime=skill_runtime,
        )
        if contextual_decision.relation == "resume":
            matched_candidates.append(candidate)
    return matched_candidates


def _build_task_selection_prompt(
    *,
    user_input: str,
    candidates: list[dict[str, str]],
    selection_action: Literal["resume", "cancel"],
) -> str:
    """
    构建多个等待任务无法唯一匹配时的编号选择提示。

    参数含义：
        user_input:
            尚未绑定到任何任务的用户输入。
        candidates:
            等待用户选择的任务候选列表。
        selection_action:
            本次编号选择用于继续任务还是取消任务。

    返回值含义：
        str:
            展示原始输入、候选任务和回复格式的中文提示。
    """

    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        prompt_suffix = (
            f"（原问题：{candidate['pending_prompt']}）"
            if candidate["pending_prompt"]
            else ""
        )
        candidate_lines.append(
            f"{index}. {candidate['title']}{prompt_suffix}"
        )
    if selection_action == "cancel":
        return (
            "当前存在多个等待任务，请选择要取消的任务。\n"
            + "\n".join(candidate_lines)
            + "\n请回复任务编号，例如“1”；如需全部取消，请回复“全部取消”。"
        )
    return (
        f"你输入的“{user_input}”可能属于多个等待任务，系统不会猜测执行。\n"
        + "\n".join(candidate_lines)
        + "\n请回复任务编号，例如“1”或“选择任务：1”。"
    )


def _resolve_contextual_relation(
    *,
    state: Mapping[str, Any],
    pending_kind: PendingTaskKind,
    initial_decision: TaskRelationDecision,
    skill_runtime: SkillRuntime | None,
) -> TaskRelationDecision:
    """
    使用等待任务已有的结构化信息补充通用关系判断。

    功能：
        通用规则无法判断时，优先读取对应业务已经保存的结构化信息。
        Tool 输入唯一命中候选值，或 Skill 输入补齐至少一个当前缺失字段时，
        都可以确定本轮是在继续旧任务。

    参数含义：
        state:
            当前主图状态。
        pending_kind:
            当前等待任务所属模块。
        initial_decision:
            不考虑业务上下文时得到的第一轮判断。
        skill_runtime:
            用来提取并检查等待 Skill 输入的运行器；为空时按需构建默认实例。

    返回值含义：
        TaskRelationDecision:
            加入确定性业务证据后的任务关系判断。
    """

    if initial_decision.relation != "ambiguous":
        return initial_decision

    if pending_kind == "tool":
        clarification = state.get("tool_agent_clarification_request")
        if (
            isinstance(clarification, Mapping)
            and match_clarification_candidate(
                user_input=initial_decision.normalized_input,
                clarification_request=clarification,
            )
            is not None
        ):
            return TaskRelationDecision(
                relation="resume",
                normalized_input=initial_decision.normalized_input,
                confidence=1.0,
                reason="用户输入唯一命中待补全工具参数的候选值。",
                source="rule",
            )

    if (
        pending_kind == "multi_agent"
        and any(
            keyword in initial_decision.normalized_input
            for keyword in MULTI_AGENT_DEGRADED_INPUTS
        )
    ):
        return TaskRelationDecision(
            relation="resume",
            normalized_input=initial_decision.normalized_input,
            confidence=1.0,
            reason=(
                "用户明确提交了等待中多智能体任务的简化执行控制指令。"
            ),
            source="explicit",
        )

    if pending_kind == "skill":
        return _resolve_pending_skill_relation(
            state=state,
            initial_decision=initial_decision,
            skill_runtime=skill_runtime,
        )
    return initial_decision


def _resolve_pending_skill_relation(
    *,
    state: Mapping[str, Any],
    initial_decision: TaskRelationDecision,
    skill_runtime: SkillRuntime | None,
) -> TaskRelationDecision:
    """
    判断模糊输入是否实际补充了等待 Skill 的必需字段。

    功能：
        使用检查点保存的技能编号和历史输入计算补充前的缺失字段，再从本轮
        用户文本提取技能输入并重新检查。只要缺失字段至少减少一个，就把
        本轮输入确定为继续旧任务；无法提取有效进展时维持模糊判断。

    参数含义：
        state:
            包含 skill_selected_id 和 skill_inputs 的当前主图状态。
        initial_decision:
            通用任务关系规则生成的模糊判断。
        skill_runtime:
            可复用的技能运行器；为空时创建项目默认技能运行器。

    返回值含义：
        TaskRelationDecision:
            有结构化补充证据时返回 resume，否则返回原始模糊判断。
    """

    selected_skill_id = str(
        state.get("skill_selected_id") or ""
    ).strip()
    if not selected_skill_id:
        return initial_decision

    if (
        initial_decision.normalized_input.casefold()
        in _SKILL_DEGRADED_EXECUTION_INPUTS
    ):
        return TaskRelationDecision(
            relation="resume",
            normalized_input=initial_decision.normalized_input,
            confidence=1.0,
            reason="用户明确选择按现有信息简化执行等待中的 Skill。",
            source="explicit",
        )

    raw_existing_inputs = state.get("skill_inputs")
    existing_inputs = (
        dict(raw_existing_inputs)
        if isinstance(raw_existing_inputs, Mapping)
        else {}
    )
    resolved_runtime = skill_runtime or build_default_skill_runtime()

    try:
        before_check = resolved_runtime.check_inputs(
            skill_id=selected_skill_id,
            provided_inputs=existing_inputs,
        )
        extraction = resolved_runtime.extract_inputs(
            skill_id=selected_skill_id,
            user_text=initial_decision.normalized_input,
            existing_inputs=existing_inputs,
        )
        after_check = resolved_runtime.check_inputs(
            skill_id=selected_skill_id,
            provided_inputs=extraction.merged_inputs,
        )
    except (LookupError, TypeError, ValueError):
        # 检查点中的技能编号或输入损坏时保持谨慎，不让门卫错误恢复旧任务。
        return initial_decision

    newly_supplied_input_ids = [
        input_id
        for input_id in before_check.missing_input_ids
        if input_id not in after_check.missing_input_ids
    ]
    if not newly_supplied_input_ids:
        return initial_decision

    return TaskRelationDecision(
        relation="resume",
        normalized_input=initial_decision.normalized_input,
        confidence=1.0,
        reason=(
            "用户本轮补充了等待 Skill 当前缺失的必需字段: "
            f"{newly_supplied_input_ids}。"
        ),
        source="rule",
    )


def _build_skill_degraded_execution_update(
    *,
    state: Mapping[str, Any],
    normalized_input: str,
    raw_user_input: str,
) -> dict[str, Any]:
    """
    根据等待中的技能检查结果构建简化执行状态。

    功能：
        只在用户明确选择简化执行，并且上一轮检查结果确认不存在强制缺失
        字段时，保存简化模式和允许忽略的字段。忽略范围完全来自系统生成的
        检查结果，不接受用户自行指定内部字段名。

    参数含义：
        state:
            包含上一轮 SkillRuntime 检查结果的主图状态。
        normalized_input:
            去除任务关系前缀后的本轮输入。
        raw_user_input:
            用户本轮未经任务关系处理的原始输入。

    返回值含义：
        dict[str, Any]:
            可以合并进 DogState 的简化执行字段；条件不满足时返回空字典。
    """

    if (
        normalized_input.casefold()
        not in _SKILL_DEGRADED_EXECUTION_INPUTS
    ):
        return {}

    raw_runtime_result = state.get("skill_runtime_result")
    if not isinstance(raw_runtime_result, Mapping):
        return {}
    raw_input_check = raw_runtime_result.get("input_check")
    if not isinstance(raw_input_check, Mapping):
        return {}
    if not bool(raw_input_check.get("can_run_degraded")):
        return {}

    raw_missing_ids = raw_input_check.get(
        "missing_degradable_input_ids"
    )
    if not isinstance(raw_missing_ids, list):
        return {}
    ignored_input_ids = [
        str(input_id).strip()
        for input_id in raw_missing_ids
        if str(input_id).strip()
    ]
    if not ignored_input_ids:
        return {}

    return {
        "skill_execution_mode": "degraded",
        "skill_ignored_input_ids": ignored_input_ids,
        "skill_degradation_reason": (
            "user_selected_degraded_execution"
        ),
        "skill_degradation_user_input": raw_user_input,
    }


def _find_pending_task_kinds(
    state: Mapping[str, Any],
) -> list[PendingTaskKind]:
    """
    查找当前状态中真正等待用户输入的业务模块。

    参数含义：
        state:
            当前主图状态。

    返回值含义：
        list[PendingTaskKind]:
            等待中的模块名称列表。
    """

    pending_kinds: list[PendingTaskKind] = []
    if (
        isinstance(state.get("tool_agent_clarification_request"), Mapping)
        and isinstance(state.get("tool_agent_pending_tool_call"), Mapping)
    ):
        pending_kinds.append("tool")

    raw_multi_agent_result = state.get("multi_agent_task_result")
    if (
        isinstance(raw_multi_agent_result, Mapping)
        and str(raw_multi_agent_result.get("status") or "").strip()
        == "awaiting_input"
    ):
        pending_kinds.append("multi_agent")

    if str(state.get("skill_status") or "").strip() == "awaiting_input":
        pending_kinds.append("skill")
    return pending_kinds


def _find_pending_prompt(
    state: Mapping[str, Any],
    pending_kind: PendingTaskKind,
) -> str:
    """
    提取旧任务原本向用户提出的补充问题。

    参数含义：
        state:
            当前主图状态。
        pending_kind:
            当前等待任务所属模块。

    返回值含义：
        str:
            原补充问题；没有找到时返回空字符串。
    """

    if pending_kind == "tool":
        clarification = state.get("tool_agent_clarification_request")
        if isinstance(clarification, Mapping):
            return str(clarification.get("question") or "").strip()
    if pending_kind == "multi_agent":
        return str(state.get("multi_agent_pending_prompt") or "").strip()
    if pending_kind == "skill":
        return str(state.get("skill_pending_prompt") or "").strip()
    return ""


def _build_pending_task_cleanup_update(
    *,
    pending_kind: PendingTaskKind,
    action: Literal["new_question", "cancelled"],
) -> dict[str, Any]:
    """
    构建只清理一个目标等待任务的状态更新。

    参数含义：
        pending_kind:
            需要清理的业务模块，只允许 tool、multi_agent 或 skill。
        action:
            清理原因，是开始新问题或用户取消。

    返回值含义：
        dict[str, Any]:
            仅包含目标模块等待字段和通用等待提示的局部状态更新。
    """

    common_cleanup = {
        "pending_prompt": "",
        "waiting_user_input": False,
    }
    if pending_kind == "tool":
        return {
            **build_clarification_cleanup_update(action=action),
            **common_cleanup,
        }
    if pending_kind == "multi_agent":
        return {
            "multi_agent_task_result": {},
            "multi_agent_resume_action": action,
            "multi_agent_resume_inputs": {},
            "multi_agent_step_resume_decisions": {},
            "multi_agent_resume_ready": False,
            "multi_agent_clarification_extraction": {},
            "multi_agent_pending_prompt": "",
            **common_cleanup,
        }
    if pending_kind == "skill":
        return {
            "skill_runtime_result": {},
            "skill_selected_id": "",
            "skill_inputs": {},
            "skill_status": "no_skill",
            "skill_pending_prompt": "",
            "skill_context": "",
            "skill_original_question": "",
            "skill_target_agent": "",
            "skill_execution_mode": "standard",
            "skill_ignored_input_ids": [],
            "skill_degradation_reason": "",
            "skill_degradation_user_input": "",
            "retrieval_question": "",
            "memory_retrieval_text": "",
            "pet_profile_recall_result": {},
            "pet_profile_suggested_attributes": [],
            "skill_required_pet_profile_attributes": [],
            "skill_profile_recall_result": {},
            "skill_profile_access_decision": {},
            "answer_profile_access_decision": {},
            "dog_query_understanding_result": {},
            **common_cleanup,
        }
    raise ValueError("定向清理必须指定单一等待任务类型")


def _build_all_pending_cleanup_update(
    action: Literal["new_question", "cancelled"],
) -> dict[str, Any]:
    """
    构建清理所有旧等待任务所需的主图字段。

    参数含义：
        action:
            清理原因，是开始新问题或取消旧任务。

    返回值含义：
        dict[str, Any]:
            Tool、Multi-Agent 和 Skill 等待字段的统一清理结果。
    """

    return {
        **build_clarification_cleanup_update(action=action),
        "multi_agent_task_result": {},
        "multi_agent_resume_action": action,
        "multi_agent_resume_inputs": {},
        "multi_agent_step_resume_decisions": {},
        "multi_agent_resume_ready": False,
        "multi_agent_clarification_extraction": {},
        "multi_agent_pending_prompt": "",
        "skill_runtime_result": {},
        "skill_selected_id": "",
        "skill_inputs": {},
        "skill_status": "no_skill",
        "skill_pending_prompt": "",
        "skill_context": "",
        "skill_original_question": "",
        "skill_target_agent": "",
        "skill_execution_mode": "standard",
        "skill_ignored_input_ids": [],
        "skill_degradation_reason": "",
        "skill_degradation_user_input": "",
        "retrieval_question": "",
        "memory_retrieval_text": "",
        "pet_profile_recall_result": {},
        "pet_profile_suggested_attributes": [],
        "skill_required_pet_profile_attributes": [],
        "skill_profile_recall_result": {},
        "skill_profile_access_decision": {},
        "answer_profile_access_decision": {},
        "dog_query_understanding_result": {},
        "pending_prompt": "",
        "waiting_user_input": False,
    }


def _build_relation_confirmation_prompt(
    *,
    pending_kind: PendingTaskKind,
    pending_prompt: str,
) -> str:
    """
    构建无法区分新旧任务时展示给用户的确认提示。

    参数含义：
        pending_kind:
            当前等待任务所属模块。
        pending_prompt:
            旧任务原本提出的补充问题。

    返回值含义：
        str:
            告诉用户如何明确继续、开始新问题或取消的提示。
    """

    kind_labels = {
        "tool": "工具调用",
        "multi_agent": "多智能体",
        "skill": "Skill 技能",
        "multiple": "多个",
    }
    original_prompt = (
        f"旧任务正在询问：{pending_prompt}\n"
        if pending_prompt
        else ""
    )
    return (
        f"检测到当前会话还有一个等待补充信息的{kind_labels[pending_kind]}任务。\n"
        f"{original_prompt}"
        "如果要继续，请回复“继续任务：你的补充内容”；"
        "如果要开始别的事情，请回复“新问题：你的新问题”；"
        "也可以回复“取消”。"
    )
