"""等待任务与本轮输入关系的主图状态适配器。"""

from __future__ import annotations

from collections.abc import Mapping
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


def resolve_pending_task_relation(
    state: Mapping[str, Any],
    *,
    skill_runtime: SkillRuntime | None = None,
) -> dict[str, Any]:
    """
    在各业务恢复适配器运行前判断本轮输入和等待任务的关系。

    功能：
        找出 Checkpoint 恢复出的 Tool、Multi-Agent 或 Skill 等待状态，
        再把本轮输入分类为继续、开始新任务、取消或无法判断。开始新任务
        和取消时会统一清理旧等待状态，避免旧任务继续拦截新问题。

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

    # 当前真正处于等待状态的业务模块，可能是工具、多智能体或 Skill。
    pending_kinds = _find_pending_task_kinds(state)
    if not pending_kinds:
        return {
            "action": "none",
            "state_update": {},
        }

    user_input = str(state.get("question") or "").strip()
    initial_decision = classify_pending_task_relation(user_input)

    # 同时存在多个任务时，明确取消或明确开始新任务可以安全地清理全部旧状态。
    if (
        len(pending_kinds) > 1
        and initial_decision.relation not in {"cancel", "new_task"}
    ):
        pending_kind: PendingTaskKind = "multiple"
        decision = TaskRelationDecision(
            relation="ambiguous",
            normalized_input=user_input,
            confidence=0.0,
            reason="同一会话同时存在多个等待任务，无法安全判断恢复目标。",
            source="fallback",
        )
    else:
        pending_kind = pending_kinds[0]
        decision = _resolve_contextual_relation(
            state=state,
            pending_kind=pending_kind,
            initial_decision=initial_decision,
            skill_runtime=skill_runtime,
        )

    common_update = {
        "question": decision.normalized_input,
        "task_relation_decision": decision.model_dump(mode="python"),
        "task_relation_pending_kind": pending_kind,
        "task_relation_requires_confirmation": False,
    }
    if pending_kind == "skill" and decision.relation == "resume":
        common_update.update(
            _build_skill_degraded_execution_update(
                state=state,
                normalized_input=decision.normalized_input,
                raw_user_input=user_input,
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
            },
        }

    if decision.relation == "cancel":
        return {
            "action": "cancel",
            "state_update": {
                **_build_all_pending_cleanup_update("cancelled"),
                **common_update,
                "final_answer": "已取消上一条等待中的任务。",
            },
        }

    confirmation_prompt = _build_relation_confirmation_prompt(
        pending_kind=pending_kind,
        pending_prompt=_find_pending_prompt(state, pending_kind),
    )
    return {
        "action": "ambiguous",
        "state_update": {
            **common_update,
            "task_relation_requires_confirmation": True,
            "pending_prompt": confirmation_prompt,
            "waiting_user_input": True,
            "final_answer": confirmation_prompt,
        },
    }


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
