"""等待任务与本轮输入关系的主图状态适配器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from src.agents.tool_agent.adapters.clarification_resume_adapter import (
    build_clarification_cleanup_update,
    match_clarification_candidate,
)
from src.runtime.resume.task_relation import (
    TaskRelationDecision,
    classify_pending_task_relation,
)


PendingTaskKind = Literal[
    "tool",
    "multi_agent",
    "skill",
    "multiple",
]


def resolve_pending_task_relation(
    state: Mapping[str, Any],
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
        )

    common_update = {
        "question": decision.normalized_input,
        "task_relation_decision": decision.model_dump(mode="python"),
        "task_relation_pending_kind": pending_kind,
        "task_relation_requires_confirmation": False,
    }

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
) -> TaskRelationDecision:
    """
    使用等待任务已有的结构化信息补充通用关系判断。

    功能：
        通用规则无法判断时，检查 Tool 澄清请求里的候选参数。用户输入
        唯一命中候选值时，可以确定它是在继续工具任务。

    参数含义：
        state:
            当前主图状态。
        pending_kind:
            当前等待任务所属模块。
        initial_decision:
            不考虑业务上下文时得到的第一轮判断。

    返回值含义：
        TaskRelationDecision:
            加入确定性业务证据后的任务关系判断。
    """

    if initial_decision.relation != "ambiguous" or pending_kind != "tool":
        return initial_decision

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
    return initial_decision

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
        "multi_agent_resume_ready": False,
        "multi_agent_pending_prompt": "",
        "skill_runtime_result": {},
        "skill_selected_id": "",
        "skill_inputs": {},
        "skill_status": "no_skill",
        "skill_pending_prompt": "",
        "skill_context": "",
        "skill_original_question": "",
        "skill_target_agent": "",
        "retrieval_question": "",
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
