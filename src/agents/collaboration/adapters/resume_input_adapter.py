"""
多 Agent 跨轮恢复输入适配器。

功能：
    读取 Checkpoint 恢复出的暂停任务和本轮用户输入，判断用户是继续任务、
    取消任务、明确开始新问题，还是仍需补充多个等待步骤的回答。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from src.agents.collaboration.contracts import (
    MultiAgentStepResumeDecision,
    MultiAgentTaskResult,
)
from src.agents.collaboration.adapters.clarification_field_resolver import (
    MultiAgentClarificationFieldResolver,
    allocate_fields_to_steps,
    build_default_multi_agent_clarification_field_resolver,
)


MultiAgentResumeAction = Literal[
    "none",
    "resume",
    "replan",
    "cancelled",
    "new_question",
    "needs_clarification",
]

MULTI_AGENT_CANCEL_INPUTS = {
    "取消",
    "算了",
    "不继续了",
    "cancel",
}

MULTI_AGENT_NEW_QUESTION_PREFIXES = (
    "新问题:",
    "新问题：",
    "换个问题:",
    "换个问题：",
)

MULTI_AGENT_RESUME_PREFIXES = (
    "继续任务:",
    "继续任务：",
    "恢复任务:",
    "恢复任务：",
)

MULTI_AGENT_DEGRADED_INPUTS = (
    "简化执行",
    "简化运行",
)


async def resolve_multi_agent_resume_input(
    state: Mapping[str, Any],
    *,
    field_resolver: MultiAgentClarificationFieldResolver | None = None,
) -> dict[str, Any]:
    """
    判断本轮用户输入与暂停中的多 Agent 任务是什么关系。

    功能：
        没有暂停任务时不做处理；命中取消词或新问题前缀时清理旧任务；
        单个等待步骤直接接收本轮文本；多个等待步骤要求 JSON 对象按编号回答。

    参数含义：
        state:
            主图当前状态，需要包含 question 和可选 multi_agent_task_result。
        field_resolver:
            可选澄清字段解析器；不传时使用项目默认高精度规则，测试可以
            注入固定解析器。

    返回值含义：
        dict[str, Any]:
            包含 action 和 state_update。action 表示判断结果，state_update
            是后续主图节点需要合并回 DogState 的字段。
    """

    pending_result = _parse_pending_task_result(
        state.get("multi_agent_task_result")
    )
    if pending_result is None:
        return {
            "action": "none",
            "state_update": {},
        }

    user_input = str(state.get("question") or "").strip()
    if user_input.casefold() in MULTI_AGENT_CANCEL_INPUTS:
        return {
            "action": "cancelled",
            "state_update": _build_resume_cleanup_update("cancelled"),
        }

    new_question = _strip_first_prefix(
        user_input,
        MULTI_AGENT_NEW_QUESTION_PREFIXES,
    )
    if new_question is not None:
        return {
            "action": "new_question",
            "state_update": {
                **_build_resume_cleanup_update("new_question"),
                "question": new_question,
            },
        }

    awaiting_step_ids = [
        result.step_id
        for result in pending_result.task_results
        if result.status == "awaiting_input"
    ]
    if not awaiting_step_ids:
        if pending_result.plan.requires_user_input and user_input:
            return {
                "action": "replan",
                "state_update": {
                    "multi_agent_task_result": pending_result.model_dump(
                        mode="python"
                    ),
                    "multi_agent_resume_action": "replan",
                    "multi_agent_resume_inputs": {
                        "planner_clarification": user_input,
                    },
                    "multi_agent_step_resume_decisions": {},
                    "multi_agent_resume_ready": True,
                    "multi_agent_clarification_extraction": {},
                    "multi_agent_pending_prompt": "",
                    "pending_prompt": "",
                    "waiting_user_input": False,
                },
            }
        return {
            "action": "none",
            "state_update": _build_resume_cleanup_update("none"),
        }

    resume_text = _strip_first_prefix(
        user_input,
        MULTI_AGENT_RESUME_PREFIXES,
    )
    normalized_input = (
        resume_text
        if resume_text is not None
        else user_input
    ).strip()

    # 开发调试仍兼容旧版“步骤编号 -> 回答”JSON，不向普通用户强制展示。
    if len(awaiting_step_ids) > 1:
        parsed_inputs = _parse_multiple_step_inputs(normalized_input)
        if parsed_inputs is not None:
            expected_ids = set(awaiting_step_ids)
            if set(parsed_inputs) == expected_ids and all(
                str(value or "").strip()
                for value in parsed_inputs.values()
            ):
                return _build_ready_resume_update(
                    pending_result=pending_result,
                    user_inputs=parsed_inputs,
                )

    field_consumers = _load_field_consumers(pending_result)
    if (
        field_consumers
        and _field_consumers_cover_waiting_steps(
            field_consumers=field_consumers,
            awaiting_step_ids=awaiting_step_ids,
        )
    ):
        resolver = (
            field_resolver
            or build_default_multi_agent_clarification_field_resolver()
        )
        existing_fields = _load_existing_extracted_fields(state)
        # 简化执行属于控制指令，不包含需要 LLM 猜测的业务字段。仍运行
        # 确定性规则，以兼容“它6岁，这个步骤简化执行”这类混合输入。
        is_degraded_control = any(
            keyword in normalized_input.casefold()
            for keyword in MULTI_AGENT_DEGRADED_INPUTS
        )
        if is_degraded_control:
            extraction = resolver.extract(
                user_text=normalized_input,
                requested_field_ids=list(field_consumers),
                existing_fields=existing_fields,
            )
        else:
            extraction = await resolver.extract_layered(
                user_text=normalized_input,
                requested_field_ids=list(field_consumers),
                existing_fields=existing_fields,
                field_descriptions=_load_field_descriptions(
                    pending_result
                ),
            )
        step_inputs = allocate_fields_to_steps(
            resolved_fields=extraction.resolved_fields,
            field_consumers=field_consumers,
        )
        extraction_data = extraction.model_dump(mode="python")
        resume_decisions = _build_step_resume_decisions(
            state=state,
            pending_result=pending_result,
            awaiting_step_ids=awaiting_step_ids,
            step_inputs=step_inputs,
            missing_field_ids=extraction.missing_field_ids,
            ambiguous_field_ids=extraction.ambiguous_field_ids,
            user_input=normalized_input,
        )
        if all(
            decision.action != "keep_waiting"
            for decision in resume_decisions.values()
        ):
            return _build_ready_resume_update(
                pending_result=pending_result,
                user_inputs={
                    step_id: step_inputs.get(step_id, {})
                    for step_id in awaiting_step_ids
                },
                clarification_extraction=extraction_data,
                resume_decisions=resume_decisions,
            )

        prompt = _build_natural_clarification_prompt(
            pending_result=pending_result,
            missing_field_ids=_collect_waiting_field_ids(
                resume_decisions,
            ),
            ambiguous_field_ids=[
                field_id
                for field_id in extraction.ambiguous_field_ids
                if field_id in _collect_waiting_field_ids(
                    resume_decisions,
                )
            ],
        )
        prompt = _append_degraded_choice_hint(
            prompt=prompt,
            user_input=normalized_input,
            pending_result=pending_result,
            resume_decisions=resume_decisions,
        )
        return {
            "action": "needs_clarification",
            "state_update": {
                "multi_agent_resume_action": "needs_clarification",
                "multi_agent_resume_inputs": step_inputs,
                "multi_agent_step_resume_decisions": {
                    step_id: decision.model_dump(mode="python")
                    for step_id, decision in resume_decisions.items()
                },
                "multi_agent_resume_ready": False,
                "multi_agent_clarification_extraction": extraction_data,
                "multi_agent_pending_prompt": prompt,
                "pending_prompt": prompt,
                "waiting_user_input": True,
            },
        }

    if len(awaiting_step_ids) == 1 and normalized_input:
        return _build_ready_resume_update(
            pending_result=pending_result,
            user_inputs={awaiting_step_ids[0]: normalized_input},
        )

    prompt = _build_resume_clarification_prompt(
        pending_result=pending_result,
        awaiting_step_ids=awaiting_step_ids,
    )
    return {
        "action": "needs_clarification",
        "state_update": {
            "multi_agent_resume_action": "needs_clarification",
            "multi_agent_resume_inputs": {},
            "multi_agent_step_resume_decisions": {},
            "multi_agent_resume_ready": False,
            "multi_agent_clarification_extraction": {},
            "multi_agent_pending_prompt": prompt,
            "pending_prompt": prompt,
            "waiting_user_input": True,
        },
    }


def _build_step_resume_decisions(
    *,
    state: Mapping[str, Any],
    pending_result: MultiAgentTaskResult,
    awaiting_step_ids: list[str],
    step_inputs: Mapping[str, Mapping[str, Any]],
    missing_field_ids: list[str],
    ambiguous_field_ids: list[str],
    user_input: str,
) -> dict[str, MultiAgentStepResumeDecision]:
    """
    为每个等待步骤分别生成正常恢复、简化执行或继续等待决定。

    功能：
        根据整批澄清包保留的字段归属关系，把字段提取结果重新投影到每个
        步骤。硬必需字段不能忽略；可简化字段只有在用户明确选择后才能
        忽略。任一步骤继续等待时，严格批次门禁不会启动整批 Worker。

    参数含义：
        state:
            当前主图状态，用于读取前几轮已经保存的步骤决定。
        pending_result:
            Checkpoint 中恢复出的多智能体暂停任务。
        awaiting_step_ids:
            当前真正等待输入的步骤编号。
        step_inputs:
            字段解析后按消费者关系分配给各步骤的输入。
        missing_field_ids:
            当前多轮输入仍未提供的字段编号。
        ambiguous_field_ids:
            当前存在冲突、不能安全采用的字段编号。
        user_input:
            用户本轮原始输入，用于识别简化执行选择和审计。

    返回值含义：
        dict[str, MultiAgentStepResumeDecision]:
            步骤编号到独立恢复决定的映射。
    """

    requests_by_step_id = _load_step_requests_by_id(pending_result)
    unresolved_field_ids = set(missing_field_ids) | set(
        ambiguous_field_ids
    )
    previous_decisions = _load_existing_step_resume_decisions(state)
    degraded_step_ids = _select_degraded_step_ids(
        user_input=user_input,
        requests_by_step_id=requests_by_step_id,
        previous_decisions=previous_decisions,
    )

    decisions: dict[str, MultiAgentStepResumeDecision] = {}
    for step_id in awaiting_step_ids:
        request = requests_by_step_id.get(step_id, {})
        raw_fields = request.get("missing_fields", [])
        unresolved_fields = [
            field
            for field in raw_fields
            if isinstance(field, Mapping)
            and str(field.get("input_id") or "")
            in unresolved_field_ids
            and str(field.get("requirement_level") or "") != "optional"
        ] if isinstance(raw_fields, list) else []
        hard_required_ids = [
            str(field.get("input_id"))
            for field in unresolved_fields
            if str(field.get("requirement_level") or "")
            in {"hard_required", "unknown"}
        ]
        degradable_ids = [
            str(field.get("input_id"))
            for field in unresolved_fields
            if str(field.get("requirement_level") or "") == "degradable"
        ]
        provided_inputs = dict(step_inputs.get(step_id, {}))

        if not unresolved_fields:
            decisions[step_id] = MultiAgentStepResumeDecision(
                step_id=step_id,
                action="resume",
                provided_inputs=provided_inputs,
                reason="当前步骤需要的输入已经齐全。",
                user_input=user_input,
            )
            continue

        if not hard_required_ids and degradable_ids and (
            step_id in degraded_step_ids
        ):
            decisions[step_id] = MultiAgentStepResumeDecision(
                step_id=step_id,
                action="degraded",
                provided_inputs=provided_inputs,
                ignored_input_ids=degradable_ids,
                reason="用户明确同意忽略当前步骤缺少的可简化输入。",
                user_input=user_input,
            )
            continue

        waiting_input_ids = [
            str(field.get("input_id"))
            for field in unresolved_fields
        ]
        decisions[step_id] = MultiAgentStepResumeDecision(
            step_id=step_id,
            action="keep_waiting",
            provided_inputs=provided_inputs,
            waiting_input_ids=waiting_input_ids,
            reason=(
                "当前步骤仍缺少不能忽略的必需输入。"
                if hard_required_ids
                else "当前步骤仍缺少输入，且用户尚未明确选择简化执行。"
            ),
            user_input=user_input,
        )
    return decisions


def _load_step_requests_by_id(
    pending_result: MultiAgentTaskResult,
) -> dict[str, Mapping[str, Any]]:
    """
    从整批澄清包读取步骤编号到澄清请求的映射。

    参数含义：
        pending_result:
            当前等待恢复的多智能体任务结果。

    返回值含义：
        dict[str, Mapping[str, Any]]:
            合法步骤编号到原始澄清请求字典的映射。
    """

    bundle = pending_result.metadata.get("clarification_bundle")
    if not isinstance(bundle, Mapping):
        return {}
    raw_requests = bundle.get("step_requests")
    if not isinstance(raw_requests, list):
        return {}
    return {
        str(request.get("step_id") or "").strip(): request
        for request in raw_requests
        if isinstance(request, Mapping)
        and str(request.get("step_id") or "").strip()
    }


def _load_existing_step_resume_decisions(
    state: Mapping[str, Any],
) -> dict[str, MultiAgentStepResumeDecision]:
    """
    读取前几轮已经保存的步骤恢复决定。

    参数含义：
        state:
            当前已恢复 Checkpoint 字段的主图状态。

    返回值含义：
        dict[str, MultiAgentStepResumeDecision]:
            校验通过的历史步骤决定；异常旧数据会被忽略。
    """

    raw_decisions = state.get("multi_agent_step_resume_decisions")
    if not isinstance(raw_decisions, Mapping):
        return {}
    decisions: dict[str, MultiAgentStepResumeDecision] = {}
    for step_id, raw_decision in raw_decisions.items():
        if not isinstance(raw_decision, Mapping):
            continue
        try:
            decision = MultiAgentStepResumeDecision.model_validate(
                raw_decision
            )
        except (TypeError, ValueError, ValidationError):
            continue
        decisions[str(step_id)] = decision
    return decisions


def _select_degraded_step_ids(
    *,
    user_input: str,
    requests_by_step_id: Mapping[str, Mapping[str, Any]],
    previous_decisions: Mapping[str, MultiAgentStepResumeDecision],
) -> set[str]:
    """
    识别用户本轮明确要求简化执行的步骤。

    功能：
        保留前几轮已经确认的简化决定；本轮包含“全部简化执行”时选择所有
        可简化步骤；只有一个候选步骤时允许直接回复“简化执行”；多个候选
        时必须通过步骤名称或编号明确目标，避免替用户猜测。

    参数含义：
        user_input:
            用户本轮自然语言输入。
        requests_by_step_id:
            步骤编号到澄清请求的映射。
        previous_decisions:
            前几轮已经保存的步骤决定。

    返回值含义：
        set[str]:
            已经得到用户明确授权、可以采用简化执行的步骤编号集合。
    """

    selected_step_ids = {
        step_id
        for step_id, decision in previous_decisions.items()
        if decision.action == "degraded"
    }
    if not any(keyword in user_input for keyword in MULTI_AGENT_DEGRADED_INPUTS):
        return selected_step_ids

    candidates = {
        step_id: request
        for step_id, request in requests_by_step_id.items()
        if bool(request.get("can_run_degraded"))
    }
    if "全部简化执行" in user_input or "全部简化运行" in user_input:
        return selected_step_ids | set(candidates)
    matched_step_ids = {
        step_id
        for step_id, request in candidates.items()
        if step_id in user_input
        or str(request.get("step_title") or "").strip() in user_input
    }
    if matched_step_ids:
        return selected_step_ids | matched_step_ids
    if len(candidates) == 1:
        return selected_step_ids | set(candidates)
    return selected_step_ids


def _collect_waiting_field_ids(
    decisions: Mapping[str, MultiAgentStepResumeDecision],
) -> list[str]:
    """
    汇总仍处于等待状态的步骤字段编号，仅用于生成用户提示。

    参数含义：
        decisions:
            当前整批步骤的恢复决定。

    返回值含义：
        list[str]:
            去重后仍需补充或明确处理的字段编号。
    """

    return list(dict.fromkeys(
        input_id
        for decision in decisions.values()
        if decision.action == "keep_waiting"
        for input_id in decision.waiting_input_ids
    ))


def _append_degraded_choice_hint(
    *,
    prompt: str,
    user_input: str,
    pending_result: MultiAgentTaskResult,
    resume_decisions: Mapping[str, MultiAgentStepResumeDecision],
) -> str:
    """
    在多个步骤均可简化但用户没有明确目标时追加选择提示。

    参数含义：
        prompt:
            已经生成的剩余字段澄清提示。
        user_input:
            用户本轮原始输入。
        pending_result:
            当前暂停任务，用于读取步骤名称。
        resume_decisions:
            本轮逐步骤恢复决定。

    返回值含义：
        str:
            必要时追加步骤名称后的最终提示文本。
    """

    if not any(keyword in user_input for keyword in MULTI_AGENT_DEGRADED_INPUTS):
        return prompt
    requests = _load_step_requests_by_id(pending_result)
    undecided_degradable_steps = [
        step_id
        for step_id, decision in resume_decisions.items()
        if decision.action == "keep_waiting"
        and bool(requests.get(step_id, {}).get("can_run_degraded"))
    ]
    if len(undecided_degradable_steps) < 2:
        return prompt
    step_names = [
        str(requests[step_id].get("step_title") or step_id)
        for step_id in undecided_degradable_steps
    ]
    return (
        f"{prompt}\n多个步骤支持简化执行，请明确回复要简化的步骤："
        f"{'、'.join(step_names)}；也可以回复“全部简化执行”。"
    )


def _parse_pending_task_result(
    raw_result: Any,
) -> MultiAgentTaskResult | None:
    """
    把 Checkpoint 中的普通字典还原成多 Agent 任务结果。

    参数含义：
        raw_result:
            DogState 中保存的普通字典或任务结果对象。

    返回值含义：
        MultiAgentTaskResult | None:
            合法且正在等待输入时返回模型，否则返回 None。
    """

    if isinstance(raw_result, MultiAgentTaskResult):
        parsed_result = raw_result
    elif isinstance(raw_result, Mapping):
        try:
            parsed_result = MultiAgentTaskResult.model_validate(raw_result)
        except (TypeError, ValueError, ValidationError):
            return None
    else:
        return None
    if parsed_result.status != "awaiting_input":
        return None
    return parsed_result


def _strip_first_prefix(
    text: str,
    prefixes: tuple[str, ...],
) -> str | None:
    """
    移除文本命中的第一个业务前缀。

    参数含义：
        text:
            用户输入文本。
        prefixes:
            允许匹配的前缀集合。

    返回值含义：
        str | None:
            命中时返回去掉前缀的文本，未命中时返回 None。
    """

    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def _parse_multiple_step_inputs(
    text: str,
) -> dict[str, Any] | None:
    """
    解析多个等待步骤使用的 JSON 回答对象。

    参数含义：
        text:
            用户提供的 JSON 文本。

    返回值含义：
        dict[str, Any] | None:
            合法 JSON 对象返回普通字典，格式不正确时返回 None。
    """

    try:
        parsed_value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed_value, Mapping):
        return None
    return {
        str(step_id): value
        for step_id, value in parsed_value.items()
    }


def _build_ready_resume_update(
    *,
    pending_result: MultiAgentTaskResult,
    user_inputs: Mapping[str, Any],
    clarification_extraction: Mapping[str, Any] | None = None,
    resume_decisions: (
        Mapping[str, MultiAgentStepResumeDecision] | None
    ) = None,
) -> dict[str, Any]:
    """
    构建已经可以恢复多 Agent 任务的状态更新。

    参数含义：
        pending_result:
            Checkpoint 中恢复出的暂停任务结果。
        user_inputs:
            等待步骤编号到用户回答的映射。
        clarification_extraction:
            可选的自然语言字段提取轨迹，供可观测系统解释本次分配。
        resume_decisions:
            可选的逐步骤恢复决定，用于把简化执行控制信息传给对应 Worker。

    返回值含义：
        dict[str, Any]:
            action 为 resume，并包含需要合并回 DogState 的恢复字段。
    """

    return {
        "action": "resume",
        "state_update": {
            "multi_agent_task_result": pending_result.model_dump(
                mode="python"
            ),
            "multi_agent_resume_action": "resume",
            "multi_agent_resume_inputs": dict(user_inputs),
            "multi_agent_step_resume_decisions": {
                step_id: decision.model_dump(mode="python")
                for step_id, decision in (resume_decisions or {}).items()
            },
            "multi_agent_resume_ready": True,
            "multi_agent_clarification_extraction": dict(
                clarification_extraction or {}
            ),
            "multi_agent_pending_prompt": "",
            "pending_prompt": "",
            "waiting_user_input": False,
        },
    }


def _build_resume_cleanup_update(
    action: MultiAgentResumeAction,
) -> dict[str, Any]:
    """
    构建取消任务或开始新问题时的清理字段。

    参数含义：
        action:
            本轮恢复意图判断结果。

    返回值含义：
        dict[str, Any]:
            清空暂停任务和恢复输入后的 DogState 局部更新。
    """

    return {
        "multi_agent_task_result": {},
        "multi_agent_resume_action": action,
        "multi_agent_resume_inputs": {},
        "multi_agent_step_resume_decisions": {},
        "multi_agent_resume_ready": False,
        "multi_agent_clarification_extraction": {},
        "multi_agent_pending_prompt": "",
        "pending_prompt": "",
        "waiting_user_input": False,
    }


def _build_resume_clarification_prompt(
    *,
    pending_result: MultiAgentTaskResult,
    awaiting_step_ids: list[str],
) -> str:
    """
    生成多个 Worker 同时等待时的结构化回答提示。

    参数含义：
        pending_result:
            Checkpoint 中恢复出的暂停任务结果，可能包含新版整批澄清包。
        awaiting_step_ids:
            当前全部等待步骤编号。

    返回值含义：
        str:
            告诉用户每个步骤缺少什么，以及如何按 step_id 提供 JSON 回答
            的提示文本。
    """

    example = {
        step_id: "请填写这个步骤的回答"
        for step_id in awaiting_step_ids
    }
    readable_step_lines = _build_readable_step_lines(
        pending_result=pending_result,
        awaiting_step_ids=awaiting_step_ids,
    )
    prompt_parts = ["当前有多个步骤等待输入："]
    if readable_step_lines:
        prompt_parts.extend(readable_step_lines)
    prompt_parts.append("请按步骤编号提供 JSON 对象：")
    prompt_parts.append(
        json.dumps(example, indent=4, ensure_ascii=False)
    )
    return "\n".join(prompt_parts)


def _load_field_consumers(
    pending_result: MultiAgentTaskResult,
) -> dict[str, list[str]]:
    """
    从暂停任务中读取字段到步骤的确定性使用关系。

    参数含义：
        pending_result:
            当前等待恢复的多智能体任务结果。

    返回值含义：
        dict[str, list[str]]:
            合法字段编号到步骤编号列表的映射；旧任务没有该数据时为空。
    """

    bundle = pending_result.metadata.get("clarification_bundle")
    if not isinstance(bundle, Mapping):
        return {}
    raw_consumers = bundle.get("field_consumers")
    if not isinstance(raw_consumers, Mapping):
        return {}
    return {
        str(field_id): [
            str(step_id)
            for step_id in step_ids
            if str(step_id).strip()
        ]
        for field_id, step_ids in raw_consumers.items()
        if str(field_id).strip() and isinstance(step_ids, list)
    }


def _load_field_descriptions(
    pending_result: MultiAgentTaskResult,
) -> dict[str, str]:
    """
    从整批澄清包读取字段的通俗名称和说明。

    功能：
        合并各等待步骤声明的缺失字段描述，为 LLM 兜底层提供有限且明确
        的字段语义；同一字段被多个步骤使用时只保留一份说明。

    参数含义：
        pending_result:
            当前等待恢复的多智能体任务结果。

    返回值含义：
        dict[str, str]:
            字段编号到“名称：说明”文本的映射。
    """

    bundle = pending_result.metadata.get("clarification_bundle")
    if not isinstance(bundle, Mapping):
        return {}
    raw_step_requests = bundle.get("step_requests")
    if not isinstance(raw_step_requests, list):
        return {}

    descriptions: dict[str, str] = {}
    for raw_request in raw_step_requests:
        if not isinstance(raw_request, Mapping):
            continue
        raw_missing_fields = raw_request.get("missing_fields")
        if not isinstance(raw_missing_fields, list):
            continue
        for raw_field in raw_missing_fields:
            if not isinstance(raw_field, Mapping):
                continue
            field_id = str(raw_field.get("input_id") or "").strip()
            if not field_id or field_id in descriptions:
                continue
            field_name = str(raw_field.get("name") or field_id).strip()
            description = str(
                raw_field.get("description") or ""
            ).strip()
            descriptions[field_id] = (
                f"{field_name}：{description}"
                if description
                else field_name
            )
    return descriptions


def _field_consumers_cover_waiting_steps(
    *,
    field_consumers: Mapping[str, list[str]],
    awaiting_step_ids: list[str],
) -> bool:
    """
    判断字段使用关系是否覆盖当前全部等待步骤。

    功能：
        防止同一批次混有未声明字段契约的等待步骤时，只给部分步骤分配
        输入却误判为整批已经可以恢复。

    参数含义：
        field_consumers:
            字段编号到需要该字段的步骤编号列表。
        awaiting_step_ids:
            当前真正处于等待输入状态的全部步骤编号。

    返回值含义：
        bool:
            每个等待步骤至少对应一个字段时返回 True，否则返回 False。
    """

    covered_step_ids = {
        step_id
        for consumer_step_ids in field_consumers.values()
        for step_id in consumer_step_ids
    }
    return set(awaiting_step_ids).issubset(covered_step_ids)


def _collect_existing_resume_fields(raw_inputs: Any) -> dict[str, Any]:
    """
    从前几轮按步骤保存的部分回答中还原已识别字段。

    参数含义：
        raw_inputs:
            DogState 中步骤编号到结构化字段的恢复输入。

    返回值含义：
        dict[str, Any]:
            合并后的字段编号到字段值；冲突值不在这里猜测处理。
    """

    if not isinstance(raw_inputs, Mapping):
        return {}
    fields: dict[str, Any] = {}
    for step_inputs in raw_inputs.values():
        if not isinstance(step_inputs, Mapping):
            continue
        for field_id, value in step_inputs.items():
            fields[str(field_id)] = value
    return fields


def _load_existing_extracted_fields(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    读取前几轮已经确认的澄清字段。

    功能：
        优先使用专门保存的字段提取轨迹；兼容旧检查点时，再从按步骤分配
        的恢复输入中还原字段。

    参数含义：
        state:
            当前已恢复检查点字段的主图状态。

    返回值含义：
        dict[str, Any]:
            前几轮已经明确识别的字段和值。
    """

    raw_extraction = state.get("multi_agent_clarification_extraction")
    if isinstance(raw_extraction, Mapping):
        raw_resolved_fields = raw_extraction.get("resolved_fields")
        if isinstance(raw_resolved_fields, Mapping):
            return dict(raw_resolved_fields)
    return _collect_existing_resume_fields(
        state.get("multi_agent_resume_inputs")
    )


def _build_natural_clarification_prompt(
    *,
    pending_result: MultiAgentTaskResult,
    missing_field_ids: list[str],
    ambiguous_field_ids: list[str],
) -> str:
    """
    为自然语言恢复生成剩余字段提示。

    参数含义：
        pending_result:
            包含字段中文名称和步骤关系的暂停任务。
        missing_field_ids:
            尚未从多轮回答中得到值的字段编号。
        ambiguous_field_ids:
            本轮出现冲突候选值、需要用户明确说明的字段编号。

    返回值含义：
        str:
            不要求用户填写 JSON 的自然语言补充提示。
    """

    remaining_ids = list(dict.fromkeys(
        [*missing_field_ids, *ambiguous_field_ids]
    ))
    readable_lines = _build_readable_step_lines(
        pending_result=pending_result,
        awaiting_step_ids=[
            result.step_id
            for result in pending_result.task_results
            if result.status == "awaiting_input"
        ],
        field_filter=set(remaining_ids),
    )
    parts = ["还需要补充以下信息："]
    parts.extend(readable_lines)
    if ambiguous_field_ids:
        parts.append("其中部分信息存在多个可能值，请明确说明最终采用的值。")
    parts.append("请直接使用自然语言回答，也可以回复“取消”。")
    return "\n".join(parts)


def _build_readable_step_lines(
    *,
    pending_result: MultiAgentTaskResult,
    awaiting_step_ids: list[str],
    field_filter: set[str] | None = None,
) -> list[str]:
    """
    从整批澄清包生成每个等待步骤的可读说明。

    功能：
        使用步骤名称和缺失字段中文名帮助用户理解需要回答什么。旧任务
        没有 clarification_bundle 时返回空列表，由调用方保留旧版提示。

    参数含义：
        pending_result:
            当前等待恢复的完整多智能体任务结果。
        awaiting_step_ids:
            本轮真正处于等待状态的步骤编号。
        field_filter:
            只展示这些字段编号；为空时展示步骤声明的全部缺失字段。

    返回值含义：
        list[str]:
            按等待步骤顺序排列的用户可读说明。
    """

    clarification_bundle = pending_result.metadata.get(
        "clarification_bundle"
    )
    if not isinstance(clarification_bundle, Mapping):
        return []
    raw_step_requests = clarification_bundle.get("step_requests")
    if not isinstance(raw_step_requests, list):
        return []

    requests_by_step_id = {
        str(request.get("step_id") or "").strip(): request
        for request in raw_step_requests
        if isinstance(request, Mapping)
        and str(request.get("step_id") or "").strip()
    }
    lines: list[str] = []
    for step_id in awaiting_step_ids:
        request = requests_by_step_id.get(step_id)
        if request is None:
            continue
        step_title = str(request.get("step_title") or step_id).strip()
        raw_fields = request.get("missing_fields")
        # 上游结构异常时不阻塞恢复，只是不展示字段中文名。
        field_names: list[str] = []
        if isinstance(raw_fields, list):
            field_names = [
                str(
                    field.get("name")
                    or field.get("input_id")
                    or ""
                ).strip()
                for field in raw_fields
                if isinstance(field, Mapping)
                and (
                    field_filter is None
                    or str(field.get("input_id") or "") in field_filter
                )
                and str(
                    field.get("name")
                    or field.get("input_id")
                    or ""
                ).strip()
            ]
        if field_filter is not None and not field_names:
            continue
        field_text = (
            "，需要补充：" + "、".join(field_names)
            if field_names
            else ""
        )
        lines.append(f"- {step_title}（{step_id}）{field_text}")
    return lines
