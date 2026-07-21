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

from src.agents.collaboration.contracts import MultiAgentTaskResult


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


def resolve_multi_agent_resume_input(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    判断本轮用户输入与暂停中的多 Agent 任务是什么关系。

    功能：
        没有暂停任务时不做处理；命中取消词或新问题前缀时清理旧任务；
        单个等待步骤直接接收本轮文本；多个等待步骤要求 JSON 对象按编号回答。

    参数含义：
        state:
            主图当前状态，需要包含 question 和可选 multi_agent_task_result。

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
                    "multi_agent_resume_ready": True,
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

    if len(awaiting_step_ids) == 1 and normalized_input:
        return _build_ready_resume_update(
            pending_result=pending_result,
            user_inputs={awaiting_step_ids[0]: normalized_input},
        )

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

    prompt = _build_resume_clarification_prompt(awaiting_step_ids)
    return {
        "action": "needs_clarification",
        "state_update": {
            "multi_agent_resume_action": "needs_clarification",
            "multi_agent_resume_inputs": {},
            "multi_agent_resume_ready": False,
            "multi_agent_pending_prompt": prompt,
            "pending_prompt": prompt,
            "waiting_user_input": True,
        },
    }


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
) -> dict[str, Any]:
    """
    构建已经可以恢复多 Agent 任务的状态更新。

    参数含义：
        pending_result:
            Checkpoint 中恢复出的暂停任务结果。
        user_inputs:
            等待步骤编号到用户回答的映射。

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
            "multi_agent_resume_ready": True,
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
        "multi_agent_resume_ready": False,
        "multi_agent_pending_prompt": "",
        "pending_prompt": "",
        "waiting_user_input": False,
    }


def _build_resume_clarification_prompt(
    awaiting_step_ids: list[str],
) -> str:
    """
    生成多个 Worker 同时等待时的结构化回答提示。

    参数含义：
        awaiting_step_ids:
            当前全部等待步骤编号。

    返回值含义：
        str:
            告诉用户如何按 step_id 提供 JSON 回答的提示文本。
    """

    example = {
        step_id: "请填写这个步骤的回答"
        for step_id in awaiting_step_ids
    }
    return (
        "当前有多个步骤等待输入，请按步骤编号提供 JSON 对象：\n"
        + json.dumps(example, indent=4, ensure_ascii=False)
    )
