"""
ToolAgent 参数澄清恢复适配器。

功能：
    判断本轮用户输入是否是上一轮缺失参数的确定性补充，
    并把补充值合并回待处理工具调用。

当前 MVP 规则：
    1. 输入唯一匹配某个缺失字段的候选值时，补全该字段。
    2. 仍有缺失字段时保留待处理调用，并继续发起下一轮澄清。
    3. 只剩无候选值字段时，允许安全短文本作为参数值。
    4. 输入明确取消词时，取消待处理调用。
    5. 其他输入视为新问题，清理旧澄清状态。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


ClarificationResumeAction = Literal[
    "none",
    "partial",
    "resumed",
    "cancelled",
    "new_question",
]

CLARIFICATION_CANCEL_INPUTS = {
    "取消",
    "算了",
    "不查了",
    "不用了",
    "cancel",
}


def resolve_tool_clarification_input(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    判断并处理本轮用户输入与待澄清工具调用的关系。

    功能：
        从 state 读取澄清请求、待补全工具调用和本轮 question。
        候选值精确匹配时补全参数；取消词清理任务；其他输入按新问题处理。

    参数：
        state:
            当前 LangGraph state，可能包含 Checkpoint 恢复出的澄清字段。

    返回值：
        dict[str, Any]:
            包含 action 和 state_update：
            action 表示判断结果，state_update 表示需要合并回 state 的普通字典。
    """

    clarification_request = state.get(
        "tool_agent_clarification_request"
    )
    pending_tool_call = state.get(
        "tool_agent_pending_tool_call"
    )

    if not isinstance(
        clarification_request,
        Mapping,
    ) or not isinstance(
        pending_tool_call,
        Mapping,
    ):
        return {
            "action": "none",
            "state_update": {},
        }

    user_input = normalize_clarification_input(
        state.get(
            "question",
            "",
        )
    )


    # casefold是python内置的比较字符串的方法 将字符串变为小写
    # 和lower类似  但比lower更强
    if user_input.casefold() in CLARIFICATION_CANCEL_INPUTS:
        return {
            "action": "cancelled",
            "state_update": build_clarification_cleanup_update(
                action="cancelled",
            ),
        }

    matched_parameter = match_clarification_candidate(
        user_input=user_input,
        clarification_request=clarification_request,
    )
    if matched_parameter is not None:
        field_name, field_value = matched_parameter
        args = pending_tool_call.get(
            "args",
            {},
        )
        normalized_args = dict(args) if isinstance(args, Mapping) else {}
        normalized_args[field_name] = field_value

        resumed_tool_call = {
            **dict(pending_tool_call),
            "args": normalized_args,
        }
        missing_fields = [
            str(item)
            for item in clarification_request.get(
                "missing_fields",
                [],
            )
            if str(item) != field_name
        ]
        if missing_fields:
            updated_request = dict(
                clarification_request
            )
            updated_options = clarification_request.get(
                "options",
                {},
            )
            updated_request["missing_fields"] = missing_fields
            updated_request["options"] = {
                missing_field: list(
                    updated_options.get(
                        missing_field,
                        [],
                    )
                )
                for missing_field in missing_fields
            } if isinstance(
                updated_options,
                Mapping,
            ) else {}
            updated_request["question"] = build_remaining_clarification_question(
                completed_field=field_name,
                completed_value=field_value,
                missing_fields=missing_fields,
                options=updated_request["options"],
            )

            return {
                "action": "partial",
                "state_update": {
                    "tool_agent_clarification_request": updated_request,
                    "tool_agent_pending_tool_call": resumed_tool_call,
                    "tool_agent_clarification_resume_ready": False,
                    "tool_agent_clarification_resolution": {
                        "action": "partial",
                        "completed_field": field_name,
                    },
                    "tool_agent_response": {},
                    "final_answer": "",
                },
            }

        return {
            "action": "resumed",
            "state_update": {
                **build_clarification_cleanup_update(
                    action="resumed",
                ),
                "tool_calls": [resumed_tool_call],
                "need_tool": True,
                "tool_agent_clarification_resume_ready": True,
            },
        }

    return {
        "action": "new_question",
        "state_update": build_clarification_cleanup_update(
            action="new_question",
        ),
    }


def match_clarification_candidate(
    user_input: str,
    clarification_request: Mapping[str, Any],
) -> tuple[str, Any] | None:
    """
    将用户输入与唯一缺失字段的候选值进行精确匹配。

    功能：
        在所有缺失字段候选值中执行精确匹配；只有唯一命中时才补全参数。
        当只剩一个没有候选值的字段时，允许安全的短标识符作为自由文本值。

    参数：
        user_input:
            归一化后的本轮用户输入。
        clarification_request:
            Checkpoint 中恢复出的结构化澄清请求。

    返回值：
        tuple[str, Any] | None:
            匹配成功返回“字段名、原始候选值”；不匹配返回 None。
    """

    missing_fields = clarification_request.get(
        "missing_fields",
        [],
    )
    options = clarification_request.get(
        "options",
        {},
    )
    if (
        not isinstance(missing_fields, list)
        or not missing_fields
        or not isinstance(options, Mapping)
    ):
        return None

    normalized_user_input = user_input.casefold()
    matches: list[tuple[str, Any]] = []
    for raw_field_name in missing_fields:
        field_name = str(
            raw_field_name
        )
        field_options = options.get(
            field_name,
            [],
        )
        if not isinstance(field_options, list):
            continue

        for option in field_options:
            if normalize_clarification_input(option).casefold() == normalized_user_input:
                matches.append(
                    (
                        field_name,
                        option,
                    )
                )

    if len(matches) == 1:
        return matches[0]

    if len(missing_fields) == 1:
        field_name = str(
            missing_fields[0]
        )
        field_options = options.get(
            field_name,
            [],
        )
        if not field_options and is_safe_free_text_parameter(
            user_input
        ):
            return field_name, user_input

    return None


def is_safe_free_text_parameter(
    user_input: str,
) -> bool:
    """
    判断用户输入是否适合作为自由文本工具参数。

    功能：
        当前 MVP 只接受不含空白、长度不超过 128 的短值，
        用于补充 SQLite 表名等没有固定候选列表的参数。

    参数：
        user_input:
            归一化后的用户输入。

    返回值：
        bool:
            输入适合作为单个参数值时返回 True，否则返回 False。
    """

    return bool(
        user_input
        and len(user_input) <= 128
        and not any(
            character.isspace()
            for character in user_input
        )
    )


def build_remaining_clarification_question(
    completed_field: str,
    completed_value: Any,
    missing_fields: list[str],
    options: Mapping[str, list[Any]],
) -> str:
    """
    构建分步补参后的下一条澄清问题。

    功能：
        告知用户刚刚记录了哪个参数，并只询问仍然缺失的字段。

    参数：
        completed_field:
            本轮已经补全的字段名。
        completed_value:
            本轮已经确认的字段值。
        missing_fields:
            仍待补全的字段列表。
        options:
            剩余字段对应的候选值列表。

    返回值：
        str:
            可直接展示给用户的中文澄清问题。
    """

    field_labels = {
        "database_name": "数据库别名",
        "table_name": "表名",
        "sql": "只读 SQL",
    }
    prompts: list[str] = []
    for field_name in missing_fields:
        label = field_labels.get(
            field_name,
            field_name,
        )
        allowed_values = options.get(
            field_name,
            [],
        )
        if allowed_values:
            label += "（可选：" + "、".join(
                str(value)
                for value in allowed_values
            ) + "）"
        prompts.append(label)

    completed_label = field_labels.get(
        completed_field,
        completed_field,
    )
    return (
        f"已记录{completed_label}为 {completed_value}。"
        f"请继续补充：{'；'.join(prompts)}。"
    )


def build_clarification_cleanup_update(
    action: ClarificationResumeAction,
) -> dict[str, Any]:
    """
    构建清理旧澄清状态的 state update。

    功能：
        清空待补全调用、澄清请求和上一轮等待提示，防止污染后续问题。

    参数：
        action:
            本轮处理结果，例如 resumed、cancelled 或 new_question。

    返回值：
        dict[str, Any]:
            可直接合并进 LangGraph state 的普通字典。
    """

    return {
        "tool_agent_clarification_request": None,
        "tool_agent_pending_tool_call": None,
        "tool_agent_pending_original_question": "",
        "tool_agent_pending_created_at": "",
        "tool_agent_clarification_resume_ready": False,
        "tool_agent_clarification_resolution": {
            "action": action,
        },
        "tool_agent_response": {},
        "tool_agent_permission": {},
        "waiting_user_input": False,
        "has_asked_user": False,
        "pending_prompt": "",
    }


def normalize_clarification_input(
    value: Any,
) -> str:
    """
    归一化用户输入或候选值。

    功能：
        将任意值转换成去除首尾空白的字符串，供确定性匹配使用。

    参数：
        value:
            用户输入或候选值。

    返回值：
        str:
            归一化后的字符串。
    """

    return str(
        value
        if value is not None
        else ""
    ).strip()
