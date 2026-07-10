"""
ToolAgent 工具调用校验适配器。

功能：
    根据 ToolAgent 工具目录校验 LLM 解析出的 tool_calls。

设计说明：
    当前模块只做独立校验能力，不直接接入 ToolAgent 主链路。
    这样可以先把工具调用契约收敛好，再决定放到 parse 节点后面，
    或者单独新增 validation 节点。

专业名词：
    ToolCall：
        工具调用请求，表示 LLM 希望调用哪个工具以及传入哪些 args。
    Validation：
        校验，检查工具名、参数结构、必填字段和基础类型是否符合工具目录。
    JSON Schema：
        JSON 数据结构描述格式，这里主要读取 properties 和 required。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall


TOOL_CALL_VALIDATION_OK_STATE_KEY = "tool_call_validation_ok"
TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY = "tool_call_validation_skipped"
TOOL_CALL_VALIDATION_ERRORS_STATE_KEY = "tool_call_validation_errors"
TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY = "tool_call_validation_invalid_calls"


def validate_tool_calls_from_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从 state 中读取工具目录和工具调用并执行校验。

    功能：
        读取 state["tool_calls"] 和 state["tool_agent_tool_catalog"]，
        校验工具名是否存在、args 是否符合 input_schema，
        并返回可写回 state 的普通 dict。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            校验后的 state update，包含 tool_calls、校验状态和错误列表。
    """

    validation_result = validate_tool_calls_against_catalog(
        tool_calls=state.get(
            "tool_calls",
            [],
        ),
        tool_catalog=state.get(
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
            [],
        ),
    )

    return build_tool_call_validation_state_update(
        validation_result=validation_result,
    )


def validate_tool_calls_against_catalog(
    tool_calls: Iterable[Any],
    tool_catalog: Iterable[Any],
) -> dict[str, Any]:
    """
    根据工具目录校验工具调用列表。

    功能：
        将工具目录按工具名建立索引，然后逐个校验 tool_call。
        如果工具目录为空，当前阶段保守跳过校验并保留原始 tool_calls。

    参数：
        tool_calls:
            LLM 解析出的工具调用列表，可以是 ToolCall 或 dict。

        tool_catalog:
            ToolAgent 工具目录列表，通常来自 state["tool_agent_tool_catalog"]。

    返回值：
        dict[str, Any]:
            普通字典格式的校验结果。
    """

    normalized_tool_calls = [
        normalize_tool_call_for_validation(
            raw_tool_call=raw_tool_call,
        )
        for raw_tool_call in tool_calls or []
    ]
    catalog_by_name = build_tool_catalog_by_name(
        tool_catalog=tool_catalog,
    )

    if not catalog_by_name:
        return {
            "is_valid": True,
            "validation_skipped": True,
            "valid_tool_calls": normalized_tool_calls,
            "invalid_tool_calls": [],
            "errors": [
                build_validation_error(
                    code="tool_catalog_missing",
                    message="工具目录为空，已跳过工具调用校验。",
                )
            ],
        }

    valid_tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []

    for tool_call in normalized_tool_calls:
        tool_errors = validate_single_tool_call(
            tool_call=tool_call,
            catalog_by_name=catalog_by_name,
        )

        if tool_errors:
            invalid_tool_calls.append(
                {
                    "tool_call": tool_call,
                    "errors": tool_errors,
                }
            )
            all_errors.extend(
                tool_errors
            )
            continue

        valid_tool_calls.append(
            tool_call
        )

    return {
        "is_valid": not all_errors,
        "validation_skipped": False,
        "valid_tool_calls": valid_tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "errors": all_errors,
    }


def build_tool_call_validation_state_update(
    validation_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将工具调用校验结果转换成 state update。

    功能：
        把 valid_tool_calls 写回 tool_calls，
        并补充校验状态、是否跳过校验、错误列表和非法调用列表。

    参数：
        validation_result:
            validate_tool_calls_against_catalog 返回的校验结果。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的普通字典。
    """

    valid_tool_calls = list(
        validation_result.get(
            "valid_tool_calls",
            [],
        )
        or []
    )
    validation_skipped = bool(
        validation_result.get(
            "validation_skipped",
            False,
        )
    )

    update: dict[str, Any] = {
        "tool_calls": valid_tool_calls,
        TOOL_CALL_VALIDATION_OK_STATE_KEY: bool(
            validation_result.get(
                "is_valid",
                False,
            )
        ),
        TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY: validation_skipped,
        TOOL_CALL_VALIDATION_ERRORS_STATE_KEY: list(
            validation_result.get(
                "errors",
                [],
            )
            or []
        ),
        TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY: list(
            validation_result.get(
                "invalid_tool_calls",
                [],
            )
            or []
        ),
    }

    if not validation_skipped and not valid_tool_calls:
        update["need_tool"] = False
        update["tool_results"] = [
            build_validation_failed_tool_result(
                errors=update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY],
            )
        ]

    return update


def build_tool_catalog_by_name(
    tool_catalog: Iterable[Any],
) -> dict[str, dict[str, Any]]:
    """
    将工具目录转换成按 name 索引的字典。

    功能：
        遍历工具目录，只保留 Mapping 且包含 name 的条目，
        方便后续用 O(1) 方式按工具名查询工具定义。

    参数：
        tool_catalog:
            工具目录列表。

    返回值：
        dict[str, dict[str, Any]]:
            key 为工具名，value 为工具目录条目的普通 dict。
    """

    catalog_by_name: dict[str, dict[str, Any]] = {}

    for raw_item in tool_catalog or []:
        if not isinstance(
            raw_item,
            Mapping,
        ):
            continue

        tool_name = str(
            raw_item.get(
                "name",
                "",
            )
            or ""
        ).strip()

        if not tool_name:
            continue

        catalog_by_name[tool_name] = dict(
            raw_item
        )

    return catalog_by_name


def normalize_tool_call_for_validation(
    raw_tool_call: Any,
) -> dict[str, Any]:
    """
    将工具调用转换成普通 dict。

    功能：
        兼容 ToolCall 对象和 dict 格式工具调用。
        归一化阶段不吞掉非法数据，只把原始结构整理成校验层可识别的 dict。
        这样后续 validate_single_tool_call 可以返回明确错误，而不是静默丢弃。

    参数：
        raw_tool_call:
            原始工具调用，可以是 ToolCall 或 Mapping。

    返回值：
        dict[str, Any]:
            归一化后的工具调用。即使原始数据非法，也会返回带错误标记的 dict。
    """

    if isinstance(
        raw_tool_call,
        ToolCall,
    ):
        return raw_tool_call.model_dump()

    if not isinstance(
        raw_tool_call,
        Mapping,
    ):
        return {
            "name": "",
            "args": None,
            "_normalization_error": "invalid_tool_call",
            "_raw_tool_call_type": type(
                raw_tool_call
            ).__name__,
            "_raw_tool_call_preview": repr(
                raw_tool_call
            ),
        }

    tool_name = str(
        raw_tool_call.get(
            "name",
            "",
        )
        or ""
    ).strip()

    raw_args = raw_tool_call.get(
        "args",
        {},
    )

    return {
        "name": tool_name,
        "args": raw_args,
    }


def validate_single_tool_call(
    tool_call: Mapping[str, Any],
    catalog_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    校验单个工具调用。

    功能：
        检查工具是否存在，并根据工具目录中的 input_schema 校验 args。

    参数：
        tool_call:
            单个工具调用。

        catalog_by_name:
            按工具名索引后的工具目录。

    返回值：
        list[dict[str, Any]]:
            错误列表。为空表示校验通过。
    """

    tool_name = str(
        tool_call.get(
            "name",
            "",
        )
        or ""
    )
    args = tool_call.get(
        "args",
        {},
    )

    normalization_error = str(
        tool_call.get(
            "_normalization_error",
            "",
        )
        or ""
    )

    if normalization_error == "invalid_tool_call":
        return [
            build_validation_error(
                code="invalid_tool_call",
                message="工具调用条目必须是字典对象或 ToolCall 对象。",
                tool_name=tool_name,
            )
        ]

    if not tool_name:
        return [
            build_validation_error(
                code="missing_tool_name",
                message="工具调用缺少工具名称 name。",
            )
        ]

    if tool_name not in catalog_by_name:
        return [
            build_validation_error(
                code="unknown_tool",
                message=f"工具 {tool_name} 不存在于工具目录中。",
                tool_name=tool_name,
            )
        ]

    if not isinstance(
        args,
        Mapping,
    ):
        return [
            build_validation_error(
                code="invalid_args",
                message=f"工具 {tool_name} 的 args 必须是字典对象。",
                tool_name=tool_name,
            )
        ]

    catalog_item = catalog_by_name[tool_name]
    input_schema = catalog_item.get(
        "input_schema",
        {},
    )

    if not isinstance(
        input_schema,
        Mapping,
    ) or not input_schema:
        return []

    return validate_args_against_input_schema(
        tool_name=tool_name,
        args=args,
        input_schema=input_schema,
    )


def validate_args_against_input_schema(
    tool_name: str,
    args: Mapping[str, Any],
    input_schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    根据 input_schema 校验工具参数。

    功能：
        当前 MVP 校验 required 必填字段和常见基础类型。
        暂不实现完整 JSON Schema 引擎，避免过度复杂。

    参数：
        tool_name:
            工具名称。

        args:
            工具调用参数。

        input_schema:
            工具输入参数结构。

    返回值：
        list[dict[str, Any]]:
            错误列表。为空表示校验通过。
    """

    errors: list[dict[str, Any]] = []
    required_fields = read_required_fields(
        input_schema=input_schema,
    )
    properties = read_schema_properties(
        input_schema=input_schema,
    )

    for field_name in required_fields:
        if field_name not in args:
            errors.append(
                build_validation_error(
                    code="missing_required_arg",
                    message=f"工具 {tool_name} 缺少必填参数 {field_name}。",
                    tool_name=tool_name,
                    field=field_name,
                )
            )

    for field_name, value in args.items():
        field_schema = properties.get(
            field_name,
            {},
        )

        if not field_schema:
            continue

        expected_type = str(
            field_schema.get(
                "type",
                "",
            )
            or ""
        )

        if not expected_type:
            continue

        if not is_value_matching_json_schema_type(
            value=value,
            expected_type=expected_type,
        ):
            errors.append(
                build_validation_error(
                    code="invalid_arg_type",
                    message=(
                        f"工具 {tool_name} 的参数 {field_name} 类型不符合要求，"
                        f"期望 {expected_type}。"
                    ),
                    tool_name=tool_name,
                    field=field_name,
                )
            )

        allowed_values = read_schema_allowed_values(
            field_schema=field_schema,
        )
        if allowed_values and value not in allowed_values:
            errors.append(
                build_validation_error(
                    code="invalid_arg_value",
                    message=(
                        f"工具 {tool_name} 的参数 {field_name} 不在允许值中，"
                        f"当前值为 {value}，允许值为 "
                        f"{', '.join(str(item) for item in allowed_values)}。"
                    ),
                    tool_name=tool_name,
                    field=field_name,
                )
            )

    return errors


def read_schema_allowed_values(
    field_schema: Mapping[str, Any],
) -> list[Any]:
    """
    从字段 schema 中读取允许值列表。

    功能：
        优先读取 enum，其次读取 allowed_values。
        这层用于校验 database_name 这类有业务白名单的字段。

    参数：
        field_schema:
            单个字段的 schema 描述。

    返回值：
        list[Any]:
            允许值列表；没有配置时返回空列表。
    """

    raw_allowed_values = field_schema.get(
        "enum",
        field_schema.get(
            "allowed_values",
            [],
        ),
    )

    if not isinstance(
        raw_allowed_values,
        list,
    ):
        return []

    return list(
        raw_allowed_values
    )


def build_validation_failed_tool_result(
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    构建工具调用校验失败的 ToolResult 字典。

    功能：
        当所有工具调用都被校验拦截时，生成一个失败工具结果，
        让后续 tool_answer_node 可以返回清晰的用户提示。

    参数：
        errors:
            工具调用校验错误列表。

    返回值：
        dict[str, Any]:
            普通 dict 格式的失败工具结果。
    """

    first_error = (
        errors[0]
        if errors
        else {}
    )
    message = str(
        first_error.get(
            "message",
            "工具调用参数校验失败。",
        )
        or "工具调用参数校验失败。"
    )

    return {
        "success": False,
        "tool_name": str(
            first_error.get(
                "tool_name",
                "tool_validation",
            )
            or "tool_validation"
        ),
        "content": None,
        "error": message,
        "metadata": {
            "source": "tool_call_validation_adapter",
            "errors": errors,
        },
    }


def read_required_fields(
    input_schema: Mapping[str, Any],
) -> list[str]:
    """
    从 input_schema 中读取 required 字段列表。

    功能：
        只保留字符串类型字段，忽略异常结构。

    参数：
        input_schema:
            工具输入参数结构。

    返回值：
        list[str]:
            必填字段名列表。
    """

    raw_required = input_schema.get(
        "required",
        [],
    )

    if not isinstance(
        raw_required,
        list,
    ):
        return []

    return [
        field_name
        for field_name in raw_required
        if isinstance(
            field_name,
            str,
        )
    ]


def read_schema_properties(
    input_schema: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    从 input_schema 中读取 properties。

    功能：
        只保留字段名为字符串且字段 schema 为 Mapping 的项目。

    参数：
        input_schema:
            工具输入参数结构。

    返回值：
        dict[str, dict[str, Any]]:
            参数字段名到字段 schema 的映射。
    """

    raw_properties = input_schema.get(
        "properties",
        {},
    )

    if not isinstance(
        raw_properties,
        Mapping,
    ):
        return {}

    return {
        str(
            field_name
        ): dict(
            field_schema
        )
        for field_name, field_schema in raw_properties.items()
        if isinstance(
            field_name,
            str,
        )
        and isinstance(
            field_schema,
            Mapping,
        )
    }


def is_value_matching_json_schema_type(
    value: Any,
    expected_type: str,
) -> bool:
    """
    判断值是否符合常见 JSON Schema 基础类型。

    功能：
        支持 string、integer、number、boolean、object、array。
        未识别类型默认放行，避免 MVP 阶段误伤扩展 schema。

    参数：
        value:
            待校验的参数值。

        expected_type:
            JSON Schema type 字段。

    返回值：
        bool:
            True 表示类型匹配或暂不校验，False 表示类型不匹配。
    """

    if expected_type == "string":
        return isinstance(
            value,
            str,
        )

    if expected_type == "integer":
        return isinstance(
            value,
            int,
        ) and not isinstance(
            value,
            bool,
        )

    if expected_type == "number":
        return isinstance(
            value,
            (
                int,
                float,
            ),
        ) and not isinstance(
            value,
            bool,
        )

    if expected_type == "boolean":
        return isinstance(
            value,
            bool,
        )

    if expected_type == "object":
        return isinstance(
            value,
            Mapping,
        )

    if expected_type == "array":
        return isinstance(
            value,
            list,
        )

    return True


def build_validation_error(
    code: str,
    message: str,
    tool_name: str = "",
    field: str = "",
) -> dict[str, Any]:
    """
    构建工具调用校验错误。

    功能：
        统一错误输出格式，方便后续写入日志、debug report 或 state。

    参数：
        code:
            错误代码，例如 unknown_tool、missing_required_arg。

        message:
            中文错误说明。

        tool_name:
            相关工具名称。没有时为空字符串。

        field:
            相关参数字段名。没有时为空字符串。

    返回值：
        dict[str, Any]:
            普通字典格式的校验错误。
    """

    return {
        "code": code,
        "message": message,
        "tool_name": tool_name,
        "field": field,
    }
