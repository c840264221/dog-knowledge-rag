"""
ToolAgent 工具调用校验适配器测试。

功能：
    验证 tool_calls 可以根据 ToolAgent 工具目录完成基础契约校验。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
)
from src.agents.tool_agent.adapters.tool_call_validation_adapter import (
    TOOL_CALL_VALIDATION_ERRORS_STATE_KEY,
    TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY,
    TOOL_CALL_VALIDATION_OK_STATE_KEY,
    TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY,
    TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY,
    TOOL_AGENT_PENDING_TOOL_CALL_STATE_KEY,
    build_tool_catalog_by_name,
    is_value_matching_json_schema_type,
    read_required_fields,
    validate_tool_calls_against_catalog,
    validate_tool_calls_from_state,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.mcp.sqlite.tool_definitions import (
    SQLITE_SELECT_ROWS_TOOL_NAME,
)


def build_test_tool_catalog() -> list[dict]:
    """
    构建测试用工具目录。

    功能：
        创建包含 weather 和 sqlite_select_rows 的工具目录。

    参数：
        无。

    返回值：
        list[dict]:
            测试用工具目录。
    """

    return [
        {
            "name": "weather",
            "description": "查询天气",
            "require_confirm": True,
            "input_schema": {},
            "source": "local",
        },
        {
            "name": SQLITE_SELECT_ROWS_TOOL_NAME,
            "description": "查看 SQLite 表前 N 行数据。",
            "require_confirm": False,
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                    },
                    "table_name": {
                        "type": "string",
                    },
                    "limit": {
                        "type": "integer",
                    },
                },
                "required": [
                    "database_name",
                    "table_name",
                ],
            },
            "source": "mcp",
        },
    ]


def test_validate_tool_calls_against_catalog_should_pass_valid_calls() -> None:
    """
    测试合法工具调用可以通过校验。

    功能：
        当工具名存在，且必填参数和类型都符合 input_schema 时，
        工具调用应进入 valid_tool_calls。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": {
                    "database_name": "memory",
                    "table_name": "dogs",
                    "limit": 5,
                },
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is True
    assert result["validation_skipped"] is False
    assert result["valid_tool_calls"] == [
        {
            "name": SQLITE_SELECT_ROWS_TOOL_NAME,
            "args": {
                "database_name": "memory",
                "table_name": "dogs",
                "limit": 5,
            },
        }
    ]
    assert result["invalid_tool_calls"] == []
    assert result["errors"] == []


def test_validate_tool_calls_against_catalog_should_reject_unknown_tool() -> None:
    """
    测试未知工具会被拦截。

    功能：
        当 tool_call.name 不存在于工具目录时，
        该调用应进入 invalid_tool_calls。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": "unknown_tool",
                "args": {},
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["valid_tool_calls"] == []
    assert result["errors"][0]["code"] == "unknown_tool"
    assert result["errors"][0]["tool_name"] == "unknown_tool"


def test_validate_tool_calls_against_catalog_should_reject_missing_required_arg() -> None:
    """
    测试缺少必填参数会被拦截。

    功能：
        sqlite_select_rows 需要 database_name 和 table_name，
        缺少 table_name 时应返回 missing_required_arg。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": {
                    "database_name": "memory",
                },
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["errors"][0]["code"] == "missing_required_arg"
    assert result["errors"][0]["field"] == "table_name"


def test_validate_tool_calls_against_catalog_should_reject_invalid_arg_type() -> None:
    """
    测试参数类型错误会被拦截。

    功能：
        limit 在 input_schema 中要求 integer，
        如果传入字符串，则应返回 invalid_arg_type。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": {
                    "database_name": "memory",
                    "table_name": "dogs",
                    "limit": "5",
                },
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["errors"][0]["code"] == "invalid_arg_type"
    assert result["errors"][0]["field"] == "limit"


def test_validate_tool_calls_against_catalog_should_reject_invalid_allowed_value() -> None:
    """
    测试参数值不在白名单时会被拦截。

    功能：
        当 database_name schema 声明 enum=["memory", "rag"]，
        但 LLM 输出 unknown_db 时，应返回 invalid_arg_value。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"]["enum"] = [
        "memory",
        "rag",
    ]

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": {
                    "database_name": "unknown_db",
                    "table_name": "dogs",
                },
            }
        ],
        tool_catalog=catalog,
    )

    assert result["is_valid"] is False
    assert result["valid_tool_calls"] == []
    assert result["errors"][0]["code"] == "invalid_arg_value"
    assert result["errors"][0]["field"] == "database_name"


def test_validate_tool_calls_against_catalog_should_reject_non_mapping_args() -> None:
    """
    测试 args 不是字典时返回 invalid_args。

    功能：
        确认归一化阶段不会把非法 args 静默转换成空字典，
        而是保留原始值交给校验阶段返回明确错误。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": [
                    "bad_args",
                ],
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["valid_tool_calls"] == []
    assert result["errors"][0]["code"] == "invalid_args"
    assert result["errors"][0]["tool_name"] == SQLITE_SELECT_ROWS_TOOL_NAME
    assert result["invalid_tool_calls"][0]["tool_call"]["args"] == [
        "bad_args",
    ]


def test_validate_tool_calls_against_catalog_should_not_drop_invalid_tool_call_item() -> None:
    """
    测试非法 tool_call 条目不会被静默丢弃。

    功能：
        当 LLM 输出的 tool_calls 中混入非 dict / 非 ToolCall 条目时，
        该条目应进入 invalid_tool_calls，并返回 invalid_tool_call 错误。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            "bad_tool_call",
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["valid_tool_calls"] == []
    assert result["errors"][0]["code"] == "invalid_tool_call"
    assert result["invalid_tool_calls"][0]["tool_call"][
        "_normalization_error"
    ] == "invalid_tool_call"
    assert result["invalid_tool_calls"][0]["tool_call"][
        "_raw_tool_call_type"
    ] == "str"


def test_validate_tool_calls_against_catalog_should_reject_missing_tool_name() -> None:
    """
    测试缺少工具名称时返回 missing_tool_name。

    功能：
        确认缺少 name 的 tool_call 不会被归一化阶段过滤，
        而是在校验阶段生成可观测错误。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "args": {},
            }
        ],
        tool_catalog=build_test_tool_catalog(),
    )

    assert result["is_valid"] is False
    assert result["valid_tool_calls"] == []
    assert result["errors"][0]["code"] == "missing_tool_name"
    assert result["invalid_tool_calls"][0]["tool_call"]["name"] == ""


def test_validate_tool_calls_from_state_should_clear_need_tool_when_no_valid_calls() -> None:
    """
    测试没有合法工具调用时清空 need_tool。

    功能：
        当所有 tool_calls 都被判定非法时，state update 应写入 need_tool=False，
        避免后续响应适配器误判仍在等待工具确认。

    参数：
        无。

    返回值：
        None。
    """

    update = validate_tool_calls_from_state(
        state={
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "unknown_tool",
                    "args": {},
                }
            ],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: build_test_tool_catalog(),
        }
    )

    assert update["need_tool"] is False
    assert update["tool_results"][0]["success"] is False
    assert update["tool_calls"] == []
    assert update[TOOL_CALL_VALIDATION_OK_STATE_KEY] is False
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY][0]["code"] == "unknown_tool"


def test_missing_required_args_should_create_clarification_state() -> None:
    """
    测试缺少必填参数时生成澄清状态。

    功能：
        缺少 database_name 时保留待补全调用，并且不生成普通失败工具结果。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"].update(
        {
            "description": "数据库别名",
            "enum": ["memory", "rag"],
        }
    )
    update = validate_tool_calls_from_state(
        state={
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "table_name": "dogs",
                    },
                }
            ],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: catalog,
        }
    )

    request = update[TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY]
    assert request["status"] == "pending"
    assert request["missing_fields"] == ["database_name"]
    assert request["options"] == {"database_name": ["memory", "rag"]}
    assert "数据库别名（可选：memory、rag）" in request["question"]
    assert update[TOOL_AGENT_PENDING_TOOL_CALL_STATE_KEY]["args"] == {
        "table_name": "dogs",
    }
    assert update["tool_results"] == []


def test_guessed_explicit_arg_should_create_clarification_state() -> None:
    """
    测试 LLM 猜测的显式输入参数会转成澄清状态。

    功能：
        当前问题没有出现 memory 时，即使 LLM 写入 database_name=memory，
        校验器也应拒绝猜测值、保留错误证据并移除待恢复调用中的该值。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"].update(
        {
            "description": "数据库别名",
            "enum": ["memory", "rag"],
            "x-requires-explicit-user-input": True,
        }
    )

    update = validate_tool_calls_from_state(
        state={
            "question": "帮我查一下数据库中都有哪些表",
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "database_name": "memory",
                        "table_name": "dogs",
                    },
                }
            ],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: catalog,
        }
    )

    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY][0]["code"] == (
        "implicit_required_arg"
    )
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY][0]["value"] == "memory"
    assert update[TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY]["missing_fields"] == [
        "database_name"
    ]
    assert update[TOOL_AGENT_PENDING_TOOL_CALL_STATE_KEY]["args"] == {
        "table_name": "dogs",
    }


def test_explicit_arg_in_current_question_should_pass_validation() -> None:
    """
    测试当前问题明确给出的参数可以通过显式输入契约。

    功能：
        用户本轮明确提到 memory 时，database_name=memory 不应被误判为猜测值。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"].update(
        {
            "enum": ["memory", "rag"],
            "x-requires-explicit-user-input": True,
        }
    )

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            {
                "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                "args": {
                    "database_name": "memory",
                    "table_name": "dogs",
                },
            }
        ],
        tool_catalog=catalog,
        question="帮我查看 memory 数据库中的 dogs 表",
    )

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_resolved_clarification_args_should_pass_explicit_contract() -> None:
    """测试经过前序澄清确认的参数不会被当成本轮 LLM 猜测值。"""

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"].update(
        {
            "enum": ["memory", "rag"],
            "x-requires-explicit-user-input": True,
        }
    )

    update = validate_tool_calls_from_state(
        state={
            "question": "dogs",
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "database_name": "memory",
                        "table_name": "dogs",
                    },
                }
            ],
            "tool_agent_clarification_resolution": {
                "action": "resumed",
            },
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: catalog,
        }
    )

    assert update[TOOL_CALL_VALIDATION_OK_STATE_KEY] is True
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY] == []


def test_invalid_allowed_value_should_not_create_clarification_state() -> None:
    """
    测试非法参数值仍按普通校验失败处理。

    功能：
        database_name 已提供但不在白名单时，不应误判为需要用户补参。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_test_tool_catalog()
    catalog[1]["input_schema"]["properties"]["database_name"]["enum"] = [
        "memory",
        "rag",
    ]
    update = validate_tool_calls_from_state(
        state={
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "database_name": "unknown",
                        "table_name": "dogs",
                    },
                }
            ],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: catalog,
        }
    )

    assert update[TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY] is None
    assert update[TOOL_AGENT_PENDING_TOOL_CALL_STATE_KEY] is None
    assert update["tool_results"][0]["success"] is False
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY][0]["code"] == "invalid_arg_value"


def test_validate_tool_calls_against_catalog_should_skip_when_catalog_missing() -> None:
    """
    测试工具目录缺失时跳过校验。

    功能：
        当前阶段为了不破坏旧链路，工具目录为空时保留原始 tool_calls，
        但通过 validation_skipped 标记说明本次没有真正校验。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_tool_calls_against_catalog(
        tool_calls=[
            ToolCall(
                name="weather",
                args={
                    "city": "成都",
                },
            )
        ],
        tool_catalog=[],
    )

    assert result["is_valid"] is True
    assert result["validation_skipped"] is True
    assert result["valid_tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert result["errors"][0]["code"] == "tool_catalog_missing"


def test_empty_tool_calls_should_not_create_failed_tool_result() -> None:
    """
    测试普通问题没有工具调用时不生成失败结果。

    功能：
        tool_calls 为空表示没有需要校验的调用，不应该被误判成校验失败。

    参数：
        无。

    返回值：
        None。
    """

    update = validate_tool_calls_from_state(
        state={
            "need_tool": False,
            "tool_calls": [],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: build_test_tool_catalog(),
        }
    )

    assert update[TOOL_CALL_VALIDATION_OK_STATE_KEY] is True
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY] == []
    assert "tool_results" not in update


def test_validate_tool_calls_from_state_should_return_state_update() -> None:
    """
    测试从 state 校验工具调用并返回 state update。

    功能：
        确认适配器输出可以直接作为 LangGraph state update 使用。

    参数：
        无。

    返回值：
        None。
    """

    update = validate_tool_calls_from_state(
        state={
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "database_name": "memory",
                        "table_name": "dogs",
                    },
                }
            ],
            TOOL_AGENT_TOOL_CATALOG_STATE_KEY: build_test_tool_catalog(),
        }
    )

    assert update["tool_calls"] == [
        {
            "name": SQLITE_SELECT_ROWS_TOOL_NAME,
            "args": {
                "database_name": "memory",
                "table_name": "dogs",
            },
        }
    ]
    assert update[TOOL_CALL_VALIDATION_OK_STATE_KEY] is True
    assert update[TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY] is False
    assert update[TOOL_CALL_VALIDATION_ERRORS_STATE_KEY] == []
    assert update[TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY] == []


def test_build_tool_catalog_by_name_should_ignore_invalid_items() -> None:
    """
    测试工具目录索引会忽略非法条目。

    功能：
        非 Mapping、缺少 name 或 name 为空的目录项不进入索引。

    参数：
        无。

    返回值：
        None。
    """

    catalog_by_name = build_tool_catalog_by_name(
        tool_catalog=[
            "bad",
            {},
            {
                "name": "",
            },
            {
                "name": "weather",
                "description": "查询天气",
            },
        ]
    )

    assert set(
        catalog_by_name
    ) == {
        "weather",
    }


def test_read_required_fields_should_ignore_invalid_required_items() -> None:
    """
    测试读取 required 时忽略非法字段。

    功能：
        required 中只有字符串字段会被保留。

    参数：
        无。

    返回值：
        None。
    """

    assert read_required_fields(
        input_schema={
            "required": [
                "database_name",
                123,
                None,
            ],
        }
    ) == [
        "database_name",
    ]


def test_is_value_matching_json_schema_type_should_support_basic_types() -> None:
    """
    测试基础 JSON Schema 类型判断。

    功能：
        覆盖 string、integer、number、boolean、object、array 的基础判断。

    参数：
        无。

    返回值：
        None。
    """

    assert is_value_matching_json_schema_type(
        value="成都",
        expected_type="string",
    ) is True
    assert is_value_matching_json_schema_type(
        value=5,
        expected_type="integer",
    ) is True
    assert is_value_matching_json_schema_type(
        value=True,
        expected_type="integer",
    ) is False
    assert is_value_matching_json_schema_type(
        value=1.5,
        expected_type="number",
    ) is True
    assert is_value_matching_json_schema_type(
        value=False,
        expected_type="boolean",
    ) is True
    assert is_value_matching_json_schema_type(
        value={},
        expected_type="object",
    ) is True
    assert is_value_matching_json_schema_type(
        value=[],
        expected_type="array",
    ) is True
