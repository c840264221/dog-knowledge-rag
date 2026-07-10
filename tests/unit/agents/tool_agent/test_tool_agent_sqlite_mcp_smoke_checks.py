"""
SQLite MCP ToolAgent smoke checks 测试。

功能：
    测试 V1.9.0 SQLite MCP ToolAgent 冒烟检查逻辑。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.smoke.v190_sqlite_mcp_smoke_checks import (
    SQLITE_LIST_TABLES_TOOL_NAME,
    SMOKE_TABLE_NAME,
    assert_v190_sqlite_mcp_tool_agent_smoke_check,
    find_tool_catalog_item,
    run_v190_sqlite_mcp_tool_agent_smoke_check,
    validate_sqlite_mcp_tool_agent_state,
)


def test_validate_sqlite_mcp_tool_agent_state_should_pass_valid_state() -> None:
    """
    测试合法 ToolAgent state 可以通过 smoke 校验。

    功能：
        构造包含工具目录、校验结果、MCP 工具结果和最终回答的 state，
        验证校验函数返回 passed=True。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_sqlite_mcp_tool_agent_state(
        state={
            "tool_agent_tool_catalog": [
                {
                    "name": SQLITE_LIST_TABLES_TOOL_NAME,
                    "source": "mcp",
                }
            ],
            "tool_call_validation_ok": True,
            "tool_results": [
                {
                    "success": True,
                    "tool_name": SQLITE_LIST_TABLES_TOOL_NAME,
                    "content": {
                        "tables": [
                            SMOKE_TABLE_NAME,
                        ],
                    },
                    "metadata": {
                        "source": "mcp",
                    },
                }
            ],
            "final_answer": f"sqlite_list_tables 工具返回：{SMOKE_TABLE_NAME}",
            "tool_agent_response": {
                "status": "completed",
            },
        }
    )

    assert result.passed is True
    assert result.errors == []
    assert result.tool_catalog_count == 1
    assert result.tool_result_count == 1


def test_validate_sqlite_mcp_tool_agent_state_should_report_errors() -> None:
    """
    测试非法 ToolAgent state 会返回错误。

    功能：
        构造缺少 MCP 工具目录和工具结果的 state，
        验证 smoke 校验能返回失败原因。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_sqlite_mcp_tool_agent_state(
        state={
            "tool_agent_tool_catalog": [],
            "tool_call_validation_ok": False,
            "tool_results": [],
            "final_answer": "",
            "tool_agent_response": {
                "status": "failed",
            },
        }
    )

    assert result.passed is False
    assert result.errors


def test_find_tool_catalog_item_should_return_matching_item() -> None:
    """
    测试工具目录查找函数。

    功能：
        从工具目录中找到指定 name 的条目。

    参数：
        无。

    返回值：
        None。
    """

    item = find_tool_catalog_item(
        tool_catalog=[
            {
                "name": "date",
                "source": "local",
            },
            {
                "name": SQLITE_LIST_TABLES_TOOL_NAME,
                "source": "mcp",
            },
        ],
        tool_name=SQLITE_LIST_TABLES_TOOL_NAME,
    )

    assert item == {
        "name": SQLITE_LIST_TABLES_TOOL_NAME,
        "source": "mcp",
    }


def test_assert_v190_sqlite_mcp_tool_agent_smoke_check_should_raise_on_failure() -> None:
    """
    测试 smoke 断言函数失败时抛出异常。

    功能：
        使用失败结果调用断言函数，验证会抛出 AssertionError。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_sqlite_mcp_tool_agent_state(
        state={}
    )

    with pytest.raises(
        AssertionError,
    ):
        assert_v190_sqlite_mcp_tool_agent_smoke_check(
            result=result,
        )


@pytest.mark.asyncio
async def test_run_v190_sqlite_mcp_tool_agent_smoke_check_should_pass() -> None:
    """
    测试真实 smoke 核心逻辑可以跑通。

    功能：
        创建临时 SQLite 数据库，构建 ToolAgent，
        并验证 sqlite_list_tables MCP 工具能执行成功。

    参数：
        无。

    返回值：
        None。
    """

    result = await run_v190_sqlite_mcp_tool_agent_smoke_check()

    assert result.passed is True
    assert result.validation_ok is True
    assert result.tool_result_count == 1
