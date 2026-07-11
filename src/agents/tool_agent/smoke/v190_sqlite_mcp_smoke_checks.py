"""
V1.9.0 ToolAgent SQLite MCP 冒烟检查。

功能：
    使用 mock parser 和临时 SQLite 数据库验证 ToolAgent 的 MCP 工具链路：
    1. tool_catalog_node 可以把 SQLite MCP 工具写入工具目录。
    2. tool_validate_node 可以校验 MCP tool_call。
    3. tool_execute_node 可以根据 source=mcp 调用 SQLite MCP tool_client。
    4. tool_answer_node 和 response_adapter 可以生成稳定响应。

专业名词：
    MCP：
        Model Context Protocol，模型上下文协议。
    Smoke Check：
        冒烟检查，用少量关键场景快速确认主能力没有明显断裂。
    Mock Parser：
        模拟解析器，用固定输出替代 LLM，保证 smoke 稳定。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.agents.tool_agent.agent import build_tool_agent
from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
)
from src.mcp.sqlite.tool_definitions import (
    SQLITE_LIST_TABLES_TOOL_NAME,
)
from src.runtime.container.providers.sqlite_mcp_provider import (
    SQLiteMcpProvider,
)


SMOKE_DATABASE_NAME = "smoke"
SMOKE_TABLE_NAME = "dogs"


@dataclass(frozen=True)
class SQLiteMcpToolAgentSmokeResult:
    """
    SQLite MCP ToolAgent 冒烟结果。

    功能：
        保存一次 smoke 检查的关键输出和错误信息。

    参数：
        passed:
            是否通过。

        final_answer_preview:
            最终答案预览。

        tool_catalog_count:
            ToolAgent 工具目录条目数量。

        tool_result_count:
            工具执行结果数量。

        validation_ok:
            工具调用校验是否通过。

        errors:
            校验错误列表。

    返回值：
        SQLiteMcpToolAgentSmokeResult:
            dataclass 数据对象本身。
    """

    passed: bool
    final_answer_preview: str
    tool_catalog_count: int
    tool_result_count: int
    validation_ok: bool
    errors: list[str]


class FakeSQLiteMcpParser:
    """
    测试用 SQLite MCP parser。

    功能：
        模拟 ToolAgent LLM 工具解析结果，固定输出 sqlite_list_tables 调用。
        这样 smoke 不依赖真实 LLM，重点验证 MCP 工具执行链路。

    参数：
        无。

    返回值：
        FakeSQLiteMcpParser:
            测试用 parser。
    """

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        parser_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        模拟异步工具解析。

        功能：
            记录 parser 输入，并返回固定 sqlite_list_tables tool_call。

        参数：
            parser_input:
                ToolAgent 工具解析节点传入的解析上下文。

        返回值：
            dict[str, Any]:
                模拟 LLM 解析出的工具调用结果。
        """

        self.inputs.append(
            parser_input
        )

        return {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": SQLITE_LIST_TABLES_TOOL_NAME,
                    "args": {
                        "database_name": SMOKE_DATABASE_NAME,
                    },
                }
            ],
            "tool_reason": "用户想查看 SQLite 数据库中有哪些表。",
        }


def create_smoke_sqlite_database(
    database_path: Path,
) -> None:
    """
    创建 smoke 用 SQLite 数据库。

    功能：
        在临时路径创建 SQLite 数据库，并写入一张 dogs 表。
        该数据库只用于 smoke 检查，不依赖用户真实项目数据库。

    参数：
        database_path:
            要创建的 SQLite 数据库文件路径。

    返回值：
        None。
    """

    connection = sqlite3.connect(
        database_path
    )

    try:
        connection.execute(
            "CREATE TABLE dogs (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO dogs (name) VALUES (?)",
            (
                "Golden Retriever",
            ),
        )
        connection.commit()
    finally:
        connection.close()


async def run_v190_sqlite_mcp_tool_agent_smoke_check() -> SQLiteMcpToolAgentSmokeResult:
    """
    执行 V1.9.0 SQLite MCP ToolAgent 冒烟检查。

    功能：
        创建临时 SQLite 数据库，构建 SQLiteMcpProvider，
        再用 mock parser 驱动 ToolAgent 调用 sqlite_list_tables。

    参数：
        无。

    返回值：
        SQLiteMcpToolAgentSmokeResult:
            smoke 检查结果。
    """

    with TemporaryDirectory() as temporary_directory:
        database_path = Path(
            temporary_directory
        ) / "tool_agent_sqlite_mcp_smoke.sqlite3"
        create_smoke_sqlite_database(
            database_path=database_path,
        )

        sqlite_mcp_provider = SQLiteMcpProvider(
            allowed_databases={
                SMOKE_DATABASE_NAME: database_path,
            },
            default_limit=10,
            max_limit=20,
        )
        await sqlite_mcp_provider.startup()

        tool_agent = build_tool_agent(
            parser=FakeSQLiteMcpParser(),
            sqlite_mcp_provider=sqlite_mcp_provider,
            runtime_context_getter=lambda: None,
        )

        result_state = await tool_agent(
            {
                # smoke 是当前测试注册的数据库别名，必须由本轮问题明确提供。
                "question": "帮我看看 smoke SQLite 数据库里有哪些表",
            }
        )

        return validate_sqlite_mcp_tool_agent_state(
            state=result_state,
        )


def validate_sqlite_mcp_tool_agent_state(
    state: dict[str, Any],
) -> SQLiteMcpToolAgentSmokeResult:
    """
    校验 SQLite MCP ToolAgent 最终 state。

    功能：
        检查工具目录、工具校验、工具执行结果和最终响应是否符合预期。

    参数：
        state:
            ToolAgent 执行后的完整 state。

    返回值：
        SQLiteMcpToolAgentSmokeResult:
            smoke 检查结果。
    """

    errors: list[str] = []
    tool_catalog = state.get(
        TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
        [],
    )
    tool_results = state.get(
        "tool_results",
        [],
    )
    final_answer = str(
        state.get(
            "final_answer",
            "",
        )
        or ""
    )
    validation_ok = bool(
        state.get(
            "tool_call_validation_ok",
            False,
        )
    )
    response = state.get(
        TOOL_AGENT_RESPONSE_STATE_KEY,
        {},
    )

    if not isinstance(
        tool_catalog,
        list,
    ):
        errors.append(
            "tool_agent_tool_catalog 缺失或不是列表。"
        )
        tool_catalog = []

    sqlite_list_tables_item = find_tool_catalog_item(
        tool_catalog=tool_catalog,
        tool_name=SQLITE_LIST_TABLES_TOOL_NAME,
    )

    if sqlite_list_tables_item is None:
        errors.append(
            "工具目录中没有 sqlite_list_tables。"
        )
    elif sqlite_list_tables_item.get(
        "source"
    ) != "mcp":
        errors.append(
            "sqlite_list_tables 的 source 不是 mcp。"
        )

    if not validation_ok:
        errors.append(
            "tool_call_validation_ok 不是 True。"
        )

    if not tool_results:
        errors.append(
            "tool_results 为空，说明 MCP 工具没有执行成功。"
        )
    else:
        first_tool_result = tool_results[0]
        if first_tool_result.get(
            "tool_name"
        ) != SQLITE_LIST_TABLES_TOOL_NAME:
            errors.append(
                "tool_results[0].tool_name 不是 sqlite_list_tables。"
            )

        if first_tool_result.get(
            "metadata",
            {},
        ).get(
            "source"
        ) != "mcp":
            errors.append(
                "tool_results[0].metadata.source 不是 mcp。"
            )

        if SMOKE_TABLE_NAME not in str(
            first_tool_result.get(
                "content",
                "",
            )
        ):
            errors.append(
                "工具结果中没有 smoke 表 dogs。"
            )

    if SMOKE_TABLE_NAME not in final_answer:
        errors.append(
            "final_answer 中没有 smoke 表 dogs。"
        )

    if not isinstance(
        response,
        dict,
    ) or response.get(
        "status"
    ) != "completed":
        errors.append(
            "tool_agent_response.status 不是 completed。"
        )

    return SQLiteMcpToolAgentSmokeResult(
        passed=not errors,
        final_answer_preview=final_answer[:120],
        tool_catalog_count=len(
            tool_catalog
        ),
        tool_result_count=len(
            tool_results
        ),
        validation_ok=validation_ok,
        errors=errors,
    )


def find_tool_catalog_item(
    tool_catalog: list[Any],
    tool_name: str,
) -> dict[str, Any] | None:
    """
    从工具目录中查找指定工具。

    功能：
        遍历工具目录，返回 name 等于 tool_name 的字典条目。

    参数：
        tool_catalog:
            ToolAgent 工具目录列表。

        tool_name:
            要查找的工具名称。

    返回值：
        dict[str, Any] | None:
            找到返回工具目录条目，找不到返回 None。
    """

    for item in tool_catalog:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get(
            "name"
        ) == tool_name:
            return item

    return None


def assert_v190_sqlite_mcp_tool_agent_smoke_check(
    result: SQLiteMcpToolAgentSmokeResult,
) -> SQLiteMcpToolAgentSmokeResult:
    """
    断言 V1.9.0 SQLite MCP ToolAgent 冒烟结果。

    功能：
        如果 smoke 未通过，则抛出 AssertionError。
        如果通过，则原样返回 result。

    参数：
        result:
            smoke 检查结果。

    返回值：
        SQLiteMcpToolAgentSmokeResult:
            原始 smoke 检查结果。
    """

    if result.passed:
        return result

    failure_text = "\n".join(
        f"- {error}"
        for error in result.errors
    )
    raise AssertionError(
        f"V1.9.0 SQLite MCP ToolAgent 冒烟测试失败:\n{failure_text}"
    )
