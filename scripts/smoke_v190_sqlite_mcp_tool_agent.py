"""
V1.9.0 SQLite MCP ToolAgent Smoke 脚本。

功能：
    运行 SQLite MCP ToolAgent 冒烟检查，验证 MCP 工具能从工具目录、
    工具校验、工具执行到最终响应完整跑通。

运行方式：
    python -m scripts.smoke_v190_sqlite_mcp_tool_agent

专业名词：
    MCP：
        Model Context Protocol，模型上下文协议。
    Smoke Test：
        冒烟测试，用少量关键场景验证主链路是否明显断裂。
"""

from __future__ import annotations

import asyncio

from src.agents.tool_agent.smoke.v190_sqlite_mcp_smoke_checks import (
    SQLiteMcpToolAgentSmokeResult,
    run_v190_sqlite_mcp_tool_agent_smoke_check,
)


def render_report(
    result: SQLiteMcpToolAgentSmokeResult,
) -> str:
    """
    渲染 SQLite MCP ToolAgent smoke 报告。

    功能：
        将 smoke 结果转换成终端可读 Markdown 文本。

    参数：
        result:
            SQLite MCP ToolAgent smoke 结果。

    返回值：
        str:
            Markdown 格式报告。
    """

    status = "PASS" if result.passed else "FAIL"
    lines = [
        "# V1.9.0 SQLite MCP ToolAgent Smoke Report",
        "",
        f"- status: {status}",
        f"- tool_catalog_count: {result.tool_catalog_count}",
        f"- tool_result_count: {result.tool_result_count}",
        f"- validation_ok: {result.validation_ok}",
        f"- final_answer_preview: {result.final_answer_preview}",
        "",
    ]

    if result.errors:
        lines.extend(
            [
                "## Errors",
                "",
            ]
        )

        for error in result.errors:
            lines.append(
                f"- {error}"
            )

    return "\n".join(
        lines
    )


async def async_main() -> int:
    """
    异步脚本入口。

    功能：
        执行 SQLite MCP ToolAgent smoke，并打印 Markdown 报告。

    参数：
        无。

    返回值：
        int:
            0 表示通过，1 表示失败。
    """

    result = await run_v190_sqlite_mcp_tool_agent_smoke_check()
    print(
        render_report(
            result=result,
        )
    )

    return 0 if result.passed else 1


def main() -> int:
    """
    同步脚本入口。

    功能：
        使用 asyncio.run 启动异步 smoke。

    参数：
        无。

    返回值：
        int:
            脚本退出码。
    """

    return asyncio.run(
        async_main()
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
