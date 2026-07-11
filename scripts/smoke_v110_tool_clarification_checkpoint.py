"""
V1.10 ToolAgent 多轮澄清 Checkpoint 真实主图冒烟脚本。

运行方式：
    python -m scripts.smoke_v110_tool_clarification_checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

import src.runtime.events.setup  # noqa: F401
from src.agents.tool_agent.smoke.v110_clarification_checkpoint_smoke_checks import (
    ClarificationCheckpointSmokeResult,
    validate_clarification_checkpoint_smoke,
)
from src.graph.graph_run import run_main_graph_with_result
from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.hooks.tool_counter_hook import ToolCounterHook
from src.runtime.resume.contracts import GraphFinalResult
from src.runtime.trace.init import trace_manager
from src.mcp.sqlite.tool_definitions import SQLITE_LIST_TABLES_TOOL_NAME


DEFAULT_FIRST_QUESTION = (
    "请帮我查一下数据库里有哪些表。"
    # "我还没有指定数据库，请不要猜测 database_name，也不要自动选择数据库。"
)


class SmokeMissingDatabaseParser:
    """
    冒烟测试专用缺参工具解析器。

    功能：
        固定生成缺少 database_name 的 sqlite_list_tables 调用，
        保证冒烟测试稳定进入参数澄清分支，而不依赖 LLM 是否主动猜测数据库。

    参数：
        无。

    返回值：
        SmokeMissingDatabaseParser:
            可通过 ainvoke 调用的异步测试解析器。
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(
        self,
        parser_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        返回固定的缺参工具调用。

        参数：
            parser_input:
                ToolAgent 传入的解析上下文。本测试只记录调用次数。

        返回值：
            dict[str, Any]:
                缺少 database_name 的标准工具解析结果。
        """

        self.call_count += 1
        return {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": SQLITE_LIST_TABLES_TOOL_NAME,
                    "args": {},
                }
            ],
            "response": "",
        }


async def create_request_runtime_context(
    trace_id: str,
    thread_id: str,
    component: str,
) -> RuntimeContext:
    """
    为一轮独立请求创建 RuntimeContext。

    参数：
        trace_id:
            当前请求追踪 ID。
        thread_id:
            当前对话线程 ID，同时写入 session_id。
        component:
            当前 smoke 阶段名称。

    返回值：
        RuntimeContext:
            已写入 runtime_ctx 的运行时上下文。
    """

    trace_manager.ensure_trace(
        trace_id
    )
    context = RuntimeContext(
        trace_id=trace_id,
        session_id=thread_id,
        user_id="smoke_user_v110",
        component=component,
    )
    context.hooks().register(
        "tool.before",
        ToolCounterHook(),
    )
    await runtime_ctx.create(
        context
    )
    return context


async def destroy_request_runtime_context() -> None:
    """
    销毁当前请求的 RuntimeContext。

    参数：
        无。

    返回值：
        None。
    """

    if runtime_ctx.get() is not None:
        await runtime_ctx.destroy()


async def read_graph_checkpoint_state(
    graph: Any,
    thread_id: str,
) -> dict[str, Any]:
    """
    按 thread_id 读取 LangGraph 最新 Checkpoint state。

    参数：
        graph:
            已编译的真实主图。
        thread_id:
            要读取的 LangGraph 对话线程 ID。

    返回值：
        dict[str, Any]:
            Checkpoint 中的 state 普通字典。
    """

    snapshot = await graph.aget_state(
        {
            "configurable": {
                "thread_id": thread_id,
            }
        }
    )
    values = getattr(
        snapshot,
        "values",
        {},
    )
    return dict(values) if isinstance(values, Mapping) else {}


def render_report(
    result: ClarificationCheckpointSmokeResult,
    thread_id: str,
    first_trace_id: str,
    second_trace_id: str,
) -> str:
    """
    渲染 V1.10 多轮澄清 Markdown 报告。

    参数：
        result:
            结构化冒烟结果。
        thread_id:
            两轮共用的 thread_id。
        first_trace_id:
            第一轮 trace_id。
        second_trace_id:
            第二轮 trace_id。

    返回值：
        str:
            终端可读 Markdown 报告。
    """

    lines = [
        "# V1.10 ToolAgent Clarification Checkpoint Smoke Report",
        "",
        f"- status: {'PASS' if result.passed else 'FAIL'}",
        f"- thread_id: {thread_id}",
        f"- first_trace_id: {first_trace_id}",
        f"- second_trace_id: {second_trace_id}",
        f"- same_thread_id: {result.same_thread_id}",
        f"- different_trace_ids: {result.different_trace_ids}",
        f"- clarification_saved: {result.clarification_saved}",
        f"- awaiting_clarification: {result.awaiting_clarification}",
        f"- pending_call_saved: {result.pending_call_saved}",
        f"- clarification_resumed: {result.clarification_resumed}",
        f"- tool_executed: {result.tool_executed}",
        f"- clarification_cleared: {result.clarification_cleared}",
        f"- final_answer_preview: {result.final_answer_preview}",
    ]

    if result.errors:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                *[
                    f"- {error}"
                    for error in result.errors
                ],
            ]
        )

    return "\n".join(
        lines
    )


async def run_smoke(
    first_question: str,
    clarification_value: str,
) -> tuple[
    ClarificationCheckpointSmokeResult,
    str,
    str,
    str,
]:
    """
    使用真实主图连续执行两轮多轮澄清请求。

    参数：
        first_question:
            第一轮用于触发缺参澄清的问题。
        clarification_value:
            第二轮用于补全 database_name 的候选值。

    返回值：
        tuple:
            冒烟结果、共用 thread_id、第一轮 trace_id、第二轮 trace_id。
    """

    from src.runtime.container.init import container

    thread_id = f"smoke_v110_thread_{uuid.uuid4().hex}"
    first_trace_id = f"smoke_v110_first_{uuid.uuid4().hex}"
    second_trace_id = f"smoke_v110_second_{uuid.uuid4().hex}"
    first_state: dict[str, Any] = {}
    second_state: dict[str, Any] = {}
    final_answer = ""
    runtime_error = ""
    container_started = False
    checkpoint_manager: Any | None = None
    graph_runtime_service: Any | None = None
    original_tool_parser: Any | None = None

    try:
        graph_runtime_service = container.get(
            "graph_runtime"
        )
        original_tool_parser = graph_runtime_service.tool_parser
        graph_runtime_service.tool_parser = SmokeMissingDatabaseParser()

        await container.startup()
        container_started = True
        graph = container.get(
            "graph_runtime"
        ).graph
        checkpoint_manager = container.get(
            "checkpoint"
        ).manager

        await create_request_runtime_context(
            trace_id=first_trace_id,
            thread_id=thread_id,
            component="smoke_v110_first_turn",
        )
        await run_main_graph_with_result(
            question=first_question,
            thread_id=thread_id,
            trace_id=first_trace_id,
        )
        first_state = await read_graph_checkpoint_state(
            graph=graph,
            thread_id=thread_id,
        )

        await destroy_request_runtime_context()
        await create_request_runtime_context(
            trace_id=second_trace_id,
            thread_id=thread_id,
            component="smoke_v110_second_turn",
        )
        second_result = await run_main_graph_with_result(
            question=clarification_value,
            thread_id=thread_id,
            trace_id=second_trace_id,
        )
        second_state = await read_graph_checkpoint_state(
            graph=graph,
            thread_id=thread_id,
        )

        if isinstance(
            second_result,
            GraphFinalResult,
        ):
            final_answer = second_result.answer

    except Exception as exc:
        runtime_error = str(
            exc
        )
    finally:
        await destroy_request_runtime_context()
        if checkpoint_manager is not None:
            try:
                checkpoint_manager.clear_checkpoint(
                    first_trace_id
                )
                checkpoint_manager.clear_checkpoint(
                    second_trace_id
                )
            except Exception as exc:
                if not runtime_error:
                    runtime_error = f"清理 Runtime Checkpoint 失败：{exc}"

        if container_started:
            try:
                await container.shutdown()
            except Exception as exc:
                if not runtime_error:
                    runtime_error = f"关闭 RuntimeContainer 失败：{exc}"

        if graph_runtime_service is not None:
            graph_runtime_service.tool_parser = original_tool_parser

    result = validate_clarification_checkpoint_smoke(
        first_state=first_state,
        second_state=second_state,
        first_thread_id=thread_id,
        second_thread_id=thread_id,
        first_trace_id=first_trace_id,
        second_trace_id=second_trace_id,
        final_answer=final_answer,
        runtime_error=runtime_error,
    )
    return (
        result,
        thread_id,
        first_trace_id,
        second_trace_id,
    )


def parse_args() -> argparse.Namespace:
    """
    解析冒烟脚本命令行参数。

    功能：
        读取第一轮问题和第二轮参数补充值。

    参数：
        无。

    返回值：
        argparse.Namespace:
            第一轮问题和第二轮澄清值。
    """

    parser = argparse.ArgumentParser(
        description="V1.10 ToolAgent clarification checkpoint smoke test."
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_FIRST_QUESTION,
        help="用于触发缺少 database_name 的第一轮问题。",
    )
    parser.add_argument(
        "--clarification-value",
        default="memory",
        help="第二轮补全的数据库候选值，默认 memory。",
    )
    return parser.parse_args()


def main() -> int:
    """
    运行真实主图冒烟脚本并返回进程退出码。

    功能：
        解析参数、执行异步双轮冒烟流程、打印 Markdown 报告。

    参数：
        无。

    返回值：
        int:
            通过返回 0，失败返回 1。
    """

    args = parse_args()
    result, thread_id, first_trace_id, second_trace_id = asyncio.run(
        run_smoke(
            first_question=args.question,
            clarification_value=args.clarification_value,
        )
    )
    print(
        render_report(
            result=result,
            thread_id=thread_id,
            first_trace_id=first_trace_id,
            second_trace_id=second_trace_id,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
