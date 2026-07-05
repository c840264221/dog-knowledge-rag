"""
V1.7.5 Checkpoint Resume Smoke Test 脚本。

功能：
    使用真实主图验证 checkpoint resume（检查点恢复）链路：
    1. 第一次运行工具类问题，预期触发 GraphInterruptResult。
    2. 模拟新的请求回合，从 checkpoint 恢复 RuntimeContext。
    3. 使用 resume_value 显式恢复 LangGraph 执行，不拼接旧 RESUME 字符串。
    4. 验证最终返回 GraphFinalResult。
    5. 验证 MetricsScope 中存在 tool_count 和 tool_before_hook_count。

使用方式：
    python -m scripts.smoke_v175_checkpoint_resume

专业名词：
    Smoke Test（冒烟测试）：
        用最小真实场景验证主链路是否能跑通。
    Checkpoint（检查点）：
        用于保存图执行状态，方便中断后继续运行。
    Resume（恢复运行）：
        从中断点继续执行图。
    RuntimeContext（运行时上下文）：
        当前请求的运行时对象，保存 trace、metrics、timeline 等运行期信息。
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import src.runtime.events.setup  # noqa: F401
from src.graph.graph_run import run_main_graph_with_result
from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.hooks.tool_counter_hook import ToolCounterHook
from src.runtime.resume.contracts import GraphFinalResult, GraphInterruptResult
from src.runtime.scopes.metrics_scope import MetricsScope
from src.runtime.trace.init import trace_manager


@dataclass(frozen=True)
class CheckpointResumeSmokeReport:
    """
    V1.7.5 checkpoint resume smoke 报告。

    功能：
        保存 smoke test 的核心验证结果，方便统一渲染 Markdown 报告。

    参数含义：
        passed:
            smoke test 是否通过。
        interrupt_count:
            运行过程中遇到的中断次数。
        final_answer_preview:
            最终答案预览。
        metrics:
            RuntimeContext 中的 MetricsScope 指标。
        errors:
            smoke test 过程中收集到的错误信息。

    返回值含义：
        CheckpointResumeSmokeReport:
            dataclass 数据对象，无额外方法返回值。
    """

    passed: bool
    interrupt_count: int
    final_answer_preview: str
    metrics: dict[str, Any]
    errors: list[str]


def ensure_trace_exists(
    trace_id: str,
) -> None:
    """
    确保 trace_manager 中存在当前 trace。

    功能：
        在新请求或 resume 前检查 trace_manager。
        如果当前 trace_id 对应的 Trace 不存在，则重新创建一个。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。

    返回值含义：
        None:
            无业务返回值，只保证 trace_manager 内部状态存在。
    """

    trace_manager.ensure_trace(
        trace_id
    )


async def create_initial_runtime_context(
    trace_id: str,
    thread_id: str,
    user_id: str,
) -> RuntimeContext:
    """
    创建首次请求使用的 RuntimeContext。

    功能：
        创建 RuntimeContext，注册 ToolCounterHook，并写入 contextvar。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。
        thread_id:
            LangGraph thread_id，同时模拟会话 ID。
        user_id:
            smoke test 用户 ID。

    返回值含义：
        RuntimeContext:
            已 startup 并写入 runtime_ctx 的运行时上下文。
    """

    ctx = RuntimeContext(
        trace_id=trace_id,
        session_id=thread_id,
        user_id=user_id,
        component="smoke_v175_initial_request",
    )
    ctx.hooks().register(
        "tool.before",
        ToolCounterHook(),
    )

    await runtime_ctx.create(ctx)

    return ctx


async def restore_runtime_context_for_resume(
    trace_id: str,
    thread_id: str,
    user_id: str,
    checkpoint_manager: Any,
) -> RuntimeContext:
    """
    从 checkpoint 恢复 resume 阶段的 RuntimeContext。

    功能：
        优先从 checkpoint_manager 恢复 RuntimeContext。
        恢复后重新补齐 trace_id、session_id、user_id、component 和 ToolCounterHook。
        如果 checkpoint 缺失，则抛出 RuntimeError，让 smoke test 明确失败。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。
        thread_id:
            LangGraph thread_id。
        user_id:
            smoke test 用户 ID。
        checkpoint_manager:
            Runtime checkpoint manager，用于恢复 RuntimeContext 快照。

    返回值含义：
        RuntimeContext:
            已 startup 并写入 runtime_ctx 的恢复上下文。
    """

    ctx = checkpoint_manager.restore_checkpoint(
        trace_id
    )

    if ctx is None:
        raise RuntimeError(
            "checkpoint 中没有恢复到 RuntimeContext。"
        )

    ctx.trace_id = trace_id
    ctx.session_id = thread_id
    ctx.user_id = user_id
    ctx.component = "smoke_v175_resume_request"
    ctx.hooks().register(
        "tool.before",
        ToolCounterHook(),
    )

    await runtime_ctx.create(ctx)

    return ctx


async def reset_current_runtime_context() -> None:
    """
    重置当前 RuntimeContext。

    功能：
        如果当前 contextvar 中存在 RuntimeContext，则调用 runtime_ctx.destroy。
        该函数用于模拟 UI 中断后进入下一次请求回合的恢复过程。

    参数含义：
        无。

    返回值含义：
        None。
    """

    ctx = runtime_ctx.get()

    if ctx is not None:
        await runtime_ctx.destroy()


def get_runtime_metrics() -> dict[str, Any]:
    """
    获取当前 RuntimeContext 中的 metrics。

    功能：
        从 MetricsScope 读取当前运行指标，并复制成普通 dict 返回。

    参数含义：
        无。

    返回值含义：
        dict[str, Any]:
            当前运行时指标。
    """

    ctx = runtime_ctx.get()

    if ctx is None:
        return {}

    return dict(
        ctx.service(
            MetricsScope
        ).get_metrics()
    )


def validate_smoke_result(
    final_result: GraphFinalResult | GraphInterruptResult | None,
    interrupt_count: int,
    metrics: dict[str, Any],
) -> list[str]:
    """
    校验 checkpoint resume smoke 结果。

    功能：
        检查是否至少发生一次中断、最终是否完成、工具指标和 hook 指标是否存在。

    参数含义：
        final_result:
            最后一次 graph_run 返回的结构化结果。
        interrupt_count:
            运行过程中累计中断次数。
        metrics:
            RuntimeContext 中的 MetricsScope 指标。

    返回值含义：
        list[str]:
            错误信息列表。空列表表示校验通过。
    """

    errors: list[str] = []

    if interrupt_count < 1:
        errors.append(
            "预期至少触发一次 GraphInterruptResult，但实际没有中断。"
        )

    if not isinstance(
        final_result,
        GraphFinalResult,
    ):
        errors.append(
            "最终结果不是 GraphFinalResult。"
        )

    if metrics.get(
        "tool_count",
        0,
    ) < 1:
        errors.append(
            "MetricsScope.tool_count 小于 1，说明工具成功统计未写入。"
        )

    if metrics.get(
        "tool_before_hook_count",
        0,
    ) < 1:
        errors.append(
            "MetricsScope.tool_before_hook_count 小于 1，说明 ToolCounterHook 未触发。"
        )

    return errors


def render_report(
    report: CheckpointResumeSmokeReport,
) -> str:
    """
    渲染 checkpoint resume smoke Markdown 报告。

    功能：
        把 smoke test 结果格式化成 Markdown 文本，方便终端查看。

    参数含义：
        report:
            CheckpointResumeSmokeReport 数据对象。

    返回值含义：
        str:
            Markdown 格式报告。
    """

    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# V1.7.5 Checkpoint Resume Smoke Report",
        "",
        f"- status: {status}",
        f"- interrupt_count: {report.interrupt_count}",
        f"- final_answer_preview: {report.final_answer_preview}",
        f"- tool_count: {report.metrics.get('tool_count', 0)}",
        f"- tool_before_hook_count: {report.metrics.get('tool_before_hook_count', 0)}",
        "",
        "## Metrics",
        "",
    ]

    for key, value in sorted(
        report.metrics.items()
    ):
        lines.append(
            f"- {key}: {value}"
        )

    if report.errors:
        lines.extend(
            [
                "",
                "## Errors",
                "",
            ]
        )

        for error in report.errors:
            lines.append(
                f"- {error}"
            )

    return "\n".join(lines)


async def run_smoke(
    question: str,
    resume_value: str,
    max_resumes: int,
) -> CheckpointResumeSmokeReport:
    """
    执行 V1.7.5 checkpoint resume smoke test。

    功能：
        启动真实 runtime container，运行一次工具类问题，
        遇到中断后模拟新请求恢复 RuntimeContext，
        并使用 resume_value 显式恢复 LangGraph 执行。

    参数含义：
        question:
            用于触发工具链路的问题。
        resume_value:
            用户确认值，默认 y。
        max_resumes:
            最多允许恢复次数，避免图异常时无限循环。

    返回值含义：
        CheckpointResumeSmokeReport:
            smoke test 报告对象。
    """

    from src.runtime.container.init import container

    trace_id = f"smoke_v175_{uuid.uuid4().hex}"
    thread_id = f"smoke_v175_thread_{uuid.uuid4().hex}"
    user_id = "smoke_user_v175"
    interrupt_count = 0
    current_result: GraphFinalResult | GraphInterruptResult | None = None

    await container.startup()

    try:
        checkpoint_manager = container.get(
            "checkpoint"
        ).manager

        ensure_trace_exists(
            trace_id
        )
        await create_initial_runtime_context(
            trace_id=trace_id,
            thread_id=thread_id,
            user_id=user_id,
        )

        current_result = await run_main_graph_with_result(
            question=question,
            thread_id=thread_id,
            trace_id=trace_id,
        )

        while isinstance(
            current_result,
            GraphInterruptResult,
        ):
            interrupt_count += 1

            if interrupt_count > max_resumes:
                break

            await reset_current_runtime_context()
            ensure_trace_exists(
                trace_id
            )
            await restore_runtime_context_for_resume(
                trace_id=trace_id,
                thread_id=thread_id,
                user_id=user_id,
                checkpoint_manager=checkpoint_manager,
            )

            current_result = await run_main_graph_with_result(
                question=resume_value,
                thread_id=thread_id,
                trace_id=trace_id,
                resume_value=resume_value,
            )

        metrics = get_runtime_metrics()
        errors = validate_smoke_result(
            final_result=current_result,
            interrupt_count=interrupt_count,
            metrics=metrics,
        )

        final_answer_preview = (
            current_result.answer[:120]
            if isinstance(
                current_result,
                GraphFinalResult,
            )
            else ""
        )

        checkpoint_manager.clear_checkpoint(
            trace_id
        )

        return CheckpointResumeSmokeReport(
            passed=not errors,
            interrupt_count=interrupt_count,
            final_answer_preview=final_answer_preview,
            metrics=metrics,
            errors=errors,
        )

    finally:
        await reset_current_runtime_context()
        await container.shutdown()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    功能：
        读取 smoke test 的问题、确认值和最大恢复次数。

    参数含义：
        无。

    返回值含义：
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="V1.7.5 checkpoint resume smoke test."
    )
    parser.add_argument(
        "--question",
        default="今天成都的天气怎么样？",
        help="用于触发工具调用和中断恢复链路的问题。",
    )
    parser.add_argument(
        "--resume-value",
        default="y",
        help="用于恢复 interrupt 的确认值。",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=3,
        help="最多允许恢复次数，避免异常链路无限循环。",
    )

    return parser.parse_args()


def main() -> int:
    """
    脚本入口函数。

    功能：
        解析参数，运行异步 smoke test，打印报告并返回退出码。

    参数含义：
        无。

    返回值含义：
        int:
            0 表示通过；1 表示失败。
    """

    args = parse_args()
    report = asyncio.run(
        run_smoke(
            question=args.question,
            resume_value=args.resume_value,
            max_resumes=args.max_resumes,
        )
    )

    print(
        render_report(report)
    )

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
