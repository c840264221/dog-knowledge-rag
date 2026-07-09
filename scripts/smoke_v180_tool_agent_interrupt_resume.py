"""
V1.8.0 ToolAgent Interrupt Resume Smoke Test 脚本。

功能：
    使用真实主图验证 ToolAgent 工具确认中断和恢复链路：
    1. 天气问题进入 ToolAgent。
    2. weather 工具因为 require_confirm=True 触发 interrupt。
    3. 用户确认 y 后恢复执行工具，并返回最终答案。
    4. 用户拒绝 n 后不执行工具，并返回取消结果。

运行方式：
    python -m scripts.smoke_v180_tool_agent_interrupt_resume

专业名词：
    ToolAgent：
        工具智能体，负责解析工具、确认权限、执行工具、生成工具结果回答。
    Interrupt：
        中断，LangGraph 在需要用户输入时暂停图执行。
    Resume：
        恢复，用户输入确认值后使用 Command(resume=...) 继续执行图。
    Smoke Test：
        冒烟测试，用少量真实场景验证主链路是否跑通。
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
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
    GraphInterruptType,
)
from src.runtime.scopes.metrics_scope import MetricsScope
from src.runtime.trace.init import trace_manager


@dataclass(frozen=True)
class ToolAgentInterruptResumeScenario:
    """
    ToolAgent 中断恢复 smoke 场景。

    功能：
        描述一次真实主图 smoke 场景，包括用户问题、恢复值和工具执行预期。

    参数含义：
        name:
            场景名称。
        question:
            触发 ToolAgent 的用户问题。
        resume_value:
            模拟用户在确认面板输入的值，例如 y 或 n。
        expect_tool_execution:
            是否预期工具真正执行。

    返回值含义：
        ToolAgentInterruptResumeScenario:
            dataclass 数据对象，无额外方法返回值。
    """

    name: str
    question: str
    resume_value: str
    expect_tool_execution: bool


@dataclass(frozen=True)
class ToolAgentInterruptResumeReport:
    """
    ToolAgent 中断恢复 smoke 报告。

    功能：
        保存单个场景的运行结果，方便渲染终端 Markdown 报告。

    参数含义：
        scenario_name:
            场景名称。
        passed:
            场景是否通过。
        interrupt_count:
            运行过程中遇到的中断次数。
        final_answer_preview:
            最终答案预览。
        metrics:
            RuntimeContext 中的 MetricsScope 指标。
        errors:
            场景校验错误列表。

    返回值含义：
        ToolAgentInterruptResumeReport:
            dataclass 数据对象，无额外方法返回值。
    """

    scenario_name: str
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
        在首次运行和 resume 前检查 trace_manager。
        如果 trace_id 不存在，则创建对应 trace，避免恢复阶段缺少追踪对象。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。

    返回值含义：
        None。
    """

    trace_manager.ensure_trace(
        trace_id
    )


async def create_runtime_context(
    trace_id: str,
    thread_id: str,
    user_id: str,
    component: str,
) -> RuntimeContext:
    """
    创建并注册 RuntimeContext。

    功能：
        为一次 smoke 请求创建 RuntimeContext，并注册 ToolCounterHook。
        ToolCounterHook 用来验证工具执行 hook 是否真的被触发。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。
        thread_id:
            LangGraph thread_id，同时模拟会话 ID。
        user_id:
            smoke 用户 ID。
        component:
            当前运行组件名称，用于日志和调试区分首次运行 / 恢复运行。

    返回值含义：
        RuntimeContext:
            已写入 runtime_ctx 的运行时上下文。
    """

    ctx = RuntimeContext(
        trace_id=trace_id,
        session_id=thread_id,
        user_id=user_id,
        component=component,
    )
    ctx.hooks().register(
        "tool.before",
        ToolCounterHook(),
    )

    await runtime_ctx.create(
        ctx
    )

    return ctx


async def restore_runtime_context_for_resume(
    trace_id: str,
    thread_id: str,
    user_id: str,
    checkpoint_manager: Any,
) -> RuntimeContext:
    """
    从 checkpoint 恢复 RuntimeContext。

    功能：
        模拟 UI 第二次请求时的恢复过程。
        优先从 checkpoint_manager 中恢复 RuntimeContext，
        然后补齐当前 smoke 需要的身份字段和 ToolCounterHook。

    参数含义：
        trace_id:
            当前请求链路追踪 ID。
        thread_id:
            LangGraph thread_id。
        user_id:
            smoke 用户 ID。
        checkpoint_manager:
            checkpoint manager，负责读取 RuntimeContext 快照。

    返回值含义：
        RuntimeContext:
            已写入 runtime_ctx 的恢复后运行时上下文。
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
    ctx.component = "smoke_v180_tool_agent_resume_request"
    ctx.state().set_agent(
        "tool_agent"
    )
    ctx.hooks().register(
        "tool.before",
        ToolCounterHook(),
    )

    await runtime_ctx.create(
        ctx
    )

    return ctx


async def reset_current_runtime_context() -> None:
    """
    重置当前 RuntimeContext。

    功能：
        如果当前 contextvar 中存在 RuntimeContext，则销毁它。
        smoke 中用它模拟“第一次请求中断后，第二次请求再恢复”的真实 UI 行为。

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
    读取当前 RuntimeContext 指标。

    功能：
        从 MetricsScope 中复制一份普通 dict，方便 smoke 校验和报告渲染。

    参数含义：
        无。

    返回值含义：
        dict[str, Any]:
            当前运行时指标。没有 RuntimeContext 时返回空 dict。
    """

    ctx = runtime_ctx.get()

    if ctx is None:
        return {}

    return dict(
        ctx.service(
            MetricsScope
        ).get_metrics()
    )


def build_default_scenarios(
    question: str,
) -> tuple[ToolAgentInterruptResumeScenario, ...]:
    """
    构建默认 ToolAgent 中断恢复场景。

    功能：
        默认覆盖用户确认和用户拒绝两条分支。

    参数含义：
        question:
            用来触发 weather 工具确认的问题。

    返回值含义：
        tuple[ToolAgentInterruptResumeScenario, ...]:
            smoke 场景集合。
    """

    return (
        ToolAgentInterruptResumeScenario(
            name="confirm_weather_tool",
            question=question,
            resume_value="y",
            expect_tool_execution=True,
        ),
        ToolAgentInterruptResumeScenario(
            name="reject_weather_tool",
            question=question,
            resume_value="n",
            expect_tool_execution=False,
        ),
    )


def select_scenarios(
    scenarios: tuple[ToolAgentInterruptResumeScenario, ...],
    scenario_name: str,
) -> tuple[ToolAgentInterruptResumeScenario, ...]:
    """
    根据命令行参数筛选 smoke 场景。

    功能：
        支持运行全部场景，也支持只运行确认或拒绝单个场景。

    参数含义：
        scenarios:
            全部候选场景。
        scenario_name:
            命令行传入的场景名称，all 表示全部。

    返回值含义：
        tuple[ToolAgentInterruptResumeScenario, ...]:
            本次需要执行的场景集合。
    """

    if scenario_name == "all":
        return scenarios

    return tuple(
        scenario
        for scenario in scenarios
        if scenario.name == scenario_name
    )


def validate_interrupt_result(
    result: GraphInterruptResult,
) -> list[str]:
    """
    校验首次运行的中断结果。

    功能：
        确认真实主图返回的是工具确认中断。
        真实 LangGraph interrupt 发生在节点内部时，节点 update 可能还没有写回 state，
        因此这里优先根据 interrupt_type 判断，缺少类型时回退检查 prompt 文本。

    参数含义：
        result:
            首次运行得到的 GraphInterruptResult。

    返回值含义：
        list[str]:
            错误信息列表。空列表表示校验通过。
    """

    errors: list[str] = []

    if not is_tool_confirmation_interrupt(
        result=result,
    ):
        errors.append(
            "首次中断类型不是 tool_confirmation。"
        )

    if (
        result.metadata.get(
            "current_agent"
        )
        and result.metadata.get(
            "current_agent"
        )
        != "tool_agent"
    ):
        errors.append(
            "中断 metadata.current_agent 不是 tool_agent。"
        )

    tool_calls = result.metadata.get(
        "tool_calls",
        [],
    )

    has_weather_in_metadata = "weather" in str(
        tool_calls
    )
    has_weather_in_prompt = "weather" in str(
        result.prompt
    )

    if not (
        has_weather_in_metadata
        or has_weather_in_prompt
    ):
        errors.append(
            "首次中断中没有识别到 weather 工具。"
        )

    if not result.prompt:
        errors.append(
            "中断 prompt 为空。"
        )

    return errors


def is_tool_confirmation_interrupt(
    result: GraphInterruptResult,
) -> bool:
    """
    判断当前中断是否属于工具确认。

    功能：
        优先使用结构化 interrupt_type 判断。
        如果真实中断发生在节点 update 写回 state 之前，interrupt_type 可能是 unknown，
        此时回退检查 prompt 是否包含工具确认提示文本。

    参数含义：
        result:
            GraphInterruptResult 中断结果。

    返回值含义：
        bool:
            True 表示这是工具确认中断；False 表示不是。
    """

    if result.interrupt_type == GraphInterruptType.TOOL_CONFIRMATION:
        return True

    prompt = str(
        result.prompt
        or ""
    )

    return (
        "工具" in prompt
        and "是否允许继续" in prompt
    )


def validate_final_result(
    scenario: ToolAgentInterruptResumeScenario,
    result: GraphFinalResult | GraphInterruptResult | None,
    interrupt_count: int,
    metrics: dict[str, Any],
) -> list[str]:
    """
    校验恢复后的最终结果。

    功能：
        根据场景预期判断最终结果是否完成、工具是否执行、hook 是否触发。

    参数含义：
        scenario:
            当前 smoke 场景。
        result:
            恢复后的最终图运行结果。
        interrupt_count:
            当前场景累计中断次数。
        metrics:
            RuntimeContext 指标。

    返回值含义：
        list[str]:
            错误信息列表。空列表表示校验通过。
    """

    errors: list[str] = []

    if interrupt_count < 1:
        errors.append(
            "预期至少发生一次工具确认中断，但实际没有中断。"
        )

    if not isinstance(
        result,
        GraphFinalResult,
    ):
        errors.append(
            "恢复后最终结果不是 GraphFinalResult。"
        )
        return errors

    if scenario.expect_tool_execution:
        if metrics.get(
            "tool_count",
            0,
        ) < 1:
            errors.append(
                "用户确认后 tool_count 小于 1，说明工具没有真正执行。"
            )

        if metrics.get(
            "tool_before_hook_count",
            0,
        ) < 1:
            errors.append(
                "用户确认后 tool_before_hook_count 小于 1，说明工具 hook 没有触发。"
            )

    if not scenario.expect_tool_execution:
        if metrics.get(
            "tool_count",
            0,
        ) != 0:
            errors.append(
                "用户拒绝后 tool_count 不为 0，说明工具被意外执行。"
            )

        if "取消" not in result.answer:
            errors.append(
                "用户拒绝后最终答案中没有体现取消工具调用。"
            )

    return errors


async def run_single_scenario(
    scenario: ToolAgentInterruptResumeScenario,
    max_resumes: int,
) -> ToolAgentInterruptResumeReport:
    """
    运行单个 ToolAgent 中断恢复 smoke 场景。

    功能：
        启动一次真实主图，遇到工具确认中断后模拟新请求恢复，
        最后校验确认 / 拒绝分支是否符合预期。

    参数含义：
        scenario:
            当前 smoke 场景。
        max_resumes:
            最大恢复次数，避免异常链路无限循环。

    返回值含义：
        ToolAgentInterruptResumeReport:
            当前场景的 smoke 报告。
    """

    from src.runtime.container.init import container

    trace_id = f"smoke_v180_{scenario.name}_{uuid.uuid4().hex}"
    thread_id = f"smoke_v180_thread_{uuid.uuid4().hex}"
    user_id = "smoke_user_v180"
    interrupt_count = 0
    errors: list[str] = []
    current_result: GraphFinalResult | GraphInterruptResult | None = None

    await container.startup()

    try:
        checkpoint_manager = container.get(
            "checkpoint"
        ).manager

        ensure_trace_exists(
            trace_id
        )
        await create_runtime_context(
            trace_id=trace_id,
            thread_id=thread_id,
            user_id=user_id,
            component="smoke_v180_tool_agent_initial_request",
        )

        current_result = await run_main_graph_with_result(
            question=scenario.question,
            thread_id=thread_id,
            trace_id=trace_id,
        )

        while isinstance(
            current_result,
            GraphInterruptResult,
        ):
            interrupt_count += 1
            errors.extend(
                validate_interrupt_result(
                    current_result
                )
            )

            if interrupt_count > max_resumes:
                errors.append(
                    "恢复次数超过 max_resumes，可能出现重复中断。"
                )
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
                question=scenario.resume_value,
                thread_id=thread_id,
                trace_id=trace_id,
                resume_value=scenario.resume_value,
            )

        metrics = get_runtime_metrics()
        errors.extend(
            validate_final_result(
                scenario=scenario,
                result=current_result,
                interrupt_count=interrupt_count,
                metrics=metrics,
            )
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

        return ToolAgentInterruptResumeReport(
            scenario_name=scenario.name,
            passed=not errors,
            interrupt_count=interrupt_count,
            final_answer_preview=final_answer_preview,
            metrics=metrics,
            errors=errors,
        )

    finally:
        await reset_current_runtime_context()
        await container.shutdown()


async def run_smoke(
    question: str,
    scenario_name: str,
    max_resumes: int,
) -> list[ToolAgentInterruptResumeReport]:
    """
    执行 V1.8.0 ToolAgent 中断恢复 smoke。

    功能：
        根据命令行选择运行一个或多个场景，并返回全部报告。

    参数含义：
        question:
            触发 ToolAgent 的用户问题。
        scenario_name:
            要执行的场景名称，all 表示全部。
        max_resumes:
            每个场景最大恢复次数。

    返回值含义：
        list[ToolAgentInterruptResumeReport]:
            全部场景的 smoke 报告。
    """

    scenarios = select_scenarios(
        scenarios=build_default_scenarios(
            question=question,
        ),
        scenario_name=scenario_name,
    )

    if not scenarios:
        raise ValueError(
            f"未知 smoke 场景: {scenario_name}"
        )

    reports: list[ToolAgentInterruptResumeReport] = []

    for scenario in scenarios:
        reports.append(
            await run_single_scenario(
                scenario=scenario,
                max_resumes=max_resumes,
            )
        )

    return reports


def render_report(
    reports: list[ToolAgentInterruptResumeReport],
) -> str:
    """
    渲染 ToolAgent 中断恢复 smoke Markdown 报告。

    功能：
        把多个场景结果格式化成终端可读的 Markdown 文本。

    参数含义：
        reports:
            smoke 报告列表。

    返回值含义：
        str:
            Markdown 格式报告。
    """

    passed = all(
        report.passed
        for report in reports
    )
    status = "PASS" if passed else "FAIL"
    lines = [
        "# V1.8.0 ToolAgent Interrupt Resume Smoke Report",
        "",
        f"- status: {status}",
        f"- scenario_count: {len(reports)}",
        "",
        "## Scenarios",
        "",
    ]

    for report in reports:
        scenario_status = "PASS" if report.passed else "FAIL"
        lines.extend(
            [
                f"### {report.scenario_name}",
                "",
                f"- status: {scenario_status}",
                f"- interrupt_count: {report.interrupt_count}",
                f"- final_answer_preview: {report.final_answer_preview}",
                f"- tool_count: {report.metrics.get('tool_count', 0)}",
                f"- tool_before_hook_count: {report.metrics.get('tool_before_hook_count', 0)}",
                "",
                "#### Metrics",
                "",
            ]
        )

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
                    "#### Errors",
                    "",
                ]
            )

            for error in report.errors:
                lines.append(
                    f"- {error}"
                )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    功能：
        读取 smoke 问题、场景名称和最大恢复次数。

    参数含义：
        无。

    返回值含义：
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="V1.8.0 ToolAgent interrupt resume smoke test."
    )
    parser.add_argument(
        "--question",
        default="今天成都的天气怎么样？",
        help="用于触发 ToolAgent weather 工具确认的问题。",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "confirm_weather_tool",
            "reject_weather_tool",
        ],
        default="all",
        help="要运行的 smoke 场景。",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=3,
        help="每个场景最多允许恢复次数。",
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
            0 表示全部通过；1 表示至少一个场景失败。
    """

    args = parse_args()
    reports = asyncio.run(
        run_smoke(
            question=args.question,
            scenario_name=args.scenario,
            max_resumes=args.max_resumes,
        )
    )

    print(
        render_report(
            reports
        )
    )

    return 0 if all(
        report.passed
        for report in reports
    ) else 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
