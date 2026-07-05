"""
ToolCounterHook 单元测试。

ToolCounterHook（工具计数钩子）：
用于学习 RuntimeHook（运行时钩子）机制，在 tool.before 时机记录钩子触发次数。
"""

import pytest

from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.hooks.tool_counter_hook import ToolCounterHook
from src.runtime.scopes.metrics_scope import MetricsScope


@pytest.mark.asyncio
async def test_tool_counter_hook_should_increment_hook_metric() -> None:
    """
    测试 ToolCounterHook 是否使用 MetricsScope 计数。

    功能：
        创建 RuntimeContext，并触发 ToolCounterHook 两次。
        验证 MetricsScope 中的 tool_before_hook_count 会递增到 2。

    参数：
        无。

    返回值：
        None。
    """

    ctx = RuntimeContext(
        trace_id="trace-hook-test",
    )

    await ctx.startup()
    runtime_ctx.set(
        ctx
    )

    hook = ToolCounterHook()

    try:
        await hook.execute()
        await hook.execute()

        metrics = ctx.service(
            MetricsScope
        ).get_metrics()
    finally:
        await ctx.shutdown()
        runtime_ctx.set(
            RuntimeContext()
        )

    assert metrics["tool_before_hook_count"] == 2


@pytest.mark.asyncio
async def test_tool_counter_hook_should_not_increment_tool_count() -> None:
    """
    测试 ToolCounterHook 不直接修改 tool_count。

    功能：
        验证学习用 hook 只记录 hook 触发次数，
        不和 MetricsListener 的工具成功统计字段 tool_count 冲突。

    参数：
        无。

    返回值：
        None。
    """

    ctx = RuntimeContext(
        trace_id="trace-hook-test",
    )

    await ctx.startup()
    runtime_ctx.set(
        ctx
    )

    hook = ToolCounterHook()

    try:
        await hook.execute()

        metrics = ctx.service(
            MetricsScope
        ).get_metrics()
    finally:
        await ctx.shutdown()
        runtime_ctx.set(
            RuntimeContext()
        )

    assert metrics["tool_count"] == 0
    assert metrics["tool_before_hook_count"] == 1
