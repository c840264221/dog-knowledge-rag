"""
LLM Provider 运行时指标测试。

功能：
    验证统一 LLM 调用入口会统计逻辑调用次数、总耗时和最终失败次数。
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.context import runtime_ctx
from src.runtime.context.runtime_context import RuntimeContext
from src.runtime.scopes.metrics_scope import MetricsScope


class FakeMetricsLLM:
    """提供固定成功或失败行为的测试 LLM。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        """
        返回固定响应或抛出预设异常。

        参数含义：
            prompt:
                本次测试调用传入的提示词。

        返回值含义：
            dict[str, Any]:
                包含原提示词的固定响应。
        """

        if self.error is not None:
            raise self.error
        return {"content": prompt}


def _build_metrics_runtime_context() -> RuntimeContext:
    """
    构建已经初始化 MetricsScope 的测试运行时上下文。

    参数含义：
        无。

    返回值含义：
        RuntimeContext:
            可以接收 LLM 指标的独立运行时上下文。
    """

    context = RuntimeContext()
    context.service(MetricsScope).init_metrics()
    return context


def test_safe_ainvoke_should_record_success_metrics() -> None:
    """
    检查成功调用是否增加一次 llm_count 并累计耗时。

    参数含义：
        无。

    返回值含义：
        None。
    """

    previous_context = runtime_ctx.get()
    context = _build_metrics_runtime_context()
    runtime_ctx.set(context)
    try:
        provider = LLMProvider()
        response = asyncio.run(
            provider.safe_ainvoke(
                llm=FakeMetricsLLM(),
                prompt="测试提示词",
                max_attempts=1,
            )
        )
        metrics = context.service(MetricsScope).get_metrics()
    finally:
        runtime_ctx.set(previous_context)

    assert response == {"content": "测试提示词"}
    assert metrics["llm_count"] == 1
    assert metrics["llm_latency"] >= 0
    assert metrics["error_count"] == 0


def test_safe_ainvoke_should_record_one_final_failure() -> None:
    """
    检查主备模型全部失败时只记录一次逻辑调用和一次最终错误。

    参数含义：
        无。

    返回值含义：
        None。
    """

    previous_context = runtime_ctx.get()
    context = _build_metrics_runtime_context()
    runtime_ctx.set(context)
    try:
        provider = LLMProvider()
        provider._backup_llm = FakeMetricsLLM(error=RuntimeError("备用失败"))
        response = asyncio.run(
            provider.safe_ainvoke(
                llm=FakeMetricsLLM(error=RuntimeError("主模型失败")),
                prompt="测试提示词",
                fallback_response="兜底回答",
                max_attempts=1,
            )
        )
        metrics = context.service(MetricsScope).get_metrics()
    finally:
        runtime_ctx.set(previous_context)

    assert response == "兜底回答"
    assert metrics["llm_count"] == 1
    assert metrics["llm_latency"] >= 0
    assert metrics["error_count"] == 1
