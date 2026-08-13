"""
LLM Provider 运行时指标测试。

功能：
    验证统一 LLM 调用入口会统计逻辑调用次数、总耗时和最终失败次数。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.context import runtime_ctx
from src.runtime.context.runtime_context import RuntimeContext
from src.runtime.scopes.metrics_scope import MetricsScope
from src.runtime.observability.llm_call_records import (
    LLMCallMetadata,
    LLMCallPurpose,
    build_llm_call_metadata,
)


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


class FakeUsageMetricsLLM:
    """返回带标准 Token usage_metadata 的测试 LLM。"""

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        """
        返回带输入、输出和总 Token 的固定响应。

        参数含义：
            prompt:
                本次测试调用传入的提示词。

        返回值含义：
            dict[str, Any]:
                包含固定 Token 用量的模拟模型响应。
        """

        return {
            "content": prompt,
            "usage_metadata": {
                "input_tokens": 12,
                "output_tokens": 5,
                "total_tokens": 17,
            },
        }


class FakeRetryMetricsLLM:
    """第一次失败、后续成功的测试 LLM。"""

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        """
        首次调用抛出异常，第二次返回固定响应。

        参数含义：
            prompt：本次测试调用传入的提示词。

        返回值含义：
            dict[str, Any]：重试成功后的固定响应。
        """

        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("第一次调用失败")
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
    assert metrics["llm_input_tokens"] == 0
    assert metrics["llm_output_tokens"] == 0
    assert metrics["llm_total_tokens"] == 0
    assert len(metrics["llm_calls"]) == 1
    assert metrics["llm_calls"][0]["status"] == "completed"
    assert metrics["llm_calls"][0]["attempt_count"] == 1


def test_safe_ainvoke_should_record_token_usage() -> None:
    """
    检查统一 LLM 入口是否累计模型响应中的 Token 用量。

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
        asyncio.run(
            provider.safe_ainvoke(
                llm=FakeUsageMetricsLLM(),
                prompt="统计 Token",
                max_attempts=1,
            )
        )
        asyncio.run(
            provider.safe_ainvoke(
                llm=FakeUsageMetricsLLM(),
                prompt="继续统计 Token",
                max_attempts=1,
            )
        )
        metrics = context.service(MetricsScope).get_metrics()
    finally:
        runtime_ctx.set(previous_context)

    assert metrics["llm_count"] == 2
    assert metrics["llm_input_tokens"] == 24
    assert metrics["llm_output_tokens"] == 10
    assert metrics["llm_total_tokens"] == 34
    assert len(metrics["llm_calls"]) == 2
    assert metrics["llm_calls"][0]["input_tokens"] == 12
    assert metrics["llm_calls"][0]["output_tokens"] == 5


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
    assert metrics["llm_total_tokens"] == 0
    assert len(metrics["llm_calls"]) == 1
    assert metrics["llm_calls"][0]["status"] == "fallback"
    assert metrics["llm_calls"][0]["attempt_count"] == 2
    assert metrics["llm_calls"][0]["backup_used"] is True
    assert metrics["llm_calls"][0]["error_type"] == "RuntimeError"


def test_safe_ainvoke_should_record_explicit_and_runtime_metadata() -> None:
    """
    检查显式调用说明优先，并由运行时上下文补充缺失组件。

    功能：
        验证 Agent 和步骤可以由调用方明确传入，component 为空时从
        RuntimeContext 获取，从而不要求 LLMProvider 依赖 DogState。

    参数含义：
        无。

    返回值含义：
        None。
    """

    previous_context = runtime_ctx.get()
    context = _build_metrics_runtime_context()
    context.trace_id = "trace_001"
    context.component = "runtime_component"
    runtime_ctx.set(context)
    try:
        provider = LLMProvider()
        asyncio.run(
            provider.safe_ainvoke(
                llm=FakeUsageMetricsLLM(),
                prompt="生成步骤回答",
                max_attempts=1,
                call_metadata=LLMCallMetadata(
                    call_purpose="answer_generation",
                    agent_name="dog_knowledge_agent",
                    step_id="step_knowledge",
                ),
            )
        )
        call_record = context.service(MetricsScope).get_metrics()[
            "llm_calls"
        ][0]
    finally:
        runtime_ctx.set(previous_context)

    assert call_record["trace_id"] == "trace_001"
    assert call_record["metadata"] == {
        "call_purpose": "answer_generation",
        "component": "runtime_component",
        "agent_name": "dog_knowledge_agent",
        "step_id": "step_knowledge",
        "extra": {},
    }
    assert call_record["requested_model"] == "FakeUsageMetricsLLM"


def test_safe_ainvoke_should_record_retry_as_one_logical_call() -> None:
    """验证主模型内部重试不会被统计成多次业务调用。"""

    previous_context = runtime_ctx.get()
    context = _build_metrics_runtime_context()
    runtime_ctx.set(context)
    try:
        provider = LLMProvider()
        llm = FakeRetryMetricsLLM()
        response = asyncio.run(
            provider.safe_ainvoke(
                llm=llm,
                prompt="重试后成功",
                max_attempts=2,
            )
        )
        metrics = context.service(MetricsScope).get_metrics()
    finally:
        runtime_ctx.set(previous_context)

    assert response == {"content": "重试后成功"}
    assert llm.call_count == 2
    assert metrics["llm_count"] == 1
    assert len(metrics["llm_calls"]) == 1
    assert metrics["llm_calls"][0]["attempt_count"] == 2
    assert metrics["llm_calls"][0]["backup_used"] is False


def test_safe_ainvoke_should_record_failed_without_fallback() -> None:
    """验证主备模型全部失败且没有兜底响应时记录 failed。"""

    previous_context = runtime_ctx.get()
    context = _build_metrics_runtime_context()
    runtime_ctx.set(context)
    try:
        provider = LLMProvider()
        provider._backup_llm = FakeMetricsLLM(error=RuntimeError("备用失败"))
        try:
            asyncio.run(
                provider.safe_ainvoke(
                    llm=FakeMetricsLLM(error=RuntimeError("主模型失败")),
                    prompt="没有兜底",
                    max_attempts=1,
                )
            )
        except RuntimeError as exc:
            assert str(exc) == "所有LLM调用失败"
        else:
            raise AssertionError("主备模型全部失败时应该抛出 RuntimeError")
        metrics = context.service(MetricsScope).get_metrics()
    finally:
        runtime_ctx.set(previous_context)

    assert metrics["llm_count"] == 1
    assert metrics["error_count"] == 1
    assert len(metrics["llm_calls"]) == 1
    assert metrics["llm_calls"][0]["status"] == "failed"
    assert metrics["llm_calls"][0]["attempt_count"] == 2
    assert metrics["llm_calls"][0]["final_model"] == ""


def test_llm_call_metadata_should_reject_unknown_purpose() -> None:
    """验证调用目的不允许使用白名单以外的任意字符串。"""

    with pytest.raises(ValidationError):
        LLMCallMetadata(call_purpose="随便填写的用途")


def test_metadata_builder_should_read_worker_step_from_state() -> None:
    """验证公共构造函数从当前 Worker 独立状态读取步骤编号。"""

    metadata = build_llm_call_metadata(
        purpose=LLMCallPurpose.QUERY_UNDERSTANDING,
        component="dog_query_understanding_node",
        agent_name="dog_knowledge_agent",
        state={
            "multi_agent_step_id": "step_training",
            "multi_agent_assigned_agent": "dog_knowledge_agent",
        },
    )

    assert metadata.call_purpose == "query_understanding"
    assert metadata.component == "dog_query_understanding_node"
    assert metadata.agent_name == "dog_knowledge_agent"
    assert metadata.step_id == "step_training"
