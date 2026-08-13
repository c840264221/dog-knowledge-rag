"""主图 LLM 调用报告收尾钩子测试。"""

from __future__ import annotations

from src.graph import graph_run
from src.runtime.context.request_scope import RequestScope
from src.runtime.observability.llm_call_records import (
    LLMCallMetadata,
    LLMCallRecord,
)
from src.runtime.scopes.metrics_scope import MetricsScope


class FakeRuntimeContext:
    """
    为报告收尾函数提供 MetricsScope 的测试运行时上下文。

    参数含义：
        metrics_scope：保存测试 LLM 调用记录的指标作用域。

    返回值含义：
        FakeRuntimeContext：支持 service(MetricsScope) 的最小对象。
    """

    def __init__(self, metrics_scope: MetricsScope) -> None:
        self.trace_id = "trace-context"
        self.metrics_scope = metrics_scope

    def service(self, service_type):
        """
        返回测试需要的运行时服务。

        参数含义：
            service_type：调用方请求的服务类型。

        返回值含义：
            MetricsScope：当前测试指标作用域。
        """

        assert service_type is MetricsScope
        return self.metrics_scope


def _build_runtime_context() -> FakeRuntimeContext:
    """
    构建包含一条 LLM 调用的测试运行时上下文。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContext：已经写入调用记录的运行时上下文。
    """

    metrics_scope = MetricsScope(RequestScope())
    metrics_scope.init_metrics()
    record = LLMCallRecord(
        trace_id="trace-hook",
        metadata=LLMCallMetadata(
            call_purpose="answer_generation",
            component="generate_node",
            agent_name="dog_knowledge_agent",
        ),
        requested_model="main-model",
        final_model="main-model",
        attempt_count=1,
        status="completed",
        latency_ms=10,
    )
    metrics_scope.append_llm_call(record.model_dump(mode="python"))
    return FakeRuntimeContext(metrics_scope)


def test_hook_should_support_log_without_file(
    monkeypatch,
) -> None:
    """
    验证日志开关打开、文件开关关闭时只输出摘要。

    参数含义：
        monkeypatch：pytest 动态替换工具。

    返回值含义：
        None。
    """

    logged_messages: list[str] = []
    monkeypatch.setattr(
        graph_run.settings.observability,
        "ENABLE_LLM_CALL_REPORT",
        True,
    )
    monkeypatch.setattr(
        graph_run.settings.observability,
        "LLM_CALL_REPORT_TO_LOG",
        True,
    )
    monkeypatch.setattr(
        graph_run.settings.observability,
        "LLM_CALL_REPORT_TO_FILE",
        False,
    )
    monkeypatch.setattr(
        graph_run.logger,
        "info",
        logged_messages.append,
    )
    monkeypatch.setattr(
        graph_run,
        "save_llm_call_report",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("文件开关关闭时不应保存报告")
        ),
    )

    graph_run.write_llm_call_report_if_enabled(
        runtime_context=_build_runtime_context(),
        trace_id="trace-hook",
    )

    assert len(logged_messages) == 1
    assert "logical_calls=1" in logged_messages[0]


def test_hook_should_swallow_report_failure(
    monkeypatch,
) -> None:
    """
    验证报告生成异常只写警告，不向主业务继续抛出。

    参数含义：
        monkeypatch：pytest 动态替换工具。

    返回值含义：
        None。
    """

    warnings: list[str] = []
    monkeypatch.setattr(
        graph_run.settings.observability,
        "ENABLE_LLM_CALL_REPORT",
        True,
    )
    monkeypatch.setattr(
        graph_run.settings.observability,
        "LLM_CALL_REPORT_TO_LOG",
        False,
    )
    monkeypatch.setattr(
        graph_run.settings.observability,
        "LLM_CALL_REPORT_TO_FILE",
        True,
    )
    monkeypatch.setattr(
        graph_run,
        "save_llm_call_report",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(graph_run.logger, "warning", warnings.append)

    graph_run.write_llm_call_report_if_enabled(
        runtime_context=_build_runtime_context(),
        trace_id="trace-hook",
    )

    assert len(warnings) == 1
    assert "disk full" in warnings[0]
