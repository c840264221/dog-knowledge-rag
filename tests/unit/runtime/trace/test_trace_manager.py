from src.runtime.trace.trace_manager import TraceManager


def test_create_span_should_create_missing_trace() -> None:
    """
    测试 create_span 在 trace 缺失时自动创建 trace。

    功能：
        模拟 resume 后内存 trace_map 丢失的情况，
        验证 create_span 不会因为 trace_id 不存在而失败。

    参数：
        无。

    返回值：
        None。
    """

    manager = TraceManager()

    span = manager.create_span(
        trace_id="trace-resume-001",
        name="weather_tool",
    )

    trace = manager.get_trace(
        "trace-resume-001"
    )

    assert trace is not None
    assert span.trace_id == "trace-resume-001"
    assert trace.root_spans == [
        span
    ]
