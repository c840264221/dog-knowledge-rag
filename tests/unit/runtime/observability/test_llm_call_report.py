"""LLM 调用报告单元测试。"""

from __future__ import annotations

from src.runtime.observability.llm_call_records import (
    LLMCallMetadata,
    LLMCallRecord,
)
from src.runtime.observability.llm_call_report import (
    build_llm_call_report,
    render_llm_call_log_summary,
    render_llm_call_report_markdown,
    save_llm_call_report,
    save_llm_call_report_json,
)


def _build_call(
    *,
    call_id: str,
    purpose: str,
    component: str,
    agent_name: str = "",
    step_id: str = "",
    attempt_count: int = 1,
) -> dict:
    """
    构建一条测试用 LLM 调用记录。

    参数含义：
        call_id：逻辑调用编号。
        purpose：受控调用目的。
        component：发起调用的组件。
        agent_name：调用所属 Agent。
        step_id：多智能体步骤编号。
        attempt_count：物理模型尝试次数。

    返回值含义：
        dict：可以交给报告构建器校验的调用记录字典。
    """

    return LLMCallRecord(
        call_id=call_id,
        trace_id="trace-report",
        metadata=LLMCallMetadata(
            call_purpose=purpose,
            component=component,
            agent_name=agent_name,
            step_id=step_id,
        ),
        requested_model="main-model",
        final_model="main-model",
        attempt_count=attempt_count,
        backup_used=False,
        status="completed",
        latency_ms=125.5,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    ).model_dump(mode="python")


def test_report_should_aggregate_calls_and_find_strict_duplicates() -> None:
    """
    验证报告按严格的 Agent、步骤和用途组合识别重复候选。

    参数含义：
        无。

    返回值含义：
        None。
    """

    raw_calls = [
        _build_call(
            call_id="call-1",
            purpose="answer_generation",
            component="generate_node",
            agent_name="dog_knowledge_agent",
            step_id="step_training",
            attempt_count=2,
        ),
        _build_call(
            call_id="call-2",
            purpose="answer_generation",
            component="generate_node",
            agent_name="dog_knowledge_agent",
            step_id="step_training",
        ),
        _build_call(
            call_id="call-3",
            purpose="answer_generation",
            component="generate_node",
            agent_name="dog_knowledge_agent",
            step_id="step_health",
        ),
    ]

    report = build_llm_call_report(
        trace_id="trace-report",
        raw_calls=raw_calls,
    )

    assert report.logical_call_count == 3
    assert report.physical_attempt_count == 4
    assert report.token_usage["total_tokens"] == 45
    assert len(report.duplicate_candidates) == 1
    candidate = report.duplicate_candidates[0]
    assert candidate.scope == "multi_agent_step"
    assert candidate.step_id == "step_training"
    assert candidate.call_purpose == "answer_generation"
    assert candidate.logical_call_count == 2


def test_report_should_not_mix_agent_and_component_scopes() -> None:
    """
    验证不同 Agent、组件或用途不会被宽松地误判为同一重复分组。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_llm_call_report(
        trace_id="trace-strict",
        raw_calls=[
            _build_call(
                call_id="agent-call",
                purpose="answer_generation",
                component="generate_node",
                agent_name="dog_knowledge_agent",
            ),
            _build_call(
                call_id="other-agent-call",
                purpose="answer_generation",
                component="generate_node",
                agent_name="general_agent",
            ),
            _build_call(
                call_id="component-call",
                purpose="memory_extraction",
                component="memory_extract_node",
            ),
        ],
    )

    assert report.duplicate_candidates == []


def test_report_should_ignore_invalid_records_and_unspecified_purpose() -> None:
    """
    验证坏记录不会中断报告，未声明用途的调用不参与重复判定。

    参数含义：
        无。

    返回值含义：
        None。
    """

    unspecified = _build_call(
        call_id="unspecified-1",
        purpose="unspecified",
        component="legacy_component",
    )
    report = build_llm_call_report(
        trace_id="trace-invalid",
        raw_calls=[unspecified, unspecified, {"bad": "record"}],
    )

    assert report.logical_call_count == 2
    assert report.invalid_record_count == 1
    assert report.duplicate_candidates == []


def test_report_should_render_log_and_save_markdown(tmp_path) -> None:
    """
    验证同一结构化报告可以生成日志摘要并保存 Markdown 文件。

    参数含义：
        tmp_path：pytest 提供的临时目录。

    返回值含义：
        None。
    """

    report = build_llm_call_report(
        trace_id="trace/file:test",
        raw_calls=[
            _build_call(
                call_id="call-render",
                purpose="routing_decision",
                component="root_supervisor",
                agent_name="root_agent",
            )
        ],
    )

    summary = render_llm_call_log_summary(report)
    markdown = render_llm_call_report_markdown(report)
    report_path = save_llm_call_report(
        report=report,
        report_dir=tmp_path,
        use_date_dir=False,
    )
    json_path = save_llm_call_report_json(
        report=report,
        markdown_report_path=report_path,
    )

    assert "logical_calls=1" in summary
    assert "# LLM 调用报告" in markdown
    assert "routing_decision" in markdown
    assert report_path.parent == tmp_path
    assert "trace_file_test" in report_path.name
    assert report_path.read_text(encoding="utf-8") == markdown
    assert json_path.exists()
    assert '"trace_id": "trace/file:test"' in json_path.read_text(
        encoding="utf-8"
    )


def test_report_should_support_request_without_llm_calls() -> None:
    """
    验证没有调用 LLM 的请求也能生成明确的零调用报告。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_llm_call_report(
        trace_id="trace-zero",
        raw_calls=[],
    )

    assert report.logical_call_count == 0
    assert report.physical_attempt_count == 0
    assert report.duplicate_candidates == []
    assert report.budget_evaluation.status == "not_configured"
    assert "logical_calls=0" in render_llm_call_log_summary(report)
