from pathlib import Path

from src.rag.evaluation import (
    RagEvalMetrics,
    RagEvalReport,
    RagEvalReportWriter,
    RagEvalResult,
    RagEvalRetrievedItem,
    write_rag_eval_report,
)


def build_sample_report() -> RagEvalReport:
    """
    构建测试用 RAG 评估报告。

    参数含义：
        无。

    返回值含义：
        RagEvalReport:
            测试用报告对象。
    """

    passed_result = RagEvalResult(
        case_id="dog_eval_001",
        question="Poodle 的训练难度怎么样？",
        expected_dog_names=["Poodle"],
        expected_filters={"dog_name": "Poodle"},
        parsed_filters={"dog_name": "Poodle"},
        retrieved_items=[
            RagEvalRetrievedItem(
                rank=1,
                chunk_id="poodle_chunk_001",
                dog_name="Poodle",
                score=0.1,
                source="data/dog_markdown/poodle.md",
                section_title="训练",
                content_preview="Poodle is intelligent and highly trainable.",
            )
        ],
        retrieved_dog_names=["Poodle"],
        hit=True,
        hit_rank=1,
        top1_hit=True,
        filter_matched=True,
        empty_retrieval=False,
        passed=True,
        latency_ms=123.456,
        extra={
            "quality_status": "good",
            "quality_score": 0.9,
        },
    )

    failed_result = RagEvalResult(
        case_id="dog_eval_002",
        question="Chihuahua 的寿命是多少？",
        expected_dog_names=["Chihuahua"],
        expected_filters={"dog_name": "Chihuahua"},
        parsed_filters={"dog_name": "Chihuahua"},
        retrieved_items=[],
        retrieved_dog_names=[],
        hit=False,
        hit_rank=None,
        top1_hit=False,
        filter_matched=True,
        empty_retrieval=True,
        passed=False,
        latency_ms=50.0,
        extra={
            "quality_status": "bad",
            "quality_failure_type": "empty",
        },
    )

    metrics = RagEvalMetrics(
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        hit_at_k=0.5,
        top1_accuracy=0.5,
        filter_match_rate=1.0,
        empty_retrieval_rate=0.5,
        average_latency_ms=86.728,
    )

    return RagEvalReport(
        run_id="test_rag_eval_report",
        dataset_path="data/rag_eval/dog_rag_eval_cases.json",
        metrics=metrics,
        results=[
            passed_result,
            failed_result,
        ],
        metadata={
            "version": "v1.6.0",
            "evaluator": "OfflineRetrievalEvaluator",
        },
    )


def test_render_report_contains_core_sections() -> None:
    """
    测试 Markdown 报告包含核心章节。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_sample_report()

    writer = RagEvalReportWriter()

    markdown_text = writer.render(report)

    assert "# RAG Evaluation Report" in markdown_text
    assert "## Summary" in markdown_text
    assert "## Metrics" in markdown_text
    assert "## Failed Cases" in markdown_text
    assert "## Case Details" in markdown_text
    assert "dog_eval_001" in markdown_text
    assert "dog_eval_002" in markdown_text


def test_write_report_to_default_output_dir(
    tmp_path: Path,
) -> None:
    """
    测试可以写入默认输出目录。

    参数含义：
        tmp_path:
            pytest 临时目录。

    返回值含义：
        None。
    """

    report = build_sample_report()

    writer = RagEvalReportWriter(
        output_dir=tmp_path,
    )

    output_path = writer.write(report)

    assert output_path.exists()
    assert output_path.name == "test_rag_eval_report.md"
    assert report.report_path == output_path.as_posix()

    content = output_path.read_text(
        encoding="utf-8",
    )

    assert "RAG Evaluation Report" in content
    assert "hit_at_k" in content
    assert "Poodle" in content


def test_write_report_to_custom_output_path(
    tmp_path: Path,
) -> None:
    """
    测试可以写入自定义输出路径。

    参数含义：
        tmp_path:
            pytest 临时目录。

    返回值含义：
        None。
    """

    report = build_sample_report()

    output_path = tmp_path / "custom_report.md"

    written_path = write_rag_eval_report(
        report=report,
        output_path=output_path,
    )

    assert written_path == output_path
    assert written_path.exists()

    content = written_path.read_text(
        encoding="utf-8",
    )

    assert "custom_report" not in content
    assert "test_rag_eval_report" in content


def test_failed_cases_section_when_no_failures() -> None:
    """
    测试没有失败用例时，报告显示 No failed cases。

    参数含义：
        无。

    返回值含义：
        None。
    """

    report = build_sample_report()

    report.results = [
        report.results[0]
    ]

    report.metrics = RagEvalMetrics(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        hit_at_k=1.0,
        top1_accuracy=1.0,
        filter_match_rate=1.0,
        empty_retrieval_rate=0.0,
        average_latency_ms=123.456,
    )

    writer = RagEvalReportWriter()

    markdown_text = writer.render(report)

    assert "No failed cases." in markdown_text