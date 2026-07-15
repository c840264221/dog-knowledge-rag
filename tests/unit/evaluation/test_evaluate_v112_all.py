from pathlib import Path

from scripts.evaluation.evaluate_v112_all import (
    build_argument_parser,
    resolve_evaluation_targets,
    resolve_report_paths,
)


def test_profile_should_default_to_full() -> None:
    """
    测试统一评估命令默认执行包含真实 RAG 的 full 档位。

    参数含义：
        无。

    返回值含义：
        None。
    """

    args = build_argument_parser().parse_args([])

    assert args.profile == "full"
    assert any(
        target.category == "rag_retrieval_behavior"
        for target in resolve_evaluation_targets(args.profile)
    )


def test_core_profile_should_use_isolated_report_paths() -> None:
    """
    测试 core 档位不执行真实 RAG 且使用独立报告路径。

    参数含义：
        无。

    返回值含义：
        None。
    """

    targets = resolve_evaluation_targets("core")
    json_path, markdown_path = resolve_report_paths("core")

    assert all(
        target.category != "rag_retrieval_behavior"
        for target in targets
    )
    assert json_path == Path(
        "evaluation/reports/v112_core_evaluation_report.json"
    )
    assert markdown_path == Path(
        "evaluation/reports/v112_core_evaluation_report.md"
    )
