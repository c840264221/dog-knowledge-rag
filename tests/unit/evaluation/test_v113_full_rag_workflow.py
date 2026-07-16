from pathlib import Path


FULL_RAG_WORKFLOW_PATH = Path(
    ".github/workflows/v112-full-rag-evaluation.yml"
)
CORE_WORKFLOW_PATH = Path(
    ".github/workflows/v112-evaluation-quality-gate.yml"
)


def read_workflow(path: Path) -> str:
    """
    读取一份 GitHub Actions 工作流文件。

    功能：
        把指定 YAML 文件完整读取成字符串，后面的测试可以检查关键步骤、
        报告路径和 Artifact 配置有没有被误删。

    参数含义：
        path:
            需要检查的 Workflow YAML 文件位置。

    返回值含义：
        str:
            YAML 文件中的全部文字。
    """

    return path.read_text(encoding="utf-8")


def test_full_rag_workflow_should_compare_report_after_evaluation() -> None:
    """
    检查 Full RAG CI 是否先完成评估，再执行历史成绩比较。

    功能：
        在 YAML 中找到完整评估命令和基线比较命令，确认比较命令排在后面。
        这样比较器读取的一定是本次 CI 刚生成的最新报告。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = read_workflow(FULL_RAG_WORKFLOW_PATH)
    evaluation_command = (
        "python -m scripts.evaluation.evaluate_v112_all --profile full"
    )
    regression_command = (
        "python -m scripts.evaluation.compare_v113_evaluation_baseline"
    )

    assert evaluation_command in workflow
    assert regression_command in workflow
    assert workflow.index(evaluation_command) < workflow.index(
        regression_command
    )


def test_full_rag_workflow_should_publish_and_upload_regression_reports() -> None:
    """
    检查回归报告是否会显示在 Summary 并保存到 Artifact。

    功能：
        确认 JSON 和 Markdown 回归报告都进入 Artifact 准备步骤，Markdown
        还会写入 GITHUB_STEP_SUMMARY，方便在 GitHub 页面直接查看。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = read_workflow(FULL_RAG_WORKFLOW_PATH)

    assert "v113_evaluation_regression_report.json" in workflow
    assert "v113_evaluation_regression_report.md" in workflow
    assert "$env:GITHUB_STEP_SUMMARY" in workflow
    assert "evaluation/artifacts/full-rag" in workflow
    assert "v113-full-rag-regression-report-${{ github.run_number }}" in workflow


def test_core_workflow_should_not_run_full_baseline_comparison() -> None:
    """
    检查快速 Core CI 是否没有误接入完整 RAG 基线。

    功能：
        Core CI 不运行真实 RAG，也没有 rag_retrieval_behavior 成绩。如果
        强行和 Full 基线比较会因为缺少 RAG 类别而失败，所以这里锁定边界。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = read_workflow(CORE_WORKFLOW_PATH)

    assert "compare_v113_evaluation_baseline" not in workflow
