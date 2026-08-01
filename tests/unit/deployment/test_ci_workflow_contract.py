"""
V1.21 Python 与 Docker CI 工作流契约测试。

功能：
    验证 Pull Request 和 main 分支会执行全量 Python 门禁，并只构建但不发布
    Docker 镜像，避免 CI 阶段意外写入容器仓库。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "v121-ci.yml"
)


def _read_ci_workflow() -> str:
    """
    读取 V1.21 CI 工作流文本。

    参数含义：
        无。

    返回值含义：
        str:
            使用 UTF-8 解码的完整 GitHub Actions Workflow。
    """

    return CI_WORKFLOW_PATH.read_text(encoding="utf-8")


def test_ci_workflow_should_run_for_pull_requests_and_main() -> None:
    """
    验证代码审查和 main 更新都会触发 CI，并允许手动复验。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_ci_workflow()

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "- main" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow


def test_ci_workflow_should_compile_and_run_full_pytest() -> None:
    """
    验证 Python Job 使用项目版本、CI 依赖和完整质量门禁。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_ci_workflow()

    assert "python-quality:" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "uses: actions/checkout@v7" in workflow
    assert "uses: actions/setup-python@v6" in workflow
    assert 'python-version: "3.13"' in workflow
    assert (
        "python -m pip install --requirement requirements-ci.txt"
        in workflow
    )
    assert "python -m compileall -q src tests scripts" in workflow
    assert "python -m pytest -q" in workflow


def test_ci_workflow_should_build_without_publishing_image() -> None:
    """
    验证 Docker Job 在 Python 门禁后运行，并且 CI 阶段禁止推送镜像。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_ci_workflow()

    assert "docker-build:" in workflow
    assert "- python-quality" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "uses: docker/setup-buildx-action@v4" in workflow
    assert "uses: docker/build-push-action@v7" in workflow
    assert "context: ." in workflow
    assert "file: Dockerfile" in workflow
    assert "push: false" in workflow
    assert "tags: dog-agent-api:ci" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to: type=gha,mode=max" in workflow
    assert "Validate release Compose configuration" in workflow
    assert "-f compose.release.yaml" in workflow
    assert "config --quiet" in workflow
    assert "docker/login-action" not in workflow
    assert "packages: write" not in workflow
