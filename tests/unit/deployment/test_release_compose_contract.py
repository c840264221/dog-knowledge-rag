"""
GHCR 生产镜像部署覆盖文件契约测试。

功能：
    验证生产部署复用基础 Compose 配置时会清除本地构建入口、强制拉取指定
    GHCR 镜像，并要求工作人员在私有环境文件中明确选择发布版本。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RELEASE_COMPOSE_PATH = PROJECT_ROOT / "compose.release.yaml"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
CI_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "v121-ci.yml"
)


def _read_text(path: Path) -> str:
    """
    使用 UTF-8 读取部署契约文件。

    参数含义：
        path:
            需要读取的仓库文件绝对路径。

    返回值含义：
        str:
            文件的完整文本内容。
    """

    return path.read_text(encoding="utf-8")


def test_release_compose_should_pull_instead_of_building() -> None:
    """
    验证生产覆盖文件删除本地构建配置并始终拉取远端镜像。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_text(RELEASE_COMPOSE_PATH)

    assert "DOG_AGENT_API_IMAGE:?" in compose
    assert "build: !reset null" in compose
    assert "pull_policy: always" in compose
    assert "context:" not in compose
    assert "dockerfile:" not in compose


def test_env_example_should_select_a_fixed_ghcr_image() -> None:
    """
    验证部署模板展示 GHCR 完整地址，并使用固定版本而不是 latest。

    参数含义：
        无。

    返回值含义：
        None。
    """

    env_example = _read_text(ENV_EXAMPLE_PATH)

    expected = (
        "DOG_AGENT_API_IMAGE="
        "ghcr.io/c840264221/dog-agent-api:1.21.0"
    )
    assert expected in env_example
    assert "DOG_AGENT_API_IMAGE=ghcr.io/" in env_example
    assert "DOG_AGENT_API_IMAGE=" + "latest" not in env_example


def test_ci_should_validate_merged_release_compose() -> None:
    """
    验证 CI 会检查基础文件和生产覆盖文件合并后的 Compose 配置。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_text(CI_WORKFLOW_PATH)

    assert "Validate release Compose configuration" in workflow
    assert "cp .env.example .env" in workflow
    assert "-f compose.yaml" in workflow
    assert "-f compose.release.yaml" in workflow
    assert "config --quiet" in workflow
    assert "rm .env" in workflow
