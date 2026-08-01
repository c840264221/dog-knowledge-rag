"""
V1.21 Docker 镜像发布工作流契约测试。

功能：
    验证版本标签会触发 GHCR 镜像发布，并锁定版本解析、最小权限、登录凭据
    和镜像标签规则，防止后续维护时意外扩大发布范围或泄露长期凭据。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLISH_WORKFLOW_PATH = (
    PROJECT_ROOT
    / ".github"
    / "workflows"
    / "v121-publish-image.yml"
)


def _read_publish_workflow() -> str:
    """
    读取 V1.21 Docker 镜像发布工作流文本。

    参数含义：
        无。

    返回值含义：
        str:
            使用 UTF-8 解码的完整 GitHub Actions Workflow。
    """

    return PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")


def test_publish_workflow_should_only_run_for_version_tags() -> None:
    """
    验证发布动作只由 V 开头的三段式版本标签触发。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_publish_workflow()

    assert '      - "V*.*.*"' in workflow
    assert "pull_request:" not in workflow
    assert "branches:" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "cancel-in-progress: false" in workflow


def test_publish_workflow_should_resolve_stable_image_tags() -> None:
    """
    验证大写 V 版本标签会转换成稳定的镜像地址与四级版本标签。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_publish_workflow()

    assert 'version="${GITHUB_REF_NAME#V}"' in workflow
    assert '^[0-9]+\\.[0-9]+\\.[0-9]+$' in workflow
    assert "GITHUB_REPOSITORY_OWNER" in workflow
    assert "tr '[:upper:]' '[:lower:]'" in workflow
    assert 'image=ghcr.io/$owner/dog-agent-api' in workflow
    assert 'major_minor=${version%.*}' in workflow
    assert 'major=${version%%.*}' in workflow
    assert "uses: docker/metadata-action@v6" in workflow
    assert "type=raw,value=${{ steps.version.outputs.version }}" in workflow
    assert (
        "type=raw,value=${{ steps.version.outputs.major_minor }}"
        in workflow
    )
    assert "type=raw,value=${{ steps.version.outputs.major }}" in workflow
    assert "type=raw,value=latest" in workflow


def test_publish_workflow_should_use_scoped_github_credentials() -> None:
    """
    验证发布工作流使用最小 GitHub 权限和自动生成的短期令牌登录 GHCR。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_publish_workflow()

    assert "contents: read" in workflow
    assert "packages: write" in workflow
    assert "uses: docker/login-action@v4" in workflow
    assert "registry: ghcr.io" in workflow
    assert "username: ${{ github.actor }}" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "GHCR_TOKEN" not in workflow
    assert "DOCKERHUB" not in workflow


def test_publish_workflow_should_build_and_push_with_cache() -> None:
    """
    验证发布工作流使用项目 Dockerfile 构建、推送镜像并复用构建缓存。

    参数含义：
        无。

    返回值含义：
        None。
    """

    workflow = _read_publish_workflow()

    assert "uses: actions/checkout@v7" in workflow
    assert "uses: docker/setup-buildx-action@v4" in workflow
    assert "uses: docker/build-push-action@v7" in workflow
    assert "context: ." in workflow
    assert "file: Dockerfile" in workflow
    assert "push: true" in workflow
    assert "tags: ${{ steps.metadata.outputs.tags }}" in workflow
    assert "labels: ${{ steps.metadata.outputs.labels }}" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to: type=gha,mode=max" in workflow
