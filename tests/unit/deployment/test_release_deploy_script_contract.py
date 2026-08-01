"""V1.22 Linux 发布部署脚本契约测试。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT_PATH = PROJECT_ROOT / "deployment" / "deploy-release.sh"
GIT_ATTRIBUTES_PATH = PROJECT_ROOT / ".gitattributes"
CI_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "v121-ci.yml"
)


def _read_text(path: Path) -> str:
    """
    使用 UTF-8 读取发布部署契约文件。

    参数含义：
        path:
            需要读取的仓库文件绝对路径。

    返回值含义：
        str:
            文件的完整文本内容。
    """

    return path.read_text(encoding="utf-8")


def test_deploy_script_should_merge_all_release_layers() -> None:
    """验证脚本合并基础、发布镜像和 Nginx 三层配置。"""

    script = _read_text(DEPLOY_SCRIPT_PATH)

    assert "set -Eeuo pipefail" in script
    assert "-f compose.yaml" in script
    assert "-f compose.release.yaml" in script
    assert "-f compose.proxy.yaml" in script
    assert 'config --quiet' in script


def test_deploy_script_should_pull_wait_and_verify() -> None:
    """验证脚本会拉取镜像、等待健康并运行自动验收。"""

    script = _read_text(DEPLOY_SCRIPT_PATH)

    assert '"${compose_command[@]}" pull' in script
    assert "--remove-orphans" in script
    assert "--wait" in script
    assert "--wait-timeout" in script
    assert "verify_v122_proxy_deployment" in script


def test_deploy_script_should_validate_private_env_before_pull() -> None:
    """验证发布脚本会在拉取镜像前执行生产配置预检。"""

    script = _read_text(DEPLOY_SCRIPT_PATH)

    validator_index = script.index("validate_v122_release_env")
    pull_index = script.index('"${compose_command[@]}" pull')
    assert validator_index < pull_index
    assert '--env-file "$ENV_FILE"' in script


def test_deploy_script_should_report_failure_without_printing_env() -> None:
    """验证失败时只输出状态和日志，不直接读取或打印私有环境文件。"""

    script = _read_text(DEPLOY_SCRIPT_PATH)

    assert "trap print_failure_context ERR" in script
    assert 'ps --all || true' in script
    assert 'logs --tail 100 api nginx || true' in script
    assert "cat $ENV_FILE" not in script
    assert "cat \"$ENV_FILE\"" not in script


def test_shell_script_should_keep_linux_line_endings_and_ci_syntax_check() -> None:
    """验证 Shell 脚本固定使用 LF，且 CI 会执行 Bash 语法检查。"""

    attributes = _read_text(GIT_ATTRIBUTES_PATH)
    workflow = _read_text(CI_WORKFLOW_PATH)

    assert "*.sh text eol=lf" in attributes
    assert "Validate deployment shell script" in workflow
    assert "bash -n deployment/deploy-release.sh" in workflow
