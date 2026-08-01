"""V1.22 Linux 发布镜像回滚脚本契约测试。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROLLBACK_SCRIPT_PATH = PROJECT_ROOT / "deployment" / "rollback-release.sh"
CI_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "v121-ci.yml"
)


def _read_text(path: Path) -> str:
    """
    使用 UTF-8 读取回滚部署契约文件。

    参数含义：
        path:
            需要读取的仓库文件绝对路径。

    返回值含义：
        str:
            文件的完整文本内容。
    """

    return path.read_text(encoding="utf-8")


def test_rollback_script_should_require_fixed_ghcr_version() -> None:
    """验证回滚脚本要求显式三段版本并拒绝 latest。"""

    script = _read_text(ROLLBACK_SCRIPT_PATH)

    assert 'ROLLBACK_API_IMAGE="${1:-}"' in script
    assert "^ghcr\\.io/" in script
    assert "[0-9]+\\.[0-9]+\\.[0-9]+" in script
    assert "禁止 latest" in script


def test_rollback_script_should_override_without_rewriting_env() -> None:
    """验证旧镜像仅覆盖当前 Compose 命令且不会改写私有环境文件。"""

    script = _read_text(ROLLBACK_SCRIPT_PATH)

    assert 'DOG_AGENT_API_IMAGE="$ROLLBACK_API_IMAGE" docker compose' in script
    assert "-f compose.release.yaml" in script
    assert "-f compose.proxy.yaml" in script
    assert "Set-Content" not in script
    assert "sed -i" not in script
    assert "> \"$ENV_FILE\"" not in script


def test_rollback_script_should_wait_verify_and_preserve_failure_logs() -> None:
    """验证回滚会等待健康、重新验收并在失败时输出上下文。"""

    script = _read_text(ROLLBACK_SCRIPT_PATH)

    assert "run_compose pull api" in script
    assert "--wait" in script
    assert "verify_v122_proxy_deployment" in script
    assert "trap print_failure_context ERR" in script
    assert "同步修改 .env" in script


def test_rollback_script_should_preflight_the_overridden_image() -> None:
    """验证回滚会在拉取镜像前预检临时覆盖值和私有安全配置。"""

    script = _read_text(ROLLBACK_SCRIPT_PATH)

    validator_index = script.index("validate_v122_release_env")
    pull_index = script.index("run_compose pull api")
    assert validator_index < pull_index
    assert '--image-override "$ROLLBACK_API_IMAGE"' in script


def test_ci_should_validate_rollback_script_syntax() -> None:
    """验证 CI 会对发布和回滚两个 Shell 脚本执行语法检查。"""

    workflow = _read_text(CI_WORKFLOW_PATH)

    assert "bash -n deployment/deploy-release.sh" in workflow
    assert "bash -n deployment/rollback-release.sh" in workflow
