"""
Docker Compose 生产运行契约测试。

功能：
    验证 API 服务使用固定镜像、生产配置、宿主机 Ollama 地址和持久化目录，
    防止容器误用 localhost 或把运行数据留在临时容器文件系统中。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"


def _read_compose() -> str:
    """
    读取 Docker Compose 配置文本。

    参数含义：
        无。

    返回值含义：
        str:
            使用 UTF-8 解码的完整 Compose 配置。
    """

    return COMPOSE_PATH.read_text(encoding="utf-8")


def test_compose_should_build_and_run_versioned_api_image() -> None:
    """
    验证 Compose 使用当前版本镜像和项目 Dockerfile。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()

    assert "image: dog-agent-api:v1.19.0" in compose
    assert "dockerfile: Dockerfile" in compose
    assert "env_file:" in compose
    assert "- .env" in compose
    assert '"${API_PORT:-8000}:8000"' in compose


def test_compose_should_apply_safe_container_runtime_settings() -> None:
    """
    验证容器固定使用生产模式、单进程和非热重载配置。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()

    assert "API_ENVIRONMENT: production" in compose
    assert "API_HOST: 0.0.0.0" in compose
    assert 'API_WORKERS: "1"' in compose
    assert 'API_RELOAD: "false"' in compose
    assert "BASE_DIR: /app" in compose
    assert "restart: unless-stopped" in compose


def test_compose_should_reach_host_ollama_without_localhost() -> None:
    """
    验证 API 容器通过 Docker 宿主机地址访问 Ollama。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()

    assert (
        "OLLAMA_BASE_URL: http://host.docker.internal:11434"
        in compose
    )
    assert (
        "OLLAMA_HOST: http://host.docker.internal:11434"
        in compose
    )
    assert '"host.docker.internal:host-gateway"' in compose


def test_compose_should_mount_all_mutable_runtime_directories() -> None:
    """
    验证向量库、模型缓存、日志和状态数据库都挂载到宿主机。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()
    expected_mounts = {
        "./chroma_db:/app/chroma_db",
        "./chroma_memory_db:/app/chroma_memory_db",
        "./models_cache:/app/models_cache",
        "./logs:/app/logs",
        "./data/checkpoints_db:/app/data/checkpoints_db",
        "./data/memory_db:/app/data/memory_db",
        "./data/user:/app/data/user",
    }

    for mount in expected_mounts:
        assert f"- {mount}" in compose
