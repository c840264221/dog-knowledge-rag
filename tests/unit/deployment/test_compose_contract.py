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

    assert "image: dog-agent-api:v1.22.0-dev" in compose
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
    assert (
        'API_TRUSTED_PROXY_CIDRS: "${API_TRUSTED_PROXY_CIDRS:-[]}"'
        in compose
    )
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


def test_compose_should_provide_healthy_persistent_redis() -> None:
    """
    验证 Compose 使用固定 Redis 镜像、持久化数据并等待健康检查。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()

    assert "image: redis:8.2.8-alpine" in compose
    assert "REDIS_ENABLED: \"true\"" in compose
    assert "REDIS_URL: redis://redis:6379/0" in compose
    assert '"127.0.0.1:${REDIS_PORT:-6379}:6379"' in compose
    assert "condition: service_healthy" in compose
    assert '["CMD", "redis-cli", "ping"]' in compose
    assert "redis_data:/data" in compose
    assert "--appendonly" in compose


def test_compose_should_run_long_task_worker_as_independent_service() -> None:
    """
    验证长任务 Worker 使用独立进程命令且不暴露 API 端口。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_compose()
    worker_section = compose.split("\n  long-task-worker:\n", 1)[1].split(
        "\n  redis:\n",
        1,
    )[0]

    assert "scripts.run_long_task_worker" in worker_section
    assert 'REDIS_ENABLED: "true"' in worker_section
    assert "REDIS_URL: redis://redis:6379/0" in worker_section
    assert "LONG_TASK_WORKER_NAME:" in worker_section
    assert "condition: service_healthy" in worker_section
    assert "healthcheck:" in worker_section
    assert "disable: true" in worker_section
    assert "init: true" in worker_section
    assert "restart: unless-stopped" in worker_section
    assert "stop_grace_period: 45s" in worker_section
    assert "ports:" not in worker_section
