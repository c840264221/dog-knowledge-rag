"""
Dockerfile 生产镜像契约测试。

功能：
    验证镜像使用生产依赖、非 root 用户、正确 API 启动命令和就绪检查，
    并阻止真实环境配置或完整开发依赖进入构建步骤。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"


def _read_dockerfile() -> str:
    """
    读取 Dockerfile 文本。

    参数含义：
        无。

    返回值含义：
        str:
            使用 UTF-8 解码的完整 Dockerfile 内容。
    """

    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def test_dockerfile_should_use_python_313_slim_and_production_requirements() -> None:
    """
    验证镜像基础版本和生产依赖安装入口。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dockerfile = _read_dockerfile()

    assert "FROM python:3.13-slim" in dockerfile
    assert (
        "COPY requirements-torch-cpu.txt requirements-prod.txt ./"
        in dockerfile
    )
    assert (
        "pip install --requirement requirements-torch-cpu.txt"
        in dockerfile
    )
    assert "pip install --requirement requirements-prod.txt" in dockerfile
    assert dockerfile.index(
        "pip install --requirement requirements-torch-cpu.txt"
    ) < dockerfile.index(
        "pip install --requirement requirements-prod.txt"
    )
    assert "COPY requirements.txt" not in dockerfile


def test_dockerfile_should_copy_only_runtime_source_and_static_data() -> None:
    """
    验证镜像只显式复制生产源码、启动入口和静态数据。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dockerfile = _read_dockerfile()

    assert "COPY --chown=app:app src/ ./src/" in dockerfile
    assert (
        "COPY --chown=app:app scripts/api_run.py "
        "./scripts/api_run.py"
    ) in dockerfile
    assert "COPY --chown=app:app data/ ./data/" in dockerfile
    assert "COPY . ." not in dockerfile
    assert "COPY .env" not in dockerfile


def test_dockerfile_should_run_as_non_root_user() -> None:
    """
    验证生产 API 不会以 root 用户身份长期运行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dockerfile = _read_dockerfile()

    assert "useradd --system --uid 10001" in dockerfile
    assert "\nUSER app\n" in dockerfile


def test_dockerfile_should_start_api_and_check_readiness() -> None:
    """
    验证容器启动命令、监听配置和健康检查契约。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dockerfile = _read_dockerfile()

    assert "API_HOST=0.0.0.0" in dockerfile
    assert "API_WORKERS=1" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/ready" in dockerfile
    assert 'CMD ["python", "-m", "scripts.api_run"]' in dockerfile
