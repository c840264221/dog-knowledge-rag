"""
生产依赖清单契约测试。

功能：
    验证 Docker 生产依赖包含 API、Agent、RAG、模型和数据库核心能力，
    版本与完整开发依赖一致，并排除测试、UI 和爬虫专用包。
"""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-prod.txt"
CPU_TORCH_REQUIREMENTS_PATH = (
    PROJECT_ROOT / "requirements-torch-cpu.txt"
)
DEVELOPMENT_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

REQUIRED_PRODUCTION_PACKAGES = {
    "aiohttp",
    "aiosqlite",
    "chromadb",
    "fastapi",
    "langchain-chroma",
    "langchain-core",
    "langchain-huggingface",
    "langchain-ollama",
    "langchain-openai",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "loguru",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "sentence-transformers",
    "sqlalchemy",
    "torch",
    "transformers",
    "uvicorn",
}
FORBIDDEN_PRODUCTION_PACKAGES = {
    "beautifulsoup4",
    "gradio",
    "gradio-client",
    "pytest",
    "pytest-asyncio",
    "selenium",
    "webdriver-manager",
}


def _canonicalize_package_name(raw_name: str) -> str:
    """
    将 Python 包名称转换成不区分大小写和分隔符的标准形式。

    参数含义：
        raw_name:
            requirements 文件中的原始包名称。

    返回值含义：
        str:
            使用小写和连字符表示的标准包名称。
    """

    return re.sub(r"[-_.]+", "-", raw_name).lower()


def _read_pinned_requirements(path: Path) -> dict[str, str]:
    """
    读取只使用双等号固定版本的 requirements 文件。

    参数含义：
        path:
            需要解析的依赖清单路径。

    返回值含义：
        dict[str, str]:
            标准包名称到固定版本号的映射。
    """

    raw_content = path.read_bytes()
    encoding = (
        "utf-16"
        if raw_content.startswith((b"\xff\xfe", b"\xfe\xff"))
        else "utf-8"
    )
    requirements: dict[str, str] = {}
    for raw_line in raw_content.decode(encoding).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        package_spec, separator, version = line.partition("==")
        if not separator:
            raise AssertionError(f"生产依赖没有固定版本: {line}")
        package_name = package_spec.split("[", 1)[0]
        canonical_name = _canonicalize_package_name(package_name)
        if canonical_name in requirements:
            raise AssertionError(f"生产依赖重复声明: {canonical_name}")
        requirements[canonical_name] = version
    return requirements


def test_production_requirements_should_include_runtime_capabilities() -> None:
    """
    验证生产清单覆盖 API 主链路需要的顶层能力。

    参数含义：
        无。

    返回值含义：
        None。
    """

    production_requirements = _read_pinned_requirements(
        PRODUCTION_REQUIREMENTS_PATH
    )
    cpu_torch_requirements = _read_pinned_requirements(
        CPU_TORCH_REQUIREMENTS_PATH
    )

    assert REQUIRED_PRODUCTION_PACKAGES <= {
        *production_requirements,
        *cpu_torch_requirements,
    }


def test_cpu_torch_should_use_official_cpu_index() -> None:
    """
    验证生产镜像从官方 CPU 软件源安装独立的 Torch 运行时。

    参数含义：
        无。

    返回值含义：
        None。
    """

    cpu_requirements_content = CPU_TORCH_REQUIREMENTS_PATH.read_text(
        encoding="utf-8"
    )
    cpu_torch_requirements = _read_pinned_requirements(
        CPU_TORCH_REQUIREMENTS_PATH
    )
    production_requirements = _read_pinned_requirements(
        PRODUCTION_REQUIREMENTS_PATH
    )

    assert (
        "--index-url https://download.pytorch.org/whl/cpu"
        in cpu_requirements_content
    )
    assert cpu_torch_requirements == {"torch": "2.11.0"}
    assert "torch" not in production_requirements


def test_production_requirements_should_exclude_non_runtime_tools() -> None:
    """
    验证测试、UI 和爬虫专用依赖不会进入生产镜像。

    参数含义：
        无。

    返回值含义：
        None。
    """

    production_requirements = _read_pinned_requirements(
        PRODUCTION_REQUIREMENTS_PATH
    )

    assert FORBIDDEN_PRODUCTION_PACKAGES.isdisjoint(
        production_requirements
    )


def test_production_versions_should_match_development_lock() -> None:
    """
    验证生产依赖版本与当前已验证的完整开发依赖一致。

    参数含义：
        无。

    返回值含义：
        None。
    """

    production_requirements = _read_pinned_requirements(
        PRODUCTION_REQUIREMENTS_PATH
    )
    cpu_torch_requirements = _read_pinned_requirements(
        CPU_TORCH_REQUIREMENTS_PATH
    )
    development_requirements = _read_pinned_requirements(
        DEVELOPMENT_REQUIREMENTS_PATH
    )
    image_requirements = {
        **production_requirements,
        **cpu_torch_requirements,
    }

    assert {
        package_name: development_requirements.get(package_name)
        for package_name in image_requirements
    } == image_requirements
