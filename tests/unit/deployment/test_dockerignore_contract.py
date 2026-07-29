"""
Docker 构建上下文排除规则测试。

功能：
    验证密钥、本机虚拟环境和运行时数据库不会进入 Docker 构建上下文，
    同时保留环境变量模板与可版本化狗狗知识源。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"

REQUIRED_EXCLUDED_PATTERNS = {
    ".git/",
    ".venv/",
    ".env",
    ".env.*",
    "models_cache/",
    "chroma_db/",
    "chroma_memory_db/",
    "logs/",
    "data/checkpoints_db/",
    "data/memory_db/",
    "data/user/",
    "tests/",
    "evaluation/",
}


def _read_dockerignore_patterns() -> list[str]:
    """
    读取 .dockerignore 中有效的模式。

    参数含义：
        无。

    返回值含义：
        list[str]:
            去除空行和注释后，按文件顺序保存的 Docker 排除模式。
    """

    return [
        line
        for raw_line in DOCKERIGNORE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if (line := raw_line.strip())
        and not line.startswith("#")
    ]


def test_dockerignore_should_exclude_sensitive_and_runtime_files() -> None:
    """
    验证敏感配置和运行时数据都在排除列表中。

    参数含义：
        无。

    返回值含义：
        None。
    """

    patterns = set(_read_dockerignore_patterns())

    assert REQUIRED_EXCLUDED_PATTERNS <= patterns


def test_dockerignore_should_keep_env_example_after_env_wildcard() -> None:
    """
    验证真实环境文件被排除后，配置说明书会被重新包含。

    参数含义：
        无。

    返回值含义：
        None。
    """

    patterns = _read_dockerignore_patterns()

    assert "!.env.example" in patterns
    assert patterns.index("!.env.example") > patterns.index(".env.*")


def test_dockerignore_should_keep_versioned_knowledge_sources() -> None:
    """
    验证构建规则不会误删 RAG 静态知识文档和规则 JSON。

    参数含义：
        无。

    返回值含义：
        None。
    """

    patterns = set(_read_dockerignore_patterns())

    assert "data/" not in patterns
    assert "data/dog_markdown/" not in patterns
    assert "*.md" not in patterns
    assert "*.json" not in patterns
