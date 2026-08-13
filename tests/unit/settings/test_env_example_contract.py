"""
环境变量示例契约测试。

功能：
    验证 .env.example 覆盖部署关键配置、不包含真实密钥、没有重复名称，
    并阻止旧版配置名称重新进入新部署链路。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

REQUIRED_ENVIRONMENT_VARIABLES = {
    "API_ENVIRONMENT",
    "API_HOST",
    "API_PORT",
    "API_WORKERS",
    "API_RELOAD",
    "API_AUTH_ENABLED",
    "API_AUTH_KEY",
    "API_CORS_ENABLED",
    "API_CORS_ALLOWED_ORIGINS",
    "API_CORS_ALLOW_CREDENTIALS",
    "API_MAX_REQUEST_BODY_BYTES",
    "API_RATE_LIMIT_ENABLED",
    "API_RATE_LIMIT_REQUESTS",
    "API_RATE_LIMIT_WINDOW_SECONDS",
    "API_TRUSTED_PROXY_CIDRS",
    "DEEPSEEK_API_KEY",
    "MAIN_MODEL",
    "REQUEST_TIMEOUT",
    "OLLAMA_BASE_URL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MEMORY_PROVIDER",
    "HUGGINGFACE_TOKEN",
    "TOOL_TIMEOUT",
    "MAX_RETRIES",
    "ENABLE_TIMEOUT_MIDDLEWARE",
    "MULTI_AGENT_STEP_TIMEOUT_SECONDS",
    "MEMORY_MINIMUM_SEMANTIC_SCORE",
    "RAG_QUALITY_MIN_SCORE",
    "RAG_RETRY_MAX_COUNT",
    "RAG_DEBUG_REPORT_DIR",
    "LLM_CALL_REPORT_DIR",
    "ENABLE_LLM_CALL_REPORT",
    "LLM_CALL_REPORT_TO_LOG",
    "LLM_CALL_REPORT_TO_FILE",
    "LLM_CALL_BUDGETS_BY_PURPOSE",
    "CHROMA_DB_DIR",
    "MEMORY_CHROMA_DB_DIR",
    "CHECKPOINTS_DB_PATH",
    "MEMORY_DB_PATH",
    "MCP_SQLITE_ALLOWED_DATABASES",
}
SENSITIVE_ENVIRONMENT_VARIABLES = {
    "DEEPSEEK_API_KEY",
    "HUGGINGFACE_TOKEN",
    "LANGSMITH_API_KEY",
    "API_AUTH_KEY",
}
LEGACY_ENVIRONMENT_VARIABLES = {
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_TOOL_TIMEOUT",
    "LLM_MODEL_NAME",
    "LLM_TIMEOUT",
}


def _read_env_assignments() -> list[tuple[str, str]]:
    """
    读取 .env.example 中未注释的变量赋值。

    参数含义：
        无。

    返回值含义：
        list[tuple[str, str]]:
            按文件顺序保存的变量名称和值；不会加载到进程环境。
    """

    assignments: list[tuple[str, str]] = []
    for raw_line in ENV_EXAMPLE_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        variable_name, value = line.split("=", 1)
        assignments.append((variable_name.strip(), value.strip()))
    return assignments


def test_env_example_should_cover_required_deployment_variables() -> None:
    """
    验证模板包含 API、模型、持久化和运行时关键变量。

    参数含义：
        无。

    返回值含义：
        None。
    """

    variable_names = {
        variable_name
        for variable_name, _ in _read_env_assignments()
    }

    assert REQUIRED_ENVIRONMENT_VARIABLES <= variable_names


def test_env_example_should_not_contain_duplicate_variables() -> None:
    """
    验证同一个变量不会因分组整理而重复出现。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assignments = _read_env_assignments()
    variable_names = [
        variable_name
        for variable_name, _ in assignments
    ]

    assert len(variable_names) == len(set(variable_names))


def test_env_example_should_leave_sensitive_values_empty() -> None:
    """
    验证示例文件不会泄露真实 API Key 或访问令牌。

    参数含义：
        无。

    返回值含义：
        None。
    """

    values_by_name = dict(_read_env_assignments())

    assert {
        variable_name: values_by_name[variable_name]
        for variable_name in SENSITIVE_ENVIRONMENT_VARIABLES
    } == {
        variable_name: ""
        for variable_name in SENSITIVE_ENVIRONMENT_VARIABLES
    }


def test_env_example_should_not_restore_legacy_variable_names() -> None:
    """
    验证新版部署模板不再传播已被替代的旧配置名称。

    参数含义：
        无。

    返回值含义：
        None。
    """

    variable_names = {
        variable_name
        for variable_name, _ in _read_env_assignments()
    }

    assert LEGACY_ENVIRONMENT_VARIABLES.isdisjoint(variable_names)


def test_env_example_should_explain_every_variable() -> None:
    """
    验证每个配置字段正上方都有包含字段名称的用途注释。

    参数含义：
        无。

    返回值含义：
        None。
    """

    lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    missing_comments: list[str] = []
    for line_number, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        variable_name = line.split("=", 1)[0].strip()
        previous_line = (
            lines[line_number - 1].strip()
            if line_number > 0
            else ""
        )
        if not (
            previous_line.startswith("#")
            and variable_name in previous_line
        ):
            missing_comments.append(variable_name)

    assert missing_comments == []
