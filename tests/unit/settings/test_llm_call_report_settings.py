"""LLM 调用报告配置测试。"""

from __future__ import annotations

from src.settings.observability import ObservabilitySettings
from src.settings.path import PathSettings


def test_llm_call_report_settings_should_have_independent_outputs() -> None:
    """
    验证日志摘要和 Markdown 文件输出可以分别控制。

    参数含义：
        无。

    返回值含义：
        None。
    """

    settings = ObservabilitySettings(
        ENABLE_LLM_CALL_REPORT=True,
        LLM_CALL_REPORT_TO_LOG=False,
        LLM_CALL_REPORT_TO_FILE=True,
    )

    assert settings.ENABLE_LLM_CALL_REPORT is True
    assert settings.LLM_CALL_REPORT_TO_LOG is False
    assert settings.LLM_CALL_REPORT_TO_FILE is True
    assert settings.LLM_CALL_REPORT_USE_DATE_DIR is True
    assert settings.LLM_CALL_BUDGETS_BY_PURPOSE == {}


def test_llm_call_budget_should_parse_purpose_specific_json(
    monkeypatch,
) -> None:
    """
    验证环境变量可以按调用目的解析不同的软预算阈值。

    参数含义：
        monkeypatch：pytest 动态设置环境变量的工具。

    返回值含义：
        None。
    """

    monkeypatch.setenv(
        "LLM_CALL_BUDGETS_BY_PURPOSE",
        '{"routing_decision":{"max_total_tokens_per_call":200},'
        '"answer_generation":{"max_total_tokens_per_call":2000}}',
    )

    settings = ObservabilitySettings()

    assert (
        settings.LLM_CALL_BUDGETS_BY_PURPOSE[
            "routing_decision"
        ].max_total_tokens_per_call
        == 200
    )
    assert (
        settings.LLM_CALL_BUDGETS_BY_PURPOSE[
            "answer_generation"
        ].max_total_tokens_per_call
        == 2000
    )


def test_llm_call_report_dir_should_resolve_from_base_dir(tmp_path) -> None:
    """
    验证相对 LLM 报告目录会基于项目根目录解析成绝对路径。

    参数含义：
        tmp_path：pytest 提供的临时目录。

    返回值含义：
        None。
    """

    settings = PathSettings(
        BASE_DIR=tmp_path,
        LLM_CALL_REPORT_DIR="evaluation/reports/llm_calls",
    )

    assert settings.LLM_CALL_REPORT_DIR == (
        tmp_path / "evaluation" / "reports" / "llm_calls"
    )
