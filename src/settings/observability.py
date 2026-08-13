from pydantic import Field

from src.runtime.observability.llm_call_budget import LLMCallBudgetLimits
from src.runtime.observability.llm_call_records import LLMCallPurpose
from src.settings.base import BaseAppSettings


class ObservabilitySettings(BaseAppSettings):
    """
    可观测系统配置。

    功能：
        控制 Runtime Report、Timeline Report、RAG Debug Report 和
        LLM Call Report 是否输出到控制台、日志或文件。

    参数：
        无。通过默认值或 .env 环境变量读取。

    返回值：
        ObservabilitySettings:
            可观测配置对象。
    """

    ENABLE_CONSOLE_TIMELINE_REPORT: bool = False

    ENABLE_CONSOLE_RUNTIME_REPORT: bool = False

    ENABLE_RAG_DEBUG_REPORT: bool = True

    RAG_DEBUG_REPORT_TO_CONSOLE: bool = False

    RAG_DEBUG_REPORT_TO_FILE: bool = True

    RAG_DEBUG_CONTEXT_MAX_CHARS: int = 1200

    RAG_DEBUG_ANSWER_MAX_CHARS: int = 1200

    RAG_DEBUG_REPORT_USE_DATE_DIR: bool = True

    RAG_DEBUG_REPORT_RETENTION_DAYS: int = 7

    RAG_DEBUG_REPORT_CLEANUP_ON_WRITE: bool = True

    ENABLE_LLM_CALL_REPORT: bool = True

    LLM_CALL_REPORT_TO_LOG: bool = True

    LLM_CALL_REPORT_TO_FILE: bool = True

    LLM_CALL_REPORT_USE_DATE_DIR: bool = True

    LLM_CALL_BUDGET_WARNING_TO_LOG: bool = True

    LLM_CALL_BUDGETS_BY_PURPOSE: dict[
        LLMCallPurpose,
        LLMCallBudgetLimits,
    ] = Field(default_factory=dict)
