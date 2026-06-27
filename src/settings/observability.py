from src.settings.base import BaseAppSettings


class ObservabilitySettings(BaseAppSettings):
    """
    可观测系统配置。

    功能：
        控制 Runtime Report、Timeline Report、RAG Debug Report
        是否输出到控制台或文件。

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