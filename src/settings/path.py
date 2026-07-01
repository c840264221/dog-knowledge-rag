from pathlib import Path

from pydantic import model_validator

from src.settings.base import BaseAppSettings


class PathSettings(BaseAppSettings):
    """
    项目路径配置。

    功能：
        统一管理项目中的数据目录、日志目录、缓存目录、
        checkpoint 路径、RAG Debug Report 路径等。

        如果 .env 中传入的是相对路径，例如：
            RAG_DEBUG_REPORT_DIR=logs/report/rag_debug

        本配置会自动将它解析为：
            BASE_DIR / "logs/report/rag_debug"

        避免因为 PyCharm / 命令行的当前工作目录不同，
        导致文件生成到错误位置。

    参数：
        无。通过默认值或 .env 环境变量读取。

    返回值：
        PathSettings:
            路径配置对象。
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    DATA_DIR: Path = BASE_DIR / "data"

    LOG_DIR: Path = BASE_DIR / "logs"

    REPORT_DIR: Path = LOG_DIR / "report"

    RAG_DEBUG_REPORT_DIR: Path = REPORT_DIR / "rag_debug"

    RAG_EVALUATE_REPORT_DIR: Path = REPORT_DIR / "rag_evaluate"

    CACHE_DIR: Path = BASE_DIR / "models_cache"

    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"

    MEMORY_CHROMA_DB_DIR: Path = BASE_DIR / "chroma_memory_db"

    CHECKPOINT_DIR: Path = DATA_DIR / "checkpoints_db"

    MEMORY_DIR: Path = DATA_DIR / "memory_db"

    USER_DIR: Path = DATA_DIR / "user"

    DOG_MD_DATA_DIR: Path = DATA_DIR / "dog_markdown"

    DOG_NAME_JSON_PATH: Path = DATA_DIR / "dog_names.json"

    DOG_NAME_ALIAS_JSON_PATH: Path = DATA_DIR / "alias_dog_name.json"

    DOG_DATA_JSON_PATH: Path = DATA_DIR / "dogs.json"

    FILTER_RULES_PATH: Path = DATA_DIR / "filter_rules.json"

    INTENT_RULES_PATH: Path = DATA_DIR / "intent_rules.json"

    TAG_RULES_PATH: Path = DATA_DIR / "tag_rules.json"

    CHECKPOINTS_DB_PATH: Path = (
        CHECKPOINT_DIR / "checkpoints.db"
    )

    MEMORY_DB_PATH: Path = (
        MEMORY_DIR / "memory.db"
    )

    USER_FILE: Path = (
        USER_DIR / "user.json"
    )

    @model_validator(mode="after")
    def resolve_relative_paths(self):
        """
        解析相对路径。

        功能：
            将 .env 中传入的相对路径统一转换成基于 BASE_DIR 的绝对路径。

            例如：
                logs/report/rag_debug

            会变成：
                BASE_DIR / logs/report/rag_debug

        参数：
            self:
                当前 PathSettings 实例。

        返回值：
            PathSettings:
                解析路径后的配置对象。

        专业名词：
            Relative Path：
                相对路径。依赖当前工作目录的路径。

            Absolute Path：
                绝对路径。从磁盘根路径开始的完整路径。

            Current Working Directory：
                当前工作目录。程序启动时所在的目录。
        """

        self.BASE_DIR = Path(
            self.BASE_DIR
        ).resolve()

        path_field_names = [
            "DATA_DIR",
            "LOG_DIR",
            "REPORT_DIR",
            "RAG_DEBUG_REPORT_DIR",
            "RAG_EVALUATE_REPORT_DIR",
            "CACHE_DIR",
            "CHROMA_DB_DIR",
            "MEMORY_CHROMA_DB_DIR",
            "CHECKPOINT_DIR",
            "MEMORY_DIR",
            "USER_DIR",
            "DOG_MD_DATA_DIR",
            "DOG_NAME_JSON_PATH",
            "DOG_NAME_ALIAS_JSON_PATH",
            "DOG_DATA_JSON_PATH",
            "FILTER_RULES_PATH",
            "INTENT_RULES_PATH",
            "TAG_RULES_PATH",
            "CHECKPOINTS_DB_PATH",
            "MEMORY_DB_PATH",
            "USER_FILE",
        ]

        for field_name in path_field_names:
            value = getattr(
                self,
                field_name,
                None,
            )

            if value is None:
                continue

            path_value = Path(
                value
            )

            if not path_value.is_absolute():
                path_value = self.BASE_DIR / path_value

            setattr(
                self,
                field_name,
                path_value
            )

        return self