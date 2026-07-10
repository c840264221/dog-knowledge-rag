from pydantic import Field
from pydantic_settings import SettingsConfigDict

from src.settings.base import BaseAppSettings


class SQLiteMcpSettings(BaseAppSettings):
    """
    SQLiteMcpSettings：SQLite MCP 配置。

    功能：
        管理 SQLite MCP 只读数据库工具相关配置。

    字段说明：
        enabled:
            是否启用 SQLite MCP 能力。

        allowed_databases:
            数据库白名单，key 是数据库别名，value 是 SQLite 文件路径。

        default_limit:
            默认返回行数。

        max_limit:
            最大返回行数。
    """

    model_config = SettingsConfigDict(
        env_prefix="MCP_SQLITE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True

    allowed_databases: dict[str, str] = Field(
        default_factory=dict
    )

    default_limit: int = 20

    max_limit: int = 100


class McpSettings(BaseAppSettings):
    """
    McpSettings：MCP 总配置。

    功能：
        作为 MCP 配置聚合对象，当前先聚合 SQLite MCP 配置。

    字段说明：
        sqlite:
            SQLite MCP 配置对象。
    """

    sqlite: SQLiteMcpSettings = Field(
        default_factory=SQLiteMcpSettings
    )
