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


class ExpenseMcpSettings(BaseAppSettings):
    """本机教学记账配置；enabled为显式开关，默认关闭；不提供生产认证。"""

    model_config = SettingsConfigDict(env_prefix="MCP_EXPENSE_", extra="ignore")
    enabled: bool = False


class McpSettings(BaseAppSettings):
    """
    McpSettings：MCP 总配置。

    功能：
        聚合SQLite只读工具与默认关闭的本机教学记账配置。

    字段说明：
        sqlite:
            SQLite MCP 配置对象。
        expense:
            记账MCP配置；MCP_EXPENSE_ENABLED=true时显式启用。
    """

    sqlite: SQLiteMcpSettings = Field(
        default_factory=SQLiteMcpSettings
    )

    expense: ExpenseMcpSettings = Field(default_factory=ExpenseMcpSettings)
