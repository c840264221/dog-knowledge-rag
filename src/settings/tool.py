from src.settings.base import BaseAppSettings
from pydantic import Field, field_validator


class ToolSettings(BaseAppSettings):

    default_timeout: int = 30

    @field_validator("default_timeout")
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError(
                "timeout必须大于0"
            )

        return v

    default_retry_count: int = 3

    retry_delay: float = 1.0

    # 工具规划技能文件：新增技能只配置路径和文件元信息，不增加解析器分支。
    planning_skill_files: list[str] = Field(default_factory=lambda: ["skills/expense-recording/SKILL.md"])

    # 粗路由的显式操作意图提示，不读取技能文件或决定具体工具参数。
    action_request_hints: list[str] = Field(default_factory=lambda: [
        "记账", "记帐", "记一笔", "记录支出", "记录消费", "用品预算", "预算核算",
    ])
