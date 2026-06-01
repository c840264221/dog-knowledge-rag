from src.settings.base import BaseAppSettings
from pydantic import field_validator


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