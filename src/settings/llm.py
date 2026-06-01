from pydantic import SecretStr, Field

from src.settings.base import BaseAppSettings


class LLMSettings(BaseAppSettings):

    main_model: str = "deepseek-v4-pro"

    backup_model: str = "deepseek-v4-flash"

    chinese_model: str = "qwen2.5:7b"

    embedding_model: str = "BAAI/bge-small-zh"

    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY"
    )

    deepseek_base_url: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_BASE_URL"
    )


    temperature: float = 0

    max_attempts: int = 3

    # 请求超时时间
    request_timeout: int = 60

    # 最大token值
    max_tokens: int = 2048

    ollama_base_url: str = (
        "http://localhost:11434"
    )
