from typing import Literal

from src.settings.base import BaseAppSettings

from pydantic_settings import SettingsConfigDict


EmbeddingProviderName = Literal[
    "huggingface",
    "ollama",
]


class EmbeddingSettings(BaseAppSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: EmbeddingProviderName = "huggingface"

    model_name: str = (
        "BAAI/bge-small-zh"
    )

    device: str = "cpu"

    normalize_embeddings: bool = True

    memory_provider: EmbeddingProviderName = "ollama"

    memory_model_name: str = (
        "qwen3-embedding:0.6b"
    )

    memory_device: str = "cpu"

    memory_normalize_embeddings: bool = True