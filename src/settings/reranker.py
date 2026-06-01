from src.settings.base import BaseAppSettings
from pydantic import SecretStr, Field


class RerankerSettings(BaseAppSettings):

    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    device: str = "cpu"

    huggingface_token: SecretStr | None = Field(
        default=None,
        validation_alias="HUGGINGFACE_TOKEN"
    )