from src.settings.base import BaseAppSettings


class EmbeddingSettings(BaseAppSettings):

    model_name: str = (
        "BAAI/bge-small-zh"
    )

    device: str = "cpu"

    normalize_embeddings: bool = True