from src.settings.base import BaseAppSettings


class VectorStoreSettings(BaseAppSettings):

    collection_name: str = "dog_knowledge"

    top_k: int = 5