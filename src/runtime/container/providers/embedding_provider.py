from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from src.settings import settings

from src.logger import logger


class EmbeddingProvider:

    def __init__(self):

        self._embedding = None

    @property
    def embedding(self):

        if self._embedding is None:

            logger.info(
                "🚀 初始化 Embedding..."
            )

            self._embedding = (
                HuggingFaceEmbeddings(
                    model_name=(
                        settings.llm.embedding_model
                    ),

                    cache_folder=str(
                        settings.path.CACHE_DIR
                    )
                )
            )

        return self._embedding

    async def startup(self):
        _ = self.embedding

        logger.info(
            "EmbeddingProvider 启动完成"
        )

    async def shutdown(self):
        logger.info(
            "EmbeddingProvider 已关闭"
        )