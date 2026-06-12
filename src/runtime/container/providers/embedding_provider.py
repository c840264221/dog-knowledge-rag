from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.settings import settings
from src.memory.embedding import MemoryEmbeddingService

from src.logger import logger


class EmbeddingProvider:

    def __init__(self):

        self._embedding = None

        self._memory_embedding = None

        self._memory_embedding_service = None

    def _create_embedding(
            self,
            provider: str,
            model_name: str,
            device: str = "cpu",
            normalize_embeddings: bool = True,
    ):
        """
        创建 Embedding 实例

        中文释义：
        - provider：模型来源，例如 huggingface / ollama
        - model_name：模型名称
        - device：运行设备，例如 cpu / cuda
        - normalize_embeddings：是否归一化向量
        """

        if provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=str(
                    settings.path.CACHE_DIR
                ),
                model_kwargs={
                    "device": device,
                },
                encode_kwargs={
                    "normalize_embeddings": normalize_embeddings,
                },
            )

        if provider == "ollama":
            return OllamaEmbeddings(
                model=model_name,
            )

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    @property
    def embedding(self):
        """
        默认 Embedding

        主要给普通 RAG / 文档检索使用。
        """

        if self._embedding is None:
            logger.info(
                "🚀 初始化默认 Embedding..."
            )

            self._embedding = self._create_embedding(
                provider=settings.embedding.provider,
                model_name=settings.embedding.model_name,
                device=settings.embedding.device,
                normalize_embeddings=(
                    settings.embedding.normalize_embeddings
                ),
            )

        return self._embedding

    @property
    def memory_embedding(self):
        """
        Memory 专用 Embedding

        主要给 Memory V7 语义召回使用。
        """

        if self._memory_embedding is None:
            logger.info(
                "🚀 初始化 Memory Embedding..."
            )

            self._memory_embedding = self._create_embedding(
                provider=settings.embedding.memory_provider,
                model_name=settings.embedding.memory_model_name,
                device=settings.embedding.memory_device,
                normalize_embeddings=(
                    settings.embedding.memory_normalize_embeddings
                ),
            )

        return self._memory_embedding

    @property
    def memory_embedding_service(
            self,
    ) -> MemoryEmbeddingService:
        """
        MemoryEmbeddingService

        中文释义：
        - service：服务层，给 Memory 模块使用
        - embedding client：真正负责生成向量的客户端
        """

        if self._memory_embedding_service is None:
            logger.info(
                "🚀 初始化 MemoryEmbeddingService..."
            )

            self._memory_embedding_service = (
                MemoryEmbeddingService(
                    embedding_client=self.memory_embedding,
                    model_name=(
                        settings.embedding.memory_model_name
                    ),
                    provider=(
                        settings.embedding.memory_provider
                    ),
                )
            )

        return self._memory_embedding_service


    async def startup(self):
        _ = self.embedding
        _ = self.memory_embedding_service

        logger.info(
            "EmbeddingProvider 启动完成"
        )

    async def shutdown(self):
        logger.info(
            "EmbeddingProvider 已关闭"
        )