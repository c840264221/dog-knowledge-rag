from __future__ import annotations

from typing import Protocol

from src.memory.embedding.schemas import EmbeddingResult

from src.logger import logger


class EmbeddingClient(Protocol):
    """
    EmbeddingClient：Embedding 客户端协议

    中文释义：
    - Protocol：协议类型，只要求对象拥有指定方法
    - embed_query：把一段文本转成向量
    """

    def embed_query(self, text: str) -> list[float]:
        ...


class MemoryEmbeddingService:
    """
    MemoryEmbeddingService：Memory 向量化服务

    职责：
    - 清洗文本
    - 调用 EmbeddingClient
    - 校验结果
    - 返回统一 EmbeddingResult
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        model_name: str,
        provider: str,
    ) -> None:
        self.embedding_client = embedding_client
        self.model_name = model_name
        self.provider = provider

    def embed_text(self, text: str) -> EmbeddingResult:
        clean_text = text.strip()

        if not clean_text:

            logger.error("Embedding text cannot be empty.")

            raise ValueError("Embedding text cannot be empty.")

        embedding = self.embedding_client.embed_query(clean_text)

        if not embedding:

            logger.error("Embedding result is empty.")

            raise ValueError("Embedding result is empty.")

        return EmbeddingResult(
            embedding=embedding,
            model_name=self.model_name,
            provider=self.provider,
            text=clean_text,
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[EmbeddingResult]:
        return [
            self.embed_text(text)
            for text in texts
        ]