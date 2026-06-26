from abc import ABC, abstractmethod

from src.rag.schemas import (
    RagDocument,
    RagChunk,
)


class BaseDocumentChunker(ABC):
    """
    文档切分器抽象基类。

    功能：
        定义所有 Chunker（文本切分器）必须遵守的统一接口。
        Chunker 负责把 RagDocument（RAG 原始文档）切分成多个 RagChunk（RAG 文本块）。

    技术名词：
        Chunker:
            文本切分器，用来把长文档切成多个较短文本块。

        Chunk:
            文本块，是 Embedding（向量化）和 Retrieval（检索）的基本单位。

    参数：
        无。

    返回值：
        这是抽象基类，本身不会直接返回业务数据。
        子类需要实现 chunk 方法。
    """

    @abstractmethod
    def chunk(
        self,
        document: RagDocument
    ) -> list[RagChunk]:
        """
        切分单篇文档。

        功能：
            把一篇 RagDocument 切分成多个 RagChunk。

        参数：
            document:
                需要切分的 RAG 原始文档。

        返回值：
            list[RagChunk]:
                切分后的文本块列表。
        """

        raise NotImplementedError

    def chunk_many(
        self,
        documents: list[RagDocument]
    ) -> list[RagChunk]:
        """
        批量切分文档。

        功能：
            把多篇 RagDocument 批量切分成 RagChunk 列表。

        参数：
            documents:
                RAG 原始文档列表。

        返回值：
            list[RagChunk]:
                所有文档切分后的文本块列表。
        """

        chunks: list[RagChunk] = []

        for document in documents:
            chunks.extend(
                self.chunk(document)
            )

        return chunks