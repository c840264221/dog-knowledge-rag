"""
RAG Index Pipeline。

Index Pipeline（索引流水线 / 入库流水线）：
负责把 Markdown 文档从加载、metadata 提取、切块，到最终写入向量数据库的过程串起来。

当前 Pipeline 负责流程编排：

1. Loader（加载器）：
    Markdown -> RagDocument

2. Metadata Extractor（元数据提取器）：
    RagDocument -> metadata 增强后的 RagDocument

3. Chunker（切块器）：
    RagDocument -> RagChunk

4. Indexer（索引器 / 入库器）：
    RagChunk -> Chroma

注意：
    当前项目中的 MarkdownDocumentLoader 在初始化时接收 input_path，
    并通过 load() 方法统一加载文件或目录。
    所以 Pipeline 不再调用 loader.load_file 或 loader.load_dir。
"""

from __future__ import annotations

from typing import Any

from src.rag.schemas import (
    RagChunk,
    RagDocument,
)


class RagIndexPipeline:
    """
    RAG 入库流水线。

    Pipeline（流水线）：
    指把多个处理步骤按照固定顺序组合起来，
    形成一个完整的数据处理流程。

    当前流程是：
        MarkdownDocumentLoader.load()
            -> DogBreedMetadataExtractor.extract_many()
            -> MarkdownChunker.chunk_many()
            -> RagChromaIndexer.index_chunks()

    设计特点：
    1. Loader 自己持有 input_path
    2. Pipeline 只负责 orchestration（编排）
    3. Pipeline 不直接读取文件
    4. Pipeline 不直接解析 metadata
    5. Pipeline 不直接写入 Chroma
    """

    PIPELINE_NAME = "rag_index_pipeline_v1"

    def __init__(
            self,
            loader: Any,
            metadata_extractor: Any,
            chunker: Any,
            indexer: Any,
    ):
        """
        初始化 RagIndexPipeline。

        参数：
            loader: Any
                Loader（加载器）。
                当前预期是 MarkdownDocumentLoader。
                需要提供 load() 方法。

            metadata_extractor: Any
                Metadata Extractor（元数据提取器）。
                当前预期是 DogBreedMetadataExtractor。
                需要提供 extract_many(documents=...) 方法。

            chunker: Any
                Chunker（切块器）。
                当前预期是 MarkdownChunker。
                需要提供 chunk_many(documents=...) 方法。

            indexer: Any
                Indexer（索引器 / 入库器）。
                当前预期是 RagChromaIndexer。
                需要提供 index_chunks(chunks=...) 方法。

        返回值：
            None：
                构造函数无返回值。
        """

        self.loader = loader
        self.metadata_extractor = metadata_extractor
        self.chunker = chunker
        self.indexer = indexer

    def index(
            self,
            source: str | None = None,
    ) -> dict[str, Any]:
        """
        执行完整 RAG 入库流程。

        功能：
            1. 调用 loader.load() 加载 Markdown 文档
            2. 调用 metadata_extractor.extract_many() 提取结构化 metadata
            3. 调用 chunker.chunk_many() 切块
            4. 调用 indexer.index_chunks() 写入向量库
            5. 返回入库统计信息

        参数：
            source: str | None
                数据来源描述。
                如果不传，则尝试从 loader.input_path 中读取。

        返回值：
            dict[str, Any]：
                入库结果统计信息。
        """

        documents = self._load_documents()

        resolved_source = self._resolve_source(
            source=source,
        )

        return self.index_documents(
            documents=documents,
            source=resolved_source,
        )

    def index_documents(
            self,
            documents: list[RagDocument],
            source: str | None = None,
    ) -> dict[str, Any]:
        """
        索引已经加载好的 RagDocument 列表。

        功能：
            这个方法适合测试，也适合后续其他 Loader 复用。
            只要外部已经准备好 list[RagDocument]，
            就可以直接复用 metadata 提取、切块、入库流程。

        参数：
            documents: list[RagDocument]
                已经加载好的 RAG 文档列表。

            source: str | None
                数据来源描述。
                例如 Markdown 目录路径、文件路径、数据库名称等。

        返回值：
            dict[str, Any]：
                入库结果统计信息。
        """

        if not documents:
            return self._build_empty_result(
                source=source,
            )

        enhanced_documents = self._extract_metadata(
            documents=documents,
        )

        chunks = self._chunk_documents(
            documents=enhanced_documents,
        )

        index_result = self._index_chunks(
            chunks=chunks,
        )

        return self._build_pipeline_result(
            source=source,
            documents=documents,
            enhanced_documents=enhanced_documents,
            chunks=chunks,
            index_result=index_result,
        )

    def _load_documents(self) -> list[RagDocument]:
        """
        调用 loader.load() 加载文档。

        功能：
            当前项目中的 MarkdownDocumentLoader 通过 input_path 持有路径，
            公开加载入口是 load()。
            Pipeline 只调用这个公开方法，不直接访问 _load_file 等私有方法。

        参数：
            无。

        返回值：
            list[RagDocument]：
                Loader 加载得到的文档列表。
        """

        if not hasattr(
                self.loader,
                "load",
        ):
            raise AttributeError(
                "loader 必须提供 load() 方法"
            )

        result = self.loader.load()

        return self._ensure_document_list(
            value=result,
        )

    def _extract_metadata(
            self,
            documents: list[RagDocument],
    ) -> list[RagDocument]:
        """
        调用 metadata_extractor 提取 metadata。

        功能：
            将原始 RagDocument 转换成 metadata 增强后的 RagDocument。

        参数：
            documents: list[RagDocument]
                原始 RAG 文档列表。

        返回值：
            list[RagDocument]：
                metadata 增强后的文档列表。
        """

        if not hasattr(
                self.metadata_extractor,
                "extract_many",
        ):
            raise AttributeError(
                "metadata_extractor 必须提供 extract_many(documents=...) 方法"
            )

        result = self.metadata_extractor.extract_many(
            documents=documents,
        )

        return self._ensure_document_list(
            value=result,
        )

    def _chunk_documents(
            self,
            documents: list[RagDocument],
    ) -> list[RagChunk]:
        """
        调用 chunker 对 RagDocument 进行切块。

        功能：
            将 list[RagDocument] 转换成 list[RagChunk]。

        参数：
            documents: list[RagDocument]
                metadata 增强后的文档列表。

        返回值：
            list[RagChunk]：
                切块后的 chunk 列表。
        """

        if not hasattr(
                self.chunker,
                "chunk_many",
        ):
            raise AttributeError(
                "chunker 必须提供 chunk_many(documents=...) 方法"
            )

        result = self.chunker.chunk_many(
            documents=documents,
        )

        return self._ensure_chunk_list(
            value=result,
        )

    def _index_chunks(
            self,
            chunks: list[RagChunk],
    ) -> dict[str, Any]:
        """
        调用 indexer 写入 chunks。

        功能：
            将 list[RagChunk] 写入 Chroma 或其他向量数据库。

        参数：
            chunks: list[RagChunk]
                待写入的 RAG chunk 列表。

        返回值：
            dict[str, Any]：
                indexer 返回的入库统计信息。
        """

        if not hasattr(
                self.indexer,
                "index_chunks",
        ):
            raise AttributeError(
                "indexer 必须提供 index_chunks(chunks=...) 方法"
            )

        return self.indexer.index_chunks(
            chunks=chunks,
        )

    def _resolve_source(
            self,
            source: str | None,
    ) -> str | None:
        """
        解析数据来源描述。

        功能：
            如果外部传入 source，则直接使用。
            如果没有传入，则尝试从 loader.input_path 中读取。

        参数：
            source: str | None
                外部传入的数据来源描述。

        返回值：
            str | None：
                最终使用的数据来源描述。
        """

        if source is not None:
            return source

        input_path = getattr(
            self.loader,
            "input_path",
            None,
        )

        if input_path is None:
            return None

        return str(
            input_path,
        )

    def _ensure_document_list(
            self,
            value: Any,
    ) -> list[RagDocument]:
        """
        确保输入值是 list[RagDocument]。

        功能：
            某些 Loader 可能返回 RagDocument，
            某些 Loader 可能返回 list[RagDocument]。
            这里统一转换成 list[RagDocument]。

        参数：
            value: Any
                待检查和转换的值。

        返回值：
            list[RagDocument]：
                文档列表。
        """

        if value is None:
            return []

        if isinstance(
                value,
                list,
        ):
            return value

        return [
            value,
        ]

    def _ensure_chunk_list(
            self,
            value: Any,
    ) -> list[RagChunk]:
        """
        确保输入值是 list[RagChunk]。

        功能：
            将 chunker 返回值统一转换成 list[RagChunk]。

        参数：
            value: Any
                chunker 返回值。

        返回值：
            list[RagChunk]：
                chunk 列表。
        """

        if value is None:
            return []

        if isinstance(
                value,
                list,
        ):
            return value

        return [
            value,
        ]

    def _build_empty_result(
            self,
            source: str | None,
    ) -> dict[str, Any]:
        """
        构建空输入时的结果。

        功能：
            当没有任何文档需要索引时，返回标准格式结果，
            避免上层代码收到 None。

        参数：
            source: str | None
                数据来源描述。

        返回值：
            dict[str, Any]：
                空入库结果。
        """

        return {
            "pipeline": self.PIPELINE_NAME,
            "source": source,
            "loaded_documents": 0,
            "enhanced_documents": 0,
            "created_chunks": 0,
            "index_result": {
                "indexed_chunks": 0,
                "skipped_chunks": 0,
                "batch_count": 0,
                "index_ids": [],
            },
        }

    def _build_pipeline_result(
            self,
            source: str | None,
            documents: list[RagDocument],
            enhanced_documents: list[RagDocument],
            chunks: list[RagChunk],
            index_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        构建完整 Pipeline 执行结果。

        功能：
            汇总 Loader、Extractor、Chunker、Indexer 四个阶段的统计信息。

        参数：
            source: str | None
                数据来源描述。

            documents: list[RagDocument]
                Loader 阶段得到的原始文档。

            enhanced_documents: list[RagDocument]
                Extractor 阶段得到的 metadata 增强文档。

            chunks: list[RagChunk]
                Chunker 阶段得到的 chunks。

            index_result: dict[str, Any]
                Indexer 阶段返回的结果。

        返回值：
            dict[str, Any]：
                Pipeline 总执行结果。
        """

        return {
            "pipeline": self.PIPELINE_NAME,
            "source": source,
            "loaded_documents": len(
                documents,
            ),
            "enhanced_documents": len(
                enhanced_documents,
            ),
            "created_chunks": len(
                chunks,
            ),
            "index_result": index_result,
        }