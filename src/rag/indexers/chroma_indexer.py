"""
RAG Chroma Indexer。

Indexer（索引器 / 入库器）：
负责把已经准备好的 RagChunk 写入向量数据库。

当前模块职责：
1. 接收 list[RagChunk]
2. 将 RagChunk 转换成 LangChain Document
3. 清洗 metadata，保证 metadata 可以写入 Chroma
4. 构造稳定的 chunk id
5. 调用 Chroma.add_documents 写入向量库

当前模块不负责：
1. Markdown 加载
2. metadata 提取
3. chunk 切分
4. Retriever 检索
5. Rerank 重排序
6. LLM 回答生成
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from langchain_core.documents import Document

from src.rag.schemas import RagChunk


MetadataValue = str | int | float | bool


class RagChromaIndexer:
    """
    RAG Chroma 索引器。

    Chroma Indexer（Chroma 索引器）：
    用于把 RagChunk 写入 Chroma 向量数据库。

    设计特点：
    1. vector_store 通过构造函数注入
    2. 不直接从 container 中读取依赖
    3. 方便单元测试
    4. 方便后续接入 VectorStoreProvider
    """

    INDEXER_NAME = "rag_chroma_indexer_v1"

    def __init__(
            self,
            vector_store: Any,
            batch_size: int = 64,
            overwrite_existing: bool = True,
    ):
        """
        初始化 RagChromaIndexer。

        参数：
            vector_store: Any
                向量数据库对象。
                当前预期是 langchain_chroma.Chroma 实例。
                只要求它支持 add_documents 方法。

            batch_size: int
                每批写入多少个 chunk。
                批量写入可以避免一次性写入过多数据导致内存压力。

            overwrite_existing: bool
                是否覆盖已经存在的 chunk。
                True 时，会先根据 ids 删除旧数据，再重新写入。

        返回值：
            None：
                构造函数不返回值。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0"
            )

        if not hasattr(
                vector_store,
                "add_documents",
        ):
            raise AttributeError(
                "vector_store 必须提供 add_documents 方法"
            )

        self.vector_store = vector_store
        self.batch_size = batch_size
        self.overwrite_existing = overwrite_existing

    def index_chunks(
            self,
            chunks: list[RagChunk],
    ) -> dict[str, Any]:
        """
        批量索引 RagChunk。

        功能：
            1. 遍历 list[RagChunk]
            2. 跳过空内容 chunk
            3. 转换成 LangChain Document
            4. 使用 chunk.chunk_id 作为 Chroma id
            5. 分批写入 Chroma
            6. 返回索引结果统计信息

        参数：
            chunks: list[RagChunk]
                已经切好的 RAG chunk 列表。

        返回值：
            dict[str, Any]：
                索引结果统计信息。
                包含 total_chunks、indexed_chunks、skipped_chunks、batch_count、index_ids。
        """

        documents: list[Document] = []
        ids: list[str] = []
        skipped_chunks = 0

        for index, chunk in enumerate(
                chunks,
        ):
            content = self._get_chunk_content(
                chunk=chunk,
            )

            if not content.strip():
                skipped_chunks += 1
                continue

            chunk_id = self._build_chunk_id(
                chunk=chunk,
                chunk_index=index,
            )

            document = self._chunk_to_document(
                chunk=chunk,
                chroma_id=chunk_id,
            )

            documents.append(
                document,
            )

            ids.append(
                chunk_id,
            )

        batch_count = 0

        for batch_documents, batch_ids in self._iter_document_batches(
                documents=documents,
                ids=ids,
        ):
            if self.overwrite_existing:
                self._delete_existing_documents(
                    ids=batch_ids,
                )

            self.vector_store.add_documents(
                documents=batch_documents,
                ids=batch_ids,
            )

            batch_count += 1

        return {
            "indexer": self.INDEXER_NAME,
            "total_chunks": len(
                chunks,
            ),
            "indexed_chunks": len(
                documents,
            ),
            "skipped_chunks": skipped_chunks,
            "batch_count": batch_count,
            "index_ids": ids,
        }

    def index_chunk(
            self,
            chunk: RagChunk,
    ) -> dict[str, Any]:
        """
        索引单个 RagChunk。

        功能：
            这是 index_chunks 的单条封装。
            方便调试时只写入一个 chunk。

        参数：
            chunk: RagChunk
                单个 RAG chunk。

        返回值：
            dict[str, Any]：
                单个 chunk 的索引结果统计信息。
        """

        return self.index_chunks(
            chunks=[
                chunk,
            ],
        )

    def _chunk_to_document(
            self,
            chunk: RagChunk,
            chroma_id: str,
    ) -> Document:
        """
        将 RagChunk 转换成 LangChain Document。

        功能：
            LangChain Chroma 写入时通常接收 Document 对象。
            Document 主要包含：
            1. page_content：正文内容
            2. metadata：元数据

            注意：
                为了召回后能从 LangChain Document 还原 RagChunk，
                这里必须把 chunk_id、doc_id、chunk_index、source、title
                写入 metadata。

        参数：
            chunk: RagChunk
                项目内部的 RAG chunk 对象。

            chroma_id: str
                写入 Chroma 时使用的稳定 id。
                通常等于 chunk.chunk_id。

        返回值：
            Document：
                可被 Chroma 写入的 LangChain Document 对象。
        """

        content = self._get_chunk_content(
            chunk=chunk,
        )

        metadata = self._build_document_metadata(
            chunk=chunk,
            chroma_id=chroma_id,
        )

        return Document(
            page_content=content,
            metadata=metadata,
        )

    def _build_document_metadata(
            self,
            chunk: RagChunk,
            chroma_id: str,
    ) -> dict[str, MetadataValue]:
        """
        构建写入 Chroma 的 metadata。

        功能：
            先继承 chunk.metadata，
            再补充 RagChunk 的核心字段：
            1. chunk_id
            2. doc_id
            3. chunk_index
            4. source
            5. title
            6. chroma_id
            7. indexer

            这些字段用于召回时还原 RagChunk。

        参数：
            chunk: RagChunk
                项目内部的 RAG chunk 对象。

            chroma_id: str
                Chroma 写入 id。

        返回值：
            dict[str, MetadataValue]：
                可以安全写入 Chroma 的扁平 metadata。
        """

        raw_metadata = self._get_chunk_metadata(
            chunk=chunk,
        )

        enriched_metadata: dict[str, Any] = {
            **raw_metadata,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "source": chunk.source,
            "title": chunk.title,
            "chroma_id": chroma_id,
            "indexer": self.INDEXER_NAME,
        }

        return self._clean_metadata(
            metadata=enriched_metadata,
        )

    def _get_chunk_content(
            self,
            chunk: RagChunk,
    ) -> str:
        """
        获取 RagChunk 正文内容。

        功能：
            当前项目中 RagChunk 的正文预期字段是 content。

        参数：
            chunk: RagChunk
                RAG chunk 对象。

        返回值：
            str：
                chunk 正文内容。
        """

        return str(
            chunk.content
            or ""
        )

    def _get_chunk_metadata(
            self,
            chunk: RagChunk,
    ) -> dict[str, Any]:
        """
        获取 RagChunk metadata。

        功能：
            从 chunk.metadata 中读取元数据。
            如果 metadata 为空，则返回空 dict。

        参数：
            chunk: RagChunk
                RAG chunk 对象。

        返回值：
            dict[str, Any]：
                chunk metadata。
        """

        if chunk.metadata is None:
            return {}

        return dict(
            chunk.metadata,
        )

    def _build_chunk_id(
            self,
            chunk: RagChunk,
            chunk_index: int,
    ) -> str:
        """
        构造稳定的 chunk id。

        功能：
            Chroma 写入时建议显式传入 ids。
            这样后续重复索引同一个 chunk 时，可以覆盖旧数据，
            避免重复写入。

            当前项目 RagChunk 已经有必填 chunk_id，
            所以优先使用 chunk.chunk_id。

            兜底逻辑只用于防御异常对象。

        参数：
            chunk: RagChunk
                RAG chunk 对象。

            chunk_index: int
                当前 chunk 在列表中的位置。

        返回值：
            str：
                稳定的 chunk id。
        """

        if chunk.chunk_id:
            return str(
                chunk.chunk_id,
            )

        if chunk.doc_id:
            return f"{chunk.doc_id}::chunk::{chunk.chunk_index}"

        content = self._get_chunk_content(
            chunk=chunk,
        )

        content_hash = hashlib.sha256(
            content.encode(
                "utf-8",
            )
        ).hexdigest()[
            :16
        ]

        return f"rag_chunk::{content_hash}::{chunk_index}"

    def _clean_metadata(
            self,
            metadata: dict[str, Any],
    ) -> dict[str, MetadataValue]:
        """
        清洗 metadata。

        功能：
            Chroma metadata 适合保存扁平基础类型：
            1. str
            2. int
            3. float
            4. bool

            因此这里会过滤掉：
            1. None
            2. list
            3. dict
            4. tuple
            5. set
            6. 其他复杂对象

        参数：
            metadata: dict[str, Any]
                原始 metadata。

        返回值：
            dict[str, MetadataValue]：
                可以安全写入 Chroma 的 metadata。
        """

        clean_metadata: dict[str, MetadataValue] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if not isinstance(
                    key,
                    str,
            ):
                key = str(
                    key,
                )

            if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
            ):
                clean_metadata[
                    key
                ] = value

        return clean_metadata

    def _delete_existing_documents(
            self,
            ids: list[str],
    ) -> None:
        """
        删除 Chroma 中已经存在的文档。

        功能：
            当 overwrite_existing=True 时，
            在重新写入同一批 ids 前先删除旧数据。

            这样可以避免重复入库。

        参数：
            ids: list[str]
                要删除的 Chroma document ids。

        返回值：
            None：
                删除操作无返回值。
        """

        if not ids:
            return

        if not hasattr(
                self.vector_store,
                "delete",
        ):
            return

        self.vector_store.delete(
            ids=ids,
        )

    def _iter_document_batches(
            self,
            documents: list[Document],
            ids: list[str],
    ) -> Iterable[tuple[list[Document], list[str]]]:
        """
        将 documents 和 ids 切分成多个批次。

        功能：
            根据 batch_size 对待写入数据进行分批。
            每一批返回：
            1. batch_documents
            2. batch_ids

        参数：
            documents: list[Document]
                待写入的 LangChain Document 列表。

            ids: list[str]
                与 documents 一一对应的 Chroma ids。

        返回值：
            Iterable[tuple[list[Document], list[str]]]：
                分批后的 documents 和 ids。
        """

        for start_index in range(
                0,
                len(
                    documents,
                ),
                self.batch_size,
        ):
            end_index = start_index + self.batch_size

            yield (
                documents[
                    start_index:end_index
                ],
                ids[
                    start_index:end_index
                ],
            )