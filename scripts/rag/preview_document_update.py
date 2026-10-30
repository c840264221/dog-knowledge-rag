"""预览单篇 Markdown 的索引差异，不写入、删除块或调用向量计算。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.rag.chunking.markdown_chunker import MarkdownChunker
from src.rag.extractors import DogBreedMetadataExtractor
from src.rag.loaders.markdown_loader import MarkdownDocumentLoader
from src.rag.schemas import RagChunk, RagDocument


def prepare_document_chunks(input_path: str | Path) -> tuple[RagDocument, list[RagChunk]]:
    """准备单文档及正文块，拒绝目录、空文件和纯标题文件。

    参数：input_path 为单个 Markdown 文件路径。
    返回：增强元数据后的文档及新块列表；不访问向量库。
    """
    path = Path(input_path).resolve()
    if not path.is_file() or path.suffix.lower() != ".md":
        raise ValueError("预览只接受一个已存在的 Markdown 文件，不接受目录")

    documents = MarkdownDocumentLoader(input_path=path).load()
    if len(documents) != 1:
        raise ValueError("没有读到有效文档；空文件不能作为删除旧索引的依据")
    document = DogBreedMetadataExtractor().extract(documents[0])
    chunks = MarkdownChunker().chunk(document)
    if not chunks:
        raise ValueError("没有生成正文块；请检查文档，不能据此清空旧索引")
    return document, chunks


def preview_document_update(input_path: str | Path, vector_store: Any) -> dict[str, Any]:
    """读取单篇文档及其现有索引，生成更新预览，不执行更新。

    参数：input_path 为单个 Markdown 文件；vector_store 为提供 get 的向量库对象。
    返回：文档身份、新旧数量、编号差异及新章节摘要组成的字典。
    """
    document, chunks = prepare_document_chunks(input_path)

    stored = vector_store.get(
        where={"doc_id": document.doc_id},
        include=["metadatas"],
    )
    old_ids = list(stored["ids"])
    old_metadata = stored["metadatas"]
    if old_metadata is None or len(old_ids) != len(old_metadata):
        raise ValueError("旧索引的编号与元数据不完整，停止生成更新预览")
    if any(
        not metadata or metadata.get("doc_id") != document.doc_id
        for metadata in old_metadata
    ):
        raise ValueError("查询结果包含非目标文档的块，停止生成更新预览")

    old_id_set = set(old_ids)
    new_id_set = {chunk.chunk_id for chunk in chunks}
    same_ids = sorted(old_id_set & new_id_set)
    added_ids = sorted(new_id_set - old_id_set)
    obsolete_ids = sorted(old_id_set - new_id_set)

    return {
        "mode": "preview_only",
        "source": document.source,
        "doc_id": document.doc_id,
        "old_chunk_count": len(old_ids),
        "new_chunk_count": len(chunks),
        "same_id_count": len(same_ids),
        "new_id_count": len(added_ids),
        "obsolete_id_count": len(obsolete_ids),
        "same_ids": same_ids,
        "new_ids": added_ids,
        "obsolete_ids": obsolete_ids,
        "new_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.metadata["section_title"],
                "heading_path": chunk.metadata["heading_path"],
                "character_count": len(chunk.content),
                "content_preview": chunk.content[:100],
            }
            for chunk in chunks
        ],
        "warnings": [
            "相同编号仍可能需要更新元数据，本预览不表示可完全跳过更新。",
            "新编号可能只是块顺序变化，不代表正文一定改变。",
            "本报告是当前快照，不能直接作为稍后删除的授权或并发校验凭据。",
        ] + (
            ["未找到旧块：请核对库配置和源路径，不能据此认定这是一篇全新文档。"]
            if not old_ids else []
        ),
    }


def main() -> None:
    """解析参数，经项目服务提供者读取向量库并打印单文档预览。

    参数：无；命令行 --file 指定 Markdown 路径。
    返回：无；打印 JSON 报告，不执行索引写入或删除。
    """
    parser = argparse.ArgumentParser(description="预览单篇 Markdown 的向量索引更新")
    parser.add_argument("--file", required=True, help="待更新的单个 Markdown 文件")
    args = parser.parse_args()

    # 延后加载容器，纯函数测试不初始化真实服务。
    from src.runtime.container.init import container

    vector_store = container.get("vectorstore").db
    report = preview_document_update(args.file, vector_store)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
