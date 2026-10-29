"""维护期间更新单文档索引；默认仅预览，不支持并发写入或无缝版本切换。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.rag.preview_document_update import (
    prepare_document_chunks,
    preview_document_update,
)
from src.rag.indexers import RagChromaIndexer


def read_document_rows(vector_store: Any, doc_id: str) -> dict[str, dict[str, Any]]:
    """读取并核对指定文档的索引记录。

    参数：vector_store 为向量库；doc_id 为目标文档编号。
    返回：以存储编号为键、正文和元数据为值的字典；结果不完整则抛出异常。
    """
    stored = vector_store.get(
        where={"doc_id": doc_id}, include=["documents", "metadatas"],
    )
    ids = stored["ids"]
    contents = stored["documents"]
    metadata = stored["metadatas"]
    if contents is None or metadata is None or not (
        len(ids) == len(contents) == len(metadata) == len(set(ids))
    ):
        raise ValueError("索引读取结果不完整，停止更新")
    rows = {}
    for chunk_id, content, fields in zip(ids, contents, metadata):
        if not fields or fields.get("doc_id") != doc_id:
            raise ValueError("读取到非目标文档，停止更新")
        rows[chunk_id] = {"content": content, "metadata": fields}
    return rows


def update_document_index(
    input_path: str | Path,
    vector_store: Any,
    *,
    confirmed_doc_id: str,
    maintenance_confirmed: bool = False,
) -> dict[str, Any]:
    """维护期间先写新块、回查成功后再清理指定文档的旧块。

    参数：input_path 为源文件；vector_store 为项目 Chroma 实例；
        confirmed_doc_id 为人工确认的文档编号；maintenance_confirmed 表示已暂停查询与写入。
    返回：成功时返回更新数量报告；失败抛出异常，可能残留部分新块，须保持维护状态。
        此标志不提供分布式锁或回滚，调用方必须实际停止并发访问。
    """
    if not maintenance_confirmed:
        raise ValueError("请先暂停相关查询和其他索引写入，再确认维护状态")
    document, chunks = prepare_document_chunks(input_path)
    if document.doc_id != confirmed_doc_id:
        raise ValueError("文件对应的文档编号与人工确认编号不一致")

    old_rows = read_document_rows(vector_store, document.doc_id)
    if not old_rows:
        raise ValueError("没有找到旧文档，请核对源路径和向量库，不执行更新")
    new_ids = {chunk.chunk_id for chunk in chunks}
    obsolete_ids = sorted(set(old_rows) - new_ids)

    # 当前 Chroma 的 add_documents 最终使用 upsert；
    # 关闭索引器的预删除逻辑，避免写入之前就删除旧记录。
    RagChromaIndexer(
        vector_store=vector_store, overwrite_existing=False,
    ).index_chunks(chunks)

    written_rows = read_document_rows(vector_store, document.doc_id)
    for chunk in chunks:
        stored = written_rows.get(chunk.chunk_id)
        if (
            stored is None
            or stored["content"] != chunk.content
            or stored["metadata"].get("heading_path") != chunk.metadata["heading_path"]
            or stored["metadata"].get("chunk_id") != chunk.chunk_id
        ):
            raise RuntimeError("新块回查不一致，未开始清理旧块；请保持维护状态并排查")

    if set(written_rows) != set(old_rows) | new_ids:
        raise RuntimeError("发现非预期编号变化，未开始清理；请检查是否存在其他写入者")
    if obsolete_ids:
        vector_store.delete(ids=obsolete_ids)

    final_rows = read_document_rows(vector_store, document.doc_id)
    if set(final_rows) != new_ids:
        raise RuntimeError("清理后编号不符合预期，请保持维护状态并重试或排查")
    return {
        "mode": "maintenance_document_update",
        "doc_id": document.doc_id,
        "source": document.source,
        "old_chunk_count": len(old_rows),
        "written_chunk_count": len(chunks),
        "removed_obsolete_count": len(obsolete_ids),
        "final_chunk_count": len(final_rows),
        "verified": True,
        "embedding_reused": False,
    }


def main() -> None:
    """提供默认预览、显式确认才更新的命令行入口。

    参数：无；命令行接受 --file、--apply、--confirm-doc-id、--maintenance-confirmed。
    返回：无；打印 JSON，失败时抛出异常，不声称已完成或自动回滚。
    """
    parser = argparse.ArgumentParser(description="维护期间更新单篇文档；默认只预览")
    parser.add_argument("--file", required=True)
    parser.add_argument("--apply", action="store_true", help="实际写入和清理旧块")
    parser.add_argument("--confirm-doc-id", help="预览中核对过的文档编号")
    parser.add_argument("--maintenance-confirmed", action="store_true",
                        help="确认相关查询及其他索引写入均已停止")
    args = parser.parse_args()
    if args.apply and not (args.confirm_doc_id and args.maintenance_confirmed):
        parser.error("实际更新必须提供文档编号并确认维护状态")

    from src.runtime.container.init import container

    store = container.get("vectorstore").db
    if args.apply:
        report = update_document_index(
            args.file, store,
            confirmed_doc_id=args.confirm_doc_id,
            maintenance_confirmed=args.maintenance_confirmed,
        )
    else:
        report = preview_document_update(args.file, store)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
