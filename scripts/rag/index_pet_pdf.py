"""项目PDF入库入口：默认仅预览，--apply才向明确指定的养护集合写入。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.rag.chunking.pdf_text_chunker import PdfTextChunker
from src.rag.indexers.chroma_indexer import RagChromaIndexer
from src.rag.loaders.pdf_loader import PdfDocumentLoader
from src.rag.pipelines.index_pipeline import RagIndexPipeline
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


async def run_pdf_index(plan_path: Path, *, apply: bool = False) -> dict:
    """读取显式计划，通过项目加载器、切块器与索引管线处理PDF。

    参数：plan_path为入库计划JSON；apply为实际写入开关，默认False只预览。
    返回值：包含文档/块数量、目标集合、输出位置及真实入库结果的报告。
        只允许写入本次pet_care_v1集合；不替换现有文档不同版本，不删除旧块。
        每次保存原始文档与待入库块；不调用聊天模型，不访问用户记忆库。
    """
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    collection_name = plan["collection_name"]
    if collection_name != "pet_care_v1":
        raise ValueError("当前入口仅允许pet_care_v1养护集合，不能写入旧犬种或记忆集合。")
    signature_payload = {
        "provider": settings.embedding.provider,
        "model": settings.embedding.model_name,
        "normalize": settings.embedding.normalize_embeddings,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    documents = []
    for item in plan["documents"]:
        source_path = root / item["path"]
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError(f"文件内容与计划不一致：{item['doc_id']}")
        loader = PdfDocumentLoader(
            source_path, doc_id=item["doc_id"], title=item["title"],
            included_pages=item["included_pages"],
            metadata={**item["metadata"], "embedding_signature": signature},
        )
        documents.extend(loader.load())
    if not documents or len({doc.doc_id for doc in documents}) != len(documents):
        raise ValueError("入库计划为空或文档编号重复。")
    chunker = PdfTextChunker(plan["chunk_size"], plan["chunk_overlap"])
    chunks = chunker.chunk_many(documents)
    if any(not any(chunk.doc_id == doc.doc_id for chunk in chunks) for doc in documents):
        raise ValueError("至少一份文档的选用页没有可入库文字，请检查提取结果。")
    output_dir = root / "output/rag_ingestion" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in (
        ("plan_snapshot.json", plan),
        ("documents.json", [doc.model_dump(mode="json") for doc in documents]),
        ("chunks.json", [chunk.model_dump(mode="json") for chunk in chunks]),
    ):
        (output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "mode": "apply" if apply else "preview_only",
        "collection_name": collection_name,
        "persist_directory": str(settings.path.CHROMA_DB_DIR),
        "embedding_config": signature_payload,
        "loaded_documents": len(documents), "prepared_chunks": len(chunks),
        "selected_pages_by_document": {doc.doc_id: doc.metadata["selected_page_numbers"] for doc in documents},
        "index_results": [], "report_path": str(output_dir / "index_report.json"),
    }
    report_path = Path(report["report_path"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not apply:
        return report

    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    container.register("pet_vectorstore", VectorStoreProvider(
        embedding, collection_name=collection_name, enable_memory=False,
    ))
    try:
        await container.startup()
        database = container.get("pet_vectorstore").db
        # 先检查整个目标集合的模型标记，禁止混用另一套向量模型。
        existing_metadata = database.get(include=["metadatas"])["metadatas"] or []
        if any((metadata or {}).get("embedding_signature") != signature for metadata in existing_metadata):
            raise ValueError("目标集合含不同或未知向量配置，请使用独立迁移流程，不直接混写。")
        pending = []
        for document in documents:
            existing = database.get(where={"doc_id": document.doc_id}, include=["metadatas"])
            expected = {chunk.chunk_id for chunk in chunks if chunk.doc_id == document.doc_id}
            if existing["ids"]:
                if set(existing["ids"]) != expected or any(
                    (metadata or {}).get("source_sha256") != document.metadata["source_sha256"]
                    for metadata in existing["metadatas"]
                ):
                    raise ValueError("文档已有不同版本或切块，不自动追加或删除，请使用文档更新流程。")
                report["index_results"].append({"doc_id": document.doc_id, "status": "already_indexed"})
            else:
                pending.append(document)
        if pending:
            pipeline = RagIndexPipeline(
                loader=None, metadata_extractor=None, chunker=chunker,
                indexer=RagChromaIndexer(database, overwrite_existing=False),
            )
            report["index_results"].append(pipeline.index_documents(pending))
        for document in documents:
            actual_ids = set(database.get(where={"doc_id": document.doc_id}, include=[])["ids"])
            expected_ids = {chunk.chunk_id for chunk in chunks if chunk.doc_id == document.doc_id}
            if actual_ids != expected_ids:
                raise RuntimeError("写入后的块编号核对未通过，请查看本轮预览文件。")
        report["verified"] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    except Exception as error:
        report["verified"] = False
        report["error_type"] = type(error).__name__
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        await container.shutdown()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="项目PDF入库：默认预览，--apply实际写入。")
    parser.add_argument("--plan", type=Path, default=root / "data/learning/pet_corpus_v1/ingestion_plan.json")
    parser.add_argument("--apply", action="store_true", help="实际向量化写入pet_care_v1集合。")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_pdf_index(args.plan, apply=args.apply)), ensure_ascii=False, indent=2))
