"""PDF 入库学习第六步：复用项目索引器，写入独立 Chroma 并检索、回查结构化内容。"""

import asyncio
import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma

from src.rag.indexers.chroma_indexer import RagChromaIndexer
from src.rag.schemas import RagChunk
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


class PdfDemoEmbeddingProvider(EmbeddingProvider):
    """复用项目默认向量模型，仅缩小本演示的启动范围。"""

    async def startup(self) -> None:
        """只初始化文档向量模型；无参数，无返回值；不初始化记忆模型。"""
        _ = self.embedding


class PdfDemoVectorStoreProvider(VectorStoreProvider):
    """复用项目提供者接口，为演示提供独立的持久化向量库。"""

    def __init__(
        self, embedding_provider: EmbeddingProvider, *, directory: Path, collection_name: str
    ):
        """保存演示库配置；参数为向量提供者、独立目录和集合名称；无返回值。"""
        super().__init__(embedding_provider)
        self.directory = directory
        self.collection_name = collection_name

    @property
    def db(self):
        """获取或创建演示Chroma集合；无参数；返回使用余弦距离的向量库对象。"""
        if self._db is None:
            self._db = Chroma(
                persist_directory=str(self.directory),
                collection_name=self.collection_name,
                embedding_function=self.embedding_provider.embedding,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._db

    async def startup(self) -> None:
        """只初始化独立演示库；无参数，无返回值；不打开原知识库和记忆库。"""
        _ = self.db


async def run_pdf_vector_demo(question: str) -> dict:
    """读取标准块，真实向量化入库，检索前三条并通过块编号回查原料。

    参数：question为手动输入的检索问题，不经过改写或模型生成。
    返回值：含向量模型配置、独立库位置、写入统计、向量维度和命中记录的报告，
        同时保存为vector_retrieval_report.json。距离为余弦距离，越小越接近，
        不代表经过校准的正确率；固定取前三条也不保证每条都相关。
        此过程只向量化文字，图片本身尚未编码为向量，也不用于生成答案。
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    bundle_path = output_dir / "rag_chunk_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    chunks = [RagChunk.model_validate(item) for item in bundle["rag_chunks"]]
    if not chunks:
        raise ValueError("没有可入库的块，请先检查切块及适配结果。")
    for chunk in chunks:
        if chunk.chunk_id not in bundle["structured_content_by_chunk_id"]:
            raise ValueError(f"缺少结构化关联记录：{chunk.chunk_id}")
        chunk.metadata["structured_bundle_path"] = str(bundle_path)

    # 内容或模型配置变化时使用另一集合，避免旧演示块混入本轮结果。
    identity = json.dumps({
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "provider": settings.embedding.provider,
        "model": settings.embedding.model_name,
        "normalize": settings.embedding.normalize_embeddings,
    }, ensure_ascii=False, sort_keys=True)
    collection_name = "pdf_demo_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    database_dir = project_root / "output/learning/pdf_chroma"

    container = RuntimeContainer()
    embedding_provider = PdfDemoEmbeddingProvider()
    container.register("embedding", embedding_provider)
    container.register("vector_store", PdfDemoVectorStoreProvider(
        embedding_provider, directory=database_dir, collection_name=collection_name,
    ))
    print(f"向量模型：{settings.embedding.provider} / {settings.embedding.model_name}", flush=True)
    print("即将加载向量模型、写入独立演示库并检索；模型未缓存时可能下载。", flush=True)
    try:
        await container.startup()
        database = container.get("vector_store").db
        indexer = RagChromaIndexer(vector_store=database, overwrite_existing=False)
        index_report = indexer.index_chunks(chunks)

        # 直接读取已存向量观察维度，不额外调用模型重复计算。
        stored = database.get(ids=index_report["index_ids"], include=["embeddings"])
        vectors = stored["embeddings"]
        vector_dimension = len(vectors[0]) if vectors is not None and len(vectors) else None
        hits = database.similarity_search_with_score(question, k=min(3, len(chunks)))
        retrieval_results = []
        for rank, (document, distance) in enumerate(hits, start=1):
            chunk_id = document.metadata["chunk_id"]
            linked_record = bundle["structured_content_by_chunk_id"][chunk_id]
            retrieval_results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "cosine_distance": float(distance),
                "text": document.page_content,
                "metadata": document.metadata,
                "structured_content": linked_record["structured_content"],
            })

        report = {
            "question": question,
            "embedding_model": settings.embedding.model_name,
            "database_directory": str(database_dir),
            "collection_name": collection_name,
            "index_report": index_report,
            "verified_stored_count": len(stored["ids"]),
            "vector_dimension": vector_dimension,
            "retrieval_results": retrieval_results,
        }
        (output_dir / "vector_retrieval_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    question = input(
        "检索问题（默认：狗狗耳朵向后、尾巴降低并抬起一只前爪，海报如何描述？）："
    ).strip() or "狗狗耳朵向后、尾巴降低并抬起一只前爪，海报如何描述？"
    print(json.dumps(asyncio.run(run_pdf_vector_demo(question)), ensure_ascii=False, indent=2))
