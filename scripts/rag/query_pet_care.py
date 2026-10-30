"""通过项目检索器查询已存在的宠物养护集合，只返回候选，不重排或生成答案。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.schemas import RagQuery
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


async def query_pet_care(question: str, *, top_k: int = 5) -> dict:
    """用本轮问题查询已入库资料，保存标准上下文和带页码的候选报告。

    参数：question为非空检索问题；top_k为召回数量，取值1到50，默认5。
    返回值：含本轮问题、集合、向量配置、候选、耗时和报告路径的字典。
        复用项目检索器，不新增或删除向量记录，不调用聊天模型或重排模型。
        集合不存在、为空或向量配置不符时明确报错，不创建空集合或切换到旧库。
        当前查询整个养护集合，不推断物种过滤条件，也不判断资料是否足以回答。
    """
    question = question.strip()
    if not question:
        raise ValueError("检索问题不能为空。")
    query = RagQuery(question=question, top_k=top_k, intent="pet_care")
    collection_name = "pet_care_v1"
    database_dir = settings.path.CHROMA_DB_DIR
    if not (database_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在，请先完成实际入库。")
    embedding_config = {
        "provider": settings.embedding.provider,
        "model": settings.embedding.model_name,
        "normalize": settings.embedding.normalize_embeddings,
    }
    expected_signature = hashlib.sha256(json.dumps(embedding_config, sort_keys=True).encode()).hexdigest()
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    container.register("pet_vectorstore", VectorStoreProvider(
        embedding, collection_name=collection_name, enable_memory=False,
        create_collection_if_not_exists=False,
    ))
    try:
        started_at = perf_counter()
        await container.startup()
        startup_seconds = perf_counter() - started_at
        database = container.get("pet_vectorstore").db
        records = await asyncio.to_thread(database.get, include=["metadatas"])
        if not records["ids"]:
            raise ValueError("养护集合存在但没有资料，请检查入库报告。")
        if any((metadata or {}).get("embedding_signature") != expected_signature
               for metadata in records["metadatas"]):
            raise ValueError("查询向量配置与目标集合记录不一致或未登记，停止检索。")
        started_at = perf_counter()
        context = await MetadataFilterRetriever(database).async_retrieve(query)
        retrieval_seconds = perf_counter() - started_at
        run_id = uuid4().hex
        root = Path(__file__).resolve().parents[2]
        output_dir = root / "output/rag_queries" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "run_id": run_id,
            "question": question,
            "collection_name": collection_name,
            "embedding_config": embedding_config,
            "requested_top_k": top_k,
            "collection_record_count": len(records["ids"]),
            "startup_seconds": round(startup_seconds, 4),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "candidates": [{
                "rank": rank,
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "collection_name": collection_name,
                "title": item.chunk.title,
                "page_number": item.chunk.metadata.get("page_number"),
                "source": item.chunk.source,
                "source_url": item.chunk.metadata.get("source_url"),
                "species": item.chunk.metadata.get("species"),
                "retrieval_score": item.retrieval_score,
                "content": item.chunk.content,
            } for rank, item in enumerate(context.chunks, start=1)],
            "rag_context": context.model_dump(mode="json"),
            "report_path": str(output_dir / "retrieval_report.json"),
            "review_status": "needs_relevance_review",
            "notice": "当前记录原始距离及接口返回顺序，非相关概率；没有重排或生成答案。",
        }
        Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只查询已存在的pet_care_v1养护集合。")
    parser.add_argument("--question", help="本轮检索问题，不填时由控制台输入。")
    parser.add_argument("--top-k", type=int, default=5, help="召回块数，默认5，范围1到50。")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("请输入本轮检索问题：")
    report = asyncio.run(query_pet_care(question, top_k=args.top_k))
    print(f"\n运行编号：{report['run_id']}")
    print(f"实际问题：{report['question']}")
    print(f"集合：{report['collection_name']}；已有记录数：{report['collection_record_count']}")
    for item in report["candidates"]:
        print(f"\n第{item['rank']}名 | 物理第{item['page_number']}页 | 原始检索值={item['retrieval_score']:.6f}")
        print(f"块编号：{item['chunk_id']}\n来源：{item['source']}")
        print(item["content"])
    print(f"\n初始化耗时：{report['startup_seconds']}秒；检索耗时：{report['retrieval_seconds']}秒")
    print(f"完整报告：{report['report_path']}")
    print("这是候选召回结果；排名第一不代表足以回答，尚未重排、选上下文或生成答案。")
