"""真实向量与 BM25 双路召回演示；使用虚构教学资料，不写入正式知识库。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from src.rag.schemas import RagChunk
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.settings import settings
from tests.learning.test_plus_rag_hybrid_retrieval import fuse_query_retrieval_results


# 仅用于检查检索行为，不是实际产品信息、养犬指南或完整业务评估集。
TEACHING_PASSAGES = (
    ("barking", "犬只频繁吠叫", "犬只频繁吠叫可按触发场景记录，例如门铃、陌生人或独处，供后续行为分析。"),
    ("grooming", "被毛梳理", "犬只被毛护理记录包含梳理工具、清理浮毛和日常清洁安排。"),
    ("walking", "外出用品", "外出用品清单包含牵引绳、拾便袋和饮水容器。"),
    ("px730", "PX-730 使用说明", "虚构产品 PX-730 的说明章节包含设备配对、指示灯与恢复设置。"),
    ("px731", "PX-731 使用说明", "虚构产品 PX-731 的说明章节包含安装、充电与维护。"),
    ("ab730", "AB-730 使用说明", "虚构产品 AB-730 的说明章节包含首次启动与配件检查。"),
    ("profile", "宠物档案管理", "宠物档案管理说明介绍如何查看和修改宠物名称、品种等档案字段。"),
    ("training", "训练记录", "训练记录表保存每次练习的目标、持续时间和完成情况。"),
)


def tokenize_retrieval_text(text: str) -> list[str]:
    """
    功能：统一中文分词与编号处理，为本地 BM25 提供词项列表。
    参数含义：text 是资料正文或查询文字，二者使用完全相同的处理方式。
    返回值含义：返回小写词项列表，保留 PX-730 这类连字符编号，不删除否定词。
        这是教学分词策略，不包含完整领域词典、别名扩展或生产分词优化。
    """
    import jieba

    parts = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(word for word in jieba.lcut(part) if word.strip())
        else:
            tokens.append(part)
    return tokens


def build_teaching_retrieval_indexes(embedding: Any) -> dict[str, Any]:
    """
    功能：把同一组教学片段分别建立真实内存向量索引和 BM25 索引。
    参数含义：embedding 是从项目容器中的向量提供者取得的客户端。
    返回值含义：返回片段、编号映射、语料词项、向量库和 BM25 实例。
        两路共享同一批片段编号，便于融合；不写入正式 Chroma 数据库。
    """
    from rank_bm25 import BM25Okapi

    chunks = [
        RagChunk(
            chunk_id=chunk_id, doc_id=f"teaching-{chunk_id}", chunk_index=0,
            title=title, content=f"{title}\n{text}", source="synthetic_teaching_corpus",
            metadata={"synthetic": True},
        )
        for chunk_id, title, text in TEACHING_PASSAGES
    ]
    tokenized_corpus = [tokenize_retrieval_text(chunk.content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    vector_store = InMemoryVectorStore(embedding)
    vector_store.add_documents(
        documents=[
            Document(page_content=chunk.content, metadata={"chunk_id": chunk.chunk_id})
            for chunk in chunks
        ],
        ids=[chunk.chunk_id for chunk in chunks],
    )
    return {
        "chunks": chunks,
        "chunks_by_id": {chunk.chunk_id: chunk for chunk in chunks},
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
        "vector_store": vector_store,
    }


def retrieve_and_fuse_teaching_query(
    *, indexes: dict[str, Any], query_id: str, question: str, candidate_count: int = 3,
) -> dict[str, Any]:
    """
    功能：对同一问题执行真实向量检索和 BM25，再复用已有排名融合函数。
    参数含义：indexes 是两路索引；query_id 是查询编号；question 是查询正文；
        candidate_count 是每路最多召回的数量，不是最终回答上下文数量。
    返回值含义：返回分词、两路原始分数/排名/耗时及融合记录。
        BM25 只保留至少有一个查询词命中的片段，不把零匹配资料强行补足。
        向量分数是余弦相似度，BM25 分数不能与其直接相加；融合不是模型重排。
    """
    query_tokens = tokenize_retrieval_text(question)
    query_terms = set(query_tokens)
    dense_started_at = time.perf_counter()
    dense_hits = indexes["vector_store"].similarity_search_with_score(
        question, k=candidate_count,
    )
    dense_ms = (time.perf_counter() - dense_started_at) * 1000
    sparse_started_at = time.perf_counter()
    scores = indexes["bm25"].get_scores(query_tokens)
    matching_indexes = [
        index for index, tokens in enumerate(indexes["tokenized_corpus"])
        if query_terms.intersection(tokens)
    ]
    sparse_indexes = sorted(
        matching_indexes, key=lambda index: (-float(scores[index]), indexes["chunks"][index].chunk_id),
    )[:candidate_count]
    sparse_ms = (time.perf_counter() - sparse_started_at) * 1000

    dense_chunks = [indexes["chunks_by_id"][doc.metadata["chunk_id"]] for doc, _ in dense_hits]
    sparse_chunks = [indexes["chunks"][index] for index in sparse_indexes]
    fusion_started_at = time.perf_counter()
    fused = fuse_query_retrieval_results(
        query_id=query_id,
        results_by_channel={"dense": dense_chunks, "sparse": sparse_chunks},
        channel_weights={"dense": 1.0, "sparse": 1.0},
        rank_constant=60.0,
    )
    fusion_ms = (time.perf_counter() - fusion_started_at) * 1000
    return {
        "query_id": query_id, "question": question, "query_tokens": query_tokens,
        "dense_latency_ms": round(dense_ms, 2),
        "sparse_latency_ms": round(sparse_ms, 2),
        "fusion_latency_ms": round(fusion_ms, 2),
        "dense_results": [
            {"rank": rank, "chunk_id": chunk.chunk_id, "text": chunk.content,
             "cosine_similarity": float(score)}
            for rank, (chunk, (_, score)) in enumerate(zip(dense_chunks, dense_hits), start=1)
        ],
        "sparse_results": [
            {"rank": rank, "chunk_id": indexes["chunks"][index].chunk_id,
             "text": indexes["chunks"][index].content, "bm25_score": float(scores[index]),
             "matched_terms": sorted(query_terms.intersection(indexes["tokenized_corpus"][index]))}
            for rank, index in enumerate(sparse_indexes, start=1)
        ],
        "fused_results": [
            {**record, "chunk": record["chunk"].model_dump(mode="json")}
            for record in fused
        ],
    }


async def run_real_hybrid_retrieval_demo(candidate_count: int = 3) -> dict[str, Any]:
    """
    功能：启动项目向量服务，建立独立索引，并观察语义、编号、无关问题的检索结果。
    参数含义：candidate_count 是每路召回数量，限制在教学语料数量以内。
    返回值含义：返回模型配置、启动和建索引耗时、教学语料及三条查询报告。
        不调用聊天模型、不生成答案、不声称准确率提升；模型未缓存时运行可能下载。
        复用现有提供者生命周期，其启动同时会初始化项目的记忆向量客户端。
    """
    missing = [name for name in ("jieba", "rank_bm25") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "缺少演示依赖：" + ", ".join(missing)
            + "。请在项目根目录手动执行：./.venv/Scripts/python.exe -m pip install rank-bm25 jieba"
        )
    if not 1 <= candidate_count <= len(TEACHING_PASSAGES):
        raise ValueError("每路召回数量必须在1到教学语料数量之间")
    container = RuntimeContainer()
    container.register("embedding", EmbeddingProvider())
    try:
        started_at = time.perf_counter()
        await container.startup()
        startup_ms = (time.perf_counter() - started_at) * 1000
        index_started_at = time.perf_counter()
        indexes = await asyncio.to_thread(
            build_teaching_retrieval_indexes, container.get("embedding").embedding,
        )
        index_ms = (time.perf_counter() - index_started_at) * 1000
        reports = []
        for query_id, question in (
            ("semantic_expression", "狗狗老是汪汪叫怎么办"),
            ("exact_identifier", "PX-730 使用说明"),
            ("outside_corpus", "如何申请企业增值税发票"),
        ):
            print(f"正在检索：{query_id} | {question}")
            reports.append(await asyncio.to_thread(
                retrieve_and_fuse_teaching_query, indexes=indexes,
                query_id=query_id, question=question, candidate_count=candidate_count,
            ))
        return {
            "mode": "real_dense_bm25_fusion",
            "embedding_provider": settings.embedding.provider,
            "embedding_model": settings.embedding.model_name,
            "candidate_count_per_channel": candidate_count,
            "provider_startup_ms": round(startup_ms, 2),
            "index_build_ms": round(index_ms, 2),
            "notice": "教学检索，不包含真实模型重排、自动适用性判断或答案生成。",
            "corpus": [chunk.model_dump(mode="json") for chunk in indexes["chunks"]],
            "results": reports,
        }
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实向量与 BM25 融合教学演示")
    parser.add_argument("--candidate-count", type=int, default=3, help="每路召回数量")
    arguments = parser.parse_args()
    report = asyncio.run(run_real_hybrid_retrieval_demo(arguments.candidate_count))
    print(json.dumps(report, ensure_ascii=False, indent=2))
