"""统一查询入口第一步：手动输入新问题，在已有PDF演示库检索并重排。"""

import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_chroma import Chroma

from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.settings import settings
from tests.learning.plus_pdf_retrieval_text_comparison import build_without_notes_chunks
from tests.learning.plus_pdf_rerank_comparison import rerank_saved_candidates
from tests.learning.plus_pdf_vector_demo import PdfDemoEmbeddingProvider, PdfDemoVectorStoreProvider


class ExistingPdfVectorStoreProvider(PdfDemoVectorStoreProvider):
    """沿用演示库配置，仅打开已存在集合，不在名称错误时新建空集合。"""

    @property
    def db(self):
        """获取已有PDF向量集合；无参数；返回Chroma实例，集合不存在时抛出异常。"""
        if self._db is None:
            self._db = Chroma(
                persist_directory=str(self.directory),
                collection_name=self.collection_name,
                embedding_function=self.embedding_provider.embedding,
                create_collection_if_not_exists=False,
            )
        return self._db


async def run_pdf_query_demo(question: str, candidate_count: int = 5) -> dict:
    """按本次问题重新检索已有集合，重排本次候选并保存独立查询报告。

    参数：question为用户本轮输入，检索与重排使用同一个问题，不使用历史固定问题；
        candidate_count为最多召回的块数，默认5。
    返回值：含本次问题、运行编号、集合、模型、检索候选、重排候选、耗时和报告
        路径的字典。只保存本次报告，不新增、更新或删除向量记录，不生成答案。
        使用without_notes作为当前演示选择，不表示它对全部问题优于其他方案。
        未实现相关性阈值、自动拒答、多子问题覆盖或线上并发治理。
    """
    question = question.strip()
    if not question or candidate_count < 1:
        raise ValueError("请输入非空问题，且候选数量必须大于0。")
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    registry_path = output_dir / "retrieval_text_comparison.json"
    bundle_path = output_dir / "rag_chunk_bundle.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))

    # 旧报告只提供集合与建库配置，不使用其中的question、ranking或目标标记。
    actual_config = (
        settings.embedding.provider,
        settings.embedding.model_name,
        settings.embedding.normalize_embeddings,
    )
    stored_config = (
        registry["embedding_provider"],
        registry["embedding_model"],
        registry["normalize_embeddings"],
    )
    if actual_config != stored_config:
        raise ValueError("当前向量模型配置与建库时不一致，请恢复原配置或重新建立演示集合。")

    # 仅重建文字对象以核对集合指纹，不生成向量、不写库。
    chunks = build_without_notes_chunks(bundle)
    identity = json.dumps({
        "mode": "without_notes",
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "provider": settings.embedding.provider,
        "model": settings.embedding.model_name,
        "normalize": settings.embedding.normalize_embeddings,
    }, ensure_ascii=False, sort_keys=True)
    expected_collection = "pdf_text_ab_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    collection_name = registry["variants"]["without_notes"]["collection_name"]
    if collection_name != expected_collection:
        raise ValueError("当前块内容与已建集合不一致，请重新运行检索正文对照脚本后再查询。")
    database_dir = project_root / "output/learning/pdf_chroma"
    if not (database_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError("演示向量库不存在，请先完成上一阶段入库。")

    container = RuntimeContainer()
    embedding_provider = PdfDemoEmbeddingProvider()
    container.register("embedding", embedding_provider)
    container.register("vector_store", ExistingPdfVectorStoreProvider(
        embedding_provider, directory=database_dir, collection_name=collection_name,
    ))
    reranker_settings = settings.reranker.model_copy(update={"model_name": "BAAI/bge-reranker-v2-m3"})
    container.register("reranker", RerankerProvider(reranker_settings=reranker_settings))

    print(f"本次问题：{question}", flush=True)
    print(f"只查询已有集合：{collection_name}，不重新入库。", flush=True)
    try:
        started_at = perf_counter()
        await container.startup()
        startup_seconds = perf_counter() - started_at
        database = container.get("vector_store").db
        stored_ids = set(database.get(include=[])["ids"])
        if stored_ids != {chunk.chunk_id for chunk in chunks}:
            raise ValueError("集合中的块编号与当前资料不一致，请检查演示库。")

        started_at = perf_counter()
        hits = database.similarity_search_with_score(question, k=min(candidate_count, len(chunks)))
        retrieval_seconds = perf_counter() - started_at
        candidates = []
        for rank, (document, distance) in enumerate(hits, start=1):
            chunk_id = document.metadata["chunk_id"]
            record = bundle["structured_content_by_chunk_id"][chunk_id]
            candidates.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "cosine_distance": float(distance),
                "caption": record["structured_content"]["caption"],
                "section_heading": record["structured_content"]["section_heading"],
                "retrieval_text": document.page_content,
                "metadata": document.metadata,
            })

        reranked = rerank_saved_candidates(
            question=question,
            candidates=candidates,
            model=container.get("reranker").reranker,
        ) if candidates else {"single_batch_inference_seconds": 0.0, "ranking": []}
        run_id = uuid4().hex
        run_dir = output_dir / "query_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "retrieval_report.json"
        report = {
            "run_id": run_id,
            "question": question,
            "report_path": str(report_path),
            "bundle_path": str(bundle_path),
            "collection_name": collection_name,
            "retrieval_mode": "without_notes",
            "embedding_model": settings.embedding.model_name,
            "reranker_model": reranker_settings.model_name,
            "requested_candidate_count": candidate_count,
            "actual_candidate_count": len(candidates),
            "startup_seconds": round(startup_seconds, 4),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "rerank_seconds": reranked["single_batch_inference_seconds"],
            "retrieval_candidates": candidates,
            "reranked_candidates": reranked["ranking"],
            "review_status": "needs_relevance_review",
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    question = input("请输入这次要检索的新问题：")
    report = asyncio.run(run_pdf_query_demo(question))
    print(f"\n本次运行编号：{report['run_id']}")
    print(f"本次实际问题：{report['question']}")
    print("\n向量检索排名：")
    for item in report["retrieval_candidates"]:
        print(f"第{item['rank']}名 | 距离={item['cosine_distance']:.6f} | {item['section_heading']} | {item['caption']}")
    print("\n重排后的排名：")
    for item in report["reranked_candidates"]:
        print(
            f"第{item['rerank_rank']}名（原第{item['retrieval_rank']}名） | "
            f"评分={item['rerank_model_score']:.6f} | {item['section_heading']} | {item['caption']}"
        )
    print(f"\n检索耗时：{report['retrieval_seconds']}秒；重排耗时：{report['rerank_seconds']}秒")
    print(f"独立报告：{report['report_path']}")
    print("这只是本轮候选，尚未判断能否回答，也没有生成答案。")
