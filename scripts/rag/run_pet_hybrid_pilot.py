"""复用已审核英文，真实对照纯向量、关键词及排名融合；不调用聊天模型。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.retrievers.english_bm25 import EnglishBM25
from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagChunk, RagQuery, RagRetrievedChunk
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS = ("breed_no_heading_v1", "pet_care_v1")


def records_digest(records: dict) -> str:
    """核对本轮与旧实验正文范围；records为编号、正文、元数据三列；返回稳定内容指纹，不含向量。"""
    rows = sorted(zip(records["ids"], records["documents"], records["metadatas"]), key=lambda row: row[0])
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def first_expected_rank(ids: list[str], expected: list[str]) -> int | None:
    """查找首个预期证据；ids为有序候选编号，expected为预期编号；返回从1开始名次或None。"""
    return next((rank for rank, record_id in enumerate(ids, 1) if record_id in expected), None)


def save_report(report: dict) -> None:
    """保存独立实验报告；report含绝对report_path；无返回值，不改源报告。"""
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_hybrid_pilot(source_report: Path, *, allow_legacy_model: bool = False) -> dict:
    """复用完成并人工审核的英文报告，运行十题关键词与向量双路查询。

    参数：source_report为上一轮pilot_report.json路径；allow_legacy_model表示显式
        允许实验查询缺少签名的旧犬种库，但已知不匹配配置始终拒绝。
    返回值：含两路原候选、完整融合候选、最终前五、目标名次及汇总的独立报告。
        同一题只用一个英文问题，不加主体过滤，不调用翻译或重排模型，不写向量库。
    """
    if not allow_legacy_model:
        raise ValueError("请显式传入--allow-legacy-model，仅用于旧集合实验。")
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    previous = json.loads(source_report.read_text(encoding="utf-8"))
    if previous.get("status") != "completed" or previous.get("human_review") != "user_approved_batch":
        raise ValueError("请使用已完成且人工审核通过的上一轮报告。")
    cases = previous["dataset"]["cases"]
    english_items = previous["rewrite_batch"]["items"]
    expected_case_ids = {case["case_id"] for case in cases}
    if (not cases or len(expected_case_ids) != len(cases) or len(english_items) != len(cases)
            or {item["case_id"] for item in english_items} != expected_case_ids):
        raise ValueError("原报告题目或英文编号不完整、重复。")
    english = {item["case_id"]: item["retrieval_question"].strip() for item in english_items}
    if not all(english.values()) or any(case["collection_name"] not in COLLECTIONS for case in cases):
        raise ValueError("英文为空或集合不在白名单中。")
    model_config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
                    "normalize": settings.embedding.normalize_embeddings}
    if previous["embedding_config"] != model_config:
        raise ValueError("当前向量模型配置与上一轮不同，不能作为同配置实验。")
    signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    old_snapshots = json.loads((source_report.parent / "source_records.json").read_text(encoding="utf-8"))
    output_dir = ROOT / "output/rag_hybrid_runs" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "source_report": str(source_report.resolve()),
              "dataset": previous["dataset"], "rewrite_batch": previous["rewrite_batch"],
              "embedding_config": model_config, "filters": {}, "top_k_per_channel": 5,
              "final_top_k": 5, "rrf_k": 60, "cases": [],
              "keyword_config": {"algorithm": "BM25_positive_idf", "k1": 1.5, "b": 0.75,
                                 "indexed_fields": ["content"], "stemming": False},
              "report_path": str(output_dir / "hybrid_report.json"),
              "notice": "英文问题固定，向量5条加关键词5条后融合取5。候选来源与工作量增加；不重排、不调用聊天模型。BM25只处理英文词语，不自动处理同义词或主体一致性。"}
    save_report(report)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    for name in COLLECTIONS:
        container.register(name, VectorStoreProvider(
            embedding, collection_name=name, enable_memory=False, create_collection_if_not_exists=False,
        ))
    try:
        await container.startup()
        keyword_indexes, retrievers, chunks_by_collection, initial_digests = {}, {}, {}, {}
        for name in COLLECTIONS:
            database = container.get(name).db
            records = await asyncio.to_thread(database.get, include=["documents", "metadatas"])
            if records_digest(records) != records_digest(old_snapshots[name]):
                raise ValueError(f"集合{name}的正文、元数据或记录范围已变化，请先重跑上一轮。")
            initial_digests[name] = records_digest(records)
            signatures = [(metadata or {}).get("embedding_signature") for metadata in records["metadatas"]]
            if any(value is not None and value != signature for value in signatures):
                raise ValueError("集合存在已知向量配置不匹配。")
            if name == "pet_care_v1" and any(value is None for value in signatures):
                raise ValueError("养护集合存在未知模型记录。")
            chunks = {}
            for record_id, text, metadata in zip(records["ids"], records["documents"], records["metadatas"]):
                if not isinstance(text, str) or not isinstance(metadata, dict) or not metadata.get("doc_id"):
                    raise ValueError("记录正文或来源信息不完整。")
                chunks[record_id] = RagChunk(
                    chunk_id=record_id, doc_id=metadata["doc_id"], content=text,
                    chunk_index=metadata.get("chunk_index", 0), source=metadata.get("source", ""),
                    title=metadata.get("title", ""), metadata=metadata,
                )
            chunks_by_collection[name] = chunks
            keyword_indexes[name] = await asyncio.to_thread(
                EnglishBM25, records["ids"], records["documents"],
            )
            retrievers[name] = MetadataFilterRetriever(database, allow_filter_relaxation=False)
        report.update(status="running", source_text_metadata_unchanged_verified=True)
        for case in cases:
            name = case["collection_name"]
            targets = case["expected_chunk_ids"]
            if not targets or not set(targets).issubset(chunks_by_collection[name]):
                raise ValueError(f"目标块缺失：{case['case_id']}")
            text = english[case["case_id"]]
            started = perf_counter()
            context = await retrievers[name].async_retrieve(RagQuery(
                question=text, top_k=5, filters={}, user_id="pet_hybrid_pilot",
            ))
            vector_seconds = round(perf_counter() - started, 4)
            vector_hits = context.chunks
            started = perf_counter()
            keyword_raw = await asyncio.to_thread(keyword_indexes[name].search, text, top_k=5)
            keyword_seconds = round(perf_counter() - started, 4)
            keyword_hits = [RagRetrievedChunk(
                chunk=chunks_by_collection[name][hit["chunk_id"]],
                retrieval_score=hit["score"], final_score=hit["score"],
                reason="英文正文关键词匹配；分数不是相关概率。",
            ) for hit in keyword_raw]
            fused = fuse_collection_results({"vector": vector_hits, "keyword": keyword_hits}, rrf_k=60)
            # 复用现有融合算法；将其历史collection_name字段转成准确的通道名称。
            fused_json = [{"chunk": hit.chunk.model_dump(mode="json"), "rrf_score": hit.rrf_score,
                           "channel_contributions": [
                               {"channel": part.collection_name, "rank": part.rank,
                                "raw_score": part.retrieval_score, "rrf_contribution": part.rrf_contribution}
                               for part in hit.contributions]} for hit in fused]
            selected = fused_json[:5]
            item = {**case, "retrieval_question": text, "vector_seconds": vector_seconds,
                    "keyword_seconds": keyword_seconds,
                    "vector_candidates": [hit.model_dump(mode="json") for hit in vector_hits],
                    "keyword_candidates": [hit.model_dump(mode="json") for hit in keyword_hits],
                    "fused_candidates": fused_json, "selected": selected,
                    "ranks": {
                        "vector_top5": first_expected_rank([hit.chunk.chunk_id for hit in vector_hits], targets),
                        "keyword_top5": first_expected_rank([hit.chunk.chunk_id for hit in keyword_hits], targets),
                        "union_all": first_expected_rank([hit["chunk"]["chunk_id"] for hit in fused_json], targets),
                        "hybrid_top5": first_expected_rank([hit["chunk"]["chunk_id"] for hit in selected], targets),
                    }}
            report["cases"].append(item)
            save_report(report)
            print(f"{case['case_id']}：向量={item['ranks']['vector_top5']}；"
                  f"关键词={item['ranks']['keyword_top5']}；混合前五={item['ranks']['hybrid_top5']}", flush=True)
        for name in COLLECTIONS:
            current = await asyncio.to_thread(container.get(name).db.get, include=["documents", "metadatas"])
            if records_digest(current) != initial_digests[name]:
                raise RuntimeError("实验期间正文或元数据发生变化，本轮不可直接对照。")
        report["summary"] = {}
        for group in ("vector_top5", "keyword_top5", "hybrid_top5"):
            ranks = [item["ranks"][group] for item in report["cases"]]
            report["summary"][group] = {
                "case_count": len(ranks), "hit_count_at_5": sum(rank is not None for rank in ranks),
                "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks),
                "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
            }
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        await container.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复用已审核英文，真实运行关键词与向量混合检索。")
    parser.add_argument("--source-report", required=True, type=Path, help="上一轮已完成的pilot_report.json完整路径。")
    parser.add_argument("--allow-legacy-model", action="store_true", help="允许旧集合实验，不伪造建库模型签名。")
    args = parser.parse_args()
    result = asyncio.run(run_hybrid_pilot(args.source_report, allow_legacy_model=args.allow_legacy_model))
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    print("报告：", result["report_path"])
