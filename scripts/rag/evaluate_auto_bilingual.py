"""实际自动翻译与双语检索对照；不生成答案、不写向量库。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.pipelines.bilingual_retrieval_pipeline import retrieve_auto_bilingual, rerank_pool
from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagQuery
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_COLLECTIONS = {"breed_no_heading_v1", "pet_care_v1"}


def save_report(report: dict) -> None:
    """保存阶段报告；参数为含绝对report_path的字典；返回None，保留已完成题目。"""
    Path(report["report_path"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def score_hits(hits: list, expected: list[str]) -> dict:
    """按最终前五条评分；参数为有序检索块和预期编号；返回首次命中名次，不影响检索。"""
    selected = hits[:5]
    rank = next((i for i, hit in enumerate(selected, 1) if hit.chunk.chunk_id in expected), None)
    return {"first_expected_rank": rank,
            "selected": [hit.model_dump(mode="json") for hit in selected]}


async def run(dataset_path: Path, *, allow_legacy_model: bool = False) -> dict:
    """逐题实际调用翻译与向量检索，比较两种召回及各自重排。

    参数：dataset_path为已标注数据集；allow_legacy_model显式允许旧犬种库缺少模型签名。
    返回值：完整报告；失败题保留并计入总分母，取消执行正常传播。集合和身份由宿主控制。
    """
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset["cases"]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("用例为空或编号重复。")
    names = {c["collection_name"] for c in cases}
    if not names.issubset(ALLOWED_COLLECTIONS):
        raise ValueError("用例集合不在允许范围内。")
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("现有向量库不存在，禁止创建空库进行评估。")
    folder = ROOT / "output/rag_auto_bilingual_runs" / uuid4().hex
    folder.mkdir(parents=True)
    config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
              "normalize": settings.embedding.normalize_embeddings}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report = {"status": "starting", "dataset": dataset,
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "embedding_config": config,
              "reranker": "BAAI/bge-reranker-v2-m3", "top_k_per_channel": 5, "final_top_k": 5,
              "filters": {}, "allow_legacy_model": allow_legacy_model, "cases": [],
              "report_path": str(folder / "report.json"),
              "notice": "自动翻译未经逐题语义审核；成绩需审核翻译及标签后使用。两组重排使用同一中文问题和同一模型；双语候选预算最多为单路两倍。不评估回答准确率。"}
    save_report(report)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    container.register("llm", LLMProvider())
    container.register("reranker", RerankerProvider(reranker_settings=settings.reranker.model_copy(
        update={"model_name": report["reranker"]})))
    for name in sorted(names):
        container.register(name, VectorStoreProvider(embedding, collection_name=name,
            enable_memory=False, create_collection_if_not_exists=False))
    try:
        await container.startup()
        retrievers = {}
        for name in sorted(names):
            db = container.get(name).db
            records = await asyncio.to_thread(db.get, include=["metadatas"])
            signatures = [(m or {}).get("embedding_signature") for m in records["metadatas"]]
            if not records["ids"] or any(s is not None and s != signature for s in signatures):
                raise ValueError(f"集合{name}为空或模型签名不匹配。")
            if any(s is None for s in signatures) and (name != "breed_no_heading_v1" or not allow_legacy_model):
                raise ValueError(f"集合{name}缺少模型签名；旧犬种库实验需--allow-legacy-model。")
            for case in cases:
                if case["collection_name"] == name and (not case["expected_chunk_ids"] or
                        not set(case["expected_chunk_ids"]).issubset(records["ids"])):
                    raise ValueError(f"预期证据不存在：{case['case_id']}")
            retrievers[name] = MetadataFilterRetriever(db, allow_filter_relaxation=False)
        report["status"] = "running"
        for case in cases:
            item = {"case_id": case["case_id"], "question": case["question"], "groups": {}}
            report["cases"].append(item)
            started = perf_counter()
            try:
                query = RagQuery(question=case["question"], user_id="auto_bilingual_evaluation",
                                 top_k=5, filters={})
                result = await retrieve_auto_bilingual(query, llm=container.get("llm").main_llm,
                    retriever=retrievers[case["collection_name"]], reranker=container.get("reranker").reranker)
                item["translation"] = result["translation"]
                item["translation_seconds"] = result["translation_seconds"]
                item["channels"] = {k: [h.model_dump(mode="json") for h in v]
                                    for k, v in result["channels"].items()}
                item["fused_candidates"] = [h.model_dump(mode="json") for h in result["fused_candidates"]]
                zh = result["channels"]["original_zh"]
                pool = fuse_collection_results({"original_zh": zh}, rrf_k=60)
                zh_ranked = await asyncio.to_thread(rerank_pool, query.question, pool,
                                                     container.get("reranker").reranker)
                expected = case["expected_chunk_ids"]
                item["groups"] = {
                    "chinese_vector": score_hits(zh, expected),
                    "bilingual_rrf": score_hits(result["fused_candidates"], expected),
                    "chinese_reranked": score_hits(zh_ranked, expected),
                    "bilingual_reranked": score_hits(result["reranked_candidates"], expected)}
                item["status"] = "completed"
                print(f"{case['case_id']}：" + str({k: v["first_expected_rank"]
                                                   for k, v in item["groups"].items()}), flush=True)
            except Exception as exc:
                item.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
                # 双语链路失败不能自动判定中文基线也失败，独立补测并保留真实结果。
                if "chinese_vector" not in item["groups"]:
                    try:
                        baseline = await retrievers[case["collection_name"]].async_retrieve(query)
                        zh_pool = fuse_collection_results({"original_zh": baseline.chunks}, rrf_k=60)
                        zh_ranked = await asyncio.to_thread(rerank_pool, query.question, zh_pool,
                                                             container.get("reranker").reranker)
                        item["groups"]["chinese_vector"] = score_hits(baseline.chunks, case["expected_chunk_ids"])
                        item["groups"]["chinese_reranked"] = score_hits(zh_ranked, case["expected_chunk_ids"])
                    except Exception as baseline_error:
                        item["baseline_error"] = {"type": type(baseline_error).__name__,
                                                  "message": str(baseline_error)}
                print(f"{case['case_id']}失败：{type(exc).__name__}，保留失败并继续。", flush=True)
            item["seconds"] = round(perf_counter() - started, 4)
            save_report(report)
        report["summary"] = {}
        for group in ("chinese_vector", "bilingual_rrf", "chinese_reranked", "bilingual_reranked"):
            ranks = [c["groups"].get(group, {}).get("first_expected_rank") for c in report["cases"]]
            report["summary"][group] = {"total_cases": len(cases),
                "completed_cases": sum(group in c["groups"] for c in report["cases"]),
                "hit_count_at_5": sum(r is not None for r in ranks),
                "hit_rate_at_5": sum(r is not None for r in ranks) / len(cases),
                "mrr_at_5": sum(1 / r for r in ranks if r is not None) / len(cases)}
        report["status"] = "completed" if all(c["status"] == "completed" for c in report["cases"]) else "completed_with_failures"
    except BaseException as exc:
        report.update(status="aborted", error_type=type(exc).__name__)
        raise
    finally:
        save_report(report)
        await container.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实际自动翻译与单中文召回对照，会调用付费模型。")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/pet_retrieval_pilot_v1.json")
    parser.add_argument("--allow-legacy-model", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args.dataset, allow_legacy_model=args.allow_legacy_model))
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    print("报告：", result["report_path"])
