"""小批量真实召回实验：人工审核英文改写后，对照无主体过滤的中英文查询。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.query_builders.english_query_rewriter import rewrite_english_queries
from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.schemas import RagQuery
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS = ("breed_no_heading_v1", "pet_care_v1")


def save_report(report: dict) -> None:
    """保存本轮阶段性报告；report须含report_path绝对路径；无返回值，不写向量库。"""
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_results(cases: list[dict]) -> dict:
    """汇总已完成的逐题命中；cases为含groups的结果列表；返回两组命中率及首个证据排名倒数均值。

    只统计预先标注的证据编号，不评估答案正确率，不声称标签穷尽所有相关资料。
    """
    output = {}
    for group in ("original_zh", "rewritten_en"):
        ranks = [case["groups"][group]["first_expected_rank"] for case in cases]
        output[group] = {
            "completed_cases": len(ranks), "hit_count_at_5": sum(rank is not None for rank in ranks),
            "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks) if ranks else None,
            "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks) if ranks else None,
        }
    return output


async def run_pilot(*, allow_legacy_model: bool = False, rewrites_path: Path | None = None) -> dict:
    """核验10道题的已存证据，生成或复用英文提议，人工确认后顺序执行两组查询。

    参数：allow_legacy_model显式允许缺少建库模型签名的旧集合；rewrites_path可选，
        为上次本脚本报告路径，复用其中英文文本以避免再次调用模型，但仍需人工确认。
    返回值：逐题候选、证据名次、耗时、用量及汇总报告；每个阶段落盘，失败保留现场。
        不修改集合、主图或配置；没有语义自动验收，也不调用回答或重排模型。
    """
    if not allow_legacy_model:
        raise ValueError("旧犬种库缺少建库模型签名，请显式传入--allow-legacy-model，仅用于实验。")
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    dataset = json.loads((ROOT / "data/evaluation/pet_retrieval_pilot_v1.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids) or not cases:
        raise ValueError("题目编号重复或题目为空。")
    output_dir = ROOT / "output/rag_pilot_runs" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    model_config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
                    "normalize": settings.embedding.normalize_embeddings}
    signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    report = {"dataset": dataset, "embedding_config": model_config, "top_k": 5,
              "filters": {}, "status": "starting", "cases": [],
              "report_path": str(output_dir / "pilot_report.json"),
              "notice": "10题开发试验，非保留测试集；模型不接收预期证据。固定每题集合，不做全库自动路由。不重排、不生成答案，中英文两路暂不融合。"}
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
        retrievers = {}
        snapshots = {}
        for name in COLLECTIONS:
            db = container.get(name).db
            records = await asyncio.to_thread(db.get, include=["documents", "metadatas"])
            if not records["ids"]:
                raise ValueError(f"集合{name}为空。")
            signatures = [(metadata or {}).get("embedding_signature") for metadata in records["metadatas"]]
            if any(value is not None and value != signature for value in signatures):
                raise ValueError(f"集合{name}存在已知向量模型配置不匹配。")
            if name == "pet_care_v1" and any(value is None for value in signatures):
                raise ValueError("养护集合存在未知模型记录。")
            snapshots[name] = {key: records[key] for key in ("ids", "documents", "metadatas")}
            retrievers[name] = MetadataFilterRetriever(db, allow_filter_relaxation=False)
        for case in cases:
            name = case["collection_name"]
            if name not in snapshots or not case["expected_chunk_ids"]:
                raise ValueError("题目集合不在白名单内或缺少预期证据。")
            if not set(case["expected_chunk_ids"]).issubset(snapshots[name]["ids"]):
                raise ValueError(f"预期证据未入库：{case['case_id']}")
        # 原始库记录另存，便于人工核对证据；不将这些记录发给翻译模型。
        (output_dir / "source_records.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
        report["legacy_model_status"] = "legacy_unverified_explicitly_allowed"
        if rewrites_path is None:
            print("即将调用项目主模型，批量生成10道题的英文检索文本，可能产生费用。", flush=True)
            llm_provider = LLMProvider()
            container.register("llm", llm_provider)
            started = perf_counter()
            report["rewrite_batch"] = await rewrite_english_queries(llm_provider.main_llm, cases)
            report["rewrite_seconds"] = round(perf_counter() - started, 4)
        else:
            previous = json.loads(rewrites_path.read_text(encoding="utf-8"))
            if previous["dataset"] != dataset:
                raise ValueError("复用报告的题目集与当前版本不同。")
            report["rewrite_batch"] = previous["rewrite_batch"]
            report["reused_rewrites_from"] = str(rewrites_path.resolve())
        proposed = report["rewrite_batch"]["items"]
        if len(proposed) != len(ids) or {item["case_id"] for item in proposed} != set(ids):
            raise ValueError("英文提议编号不完整或重复。")
        english = {item["case_id"]: item["retrieval_question"].strip() for item in proposed}
        if any(not text for text in english.values()):
            raise ValueError("英文检索文本不能为空。")
        report["status"] = "awaiting_human_review"
        save_report(report)
        for case in cases:
            print(f"\n{case['case_id']}\n中文：{case['question']}\n英文：{english[case['case_id']]}\n检查点：{case['review_points']}", flush=True)
        print(f"\n提议已保存：{report['report_path']}", flush=True)
        approval = await asyncio.to_thread(input, "确认翻译忠实后输入 继续；其他输入停止：")
        if approval.strip() != "继续":
            report["status"] = "stopped_before_retrieval"
            save_report(report)
            return report
        report["human_review"] = "user_approved_batch"
        report["status"] = "running"
        save_report(report)
        for case in cases:
            item = {**case, "english_question": english[case["case_id"]], "groups": {}}
            for group, text in (("original_zh", case["question"]), ("rewritten_en", item["english_question"])):
                started = perf_counter()
                context = await retrievers[case["collection_name"]].async_retrieve(
                    RagQuery(question=text, user_id="pet_retrieval_pilot", top_k=5, filters={}),
                )
                hits = context.chunks
                target_ranks = [rank for rank, hit in enumerate(hits, 1)
                                if hit.chunk.chunk_id in case["expected_chunk_ids"]]
                item["groups"][group] = {
                    "retrieval_question": text, "seconds": round(perf_counter() - started, 4),
                    "first_expected_rank": min(target_ranks) if target_ranks else None,
                    "chunks": [hit.model_dump(mode="json") for hit in hits],
                }
            report["cases"].append(item)
            save_report(report)
            print(f"{case['case_id']}：中文目标排名={item['groups']['original_zh']['first_expected_rank']} → "
                  f"英文目标排名={item['groups']['rewritten_en']['first_expected_rank']}", flush=True)
        report["summary"] = summarize_results(report["cases"])
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        await container.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实小批量中英文检索对照，不修改向量库。")
    parser.add_argument("--allow-legacy-model", action="store_true", help="显式允许实验查询未知建库模型的旧集合。")
    parser.add_argument("--rewrites", type=Path, help="复用之前本脚本报告中的英文文本，不再次调用翻译模型。")
    args = parser.parse_args()
    result = asyncio.run(run_pilot(allow_legacy_model=args.allow_legacy_model, rewrites_path=args.rewrites))
    print("汇总：", json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    print("状态：", result["status"], "报告：", result["report_path"])
