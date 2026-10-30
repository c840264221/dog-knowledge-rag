"""冻结双语完整候选池，使用跨语言模型重排后取前五，不重查向量库。"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.settings import settings


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def rank_candidates(model, question: str, candidates: list[dict]) -> list[dict]:
    """对固定候选正文评分，保留原融合位置和模型输出，不做批次归一化。

    参数：model为Provider提供且支持predict的重排模型；question为原中文问题；
        candidates为完整融合候选，包含chunk及原rrf_score、来源贡献。
    返回值：按模型分值降序的候选列表，新增rerank_score、rerank_rank及original_fused_rank；
        分数相同保持原融合顺序，数量或数值异常直接报错，不截断后才评分。
    """
    if not question.strip() or not candidates:
        raise ValueError("问题与候选池不能为空。")
    keys = [(item["chunk"]["doc_id"], item["chunk"]["chunk_id"]) for item in candidates]
    if len(set(keys)) != len(keys) or any(not item["chunk"]["content"].strip() for item in candidates):
        raise ValueError("候选重复或正文为空。")
    pairs = [(question, item["chunk"]["content"]) for item in candidates]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    if len(scores) != len(candidates):
        raise ValueError("模型评分数量与候选数量不同。")
    ranked = []
    for index, (candidate, value) in enumerate(zip(candidates, scores), 1):
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("模型评分不是有限数值。")
        ranked.append({**candidate, "original_fused_rank": index, "rerank_score": score})
    ranked.sort(key=lambda item: -item["rerank_score"])
    return [{**item, "rerank_rank": index} for index, item in enumerate(ranked, 1)]


def first_rank(candidates: list[dict], expected: list[str]) -> int | None:
    """检查候选中的预期证据；candidates为有序候选，expected为预期编号；返回从1开始名次或None。"""
    return next((rank for rank, item in enumerate(candidates, 1)
                 if item["chunk"]["chunk_id"] in expected), None)


def save_report(report: dict) -> None:
    """保存阶段报告；report须含绝对report_path；无返回值，不改旧报告或数据库。"""
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def rerank_bilingual_report(source_path: Path) -> dict:
    """复用已完成双语融合报告的完整候选池，逐题重排和检查最终前五。

    参数：source_path为bilingual_report.json完整路径，源实验每路与最终候选数须为5。
    返回值：保存独立报告的路径、模型配置、逐题完整排名及两组汇总。
        不调用聊天模型、不访问向量库；本地模型未缓存时Provider可能下载模型。
        原中文问题用于全部候选评分，预期证据只用于事后检查，不参与评分。
    """
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if (source.get("status") != "completed" or source.get("experiment_mode") != "frozen_candidate_fusion"
            or source.get("top_k_per_channel") != 5 or source.get("final_top_k") != 5):
        raise ValueError("请提供已完成的双语冻结候选融合报告，两处候选数量必须为5。")
    definitions = source["dataset"]["cases"]
    cases = source["cases"]
    by_id = {case["case_id"]: case for case in definitions}
    if (not cases or len(by_id) != len(definitions) or len(cases) != len(by_id)
            or len({case["case_id"] for case in cases}) != len(cases)
            or {case["case_id"] for case in cases} != set(by_id)):
        raise ValueError("题目与结果编号缺失或重复。")
    for case in cases:
        definition = by_id[case["case_id"]]
        if any(case[key] != definition[key] for key in ("question", "collection_name", "expected_chunk_ids")):
            raise ValueError("题目、集合或预期证据错位。")
        candidates = case["fused_candidates"]
        if not candidates or len(candidates) > 10 or case["selected"] != candidates[:5]:
            raise ValueError("原融合候选池或最终选择不符合本次实验前提。")
    output_dir = Path(__file__).resolve().parents[2] / "output/rag_bilingual_rerank_runs" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting", "source_report": str(source_path.resolve()),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": source["dataset"], "embedding_config": source["embedding_config"],
        "configured_reranker": MODEL_NAME, "reranker_device": settings.reranker.device,
        "rerank_question_source": "original_chinese_question", "scored_fields": ["chunk.content"],
        "score_handling": "保留predict输出，不做批次归一化；不是正确概率，不与RRF分数相加。",
        "final_top_k": 5, "cases": [],
        "report_path": str(output_dir / "bilingual_rerank_report.json"),
        "notice": "冻结同一批真实候选，仅改变最终排序。不能补回未召回证据，不代表完整在线链路或答案评测。",
    }
    save_report(report)
    container = RuntimeContainer()
    local_settings = settings.reranker.model_copy(update={"model_name": MODEL_NAME})
    container.register("reranker", RerankerProvider(reranker_settings=local_settings))
    try:
        print(f"准备加载重排模型：{MODEL_NAME}。复用本地缓存，不重查数据库。", flush=True)
        started = perf_counter()
        await container.startup()
        report["model_startup_seconds"] = round(perf_counter() - started, 4)
        model = container.get("reranker").reranker
        report["status"] = "running"
        for case in cases:
            started = perf_counter()
            ranked = await asyncio.to_thread(rank_candidates, model, case["question"], case["fused_candidates"])
            elapsed = round(perf_counter() - started, 4)
            expected = case["expected_chunk_ids"]
            selected = ranked[:5]
            report["cases"].append({
                "case_id": case["case_id"], "question": case["question"],
                "collection_name": case["collection_name"], "expected_chunk_ids": expected,
                "candidate_count": len(ranked), "rerank_seconds": elapsed,
                "original_fused_candidates": case["fused_candidates"], "reranked_candidates": ranked,
                "selected": selected,
                "ranks": {"original_pool": first_rank(case["fused_candidates"], expected),
                          "rrf_top5": first_rank(case["selected"], expected),
                          "reranked_all": first_rank(ranked, expected),
                          "reranked_top5": first_rank(selected, expected)},
            })
            save_report(report)
            ranks = report["cases"][-1]["ranks"]
            print(f"{case['case_id']}：融合前五={ranks['rrf_top5']} → 重排前五={ranks['reranked_top5']}；"
                  f"本题重排耗时={elapsed}秒", flush=True)
        report["summary"] = {}
        for group in ("rrf_top5", "reranked_top5"):
            ranks = [case["ranks"][group] for case in report["cases"]]
            report["summary"][group] = {
                "case_count": len(ranks), "hit_count_at_5": sum(rank is not None for rank in ranks),
                "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks),
                "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
            }
        report["pool_hit_case_count"] = sum(case["ranks"]["original_pool"] is not None for case in report["cases"])
        report["total_rerank_seconds"] = sum(case["rerank_seconds"] for case in report["cases"])
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        await container.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整双语候选池重排对照，不查询向量库、不调用聊天模型。")
    parser.add_argument("--source-report", required=True, type=Path, help="上一轮bilingual_report.json完整路径。")
    args = parser.parse_args()
    report = asyncio.run(rerank_bilingual_report(args.source_report))
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    print("报告：", report["report_path"])
