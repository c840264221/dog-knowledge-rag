"""冻结真实中英文候选，单独评估双路融合；不重查数据库、不调用任何模型。"""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagRetrievedChunk


def first_rank(ids: list[str], expected: list[str]) -> int | None:
    """定位首个预期块；ids为有序候选编号，expected为预期编号；返回从1开始名次或None。"""
    return next((rank for rank, record_id in enumerate(ids, 1) if record_id in expected), None)


def fuse_bilingual_report(source_path: Path) -> dict:
    """对已完成真实召回报告执行中英文等权融合，保留来源名次和最终前五。

    参数：source_path为上一轮人工审核通过、top_k为5的pilot_report.json路径。
    返回值：独立报告字典，并保存至output/rag_bilingual_runs/本轮编号。
        复用当时真实候选，不代表重新运行完整在线链路；不读取当前数据库状态。
        输入编号重复、题目错位、缺少结果等情况拒绝；不根据预期答案调整融合。
    """
    raw = source_path.read_bytes()
    previous = json.loads(raw)
    if (previous.get("status") != "completed"
            or previous.get("human_review") != "user_approved_batch" or previous.get("top_k") != 5):
        raise ValueError("源报告必须已完成、人工审核通过，且每路候选数为5。")
    definitions = previous["dataset"]["cases"]
    cases = previous["cases"]
    definitions_by_id = {item["case_id"]: item for item in definitions}
    if (not cases or len(definitions_by_id) != len(definitions) or len(cases) != len(definitions)
            or len({case["case_id"] for case in cases}) != len(cases)
            or {case["case_id"] for case in cases} != set(definitions_by_id)):
        raise ValueError("源报告题目或结果编号缺失、重复。")
    report = {
        "status": "processing", "experiment_mode": "frozen_candidate_fusion",
        "source_report": str(source_path.resolve()),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": previous["dataset"], "embedding_config": previous["embedding_config"],
        "top_k_per_channel": 5, "final_top_k": 5, "rrf_k": 60, "cases": [],
        "tie_policy": "同分按原融合函数首次出现顺序；先中文后英文，不按预期答案打破平局。",
        "notice": "复用此前真实召回的中英文候选，只评估冻结候选后的融合策略。无新模型调用、无数据库查询；不能据此计算在线耗时或成本。每路5条，融合池最多10条，最终仍取5。",
    }
    for case in cases:
        definition = definitions_by_id[case["case_id"]]
        if any(case[key] != definition[key] for key in ("question", "collection_name", "expected_chunk_ids")):
            raise ValueError("题目、集合或预期证据与题目集不一致。")
        if not definition["expected_chunk_ids"]:
            raise ValueError("题目必须有预期证据。")
        groups = case["groups"]
        if (groups["original_zh"]["retrieval_question"] != definition["question"]
                or groups["rewritten_en"]["retrieval_question"] != case["english_question"]):
            raise ValueError("中英文检索文本与本题不一致。")
        channels = {}
        for channel in ("original_zh", "rewritten_en"):
            hits = [RagRetrievedChunk.model_validate(item) for item in groups[channel]["chunks"]]
            keys = [(hit.chunk.doc_id, hit.chunk.chunk_id) for hit in hits]
            if len(hits) > 5 or len(set(keys)) != len(keys):
                raise ValueError("单路候选超过5条或含重复块，拒绝直接融合。")
            channels[channel] = hits
        fused = fuse_collection_results(channels, rrf_k=60)
        fused_json = [{
            "chunk": hit.chunk.model_dump(mode="json"), "rrf_score": hit.rrf_score,
            "channel_contributions": [
                {"channel": part.collection_name, "rank": part.rank,
                 "raw_score": part.retrieval_score, "rrf_contribution": part.rrf_contribution}
                for part in hit.contributions],
        } for hit in fused]
        selected = fused_json[:5]
        expected = definition["expected_chunk_ids"]
        ranks = {
            channel: first_rank([hit.chunk.chunk_id for hit in hits], expected)
            for channel, hits in channels.items()
        }
        ranks["union_all"] = first_rank([hit["chunk"]["chunk_id"] for hit in fused_json], expected)
        ranks["bilingual_top5"] = first_rank([hit["chunk"]["chunk_id"] for hit in selected], expected)
        report["cases"].append({
            **definition, "english_question": case["english_question"],
            "channels": {key: groups[key] for key in channels}, "ranks": ranks,
            "candidate_count_after_dedup": len(fused_json),
            "fused_candidates": fused_json, "selected": selected,
            "target_lost_at_selection": ranks["union_all"] is not None and ranks["bilingual_top5"] is None,
        })
    report["summary"] = {}
    for group in ("original_zh", "rewritten_en", "bilingual_top5"):
        ranks = [case["ranks"][group] for case in report["cases"]]
        report["summary"][group] = {
            "case_count": len(ranks), "hit_count_at_5": sum(rank is not None for rank in ranks),
            "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks),
            "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
        }
    report["union_hit_case_count"] = sum(case["ranks"]["union_all"] is not None for case in report["cases"])
    report["target_lost_at_selection_count"] = sum(case["target_lost_at_selection"] for case in report["cases"])
    output_dir = Path(__file__).resolve().parents[2] / "output/rag_bilingual_runs" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report.update(status="completed", report_path=str(output_dir / "bilingual_report.json"))
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复用真实中英文候选，离线对照排名融合，不调用模型或数据库。")
    parser.add_argument("--source-report", required=True, type=Path, help="原中英文对照pilot_report.json完整路径。")
    args = parser.parse_args()
    report = fuse_bilingual_report(args.source_report)
    for case in report["cases"]:
        print(f"{case['case_id']}：中文={case['ranks']['original_zh']}；"
              f"英文={case['ranks']['rewritten_en']}；双路前五={case['ranks']['bilingual_top5']}；"
              f"目标是否被最终截断丢弃={case['target_lost_at_selection']}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("完整候选池命中的题数：", report["union_hit_case_count"])
    print("最终截断丢失的题数：", report["target_lost_at_selection_count"])
    print("报告：", report["report_path"])
