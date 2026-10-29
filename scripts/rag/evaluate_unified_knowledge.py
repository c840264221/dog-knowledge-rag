"""真实模型查询理解与统一分组检索评估；冻结每题计划后比较四种候选方案。"""

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.agents.dog_knowledge_agent.nodes.query_understanding_node import build_dog_query_understanding_node
from src.rag.pipelines.grouped_knowledge_retrieval import retrieve_groups, rank_and_select_groups, select_group_context
from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagRetrievedChunk
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.container.providers.retriever_provider import RetrieverProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.runtime.context import RuntimeContext, runtime_ctx
from src.settings import settings

ROOT = Path(__file__).resolve().parents[2]
MODES = ("chinese_retrieval", "bilingual_fusion", "chinese_reranked", "bilingual_reranked")


def save_report(report: dict) -> None:
    """保存阶段报告；参数含绝对report_path；返回None，同一事件循环内同步写入避免半成品。"""
    path = Path(report["report_path"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def score_selection(selected: dict, expected_groups: list[dict]) -> dict:
    """事后核对已标注证据；参数为选取结果和标签；返回逐主题前五名次，不参与任何检索。"""
    chunks = selected["rag_context"]["chunks"]
    return {"topics": [{"topic_id": expected["topic_id"],
        "first_expected_rank": next((rank for rank, hit in enumerate(chunks[:5], 1)
            if hit["chunk"]["chunk_id"] in expected["expected_chunk_ids"]), None)}
        for expected in expected_groups], "context": selected["rag_context"],
        "coverage": selected["coverage"], "citation_map": selected["citation_map"]}


def original_channels_only(groups: list[dict]) -> list[dict]:
    """从同次真实通道结果提取中文基线；参数为双语分组；返回同集合中文融合，不重新翻译或查询。"""
    output = []
    for group in groups:
        channels = {name: [RagRetrievedChunk.model_validate(raw) for raw in hits]
                    for name, hits in group["channels"].items() if name.endswith("/zh")}
        pool = fuse_collection_results(channels)
        hits = [RagRetrievedChunk(chunk=item.chunk, retrieval_score=item.contributions[0].retrieval_score)
                for item in pool]
        output.append({**group, "channels": {k: [h.model_dump(mode="json") for h in v]
                        for k, v in channels.items()},
                       "candidates": [h.model_dump(mode="json") for h in hits]})
    return output


def summarize(report: dict) -> dict:
    """汇总计划完成与证据命中；参数为含全部用例的报告；返回四组指标，失败始终在总分母。"""
    cases = report["cases"]
    topic_count = sum(len(c["expected_groups"]) for c in report["dataset"]["cases"])
    summary = {}
    for mode in MODES:
        values = [c["variants"].get(mode) for c in cases]
        ranks = [t["first_expected_rank"] for value in values if value for t in value["topics"]]
        summary[mode] = {"request_count": len(cases), "evidence_topic_count": topic_count,
            "completed_requests": sum(value is not None for value in values),
            "hit_count_at_5": sum(rank is not None for rank in ranks),
            "hit_rate_at_5": sum(rank is not None for rank in ranks) / topic_count,
            "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / topic_count,
            "fully_covered_requests": sum(value is not None and
                all(t["first_expected_rank"] is not None for t in value["topics"]) for value in values)}
    return summary


async def run(dataset_path: Path, *, allow_legacy_model: bool, source_report: Path | None = None) -> dict:
    """运行开发评估或复用已生成计划。

    参数：dataset_path为冻结标签集；allow_legacy_model显式允许旧犬种集合缺少签名；
        source_report可选，复用其同题计划，不改写旧报告、不复用历史候选。
    返回：新报告；模型仅接收问题与目录，不接收预期集合、编号或原文标签。
        比较范围为查询理解后候选召回与选取，不调用回答节点或现有质量门槛。
    """
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset["cases"]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("评估题为空或编号重复。")
    fingerprint = hashlib.sha256(raw).hexdigest()
    previous = json.loads(source_report.read_text(encoding="utf-8")) if source_report else None
    if previous and previous["dataset_sha256"] != fingerprint:
        raise ValueError("不能将另一版本题目的计划复用到当前数据集。")
    frozen = {c["case_id"]: c.get("query_understanding") for c in previous["cases"]} if previous else {}
    folder = ROOT / "output/rag_unified_evaluations" / uuid4().hex
    folder.mkdir(parents=True)
    config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
              "normalize": settings.embedding.normalize_embeddings}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report = {"status": "starting", "dataset": dataset, "dataset_sha256": fingerprint,
        "embedding_config": config, "reranker": settings.reranker.model_name,
        "catalog": settings.rag.knowledge_collections, "top_k_per_channel": 5, "final_top_k": 5,
        "filter_policy": "保留统一查询理解生成的各题条件，两组相同，不与历史无主体过滤成绩直接比较",
        "case_concurrency": 2, "source_report": str(source_report) if source_report else None,
        "report_path": str(folder / "report.json"),
        "notice": "14请求18项已标注证据需求的开发评估；非独立保留集，非回答准确率。自动改写、翻译和选择集合需人工复核。双语候选预算增加；四组复用同一查询计划，标签只用于事后评分。质量评估及回答节点不在本次比较范围。",
        "cases": [{"case_id": c["case_id"], "question": c["question"], "status": "pending", "variants": {}} for c in cases]}
    save_report(report)
    print("评估报告：", report["report_path"], flush=True)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    vectors = {key: VectorStoreProvider(embedding, collection_name=spec["collection_name"],
        enable_memory=False, create_collection_if_not_exists=False)
        for key, spec in settings.rag.knowledge_collections.items()}
    provider = RetrieverProvider(next(iter(vectors.values())), collection_providers=vectors,
        collection_descriptions={key: spec["description"] for key, spec in settings.rag.knowledge_collections.items()})
    container.register("embedding", embedding)
    container.register("retriever", provider)
    container.register("reranker", RerankerProvider())
    model_lock, case_slots = asyncio.Lock(), asyncio.Semaphore(2)
    try:
        await container.startup()
        container.register("llm", LLMProvider())
        understand = build_dog_query_understanding_node(llm_provider=container.get("llm"),
            query_filter_parser=provider.dog_query_filter_parser, collection_catalog=provider.knowledge_catalog())
        report["source_fingerprints"] = {}
        for key, vector in vectors.items():
            records = await asyncio.to_thread(vector.db.get, include=["documents", "metadatas"])
            signatures = [(m or {}).get("embedding_signature") for m in records["metadatas"]]
            if not records["ids"] or any(s is not None and s != signature for s in signatures):
                raise ValueError(f"集合{key}为空或向量签名不符。")
            if any(s is None for s in signatures) and not (key == "dog_breeds" and allow_legacy_model):
                raise ValueError(f"集合{key}缺少模型签名，需要显式允许旧犬种库实验。")
            report["source_fingerprints"][key] = hashlib.sha256(
                json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            for case in cases:
                for expected in case["expected_groups"]:
                    if expected["collection_id"] == key and not set(expected["expected_chunk_ids"]).issubset(records["ids"]):
                        raise ValueError(f"预期证据未入库：{case['case_id']}")
        if previous and (previous["embedding_config"] != config or previous["catalog"] != report["catalog"]):
            raise ValueError("复用计划时需保持目录和向量配置一致。")
        report["status"] = "running"
        save_report(report)

        async def evaluate_case(case: dict, item: dict) -> None:
            """评估一个请求；参数为题目定义及结果容器；返回None，失败保留、独立上下文不串题。"""
            async with case_slots:
                started = perf_counter()
                runtime = RuntimeContext(trace_id=uuid4().hex, user_id="unified_evaluation_user",
                    session_id=item["case_id"], component="unified_retrieval_evaluation")
                old = runtime_ctx.get()
                runtime_ctx.set(runtime)
                try:
                    await runtime.startup()
                    item["status"] = "understanding"
                    print("开始：", case["case_id"], flush=True)
                    update = copy.deepcopy(frozen.get(case["case_id"]))
                    if update is None:
                        update = await understand({"question": case["question"], "user_id": runtime.user_id, "top_k": 5})
                    item["query_understanding"] = update
                    item["status"] = "retrieving"
                    save_report(report)
                    groups = await retrieve_groups(update["rag_sub_queries"], provider=provider, bilingual=True,
                        concurrency=settings.rag.retrieval_concurrency, max_groups=settings.rag.max_sub_queries)
                    item["retrieval_groups"] = groups
                    chinese = original_channels_only(groups)
                    for name, candidate_groups in (("chinese_retrieval", chinese), ("bilingual_fusion", groups)):
                        ranked = [{**g, "ranked_candidates": g["candidates"]} for g in candidate_groups]
                        selection = select_group_context(case["question"], ranked, max_chunks=5,
                            max_characters=settings.rag.context_max_characters)
                        item["variants"][name] = score_selection(selection, case["expected_groups"])
                    async with model_lock:
                        for name, candidate_groups in (("chinese_reranked", chinese), ("bilingual_reranked", groups)):
                            selection = await asyncio.to_thread(rank_and_select_groups, case["question"], candidate_groups,
                                reranker=container.get("reranker").reranker, max_chunks=5,
                                max_characters=settings.rag.context_max_characters)
                            item["variants"][name] = score_selection(selection, case["expected_groups"])
                            item.setdefault("ranked_groups", {})[name] = selection["groups"]
                    item["status"] = "completed"
                    print("完成：", case["case_id"], str({k: [t["first_expected_rank"] for t in v["topics"]]
                                                         for k, v in item["variants"].items()}), flush=True)
                except Exception as error:
                    item.update(status="failed", error_type=type(error).__name__, error_message=str(error))
                    print("失败：", case["case_id"], type(error).__name__, flush=True)
                finally:
                    item["seconds"] = round(perf_counter() - started, 4)
                    save_report(report)
                    try:
                        await runtime.shutdown()
                    finally:
                        runtime_ctx.set(old)

        await asyncio.gather(*(evaluate_case(case, item) for case, item in zip(cases, report["cases"])))
        report["summary"] = summarize(report)
        report["status"] = "completed" if all(c["status"] == "completed" for c in report["cases"]) else "completed_with_failures"
    except BaseException as error:
        report.update(status="aborted", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        await container.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一查询理解及分组检索真实评估，不生成答案。")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/unified_knowledge_development_v1.json")
    parser.add_argument("--allow-legacy-model", action="store_true")
    parser.add_argument("--source-report", type=Path, help="复用原报告中的同题查询计划，重新真实检索。")
    args = parser.parse_args()
    result = asyncio.run(run(args.dataset, allow_legacy_model=args.allow_legacy_model, source_report=args.source_report))
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    print("报告：", result["report_path"])
