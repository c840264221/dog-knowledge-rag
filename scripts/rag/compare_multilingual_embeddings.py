"""保留业务块，仅替换向量模型；默认预览，显式构建后用中文原题查询。"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.indexers.chroma_collection_copy import source_hnsw_configuration
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


MODEL = "BAAI/bge-m3"
DIMENSION = 1024
COLLECTIONS = {
    "breed_no_heading_v1": "breed_bge_m3_v1",
    "pet_care_v1": "pet_care_bge_m3_v1",
}
MODEL_METADATA_KEYS = {"embedding_signature", "embedding_model"}


def business_digest(records: dict) -> str:
    """比较不含模型标记的记录内容；records为编号、正文、元数据；返回稳定指纹。

    只排除本实验改动的embedding_signature及embedding_model，保留所有其他来源信息。
    """
    ids = records["ids"]
    if len(set(ids)) != len(ids) or any(records[key] is None or len(records[key]) != len(ids)
                                      for key in ("documents", "metadatas")):
        raise ValueError("记录编号重复或列长度不一致。")
    rows = []
    for record_id, text, metadata in zip(ids, records["documents"], records["metadatas"]):
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
            raise ValueError("正文或元数据缺失。")
        rows.append([record_id, text, {key: value for key, value in metadata.items()
                                     if key not in MODEL_METADATA_KEYS}])
    return hashlib.sha256(json.dumps(sorted(rows, key=lambda row: row[0]),
                                    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_vectors(vectors, count: int) -> None:
    """检查新模型批次输出；vectors为模型向量，count为本批正文数；无返回值，异常直接报错。"""
    if len(vectors) != count or any(len(vector) != DIMENSION or
                                   not all(math.isfinite(float(value)) for value in vector)
                                   for vector in vectors):
        raise ValueError("模型向量数量、1024维长度或数值不符合预期。")


def save_report(report: dict) -> None:
    """保存阶段报告；report含绝对report_path；无返回值，不覆盖旧实验报告。"""
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def compare_models(source_report: Path, *, build: bool = False, query_existing: bool = False) -> dict:
    """预览或重建独立跨语言集合并实际查询10道中文原题。

    参数：source_report为原中英文pilot_report.json；build表示显式允许新模型下载及
        独立集合写入；query_existing表示只读取已完整建好的新集合，二者互斥。
    返回值：含计划、原样业务记录校验、构建耗时、中文查询及旧基线对照的报告。
        默认不加载模型、不创建目标集合。完整目标可复用；部分写入不自动覆盖或删除。
        不翻译、不加主体过滤、不重排、不改主图；近似索引重建也可能影响排名。
    """
    if build and query_existing:
        raise ValueError("构建与只查询模式不能同时启用。")
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("原向量库不存在。")
    previous = json.loads(source_report.read_text(encoding="utf-8"))
    if previous.get("status") != "completed" or previous.get("top_k") != 5:
        raise ValueError("需要已完成且前五候选的原始对照报告。")
    cases = previous["dataset"]["cases"]
    prior_cases = {case["case_id"]: case for case in previous["cases"]}
    if (not cases or len({case["case_id"] for case in cases}) != len(cases)
            or len(previous["cases"]) != len(prior_cases)
            or {case["case_id"] for case in cases} != set(prior_cases)):
        raise ValueError("原题目与结果编号缺失或重复。")
    old_snapshots = json.loads((source_report.parent / "source_records.json").read_text(encoding="utf-8"))
    model_config = {"provider": "huggingface", "model": MODEL, "normalize": True}
    signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    local_settings = settings.embedding.model_copy(update={
        "provider": "huggingface", "model_name": MODEL, "normalize_embeddings": True,
    })
    output_dir = Path(__file__).resolve().parents[2] / "output/rag_multilingual_runs" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": "build_and_query" if build else "query_existing" if query_existing else "preview_only",
        "status": "starting", "source_report": str(source_report.resolve()),
        "dataset": previous["dataset"], "embedding_config": model_config,
        "device": local_settings.device, "expected_dimension": DIMENSION,
        "document_batch_size": 8, "filters": {}, "top_k": 5,
        "collections": [], "cases": [],
        "report_path": str(output_dir / "model_comparison_report.json"),
        "notice": "同编号、正文及业务元数据，新模型标记与新向量。保留源距离及索引参数，但新建近似索引图不完全相同；本实验比较整套模型替换效果，不是排除所有数值精度和索引差异的纯模型因果实验。",
    }
    save_report(report)
    stores = RuntimeContainer()
    models = RuntimeContainer()
    for name in COLLECTIONS:
        stores.register(name, VectorStoreProvider(
            None, collection_name=name, enable_memory=False, create_collection_if_not_exists=False,
        ))
    try:
        await stores.startup()
        snapshots, configurations = {}, {}
        for name, target_name in COLLECTIONS.items():
            collection = stores.get(name).native_collection
            records = await asyncio.to_thread(collection.get, include=["documents", "metadatas"])
            digest = business_digest(records)
            if digest != business_digest(old_snapshots[name]):
                raise ValueError(f"源集合{name}业务数据与上一轮不同，请先确认实验资料。")
            snapshots[name] = records
            configurations[name] = source_hnsw_configuration(collection)
            if configurations[name]["hnsw"]["space"] != "l2":
                raise ValueError("本轮对照只支持已确认的l2距离。")
            report["collections"].append({
                "source_collection_name": name, "target_collection_name": target_name,
                "record_count": len(records["ids"]), "business_records_fingerprint": digest,
                "collection_configuration": configurations[name],
            })
        for case in cases:
            if (case["collection_name"] not in snapshots or not case["expected_chunk_ids"]
                    or any(case[key] != prior_cases[case["case_id"]][key]
                           for key in ("question", "collection_name", "expected_chunk_ids"))):
                raise ValueError("题目来源、问题或预期证据不一致。")
            if not set(case["expected_chunk_ids"]).issubset(snapshots[case["collection_name"]]["ids"]):
                raise ValueError("预期证据不存在。")
        report["total_records_to_reembed"] = sum(len(records["ids"]) for records in snapshots.values())
        if not build and not query_existing:
            report["status"] = "preview_completed"
            return report
        targets = {}
        for item in report["collections"]:
            name, target_name = item["source_collection_name"], item["target_collection_name"]
            provider = VectorStoreProvider(
                None, collection_name=target_name, enable_memory=False,
                create_collection_if_not_exists=build, collection_configuration=configurations[name],
            )
            stores.register(target_name, provider)
            target = provider.native_collection
            targets[name] = target
            if source_hnsw_configuration(target) != configurations[name]:
                raise ValueError("新集合距离或索引参数与源集合不同。")
            existing = await asyncio.to_thread(target.get, include=["documents", "metadatas"])
            if existing["ids"]:
                if (business_digest(existing) != item["business_records_fingerprint"]
                        or any(metadata.get("embedding_signature") != signature
                               or metadata.get("embedding_model") != MODEL for metadata in existing["metadatas"])):
                    raise ValueError(f"目标{target_name}数据不完整或配置不同；拒绝覆盖，请保留现场检查。")
                vector_sample = await asyncio.to_thread(target.get, ids=[existing["ids"][0]], include=["embeddings"])
                validate_vectors(vector_sample["embeddings"], 1)
                item["build_status"] = "verified_existing_complete"
            elif query_existing:
                raise ValueError("新集合为空，不能执行只查询模式。")
            else:
                item["build_status"] = "pending"
        save_report(report)
        print(f"将加载{MODEL}；未缓存时可能下载约2.27GB主权重。共有{report['total_records_to_reembed']}条资料。", flush=True)
        embedding = EmbeddingProvider(enable_memory=False, embedding_settings=local_settings)
        models.register("embedding", embedding)
        started = perf_counter()
        await models.startup()
        report["model_startup_seconds"] = round(perf_counter() - started, 4)
        model = models.get("embedding").embedding
        for item in report["collections"]:
            name = item["source_collection_name"]
            target, records = targets[name], snapshots[name]
            if item["build_status"] == "pending":
                item.update(build_status="building", written_count=0)
                save_report(report)
                started = perf_counter()
                for start in range(0, len(records["ids"]), 8):
                    end = min(start + 8, len(records["ids"]))
                    vectors = await asyncio.to_thread(model.embed_documents, records["documents"][start:end])
                    validate_vectors(vectors, end - start)
                    await asyncio.to_thread(
                        target.add, ids=records["ids"][start:end], documents=records["documents"][start:end],
                        metadatas=[{**metadata, "embedding_signature": signature, "embedding_model": MODEL}
                                   for metadata in records["metadatas"][start:end]], embeddings=vectors,
                    )
                    item["written_count"] = end
                    if end % 64 == 0 or end == len(records["ids"]):
                        print(f"{item['target_collection_name']}：已写入{end}/{len(records['ids'])}", flush=True)
                        save_report(report)
                item["build_seconds"] = round(perf_counter() - started, 4)
                copied = await asyncio.to_thread(target.get, include=["documents", "metadatas"])
                if business_digest(copied) != item["business_records_fingerprint"]:
                    raise RuntimeError("写入后业务数据与源不一致，不执行查询。")
                item["build_status"] = "built_and_verified"
                save_report(report)
        for case in cases:
            target = targets[case["collection_name"]]
            started = perf_counter()
            vector = await asyncio.to_thread(model.embed_query, case["question"])
            validate_vectors([vector], 1)
            raw = await asyncio.to_thread(target.query, query_embeddings=[vector], n_results=5,
                                          include=["documents", "metadatas", "distances"])
            elapsed = round(perf_counter() - started, 4)
            hits = [{"chunk_id": record_id, "content": text, "metadata": metadata,
                     "distance": float(distance), "rank": rank}
                    for rank, (record_id, text, metadata, distance) in enumerate(zip(
                        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), 1)]
            target_rank = next((hit["rank"] for hit in hits if hit["chunk_id"] in case["expected_chunk_ids"]), None)
            original = prior_cases[case["case_id"]]["groups"]["original_zh"]
            original_rank = next((rank for rank, hit in enumerate(original["chunks"], 1)
                                  if hit["chunk"]["chunk_id"] in case["expected_chunk_ids"]), None)
            report["cases"].append({**case, "target_collection_name": COLLECTIONS[case["collection_name"]],
                                    "old_model_zh_rank": original_rank, "new_model_zh_rank": target_rank,
                                    "query_seconds": elapsed, "chunks": hits})
            print(f"{case['case_id']}：旧模型中文={original_rank} → 新模型中文={target_rank}", flush=True)
            save_report(report)
        for item in report["collections"]:
            current = await asyncio.to_thread(stores.get(item["source_collection_name"]).native_collection.get,
                                              include=["documents", "metadatas"])
            if business_digest(current) != item["business_records_fingerprint"]:
                raise RuntimeError("实验期间源业务数据发生变化，本轮对照不可直接使用。")
        report["summary"] = {}
        for key in ("old_model_zh_rank", "new_model_zh_rank"):
            ranks = [case[key] for case in report["cases"]]
            report["summary"][key] = {"case_count": len(ranks),
                                      "hit_count_at_5": sum(rank is not None for rank in ranks),
                                      "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks),
                                      "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks)}
        report.update(status="completed", source_business_data_unchanged_verified=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        await models.shutdown()
        await stores.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="默认预览；独立集合跨语言模型对照，不修改源集合。")
    parser.add_argument("--source-report", required=True, type=Path, help="原始中英文对照pilot_report.json完整路径。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="允许下载模型、构建独立新集合并查询。")
    mode.add_argument("--query-existing", action="store_true", help="只核对并查询已完整构建的新集合。")
    args = parser.parse_args()
    result = asyncio.run(compare_models(args.source_report, build=args.build, query_existing=args.query_existing))
    print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "dataset"}},
                     ensure_ascii=False, indent=2))
