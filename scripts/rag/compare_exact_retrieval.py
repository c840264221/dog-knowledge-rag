"""固定问题向量，对照逐条计算与数据库快速查找；不修改向量库。"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from src.rag.evaluation.exact_vector_ranking import exact_l2_ranking
from src.rag.indexers.chroma_collection_copy import records_fingerprint, source_hnsw_configuration
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


TARGET_ID = "chunk_46d049fe6e716d96"
QUESTIONS = {
    "zh": "金毛日常运动有什么要求？",
    "en": "What are the daily exercise needs of a Golden Retriever?",
}


async def compare_exact_retrieval() -> dict:
    """执行中文/英文和不限犬种/限定金毛的四组只读对照。

    参数：无；集合固定breed_no_heading_v1，前五条候选，当前模型由settings提供。
    返回值：含四组逐条计算结果、数据库结果和金毛运动目标名次的独立JSON报告。
        每种语言只生成一次问题向量，两种查找方式使用同一向量和相同过滤条件。
        只支持当前l2距离；源记录发生变化就报错，防止比较不同数据快照。
    """
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    container = RuntimeContainer()
    embedding_provider = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding_provider)
    container.register("vectors", VectorStoreProvider(
        None, collection_name="breed_no_heading_v1", enable_memory=False,
        create_collection_if_not_exists=False,
    ))
    report = {
        "collection_name": "breed_no_heading_v1", "top_k": 5,
        "embedding_config": {"provider": settings.embedding.provider,
                             "model": settings.embedding.model_name,
                             "normalize": settings.embedding.normalize_embeddings},
        "target_chunk_id": TARGET_ID, "cases": [],
        "notice": "逐条计算只保证按当前向量距离排序，不保证语义正确；英文查询是诊断对照，不是上线翻译策略。旧库模型身份仍未知。",
    }
    try:
        await container.startup()
        collection = container.get("vectors").native_collection
        if source_hnsw_configuration(collection)["hnsw"]["space"] != "l2":
            raise ValueError("本实验仅支持l2距离，拒绝混用距离算法。")
        records = await asyncio.to_thread(collection.get, include=["documents", "metadatas", "embeddings"])
        before = records_fingerprint(records)
        positions = {record_id: index for index, record_id in enumerate(records["ids"])}
        if TARGET_ID not in positions:
            raise ValueError("目标运动正文不存在。")
        report["source_records_fingerprint"] = before
        report["record_count"] = len(positions)
        for language, question in QUESTIONS.items():
            vector = await asyncio.to_thread(embedding_provider.embedding.embed_query, question)
            for dog_name in (None, "Golden Retriever"):
                ranked = await asyncio.to_thread(exact_l2_ranking, vector, records, dog_name=dog_name)
                if not ranked:
                    raise ValueError("过滤后没有可比较记录。")
                where = {"dog_name": {"$eq": dog_name}} if dog_name is not None else None
                raw = await asyncio.to_thread(
                    collection.query, query_embeddings=[vector], n_results=min(5, len(ranked)),
                    where=where, include=["distances"],
                )
                lookup = {item["chunk_id"]: (rank, item["distance"])
                          for rank, item in enumerate(ranked, 1)}
                exact_top = []
                for rank, item in enumerate(ranked[:5], 1):
                    index = positions[item["chunk_id"]]
                    metadata = records["metadatas"][index] or {}
                    exact_top.append({**item, "rank": rank, "dog_name": metadata.get("dog_name"),
                                      "section_title": metadata.get("section_title"),
                                      "content": records["documents"][index]})
                approximate = []
                for rank, (record_id, distance) in enumerate(zip(raw["ids"][0], raw["distances"][0]), 1):
                    exact_rank, exact_distance = lookup[record_id]
                    index = positions[record_id]
                    metadata = records["metadatas"][index] or {}
                    approximate.append({"rank": rank, "chunk_id": record_id, "distance": float(distance),
                                        "exact_rank": exact_rank,
                                        "distance_error": abs(float(distance) - exact_distance),
                                        "dog_name": metadata.get("dog_name"),
                                        "section_title": metadata.get("section_title"),
                                        "content": records["documents"][index]})
                target_rank, target_distance = lookup[TARGET_ID]
                target_approximate_rank = next((item["rank"] for item in approximate
                                                if item["chunk_id"] == TARGET_ID), None)
                report["cases"].append({
                    "language": language, "question": question, "filter": where,
                    "candidate_count": len(ranked), "target_exact_rank": target_rank,
                    "target_exact_distance": target_distance,
                    "target_near_tie_count": sum(abs(item["distance"] - target_distance) <= 1e-6
                                                 for item in ranked),
                    "target_hnsw_rank_in_top5": target_approximate_rank,
                    "top5_id_overlap_count": len({item["chunk_id"] for item in exact_top}
                                                  & {item["chunk_id"] for item in approximate}),
                    "exact_top5": exact_top, "hnsw_top5": approximate,
                })
        after = await asyncio.to_thread(collection.get, include=["documents", "metadatas", "embeddings"])
        if records_fingerprint(after) != before:
            raise RuntimeError("实验期间集合数据变化，本轮结果不可直接对照。")
        report["source_unchanged_verified"] = True
    finally:
        await container.shutdown()
    output_dir = Path(__file__).resolve().parents[2] / "output/rag_exact_comparisons" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report["report_path"] = str(output_dir / "exact_comparison_report.json")
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = asyncio.run(compare_exact_retrieval())
    for case in report["cases"]:
        scope = "限定金毛" if case["filter"] else "不限犬种"
        print(f"{case['language']} / {scope} / 比较{case['candidate_count']}条："
              f"逐条计算目标排名={case['target_exact_rank']}；"
              f"快速查找前五目标排名={case['target_hnsw_rank_in_top5']}；"
              f"前五共同记录数={case['top5_id_overlap_count']}")
    print("None表示没有进入快速查找前五，不表示库中没有该记录。")
    print("完整报告：", report["report_path"])
