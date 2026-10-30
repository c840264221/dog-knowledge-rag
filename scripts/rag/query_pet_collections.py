"""项目双集合查询入口：原犬种集合和养护集合召回，再做RRF；不调用聊天或重排模型。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.retrievers.multi_collection_retriever import MultiCollectionRetriever
from src.rag.schemas import RagQuery
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


async def query_pet_collections(
    question: str, *, top_k_per_collection: int = 5, allow_legacy_model: bool = False,
    subquestions: list[str] | None = None,
    breed_collection_name: str = "langchain",
    routed_comparison: bool = False,
) -> dict:
    """通过项目多集合检索器查询两个已存在的公共知识集合并保存报告。

    参数：question为本轮问题；top_k_per_collection为每集合候选数；
        allow_legacy_model为是否明确允许以当前向量配置查询未登记模型的旧资料。
        subquestions为可选人工子问题列表；传入时在同一容器中额外运行拆分对照，
            不调用模型拆解、不跨子问题融合，额外查询成本单独登记。
        breed_collection_name为犬种集合名称，只允许原集合或无纯标题对照集合。
        routed_comparison为是否运行固定金毛/新猫问题的人工路由对照，不自动理解其他问题。
    返回值：包含各集合候选、融合结果、耗时及模型核验说明的报告；不写入向量记录。
        已知配置不匹配始终拒绝，旧库未登记时默认拒绝，不把同维度当成兼容证明。
        犬种集合与pet_care_v1均不创建空集合，不涉及主图路由。
    """
    query = RagQuery(question=question.strip(), top_k=top_k_per_collection, intent="pet_care")
    if not query.question:
        raise ValueError("问题不能为空。")
    if routed_comparison and (
        question.strip() != "金毛日常运动有什么要求？新猫到家前要准备哪些用品？"
        or subquestions != ["金毛日常运动有什么要求？", "新猫到家前要准备哪些用品？"]
    ):
        raise ValueError("人工路由对照只适用于本案例固定整题和两个子问题，不能用于任意问题。")
    subqueries = {}
    if subquestions is not None:
        if not subquestions or any(not text.strip() for text in subquestions):
            raise ValueError("拆分对照需要非空子问题列表。")
        subqueries = {
            f"q{index}": RagQuery(question=text.strip(), top_k=top_k_per_collection, intent=query.intent)
            for index, text in enumerate(subquestions, start=1)
        }
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    if breed_collection_name not in {"langchain", "breed_no_heading_v1"}:
        raise ValueError("犬种集合不在本次实验白名单内。")
    collection_names = (breed_collection_name, "pet_care_v1")
    model_config = {
        "provider": settings.embedding.provider, "model": settings.embedding.model_name,
        "normalize": settings.embedding.normalize_embeddings,
    }
    signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    for name in collection_names:
        container.register(name, VectorStoreProvider(
            embedding, collection_name=name, enable_memory=False,
            create_collection_if_not_exists=False,
        ))
    try:
        started_at = perf_counter()
        await container.startup()
        startup_seconds = perf_counter() - started_at
        retrievers = {}
        verification = {}
        for name in collection_names:
            database = container.get(name).db
            records = await asyncio.to_thread(database.get, include=["metadatas"])
            if not records["ids"]:
                raise ValueError(f"集合{name}没有资料。")
            signatures = [(metadata or {}).get("embedding_signature") for metadata in records["metadatas"]]
            if any(value is not None and value != signature for value in signatures):
                raise ValueError(f"集合{name}存在已知向量配置不匹配，不允许继续。")
            unsigned_count = signatures.count(None)
            if unsigned_count and (name != breed_collection_name or not allow_legacy_model):
                raise ValueError(
                    f"集合{name}有记录未登记建库模型。旧犬种库仅可通过"
                    "--allow-legacy-model显式允许实验查询；不代表已证明模型兼容。"
                )
            verification[name] = {
                "record_count": len(records["ids"]),
                "unsigned_record_count": unsigned_count,
                "status": "legacy_unverified_explicitly_allowed" if unsigned_count else "recorded_signature_matches",
            }
            if unsigned_count:
                print(f"注意：{name}旧集合缺少建库模型标记，按显式授权使用当前配置做实验；模型兼容性仍未核实。", flush=True)
            retrievers[name] = MetadataFilterRetriever(database, allow_filter_relaxation=False)
        started_at = perf_counter()
        multi_retriever = MultiCollectionRetriever(retrievers)
        result = await multi_retriever.async_retrieve(query)
        retrieval_seconds = perf_counter() - started_at
        root = Path(__file__).resolve().parents[2]
        output_dir = root / "output/rag_queries" / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "run_id": output_dir.name, "embedding_config": model_config,
            "model_verification": verification,
            "top_k_per_collection": top_k_per_collection,
            "startup_seconds": round(startup_seconds, 4),
            "retrieval_and_fusion_seconds": round(retrieval_seconds, 4),
            **result.model_dump(mode="json"),
            "report_path": str(output_dir / "multi_collection_report.json"),
            "review_status": "needs_relevance_review",
            "notice": "无过滤、无重排、无最终上下文选择；RRF同分不代表同等相关，未截断融合候选。",
        }
        Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if subqueries:
            report["subquestion_comparison"] = {
                "source": "manual_controlled_experiment",
                "status": "running",
                "original_question": query.question,
                "subquestions": {key: value.question for key, value in subqueries.items()},
                "whole_question_search_requests": len(collection_names),
                "split_question_search_requests": len(collection_names) * len(subqueries),
                "whole_question_candidate_budget": len(collection_names) * top_k_per_collection,
                "split_question_candidate_budget": len(collection_names) * len(subqueries) * top_k_per_collection,
                "notice": "每次查询的候选数不变，但拆分增加查询次数与总候选预算；不能将改善全部归因于措辞，也不能用一次顺序运行证明速度变化。",
            }
            Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            started_at = perf_counter()
            try:
                split_results = await multi_retriever.async_retrieve_subqueries(subqueries)
            except Exception as error:
                report["subquestion_comparison"].update(status="failed", error_type=type(error).__name__)
                Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
            report["subquestion_comparison"].update(
                status="completed",
                retrieval_and_fusion_seconds=round(perf_counter() - started_at, 4),
                results_by_subquestion={key: value.model_dump(mode="json") for key, value in split_results.items()},
            )
            Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if routed_comparison:
            routes = {
                "q1": {breed_collection_name: subqueries["q1"].model_copy(
                    update={"filters": {"dog_name": {"$eq": "Golden Retriever"}}}, deep=True,
                )},
                "q2": {"pet_care_v1": subqueries["q2"].model_copy(deep=True)},
            }
            report["routed_comparison"] = {
                "source": "manual_controlled_experiment", "status": "running",
                "routes": {key: {name: item.model_dump(mode="json") for name, item in targets.items()}
                           for key, targets in routes.items()},
                "search_requests": 2, "candidate_budget": 2 * top_k_per_collection,
                "notice": "本组同时改变集合选择和犬种条件，不能把全部变化归因于过滤；单变量过滤效果参照之前四组距离实验。无模型路由、无翻译、无重排。",
            }
            Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                routed = await multi_retriever.async_retrieve_routed_subqueries(routes)
            except Exception as error:
                report["routed_comparison"].update(status="failed", error_type=type(error).__name__)
                Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
            report["routed_comparison"].update(
                status="completed",
                results_by_subquestion={key: value.model_dump(mode="json") for key, value in routed.items()},
            )
            Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询原犬种与新养护集合，再融合候选。")
    parser.add_argument("--question", help="本次问题，不填时手动输入。")
    parser.add_argument("--top-k-per-collection", type=int, default=5, help="每集合召回块数，默认5。")
    parser.add_argument("--allow-legacy-model", action="store_true", help="允许按当前模型查询建库配置未登记的旧犬种集合，仅用于实验。")
    parser.add_argument("--subquestion", action="append", help="人工提供一个子问题；重复此参数可提供多个，额外运行拆分对照。")
    parser.add_argument("--breed-collection", choices=["langchain", "breed_no_heading_v1"], default="langchain", help="犬种集合名称，默认原集合；可显式选择无纯标题对照集合。")
    parser.add_argument("--routed-comparison", action="store_true", help="额外运行本案例固定问题的人工集合路由与严格犬种过滤对照。")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("请输入双集合检索问题：")
    report = asyncio.run(query_pet_collections(
        question, top_k_per_collection=args.top_k_per_collection,
        allow_legacy_model=args.allow_legacy_model,
        subquestions=args.subquestion,
        breed_collection_name=args.breed_collection,
        routed_comparison=args.routed_comparison,
    ))
    for collection in report["collections"]:
        print(f"\n集合：{collection['collection_name']}，检索耗时：{collection['retrieval_seconds']}秒")
        for rank, item in enumerate(collection["chunks"], start=1):
            chunk = item["chunk"]
            print(f"第{rank}名 | {chunk['title']} | 页码={chunk['metadata'].get('page_number')} | 块编号={chunk['chunk_id']}")
    print("\n融合候选（全部保留，尚未重排）：")
    for rank, item in enumerate(report["fused_candidates"], start=1):
        sources = ", ".join(f"{part['collection_name']}原第{part['rank']}名" for part in item["contributions"])
        print(f"\n第{rank}名 | RRF={item['rrf_score']:.6f} | {sources}")
        print(item["chunk"]["content"])
    if "subquestion_comparison" in report:
        comparison = report["subquestion_comparison"]
        print("\n===== 人工子问题拆分对照：各问题单独保存，不跨问题融合 =====")
        for query_id, result in comparison["results_by_subquestion"].items():
            print(f"\n{query_id}：{result['question']}")
            for collection in result["collections"]:
                print(f"集合：{collection['collection_name']}")
                for rank, item in enumerate(collection["chunks"], start=1):
                    chunk = item["chunk"]
                    print(f"第{rank}名 | {chunk['title']} | 页码={chunk['metadata'].get('page_number')} | 块编号={chunk['chunk_id']}")
                    print(chunk["content"][:240])
            print(f"本问题融合后保留：{len(result['fused_candidates'])}条。完整正文见报告。")
        print(f"整题查询次数：{comparison['whole_question_search_requests']}；拆分查询次数：{comparison['split_question_search_requests']}")
    if "routed_comparison" in report:
        print("\n===== 人工集合路由与严格犬种过滤对照 =====")
        for query_id, result in report["routed_comparison"]["results_by_subquestion"].items():
            print(f"\n{query_id}：{result['question']}")
            for collection in result["collections"]:
                print(f"集合：{collection['collection_name']}")
                for rank, item in enumerate(collection["chunks"], 1):
                    chunk = item["chunk"]
                    print(f"第{rank}名 | 犬种={chunk['metadata'].get('dog_name')} | "
                          f"章节={chunk['metadata'].get('section_title')} | "
                          f"页码={chunk['metadata'].get('page_number')} | {chunk['chunk_id']}")
    print(f"\n完整报告：{report['report_path']}")
    print("RRF只融合排名，不证明资料足以回答；当前未生成答案，也未修改向量记录。")
