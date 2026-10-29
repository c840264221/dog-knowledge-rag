"""固定已保存的图示候选，用多语言重排模型比较重排前后的排名。"""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.settings import settings


def rerank_saved_candidates(*, question: str, candidates: list[dict], model) -> dict:
    """对固定问题与候选正文评分，保留原始排名后按模型评分降序排列。

    参数：question为已保存的检索问题；candidates为上一报告中的ranking列表；
        model为容器中的CrossEncoder（交叉编码重排模型），支持predict方法。
    返回值：包含本批推理耗时及重排记录的字典。记录保存原排名、新排名、
        检索距离、模型评分及正文；评分是predict返回值，不是概率或归一化分数。
        不重新检索、不改写正文，标准答案标记不进入模型输入。
    """
    pairs = [(question, item["retrieval_text"]) for item in candidates]
    started_at = perf_counter()
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    inference_seconds = perf_counter() - started_at
    if len(scores) != len(candidates):
        raise ValueError("评分数量与候选数量不一致，停止重排。")

    reranked = []
    for candidate, score in zip(candidates, scores):
        result = deepcopy(candidate)
        result["retrieval_rank"] = result.pop("rank")
        result["rerank_model_score"] = float(score)
        reranked.append(result)

    reranked.sort(key=lambda item: (-item["rerank_model_score"], item["retrieval_rank"]))
    for rank, item in enumerate(reranked, start=1):
        item["rerank_rank"] = rank

    return {
        "single_batch_inference_seconds": round(inference_seconds, 4),
        "ranking": reranked,
    }


async def run_pdf_rerank_comparison() -> dict:
    """读取两组已保存候选，复用重排提供者，保存各组前后对照报告。

    参数：无，固定读取本PDF演示的retrieval_text_comparison.json。
    返回值：含问题、模型名称、加载耗时、各组完整排名及目标排名的报告，
        同时写入rerank_comparison.json。只加载一个重排模型，无向量库写入、
        无聊天模型调用；首次推理可能含预热开销，不能作为P95延迟或正式基准。
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    comparison = json.loads(
        (output_dir / "retrieval_text_comparison.json").read_text(encoding="utf-8-sig")
    )
    model_name = "BAAI/bge-reranker-v2-m3"
    reranker_settings = settings.reranker.model_copy(update={"model_name": model_name})
    container = RuntimeContainer()
    container.register("reranker", RerankerProvider(reranker_settings=reranker_settings))

    print(f"准备重排模型：{model_name}；未缓存时可能下载。", flush=True)
    try:
        load_started_at = perf_counter()
        await container.startup()
        load_seconds = perf_counter() - load_started_at
        model = container.get("reranker").reranker

        groups = {}
        for mode, variant in comparison["variants"].items():
            candidates = variant["ranking"]
            print(f"重排 {mode}：固定{len(candidates)}条候选……", flush=True)
            result = rerank_saved_candidates(
                question=comparison["question"], candidates=candidates, model=model,
            )
            target = next(
                (item for item in result["ranking"] if item["chunk_id"] == comparison["target_chunk_id"]),
                None,
            )
            groups[mode] = {
                "candidate_count": len(candidates),
                "input_chunk_ids": [item["chunk_id"] for item in candidates],
                "target_retrieval_rank": variant["target_rank"],
                "target_rerank_rank": target["rerank_rank"] if target else None,
                "target_in_top3_after_rerank": target is not None and target["rerank_rank"] <= 3,
                **result,
            }

        report = {
            "question": comparison["question"],
            "embedding_model": comparison["embedding_model"],
            "reranker_model": model_name,
            "device": reranker_settings.device,
            "model_load_seconds": round(load_seconds, 4),
            "target_chunk_id": comparison["target_chunk_id"],
            "target_caption": comparison["target_caption"],
            "variants": groups,
        }
        (output_dir / "rerank_comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    report = asyncio.run(run_pdf_rerank_comparison())
    print(f"\n模型加载耗时：{report['model_load_seconds']}秒")
    for mode, result in report["variants"].items():
        print(
            f"\n{mode}：目标排名 {result['target_retrieval_rank']} → "
            f"{result['target_rerank_rank']}；本批推理耗时 {result['single_batch_inference_seconds']}秒"
        )
        for hit in result["ranking"][:3]:
            print(
                f"重排第{hit['rerank_rank']}名 | 原第{hit['retrieval_rank']}名 | "
                f"模型评分={hit['rerank_model_score']:.6f} | {hit['caption']}"
            )
    print("\n完整报告位于图片目录的 rerank_comparison.json；单样本结果不代表整体质量。")
