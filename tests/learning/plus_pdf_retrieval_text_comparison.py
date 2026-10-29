"""控制变量比较：仅移除检索正文末尾的页面注意事项，观察目标图示排名。"""

import asyncio
import hashlib
import json
from pathlib import Path

from src.rag.indexers.chroma_indexer import RagChromaIndexer
from src.rag.schemas import RagChunk
from src.runtime.container.core import RuntimeContainer
from src.settings import settings
from tests.learning.plus_pdf_vector_demo import (
    PdfDemoEmbeddingProvider,
    PdfDemoVectorStoreProvider,
)


QUESTION = "狗狗耳朵向后、尾巴降低并抬起一只前爪，海报如何描述？"
TARGET_CAPTION = "Pausing with ears back, tail lowered, paw lifted"


def build_without_notes_chunks(bundle: dict) -> list[RagChunk]:
    """复制标准块，只移除检索正文末尾的注意事项，不修改关联原料。

    参数：bundle为rag_chunk_bundle.json的完整字典。
    返回值：新的RagChunk列表，除正文指定后缀外其他内容与基线一致。
        保留原chunk_id以便跨实验集合对应同一图示，仅用于独立对照集合；
        不代表应在正式库中忽略正文版本。后缀与预期不符时拒绝猜测删除范围。
    """
    chunks = []
    for raw_chunk in bundle["rag_chunks"]:
        chunk = RagChunk.model_validate(raw_chunk).model_copy(deep=True)
        notes = bundle["structured_content_by_chunk_id"][chunk.chunk_id]["structured_content"]["notes"]
        if notes:
            suffix = "\n\n页面适用限制与注意事项原文：\n" + "\n".join(notes)
            if not chunk.content.endswith(suffix):
                raise ValueError(f"注意事项后缀与预期不符，不自动裁剪：{chunk.chunk_id}")
            chunk.content = chunk.content[:-len(suffix)]
        chunks.append(chunk)
    return chunks


async def run_retrieval_text_comparison() -> dict:
    """对比包含和不包含注意事项的检索正文，返回排名报告并保存到文件。

    参数：无，固定使用本页13个图示、同一问题和项目当前向量模型。
    返回值：包含固定配置、目标块编号、两组完整排名、目标排名与检索正文的报告。
        只在独立演示集合中真实生成向量，不改原始JSON或正式库，不使用重排。
        标准答案只用于结果核对，不加入向量模型的输入。
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    bundle = json.loads((output_dir / "rag_chunk_bundle.json").read_text(encoding="utf-8-sig"))
    baseline = [RagChunk.model_validate(item) for item in bundle["rag_chunks"]]
    variants = {
        "with_notes": baseline,
        "without_notes": build_without_notes_chunks(bundle),
    }
    target_ids = [
        chunk.chunk_id for chunk in baseline
        if bundle["structured_content_by_chunk_id"][chunk.chunk_id]["structured_content"]["caption"] == TARGET_CAPTION
    ]
    if len(target_ids) != 1:
        raise ValueError("无法唯一定位本次核对的目标图示，请检查其原文说明。")
    target_id = target_ids[0]

    container = RuntimeContainer()
    embedding_provider = PdfDemoEmbeddingProvider()
    container.register("embedding", embedding_provider)
    collections = {}
    for mode, chunks in variants.items():
        identity = json.dumps({
            "mode": mode,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "provider": settings.embedding.provider,
            "model": settings.embedding.model_name,
            "normalize": settings.embedding.normalize_embeddings,
        }, ensure_ascii=False, sort_keys=True)
        collection_name = "pdf_text_ab_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        collections[mode] = collection_name
        container.register(mode, PdfDemoVectorStoreProvider(
            embedding_provider,
            directory=project_root / "output/learning/pdf_chroma",
            collection_name=collection_name,
        ))

    print(f"固定问题：{QUESTION}", flush=True)
    print(f"固定向量模型：{settings.embedding.model_name}；无重排、无聊天模型调用。", flush=True)
    try:
        await container.startup()
        results = {}
        for mode, chunks in variants.items():
            print(f"正在处理 {mode}，共{len(chunks)}块……", flush=True)
            database = container.get(mode).db
            index_report = RagChromaIndexer(
                vector_store=database, overwrite_existing=False,
            ).index_chunks(chunks)
            hits = database.similarity_search_with_score(QUESTION, k=len(chunks))
            ranking = []
            for rank, (document, distance) in enumerate(hits, start=1):
                chunk_id = document.metadata["chunk_id"]
                ranking.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "cosine_distance": float(distance),
                    "caption": bundle["structured_content_by_chunk_id"][chunk_id]["structured_content"]["caption"],
                    "retrieval_text": document.page_content,
                    "is_target": chunk_id == target_id,
                })
            target = next((item for item in ranking if item["is_target"]), None)
            results[mode] = {
                "collection_name": collections[mode],
                "index_report": index_report,
                "total_retrieval_characters": sum(len(chunk.content) for chunk in chunks),
                "target_rank": target["rank"] if target else None,
                "target_cosine_distance": target["cosine_distance"] if target else None,
                "target_in_top3": target is not None and target["rank"] <= 3,
                "ranking": ranking,
            }

        report = {
            "question": QUESTION,
            "embedding_model": settings.embedding.model_name,
            "embedding_provider": settings.embedding.provider,
            "normalize_embeddings": settings.embedding.normalize_embeddings,
            "target_chunk_id": target_id,
            "target_caption": TARGET_CAPTION,
            "target_notes_retained_in_structured_content": bundle["structured_content_by_chunk_id"][target_id]["structured_content"]["notes"],
            "variants": results,
        }
        (output_dir / "retrieval_text_comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    report = asyncio.run(run_retrieval_text_comparison())
    for mode, result in report["variants"].items():
        print(f"\n{mode}：目标排名={result['target_rank']}，进入前三={result['target_in_top3']}")
        print(f"检索正文总字符数：{result['total_retrieval_characters']}")
        for hit in result["ranking"][:3]:
            print(f"第{hit['rank']}名 | 距离={hit['cosine_distance']:.6f} | {hit['caption']}")
    print("\n完整报告保存在图片所在目录的 retrieval_text_comparison.json。")
    print("原始注意事项仍保留在结构化记录中；单个问题的结果不能代表整体质量。")
