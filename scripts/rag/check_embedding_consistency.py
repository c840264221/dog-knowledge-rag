"""只读抽样检查旧向量与当前模型配置的数值一致性，不写入集合。"""

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.rag.evaluation.embedding_consistency import compare_vectors
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


SAMPLES = [
    ("langchain", "chunk_46d049fe6e716d96", "金毛运动正文"),
    ("langchain", "chunk_a5c203eb7763cd55", "迷你雪纳瑞饮食正文"),
    ("pet_care_v1", "chunk_96d1e4328c7d42bb", "新猫用品第4页，对照样本"),
]


async def check_embedding_consistency() -> dict:
    """读取固定三条记录并用当前模型重新向量化原样正文，输出逐条比较报告。

    参数：无；模型配置读取项目settings，样本由SAMPLES固定，不调用聊天模型。
    返回值：含模型配置、样本编号、原文哈希及数值指标的报告，并保存独立JSON文件。
        集合或样本缺失直接报错，不创建集合；新向量仅用于比较，不写回数据库。
    """
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    container = RuntimeContainer()
    embedding_provider = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding_provider)
    for name in dict.fromkeys(sample[0] for sample in SAMPLES):
        container.register(name, VectorStoreProvider(
            None, collection_name=name, enable_memory=False,
            create_collection_if_not_exists=False,
        ))
    report = {
        "embedding_config": {"provider": settings.embedding.provider,
                             "model": settings.embedding.model_name,
                             "normalize": settings.embedding.normalize_embeddings,
                             "device": settings.embedding.device},
        "samples": [],
        "notice": "只比较少量样本，不证明旧模型身份；不写回向量库。差异也可能来自输入预处理、模型版本或运行精度。",
    }
    try:
        await container.startup()
        texts = []
        stored_vectors = []
        for collection_name, record_id, label in SAMPLES:
            collection = container.get(collection_name).native_collection
            records = await asyncio.to_thread(
                collection.get, ids=[record_id], include=["documents", "metadatas", "embeddings"],
            )
            if records["ids"] != [record_id]:
                raise ValueError(f"样本不存在：{collection_name}/{record_id}")
            text = records["documents"][0]
            if not isinstance(text, str) or not text.strip() or records["embeddings"] is None:
                raise ValueError(f"样本正文或向量无效：{record_id}")
            texts.append(text)
            stored_vectors.append(records["embeddings"][0])
            report["samples"].append({
                "collection_name": collection_name, "chunk_id": record_id, "label": label,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text, "metadata": records["metadatas"][0],
            })
        recomputed = await asyncio.to_thread(embedding_provider.embedding.embed_documents, texts)
        if len(recomputed) != len(texts):
            raise ValueError("模型返回向量数量与正文数量不一致。")
        for sample, stored, fresh in zip(report["samples"], stored_vectors, recomputed):
            sample["comparison"] = compare_vectors(stored, fresh)
    finally:
        await container.shutdown()
    output_dir = Path(__file__).resolve().parents[2] / "output/rag_embedding_checks" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report["report_path"] = str(output_dir / "embedding_consistency_report.json")
    Path(report["report_path"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    report = asyncio.run(check_embedding_consistency())
    for sample in report["samples"]:
        print(sample["label"], json.dumps(sample["comparison"], ensure_ascii=False))
    print("完整报告：", report["report_path"])
    print(report["notice"])
