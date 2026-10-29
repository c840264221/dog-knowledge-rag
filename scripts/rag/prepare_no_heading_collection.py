"""复制原向量建立无纯标题对照集合；默认只预览，--apply才创建及写入目标。"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from src.rag.evaluation.index_quality_audit import is_heading_only
from src.rag.indexers.chroma_collection_copy import copy_without_headings, records_fingerprint, source_hnsw_configuration
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


async def prepare_no_heading_collection(*, apply: bool = False) -> dict:
    """读取原集合记录并预览或创建对照集合，不加载任何模型。

    参数：apply为是否实际创建breed_no_heading_v1，默认False。
    返回值：包含源指纹、排除编号、预期条数、索引参数与复制核对结果的报告。
        源固定为langchain，不删除记录，不发布正式流量；失败保留报告与目标现场。
        原模型信息未知状态保留，不伪造embedding_signature（向量配置指纹）。
    """
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    source_container = RuntimeContainer()
    target_container = RuntimeContainer()
    source_container.register("source", VectorStoreProvider(
        None, collection_name="langchain", enable_memory=False, create_collection_if_not_exists=False,
    ))
    try:
        await source_container.startup()
        source = source_container.get("source").native_collection
        records = await asyncio.to_thread(source.get, include=["documents", "metadatas", "embeddings"])
        source_fingerprint = records_fingerprint(records)
        configuration = source_hnsw_configuration(source)
        excluded = [record_id for record_id, text in zip(records["ids"], records["documents"])
                    if is_heading_only(text or "")]
        root = Path(__file__).resolve().parents[2]
        output_dir = root / "output/rag_index_copies" / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "mode": "apply" if apply else "preview_only",
            "source_collection_name": "langchain", "target_collection_name": "breed_no_heading_v1",
            "source_record_count": len(records["ids"]), "excluded_heading_count": len(excluded),
            "expected_target_count": len(records["ids"]) - len(excluded),
            "excluded_record_ids": excluded, "source_records_fingerprint": source_fingerprint,
            "collection_configuration": configuration,
            "report_path": str(output_dir / "copy_report.json"),
            "notice": "只去除严格纯标题块，保留原向量及未知建库模型状态。目标重新构建近似索引图，结果变化不一定只来自候选范围变化。未修改原集合。",
        }
        report_path = Path(report["report_path"])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not apply:
            return report
        try:
            target_container.register("target", VectorStoreProvider(
                None, collection_name="breed_no_heading_v1", enable_memory=False,
                collection_configuration=configuration,
            ))
            await target_container.startup()
            target = target_container.get("target").native_collection
            report["copy_result"] = await asyncio.to_thread(copy_without_headings, source, target, records)
            report["verified"] = True
        except Exception as error:
            report["verified"] = False
            report["error_type"] = type(error).__name__
            raise
        finally:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        await target_container.shutdown()
        await source_container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="默认预览；--apply复制原向量，创建无标题对照集合。")
    parser.add_argument("--apply", action="store_true", help="实际创建并写入目标集合，不修改源集合。")
    args = parser.parse_args()
    report = asyncio.run(prepare_no_heading_collection(apply=args.apply))
    print(json.dumps({key: value for key, value in report.items() if key != "excluded_record_ids"}, ensure_ascii=False, indent=2))
    print("完整排除编号名单保存在报告中；本次未删除任何原集合记录，也未调用模型。")
