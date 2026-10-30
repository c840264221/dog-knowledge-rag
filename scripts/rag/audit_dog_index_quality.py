"""不加载模型的旧犬种索引审计：只读取记录，输出标题噪声和目标运动证据。"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from src.rag.evaluation.index_quality_audit import audit_index_records
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.settings import settings


async def run_index_audit() -> dict:
    """读取langchain原集合，保存记录快照及审计报告，不执行向量搜索。

    参数：无；使用已核实的项目数据目录和原集合名，不扫描用户记忆库。
    返回值：包含统计、完整命中记录和输出路径的审计字典。
        不注册向量模型，只使用现有提供者读取文档；不创建集合、不写入或删除
        向量记录，不改主图。Chroma连接不是数据库级只读权限，入口只调用get。
    """
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    container = RuntimeContainer()
    container.register("audit_vectorstore", VectorStoreProvider(
        None, collection_name="langchain", enable_memory=False,
        create_collection_if_not_exists=False,
    ))
    try:
        await container.startup()
        database = container.get("audit_vectorstore").db
        records = await asyncio.to_thread(database.get, include=["documents", "metadatas"])
        report = audit_index_records(records)
        root = Path(__file__).resolve().parents[2]
        output_dir = root / "output/rag_index_audits" / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        snapshot_path = output_dir / "index_records_snapshot.json"
        snapshot_path.write_text(json.dumps({
            "collection_name": "langchain",
            "ids": records["ids"], "documents": records["documents"], "metadatas": records["metadatas"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        report.update(
            collection_name="langchain",
            snapshot_path=str(snapshot_path),
            report_path=str(output_dir / "index_quality_report.json"),
        )
        Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        await container.shutdown()


if __name__ == "__main__":
    report = asyncio.run(run_index_audit())
    print(f"\n集合：{report['collection_name']}")
    print(f"总记录数：{report['record_count']}；空正文数：{report['empty_content_count']}")
    print(f"纯标题块数：{report['heading_only_count']}；占比：{report['heading_only_fraction']:.2%}")
    print(f"已识别受影响文档数：{report['affected_document_count']}")
    print("\n纯标题分布：")
    for item in report["heading_patterns"]:
        print(f"{item['count']}条 | {item['content']}")
    print(f"\n金毛记录数：{report['target_record_count']}；明确运动章节正文数：{report['explicit_exercise_record_count']}")
    for record in report["explicit_exercise_records"]:
        print(f"\n运动证据编号：{record['record_id']}\n{record['content']}")
    print(f"\n原始记录快照：{report['snapshot_path']}")
    print(f"审计报告：{report['report_path']}")
    print("未调用任何模型，未修改向量记录。下一步再根据报告设计排除噪声的对照实验。")
