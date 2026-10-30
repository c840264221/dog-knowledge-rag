"""PDF 入库学习第五步：适配项目 RagChunk，并保存关联的结构化内容。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.rag.schemas import RagChunk


def adapt_pdf_chunks(demo_chunks: list[dict]) -> dict:
    """转换演示块，并按同一块编号保留完整结构化原料。

    参数：
        demo_chunks：page_chunks.json中的块列表，每项包含chunk_id、text、
            content（结构化字典）和metadata。
    返回值：
        包含rag_chunks（RagChunk对象列表）和structured_content_by_chunk_id
        （按chunk_id查询的完整结构化记录）的字典。
        标准块只携带简单标量元数据；完整原料与原始元数据另行保留。
        此函数不修改输入，不调用模型，不生成向量，也不写数据库。
    """
    rag_chunks = []
    structured_records = {}

    for chunk_index, demo_chunk in enumerate(demo_chunks):
        chunk_id = demo_chunk["chunk_id"]
        if chunk_id in structured_records:
            raise ValueError(f"块编号重复，不能覆盖关联记录：{chunk_id}")

        original_metadata = demo_chunk["metadata"]
        structured_content = demo_chunk["content"]
        source = str(Path(original_metadata["source"]).resolve())
        doc_id = "pdf_doc_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        # 只把需要随检索结果返回的简单字段带进标准块。
        metadata = {
            "page_number": original_metadata["page_number"],
            "image_path": original_metadata["image_path"],
            "section_index": original_metadata["section_index"],
            "content_type": original_metadata["content_type"],
            "review_status": original_metadata["review_status"],
        }
        if original_metadata["item_index"] is not None:
            metadata["item_index"] = original_metadata["item_index"]
        if structured_content["section_heading"] is not None:
            metadata["section_heading"] = structured_content["section_heading"]

        rag_chunks.append(RagChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=demo_chunk["text"],
            chunk_index=chunk_index,
            source=source,
            title=structured_content["page_title"] or "",
            metadata=metadata,
        ))

        # 完整保留，不将嵌套字典、列表强行塞入向量库元数据。
        structured_records[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "structured_content": deepcopy(structured_content),
            "original_metadata": deepcopy(original_metadata),
        }

    return {
        "rag_chunks": rag_chunks,
        "structured_content_by_chunk_id": structured_records,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    input_path = output_dir / "page_chunks.json"
    if not input_path.is_file():
        raise SystemExit("请先运行plus_pdf_chunking_demo.py生成page_chunks.json，再运行本演示。")

    demo_chunks = json.loads(input_path.read_text(encoding="utf-8-sig"))
    adapted = adapt_pdf_chunks(demo_chunks)
    payload = {
        "rag_chunks": [chunk.model_dump(mode="json") for chunk in adapted["rag_chunks"]],
        "structured_content_by_chunk_id": adapted["structured_content_by_chunk_id"],
    }
    output_path = output_dir / "rag_chunk_bundle.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"标准块数量：{len(payload['rag_chunks'])}")
    print(f"结构化记录数量：{len(payload['structured_content_by_chunk_id'])}")
    print(f"转换结果已保存：{output_path}")
    print("本次没有生成向量，也没有写入向量库。")

    if payload["rag_chunks"]:
        first_chunk = payload["rag_chunks"][0]
        linked_record = payload["structured_content_by_chunk_id"][first_chunk["chunk_id"]]
        print("\n第一个标准块及按chunk_id回查到的结构化记录：")
        print(json.dumps({
            "rag_chunk": first_chunk,
            "linked_record": linked_record,
        }, ensure_ascii=False, indent=2))
