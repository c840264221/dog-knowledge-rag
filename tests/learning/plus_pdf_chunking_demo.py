"""PDF 入库学习第四步：将已人工修正的页面结构按图示切块，不生成向量。"""

import hashlib
import json
from pathlib import Path


def build_page_chunks(page_report: dict) -> list[dict]:
    """把页面结构按图示切块，并为每块保留分组、原始字段和页面来源。

    参数：
        page_report：上一阶段vision_extraction.json中的完整字典，包含
            source、page_number、image_path和structure；图文内容由用户先行核对。
    返回值：
        内容块字典列表，每块有chunk_id、text、content、metadata。
        有图示时一项图示生成一块，无图示但有正文时整组正文生成一块。
        text用于后续文字检索，content保留不同来源的内容；metadata保存页码
        和整页图片路径，并不包含图示裁剪坐标。此阶段不进行向量化或入库。
    """
    structure = page_report["structure"]
    chunks = []

    for section_index, section in enumerate(structure["sections"], start=1):
        # 没有图示的文字分组也能保留，避免只处理图片而丢掉正文。
        items = section["items"] or [None]
        for item_index, item in enumerate(items, start=1):
            caption = item["caption"] if item else None
            visual_description = item["visual_description"] if item else None
            if not (section["paragraphs"] or caption or visual_description):
                continue

            parts = []
            if structure["title"]:
                parts.append(f"页面标题：{structure['title']}")
            if section["heading"]:
                parts.append(f"分组标题：{section['heading']}")
            if section["paragraphs"]:
                parts.append("分组原文：\n" + "\n".join(section["paragraphs"]))
            if caption:
                parts.append(f"图示原文说明：{caption}")
            if visual_description:
                parts.append(f"视觉描述（模型生成，人工修正后输入）：{visual_description}")
            if structure["notes"]:
                parts.append("页面适用限制与注意事项原文：\n" + "\n".join(structure["notes"]))
            text = "\n\n".join(parts)

            identity = json.dumps({
                "source": page_report["source"],
                "page_number": page_report["page_number"],
                "section_index": section_index,
                "item_index": item_index,
                "text": text,
            }, ensure_ascii=False, sort_keys=True)
            chunk_id = "pdf_chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "content": {
                    "page_title": structure["title"],
                    "section_heading": section["heading"],
                    "section_paragraphs": list(section["paragraphs"]),
                    "caption": caption,
                    "visual_description": visual_description,
                    "notes": list(structure["notes"]),
                },
                "metadata": {
                    "source": page_report["source"],
                    "page_number": page_report["page_number"],
                    "image_path": page_report["image_path"],
                    "section_index": section_index,
                    "item_index": item_index if item else None,
                    "content_type": "figure" if item else "paragraph",
                    "review_status": "needs_chunk_review",
                    "extraction_uncertainties": list(structure["uncertainties"]),
                },
            })

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    report_path = output_dir / "vision_extraction.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    chunks = build_page_chunks(report)

    chunks_path = output_dir / "page_chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"已生成 {len(chunks)} 个内容块，尚未生成向量或写入数据库。")
    print(f"保存位置：{chunks_path}")
    print("\n各块对应的分组和图示原文：")
    for chunk in chunks:
        metadata = chunk["metadata"]
        content = chunk["content"]
        print(
            f"分组{metadata['section_index']} / 图示{metadata['item_index']} | "
            f"{content['section_heading']} | {content['caption'] or '（无图示原文说明）'}"
        )
    if chunks:
        print("\n第一个内容块的完整结构：")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))
