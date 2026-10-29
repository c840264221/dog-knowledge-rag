"""多页宠物PDF学习第一步：校验文件、逐页提取原文并生成抽查图片，不入库。"""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from tests.learning.plus_pdf_ingestion_demo import read_pdf_pages, render_pdf_page


def inspect_cat_home_pdf() -> dict:
    """校验已下载指南，保存每页完整原文及三个抽查页面的图片。

    参数：无；使用项目内固定样本与下载记录，避免误读之前的海报。
    返回值：含源文件、校验值、总页数、逐页统计、抽查图片及输出路径的报告。
        原始文字保存在pages.json，不清洗、不切块、不翻译、不调用模型或向量库。
        物理页码从1开始，不能假定与页面印刷页码一致；字符数不代表提取质量。
    """
    project_root = Path(__file__).resolve().parents[2]
    corpus_dir = project_root / "data/learning/pet_corpus_v1"
    record = json.loads((corpus_dir / "bringing_cat_home_download.json").read_text(encoding="utf-8-sig"))
    pdf_path = project_root / record["local_path"]
    payload = pdf_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if not payload.startswith(b"%PDF-"):
        raise ValueError("文件头不是PDF，请检查是否下载成了网页。")
    if actual_hash != record["sha256"] or len(payload) != record["byte_count"]:
        raise ValueError("本地文件与下载记录不一致，请先核对来源，不直接覆盖校验记录。")

    pages = read_pdf_pages(str(pdf_path))
    run_id = uuid4().hex
    output_dir = project_root / "output/learning/pet_corpus_v1/cp_bringing_cat_home" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    pages_path = output_dir / "pages.json"
    pages_path.write_text(json.dumps({
        "doc_id": record["doc_id"], "source_url": record["requested_url"],
        "source_sha256": actual_hash, "pages": pages,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    page_stats = [{
        "page_number": page["page_number"],
        "character_count": len(page["text"]),
        "non_whitespace_character_count": len("".join(page["text"].split())),
        "has_replacement_character": "\ufffd" in page["text"],
    } for page in pages]
    sample_numbers = [number for number in (4, 8, 12) if number <= len(pages)]
    images = [render_pdf_page(
        str(pdf_path), output_dir=str(output_dir / "page_images"),
        page_number=number, scale=1.5,
    ) for number in sample_numbers]
    report = {
        "run_id": run_id,
        "doc_id": record["doc_id"],
        "source": str(pdf_path),
        "source_sha256": actual_hash,
        "page_count": len(pages),
        "empty_text_pages": [item["page_number"] for item in page_stats if item["non_whitespace_character_count"] == 0],
        "page_stats": page_stats,
        "sample_images": images,
        "pages_path": str(pages_path),
        "report_path": str(output_dir / "extraction_report.json"),
        "review_status": "pending_human_review",
        "review_questions": [
            "原图标题、列表及其顺序是否与提取文字一致？",
            "页眉页脚是否混入正文，段落是否跨页？",
            "成年猫与幼猫的条件是否仍然可辨？",
            "是否有仅凭文字无法表达的图示、标注或布局关系？"
        ],
    }
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for number in sample_numbers:
        print(f"\n物理第{number}页，原文预览（最多1800字符）：")
        print(pages[number - 1]["text"][:1800])
    return report


if __name__ == "__main__":
    report = inspect_cat_home_pdf()
    print(f"\n总页数：{report['page_count']}")
    print(f"没有提取到非空白文字的页面：{report['empty_text_pages']}")
    print("\n每页字符数（不是质量评分）：")
    for item in report["page_stats"]:
        print(f"第{item['page_number']}页：{item['character_count']}字符；去空白后{item['non_whitespace_character_count']}字符")
    print(f"\n完整原文：{report['pages_path']}")
    print(f"检查报告：{report['report_path']}")
    print("抽查图片：")
    for image in report["sample_images"]:
        print(image["image_path"])
    print("请将物理第4、8、12页原图与pages.json对应文字对照。当前没有清洗、切块或入库。")
