"""多页PDF提取对照：保留已有原始文字，另外生成布局提取结果供人工检查。"""

from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from tests.learning.plus_pdf_ingestion_demo import render_pdf_page


def compare_pdf_layout(raw_pages_path: Path) -> dict:
    """将已保存的普通提取与同一PDF的布局提取并列保存，不替换原始文字。

    参数：raw_pages_path为上一步生成的pages.json，含源文件校验值与逐页文字。
    返回值：包含原始报告路径、文件校验值、各页两种提取文字、差异提示、
        抽查图片及输出位置的字典。所有页面保留；不自动认定哪种结果更好，
        不清洗、不切块、不调用模型或向量库。页码均指从1开始的物理页码。
    """
    raw_pages_path = raw_pages_path.resolve()
    raw = json.loads(raw_pages_path.read_text(encoding="utf-8-sig"))
    pages = raw["pages"]
    if not pages:
        raise ValueError("原始报告没有页面，无法进行对照。")
    source = Path(pages[0]["source"]).resolve()
    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    if source_hash != raw["source_sha256"]:
        raise ValueError("PDF已发生变化，不能与旧提取结果直接比较。")
    if any(Path(page["source"]).resolve() != source for page in pages):
        raise ValueError("原始报告包含不同源文件，无法进行本次单文档对照。")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pet_corpus_v1/layout_comparisons" / uuid4().hex
    comparisons = []
    with source.open("rb") as stream:
        reader = PdfReader(stream)
        if len(reader.pages) != len(pages):
            raise ValueError("原始提取报告与PDF的页面数量不同。")
        for number, (saved_page, pdf_page) in enumerate(zip(pages, reader.pages), start=1):
            if saved_page["page_number"] != number:
                raise ValueError("原始报告页码不连续或顺序异常。")
            plain_text = saved_page["text"]
            layout_text = pdf_page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
                layout_mode_strip_rotated=False,
            ) or ""
            # 忽略空白，对比每个字符的出现次数；不能据此判断语义或阅读顺序。
            plain_chars = Counter(char for char in plain_text if not char.isspace())
            layout_chars = Counter(char for char in layout_text if not char.isspace())
            comparisons.append({
                "page_number": number,
                "plain_text": plain_text,
                "layout_text": layout_text,
                "same_text_ignoring_whitespace": "".join(plain_text.split()) == "".join(layout_text.split()),
                "same_character_counts_ignoring_order": plain_chars == layout_chars,
                "plain_only_character_counts": dict(plain_chars - layout_chars),
                "layout_only_character_counts": dict(layout_chars - plain_chars),
                "review_status": "pending_human_review",
            })

    output_dir.mkdir(parents=True, exist_ok=False)
    sample_numbers = [number for number in (4, 8, 12, 20) if number <= len(pages)]
    images = []
    for number in sample_numbers:
        item = comparisons[number - 1]
        for mode in ("plain", "layout"):
            (output_dir / f"page_{number:03d}_{mode}.txt").write_text(
                item[f"{mode}_text"], encoding="utf-8",
            )
        images.append(render_pdf_page(
            str(source), output_dir=str(output_dir / "page_images"),
            page_number=number, scale=1.5,
        ))

    report = {
        "raw_pages_path": str(raw_pages_path),
        "source": str(source),
        "source_sha256": source_hash,
        "page_count": len(comparisons),
        "layout_options": {
            "extraction_mode": "layout", "layout_mode_space_vertically": False,
            "layout_mode_strip_rotated": False,
        },
        "pages": comparisons,
        "sample_images": images,
        "report_path": str(output_dir / "layout_comparison.json"),
        "review_status": "pending_human_review",
        "notice": "字符相同不能证明顺序正确；布局提取也可能出错，需要对照原页面。",
    }
    Path(report["report_path"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    path_text = input("粘贴上一步生成的pages.json完整路径：").strip().strip('"')
    report = compare_pdf_layout(Path(path_text))
    print(f"\n对照页数：{report['page_count']}")
    for item in report["pages"]:
        print(
            f"第{item['page_number']}页 | 忽略空白后顺序相同：{item['same_text_ignoring_whitespace']}"
            f" | 忽略顺序后字符及次数相同：{item['same_character_counts_ignoring_order']}"
        )
    print(f"\n报告：{report['report_path']}")
    print("同目录有第4、8、12、20页的plain与layout文字文件，以及page_images中的页面图片。")
    print("先看第20页顶部文字是否回到前面，再检查正文标题、列表、条件有无改变或遗漏。")
    print("这是提取方式对照，不是自动修复。原PDF和旧pages.json均未修改。")
