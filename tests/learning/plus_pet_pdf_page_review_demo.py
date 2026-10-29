"""准备逐页人工审核材料：页面总览、完整原始文字和审核清单，不自动清洗。"""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from tests.learning.plus_pdf_ingestion_demo import render_pdf_page


def make_page_overviews(images: list[dict], output_dir: Path) -> list[str]:
    """将页面图片按物理页序拼成缩略总览，帮助定位封面、正文及复杂页面。

    参数：images为渲染函数返回的图片记录列表；output_dir为总览保存目录。
    返回值：总览PNG的绝对路径列表；每张最多12页，标签P表示物理页码。
        缩略图只能浏览页面结构，不能替代原PDF上的正文核对。
    """
    overview_paths = []
    for start in range(0, len(images), 12):
        group = images[start:start + 12]
        rows = (len(group) + 3) // 4
        with Image.new("RGB", (960, rows * 350), "#e8e8e8") as canvas:
            draw = ImageDraw.Draw(canvas)
            for index, image_record in enumerate(group):
                x, y = (index % 4) * 240, (index // 4) * 350
                draw.text((x + 10, y + 8), f"P{image_record['page_number']}", fill="black")
                with Image.open(image_record["image_path"]) as image:
                    image.thumbnail((220, 310))
                    canvas.paste(image, (x + (240 - image.width) // 2, y + 30))
            overview_path = output_dir / f"overview_{start // 12 + 1:02d}.png"
            canvas.save(overview_path)
        overview_paths.append(str(overview_path))
    return overview_paths


def prepare_page_review(raw_pages_path: Path) -> dict:
    """根据已有普通提取报告准备逐页审核材料，不替用户作出页面取舍。

    参数：raw_pages_path为第一步生成的pages.json路径，不使用布局提取结果。
    返回值：包含原文件校验值、逐页审核字段、总览路径与清单位置的字典。
        每页默认待审核，是否入知识处理为null（未决定），不删除任何页面。
        完整文字另存为可阅读文件；原PDF和原提取报告不修改，不调用模型。
    """
    raw_pages_path = raw_pages_path.resolve()
    raw = json.loads(raw_pages_path.read_text(encoding="utf-8-sig"))
    pages = raw["pages"]
    if not pages:
        raise ValueError("原报告中没有页面。")
    source = Path(pages[0]["source"]).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != raw["source_sha256"]:
        raise ValueError("原PDF校验值发生变化，不能沿用旧文字报告审核。")
    for number, page in enumerate(pages, start=1):
        if page["page_number"] != number or Path(page["source"]).resolve() != source:
            raise ValueError("原报告页序或来源异常，不能建立逐页审核对应。")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pet_corpus_v1/page_reviews" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    images = []
    decisions = []
    text_sections = []
    for page in pages:
        number = page["page_number"]
        print(f"准备物理第{number}/{len(pages)}页……", flush=True)
        image = render_pdf_page(
            str(source), output_dir=str(output_dir / "page_images"),
            page_number=number, scale=1.2,
        )
        images.append(image)
        decisions.append({
            "page_number": number,
            "image_path": image["image_path"],
            "text_preview": page["text"][:400],
            "page_type": "pending",
            "include_in_knowledge": None,
            "reading_order": "pending",
            "notes": "",
        })
        text_sections.append(f"===== PHYSICAL PAGE {number} =====\n{page['text']}")
    text_path = output_dir / "plain_text_by_page.txt"
    text_path.write_text("\n\n".join(text_sections), encoding="utf-8")
    report = {
        "doc_id": raw["doc_id"],
        "source": str(source),
        "source_sha256": source_hash,
        "raw_pages_path": str(raw_pages_path),
        "plain_text_path": str(text_path),
        "overview_paths": make_page_overviews(images, output_dir),
        "allowed_page_types": ["pending", "body", "non_knowledge", "mixed"],
        "allowed_reading_order": ["pending", "ok", "problem", "not_applicable"],
        "reviewed_by": "",
        "reviewed_at": "",
        "pages": decisions,
        "review_path": str(output_dir / "page_review.json"),
        "notice": "这是待填写的审核记录，不是已实现的页面过滤器；混合页不能直接整页删除。",
    }
    Path(report["review_path"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    raw_path = input("粘贴普通提取的pages.json完整路径：").strip().strip('"')
    report = prepare_page_review(Path(raw_path))
    print("\n页面总览：")
    for path in report["overview_paths"]:
        print(path)
    print(f"完整文字：{report['plain_text_path']}")
    print(f"待填写的审核清单：{report['review_path']}")
    print("先看总览区分页面用途，再对照原PDF和完整文字检查正文顺序。")
    print("所有页面仍保留，脚本没有自动分类、去噪、重排、切块或入库。")
