"""PDF 入库学习：提取页面文字、渲染页面图片，保留来源与页码。"""

import json
from pathlib import Path

from pypdf import PdfReader


def read_pdf_pages(pdf_path: str) -> list[dict]:
    """读取本地 PDF，按页面返回文字和来源信息。

    参数：
        pdf_path：本地 PDF 文件路径；可使用 Windows 绝对路径。
    返回值：
        按原始页序排列的字典列表，每项包含 source（绝对路径）、
        page_number（从 1 开始的物理页码）和 text（当前页提取的文字）。
        未提取到文字的页面仍保留，text 为空字符串，不自动判断为扫描页。
        文件不存在或 PDF 无法读取时，由文件系统或解析器抛出异常。
    """
    source = Path(pdf_path).expanduser().resolve()
    pages = []

    with source.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "source": str(source),
                "page_number": page_number,
                "text": text,
            })

    return pages


def render_pdf_page(
    pdf_path: str,
    *,
    output_dir: str,
    page_number: int = 1,
    scale: float = 2.0,
) -> dict:
    """将一页 PDF 渲染为 PNG 图片，保留该图片对应的来源信息。

    参数：
        pdf_path：本地 PDF 文件路径。
        output_dir：生成图片的保存目录，不修改原始 PDF。
        page_number：要渲染的物理页码，从 1 开始，默认第一页。
        scale：渲染倍率，默认 2.0，即约 144 DPI；越大越占内存和磁盘。
    返回值：
        字典包含 source（原始 PDF 绝对路径）、page_number（页码）、
        image_path（生成图片的绝对路径）、width 和 height（像素尺寸）
        以及 scale（渲染倍率）。此过程不执行 OCR 或图像语义识别。
    """
    import pypdfium2 as pdfium

    source = Path(pdf_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if scale <= 0:
        raise ValueError("渲染倍率必须大于0。")

    with pdfium.PdfDocument(str(source)) as document:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"页码必须在1到{len(document)}之间。")

        destination.mkdir(parents=True, exist_ok=True)
        image_path = destination / f"page_{page_number:03d}.png"
        page = document[page_number - 1]
        bitmap = page.render(scale=scale)
        try:
            with bitmap.to_pil() as image:
                image.save(image_path, format="PNG")
                width, height = image.size
        finally:
            bitmap.close()
        # 退出 with 时，document 会一并关闭所属页面，释放 PDF 文件资源。

    return {
        "source": str(source),
        "page_number": page_number,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "scale": scale,
    }


if __name__ == "__main__":
    pdf_path = input("请输入本地 PDF 完整路径：").strip().strip('"')
    pages = read_pdf_pages(pdf_path)

    print(f"总页数：{len(pages)}")
    print(f"未提取到有效文字的页码：{[p['page_number'] for p in pages if not p['text'].strip()]}")

    # 仅缩短控制台预览，pages 中仍保留全部页面的完整提取文字。
    for page in pages[:2]:
        print(f"\n第 {page['page_number']} 页 | 来源：{page['source']}")
        print(f"提取字符数：{len(page['text'])}")
        print(page["text"][:1000] or "（本页没有提取到文字）")
        if len(page["text"]) > 1000:
            print("（控制台仅预览前1000个字符，完整文字仍在 pages 中）")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output" / "learning" / "pdf_pages" / Path(pdf_path).stem
    rendered_page = render_pdf_page(
        pdf_path,
        output_dir=str(output_dir),
        page_number=1,
        scale=2.0,
    )
    print("\n第一页图片已生成：")
    print(json.dumps(rendered_page, ensure_ascii=False, indent=2))
