from pathlib import Path

from pypdf import PdfReader

from tests.learning.test_day08_unified_document import (
    DocumentBlock,
    SourceLocation,
    UnifiedDocument,
)

from tests.learning.test_day09_parser_registry import (
    DocumentParser,
    DocumentSourceType,
    parse_document,
)

def build_pdf_page_block(
    *,
    source: str,
    page_number: int,
    page_text: str | None,
    block_index: int,
) -> DocumentBlock | None:
    """
    把一页 PDF 文字转换成统一文档内容块。

    功能：
        接收 PDF 解析器从某一页提取出来的文字，
        清理正文首尾空格。
        正文有效时创建 paragraph 类型的 DocumentBlock，
        正文为空或者解析器返回 None 时返回 None。

    参数：
        source:
            PDF 原文件路径。
        page_number:
            当前文字所在的 PDF 页码，从 1 开始。
        page_text:
            PDF 解析器提取出的当前页文字。
            没有提取到文字时可能是 None。
        block_index:
            当前内容块在统一文档中的顺序，从 0 开始。

    返回值：
        DocumentBlock | None：
            当前页有有效文字时返回 paragraph 内容块；
            当前页没有有效文字时返回 None。

    异常：
        ValueError：
            当 source 为空、page_number 小于 1，
            或者 block_index 小于 0 时抛出。
    """

    # 具体逻辑由你完成
    normalized_source = source.strip()
    if not normalized_source:
        raise ValueError("文档路径不能为空")

    if page_number < 1:
        raise ValueError("文字所在页码必须大于等于1")

    if block_index < 0:
        raise ValueError("内容在文档中的顺序必须大于等于0")

    if page_text is None:
        return None

    normalized_page_text = page_text.strip()
    if not normalized_page_text:
        return None

    location = SourceLocation(
        source=normalized_source,
        page_number=page_number,
    )
    block_id = f"PDF-PAGE-{page_number}"

    return DocumentBlock(
        block_id=block_id,
        block_type="paragraph",
        content=normalized_page_text,
        block_index=block_index,
        location=location,
        metadata={"parser": "pdf"},
    )

def test_build_pdf_page_block_should_preserve_page_location() -> None:
    """
    验证 PDF 页面文字可以转换成带页码的统一内容块。

    功能：
        传入带首尾空格的 PDF 路径和页面文字，
        验证函数会清理文字与路径，
        并正确保存页码、内容块顺序和解析器信息。

    参数：
        无。

    返回值：
        None：
            所有断言通过时测试结束。
    """

    # 测试逻辑由你完成
    page_block = build_pdf_page_block(
        source=" docs/dogs.pdf ",
        page_number=2,
        page_text=" 柴犬原产于日本。 ",
        block_index=1,
    )
    assert page_block is not None
    assert page_block.block_id == "PDF-PAGE-2"
    assert page_block.block_type == "paragraph"
    assert page_block.content == "柴犬原产于日本。"
    assert page_block.block_index == 1
    assert page_block.location.source == "docs/dogs.pdf"
    assert page_block.location.page_number == 2
    assert page_block.metadata == {"parser": "pdf"}


def load_text_pdf(file_path: str) -> UnifiedDocument:
    """
    读取文字型 PDF 并转换成统一文档。

    功能：
        检查 PDF 文件路径是否合法并且文件真实存在，
        使用 pypdf 按页提取文字，
        把每一页有效文字转换成 DocumentBlock，
        最后将所有内容块组成 UnifiedDocument。

        没有提取到文字的页面不会生成 DocumentBlock，
        但 PDF 总页数仍然会保存在文档 metadata 中。

    参数：
        file_path:
            需要读取的 PDF 文件路径。

    返回值：
        UnifiedDocument：
            包含 PDF 基本信息、有效页面内容块和页数统计的统一文档。

    异常：
        ValueError：
            当路径为空、目标不是 PDF 文件，
            或者路径指向的不是普通文件时抛出。
        FileNotFoundError：
            当指定 PDF 文件不存在时抛出。
        pypdf 相关异常：
            PDF 文件损坏、加密或无法正常读取时，
            暂时保留 pypdf 原始异常向外抛出。
    """

    # 具体逻辑由你完成
    normalized_path = file_path.strip()
    if not normalized_path:
        raise ValueError("PDF文档路径不能为空")

    pdf_path = Path(normalized_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF文件不存在：{pdf_path}"
        )
    if not pdf_path.is_file():
        raise ValueError("路径不是文件")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("文档不是PDF格式")

    pdf_reader = PdfReader(pdf_path)
    page_count = len(pdf_reader.pages)
    page_blocks: list[DocumentBlock] = []
    pages_without_text: list[int] = []

    for page_number, page in enumerate(
            pdf_reader.pages,
            start=1,
    ):
        page_text = page.extract_text()
        page_block = build_pdf_page_block(
            source=normalized_path,
            page_number=page_number,
            page_text=page_text,
            block_index=len(page_blocks),
        )
        if page_block is None:
            pages_without_text.append(page_number)
            continue
        page_blocks.append(page_block)

    text_page_count = len(page_blocks)
    requires_ocr = page_count > 0 and len(pages_without_text) == page_count

    return UnifiedDocument(
        doc_id=f"PDF-{pdf_path.stem}",
        source=str(pdf_path),
        source_type="pdf",
        title=pdf_path.stem,
        blocks=page_blocks,
        metadata={
            "parser": "pypdf",
            "page_count": page_count,
            "text_page_count": text_page_count,
            "pages_without_text": pages_without_text,
            "requires_ocr": requires_ocr,
        },
    )

def test_load_text_pdf_should_preserve_real_page_numbers() -> None:
    """
    验证文字型 PDF 可以按页转换成统一文档。

    功能：
        读取项目中准备好的两页 PDF 样例，
        验证 Loader 能提取两页文字，
        为每页生成一个 DocumentBlock，
        并保留页码、内容块顺序和文档统计信息。

    参数：
        无。

    返回值：
        None：
            所有断言通过时测试结束。
    """

    # 测试逻辑由你完成
    project_root = Path(__file__).resolve().parents[2]

    sample_pdf_path = (
            project_root
            / "tests"
            / "fixtures"
            / "pdf"
            / "day10_text_sample.pdf"
    )
    document = load_text_pdf(str(sample_pdf_path))
    assert document.source_type == "pdf"
    assert document.title == "day10_text_sample"
    assert len(document.blocks) == 2
    assert document.metadata["page_count"] == 2
    assert document.metadata["text_page_count"] == 2
    assert document.metadata["requires_ocr"] is False
    assert document.metadata["pages_without_text"] == []

    first_block = document.blocks[0]
    second_block = document.blocks[1]

    assert first_block.block_id == "PDF-PAGE-1"
    assert first_block.block_index == 0
    assert first_block.location.page_number == 1
    assert "柴犬介绍" in first_block.content
    assert "柴犬原产于日本" in first_block.content

    assert second_block.block_id == "PDF-PAGE-2"
    assert second_block.block_index == 1
    assert second_block.location.page_number == 2
    assert "日常护理" in second_block.content
    assert "定期健康检查" in second_block.content


def test_load_text_pdf_should_record_pages_without_text() -> None:
    """
    验证 PDF 部分页面没有文字时可以保留诊断信息。

    功能：
        读取一个包含三页的真实 PDF，
        其中第 1 页和第 3 页有文字，第 2 页没有文字。
        验证 Loader 会跳过空白页、记录空白页的真实页码，
        同时保持有效 Block 的 block_index 连续。

    参数：
        无。

    返回值：
        None：
            所有断言通过时测试结束。
    """

    project_root = Path(__file__).resolve().parents[2]
    sample_pdf_path = (
        project_root
        / "tests"
        / "fixtures"
        / "pdf"
        / "day10_partial_text_sample.pdf"
    )

    document = load_text_pdf(str(sample_pdf_path))

    assert document.metadata["page_count"] == 3
    assert document.metadata["text_page_count"] == 2
    assert document.metadata["pages_without_text"] == [2]
    assert document.metadata["requires_ocr"] is False
    assert len(document.blocks) == 2

    page_numbers = [
        block.location.page_number
        for block in document.blocks
    ]
    block_indexes = [
        block.block_index
        for block in document.blocks
    ]

    assert page_numbers == [1, 3]
    assert block_indexes == [0, 1]

class TextPdfParser:
    """
    提供文字型 PDF 的统一解析器。

    功能：
        实现 DocumentParser 约定的 parse 方法，
        将统一解析入口收到的 PDF 路径交给 load_text_pdf，
        最终返回 UnifiedDocument。

    返回值：
        TextPdfParser：
            可以登记到 parser_registry 中的 PDF 解析器。
    """

    def parse(self, file_path: str) -> UnifiedDocument:
        """
        读取文字型 PDF 并返回统一文档。

        参数：
            file_path:
                需要读取的 PDF 文件路径。

        返回值：
            UnifiedDocument：
                load_text_pdf 生成的统一文档。
        """

        # 具体逻辑由你完成

        return load_text_pdf(file_path)


def build_document_parser_registry() -> dict[
    DocumentSourceType,
    DocumentParser,
]:
    """
    创建当前已经可用的文档解析器登记表。

    功能：
        创建文件类型与解析器对象之间的对应关系。
        当前只登记已经实现的文字型 PDF 解析器，
        后续再逐步加入 Markdown、Excel 和 CSV 解析器。

    参数：
        无。

    返回值：
        dict[DocumentSourceType, DocumentParser]：
            当前可供统一解析入口使用的解析器登记表。
    """

    # 具体逻辑由你完成

    return {"pdf": TextPdfParser()}

def run_pdf_registry_demo() -> UnifiedDocument:
    """
    通过统一解析入口读取真实 PDF 样例。

    功能：
        找到项目中的两页 PDF 样例，
        创建当前可用的解析器登记表，
        通过第 9 天的 parse_document 选择 TextPdfParser，
        最终取得统一文档并打印文档和页面信息。

    参数：
        无。

    返回值：
        UnifiedDocument：
            通过统一解析入口得到的 PDF 统一文档。
    """

    # 具体逻辑由你完成
    project_root = Path(__file__).resolve().parents[2]

    sample_pdf_path = (
            project_root
            / "output"
            / "pdf"
            / "day10_text_sample.pdf"
    )

    parser_registry = build_document_parser_registry()
    document = parse_document(
        file_path=str(sample_pdf_path),
        parser_registry=parser_registry,
    )
    print(f"文档标题：{document.title}")
    print(f"文件类型：{document.source_type}")
    print(f"PDF总页数：{document.metadata['page_count']}")
    print(f"文字页数：{document.metadata['text_page_count']}")
    print(f"是否可能需要OCR：{document.metadata['requires_ocr']}")

    for block in document.blocks:
        print(
            f"\n第{block.location.page_number}页，"
            f"Block顺序：{block.block_index}"
        )
        print(block.content)
    return document


if __name__ == "__main__":
    run_pdf_registry_demo()
