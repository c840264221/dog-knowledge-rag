from pydantic import BaseModel, Field
from typing import Any, Literal


class SourceLocation(BaseModel):
    """
    表示一段内容在原文件中的位置。

    功能：
        统一记录 PDF、Excel、CSV、Markdown 等不同格式文件的原文位置，
        方便检索结果展示来源，也方便后续回到原文件中核对内容。

    字段说明：
        source:
            原文件来源，例如文件路径或网页地址。
        page_number:
            PDF 页码；不是 PDF 内容时可以为空。
        sheet_name:
            Excel 工作表名称；不是 Excel 内容时可以为空。
        row_start:
            内容在表格中的起始行号；没有行号时可以为空。
        row_end:
            内容在表格中的结束行号；只有一行或没有行号时可以为空。

    返回值：
        SourceLocation：保存原文件位置的数据模型。
    """

    source: str
    page_number: int | None = Field(default=None,ge=1)
    sheet_name: str | None = None
    row_start: int | None = Field(default=None,ge=1)
    row_end: int | None = Field(default=None,ge=1)


def format_source_location(location: SourceLocation) -> str:
    """
    把原文位置整理成方便人查看的中文说明。

    参数：
        location:
            需要格式化的原文位置，其中可以包含来源、页码、工作表和行号。

    返回值：
        str：
            格式化后的位置文字，例如“docs/health.pdf，第5页”。

    异常：
        ValueError：
            当来源为空或只包含空格时抛出。
    """

    if location.source is not None:
        if not location.source.strip():
            raise ValueError("来源字段不能为空")
    else:
        raise ValueError("来源字段不能为空")
    location_parts = []
    source_text = location.source.strip()
    location_parts.append(source_text)

    page_number = location.page_number
    if page_number is not None:
        page_text = f"第{page_number}页"
        location_parts.append(page_text)

    sheet_name = location.sheet_name
    if sheet_name is not None:
        sheet_text = f"工作表：{sheet_name}"
        location_parts.append(sheet_text)

    row_start = location.row_start
    row_end = location.row_end
    if row_start is not None:
        if row_end is not None:
            row_text = f"第{row_start}-{row_end}行"
            location_parts.append(row_text)
        else:
            row_text = f"第{row_start}行"
            location_parts.append(row_text)

    return "，".join(location_parts)

class DocumentBlock(BaseModel):
    """
    表示从原文件中读取出来的一块内容。

    功能：
        统一表示标题、普通段落、表格和列表等不同类型的内容。
        每个内容块都保留自己的正文、原文位置和排列顺序。

    字段说明：
        block_id:
            内容块的唯一编号，用来区分同一篇文档中的不同内容块。
        block_type:
            内容块类型，目前支持标题、段落、表格和列表。
        content:
            内容块中保存的文字。
        block_index:
            内容块在当前文档中的排列顺序，从 0 开始。
        location:
            内容块在原文件中的位置。
        metadata:
            额外信息，例如标题级别、表格名称或者解析器名称。

    返回值：
        DocumentBlock：保存一块原文内容及其位置的数据模型。
    """

    block_id: str
    block_type: Literal["title", "paragraph", "table", "list"]
    content: str
    block_index: int = Field(default=0, ge=0)
    location: SourceLocation
    metadata: dict[str, Any] = Field(default_factory=dict)

def format_document_block(block: DocumentBlock) -> str:
    """
    把文档内容块整理成方便人查看的文本。

    功能：
        根据内容块类型生成中文名称，并把正文和原文位置组合起来，
        方便查看解析后的内容是否正确。

    参数：
        block:
            需要格式化的文档内容块。

    返回值：
        str：
            格式化后的内容块文字，格式为：
            “[内容类型] 正文（原文位置）”。
    """

    # 具体逻辑由你完成
    block_type_names = {
        "title": "标题",
        "paragraph": "段落",
        "table": "表格",
        "list": "列表",
    }

    block_type_text = block_type_names.get(block.block_type.strip(), None)
    if block_type_text is None:
        raise ValueError("暂不支持当前内容类型")
    location_text = format_source_location(block.location)
    content_text = block.content.strip()
    result_text = f"[{block_type_text}] {content_text}（{location_text}）"
    return result_text

class UnifiedDocument(BaseModel):
    """
    表示不同格式文件经过读取后得到的统一文档。

    功能：
        把 Markdown、PDF、Excel 和 CSV 等不同格式的文件，
        统一整理成相同的文档结构，供后续切分和 RAG 入库使用。

    字段说明：
        doc_id:
            文档唯一编号。
        source:
            原文件来源，例如本地文件路径。
        source_type:
            原文件格式，目前支持 Markdown、PDF、Excel 和 CSV。
        title:
            文档标题；没有明确标题时可以为空字符串。
        blocks:
            从原文件中读取出来的内容块，按照 block_index 表示原始顺序。
        metadata:
            文档级别的额外信息，例如文件名、文件大小或者解析器名称。

    返回值：
        UnifiedDocument：包含文档信息和全部内容块的统一文档模型。
    """

    doc_id: str
    source: str
    source_type: Literal["markdown", "pdf", "excel", "csv"]
    title: str = ""
    blocks: list[DocumentBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

def build_document_content(document: UnifiedDocument) -> str:
    """
    根据统一文档中的内容块生成完整正文。

    功能：
        按照 block_index 从小到大排列内容块，
        清理每块正文开头和结尾的空格，
        忽略正文为空的内容块，
        最后使用一个空行连接所有有效正文。

    参数：
        document:
            需要生成完整正文的统一文档。

    返回值：
        str：
            按照原始顺序组合后的完整正文；
            如果没有有效内容块，则返回空字符串。
    """

    # 具体逻辑由你完成
    ordered_blocks = sorted(document.blocks, key=lambda x: x.block_index)
    content_parts: list[str] = []

    for block in ordered_blocks:
        content_text = block.content.strip()

        if not content_text:
            continue

        content_parts.append(content_text)
    content_text = "\n\n".join(content_parts)
    return content_text

class DocumentChunk(BaseModel):
    """
    表示由一个或多个文档内容块组合成的检索文本块。

    功能：
        保存准备进入 RAG 索引和检索流程的正文，
        同时记录它由哪些原始内容块组成，以及这些内容位于原文件的哪里。

    字段说明：
        chunk_id:
            检索文本块的唯一编号。
        doc_id:
            当前文本块所属的统一文档编号。
        content:
            组合后的正文。
        chunk_index:
            当前文本块在文档中的顺序，从 0 开始。
        block_ids:
            组成当前文本块的原始内容块编号。
        locations:
            当前文本块涉及的所有原文位置。
        metadata:
            文本块的额外信息。

    返回值：
        DocumentChunk：可以继续转换成项目现有 RagChunk 的统一检索块。
    """

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int = Field(default=0, ge=0)
    block_ids: list[str] = Field(default_factory=list)
    locations: list[SourceLocation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

def build_document_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    chunk_index: int,
    blocks: list[DocumentBlock],
) -> DocumentChunk:
    """
    根据一组文档内容块创建统一检索块。

    功能：
        按照 block_index 排列传入的内容块，
        忽略正文为空的内容块，
        使用一个换行符连接有效正文，
        并收集有效内容块的编号和原文位置。

    参数：
        chunk_id:
            新检索文本块的唯一编号。
        doc_id:
            检索文本块所属的统一文档编号。
        chunk_index:
            检索文本块在文档中的顺序。
        blocks:
            用来组成检索文本块的原始内容块列表。

    返回值：
        DocumentChunk：
            组合完成的统一检索文本块。

    异常：
        ValueError：
            当 blocks 为空，或者所有内容块的正文都为空时抛出。
    """

    # 具体逻辑由你完成
    if len(blocks) == 0:
        raise ValueError("原始内容块列表不能为空")
    ordered_blocks = sorted(blocks, key=lambda x: x.block_index)
    content_parts = []
    block_ids = []
    locations = []
    for block in ordered_blocks:
        block_content_text = block.content.strip()
        if not block_content_text:
            continue
        content_parts.append(block_content_text)
        block_ids.append(block.block_id)
        locations.append(block.location)

    content_text = "\n".join(content_parts)

    if len(content_parts) == 0:
        raise ValueError("所有正文都是空的")

    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content_text,
        chunk_index=chunk_index,
        block_ids=block_ids,
        locations=locations,
    )


def test_build_document_chunk_preserves_order_and_source_locations() -> None:
    """
    验证统一检索块会保留内容顺序和原文位置。

    功能：
        使用顺序混乱的 PDF 内容块创建 DocumentChunk，
        验证函数会按照 block_index 排列正文，忽略空内容块，
        并保留有效内容块的编号和页码。

    参数：
        无。

    返回值：
        None：测试通过时不返回业务数据。
    """

    paragraph_block = DocumentBlock(
        block_id="BLOCK-2",
        block_type="paragraph",
        content=" 柴犬原产于日本。 ",
        block_index=1,
        location=SourceLocation(
            source="docs/dogs.pdf",
            page_number=2,
        ),
    )
    empty_block = DocumentBlock(
        block_id="BLOCK-3",
        block_type="paragraph",
        content="   ",
        block_index=2,
        location=SourceLocation(
            source="docs/dogs.pdf",
            page_number=3,
        ),
    )
    title_block = DocumentBlock(
        block_id="BLOCK-1",
        block_type="title",
        content=" 柴犬介绍 ",
        block_index=0,
        location=SourceLocation(
            source="docs/dogs.pdf",
            page_number=1,
        ),
    )
    source_blocks = [
        paragraph_block,
        empty_block,
        title_block,
    ]

    chunk = build_document_chunk(
        chunk_id="CHUNK-1",
        doc_id="DOC-1",
        chunk_index=0,
        blocks=source_blocks,
    )

    page_numbers = [
        location.page_number
        for location in chunk.locations
    ]
    original_block_ids = [
        block.block_id
        for block in source_blocks
    ]

    assert chunk.content == "柴犬介绍\n柴犬原产于日本。"
    assert chunk.block_ids == ["BLOCK-1", "BLOCK-2"]
    assert page_numbers == [1, 2]
    assert chunk.chunk_index == 0
    assert original_block_ids == ["BLOCK-2", "BLOCK-3", "BLOCK-1"]
