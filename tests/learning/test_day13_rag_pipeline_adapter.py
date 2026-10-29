from typing import Any

from src.rag.schemas import RagChunk

from tests.learning.test_day08_unified_document import (
    DocumentBlock,
    DocumentChunk,
    SourceLocation,
    UnifiedDocument,
    build_document_chunk,
)

from tests.learning.test_day12_excel_csv_loader import (
    build_multiformat_parser_registry,
)

from tests.learning.test_day11_pdf_chunk_reference import (
    split_pdf_blocks_into_chunks,
)

from tests.learning.test_day09_parser_registry import (
    parse_document,
    resolve_document_source_type,
)

import json


def build_source_location_metadata(
    location: SourceLocation,
) -> dict[str, Any]:
    """
    把一个原文位置转换成适合保存到 RAG Chunk 元数据中的字典。

    功能：
        读取 SourceLocation 中保存的文件来源、PDF 页码、Excel 工作表名称
        和表格行号，生成结构化的元数据字典。

        source 必须始终保留。
        page_number、sheet_name、row_start 和 row_end 只有在不为 None 时
        才加入返回字典，避免元数据中出现大量没有实际意义的空字段。

    参数：
        location:
            需要转换的原文位置对象。它记录内容来自哪个文件，以及可能存在的
            页码、工作表名称、起始行号和结束行号。

    返回值：
        dict[str, Any]:
            结构化的来源位置元数据。

            PDF 来源示例：
            {
                "source": "docs/dogs.pdf",
                "page_number": 2,
            }

            Excel 来源示例：
            {
                "source": "docs/dogs.xlsx",
                "sheet_name": "犬种资料",
                "row_start": 3,
                "row_end": 3,
            }
    """
    location_metadata: dict[str, Any] = {}
    location_metadata["source"] = location.source

    page_number = location.page_number
    if page_number is not None:
        location_metadata["page_number"] = page_number

    sheet_name = location.sheet_name
    if sheet_name is not None:
        location_metadata["sheet_name"] = sheet_name

    row_start = location.row_start
    if row_start is not None:
        location_metadata["row_start"] = row_start

    row_end = location.row_end
    if row_end is not None:
        location_metadata["row_end"] = row_end

    return location_metadata

def build_rag_chunk_metadata(
    *,
    document: UnifiedDocument,
    chunk: DocumentChunk,
) -> dict[str, Any]:
    """
    为一个统一文档检索块构建准备交给 RagChunk 的完整元数据。

    功能：
        合并统一文档的文档级元数据和 DocumentChunk 的块级元数据，
        并加入文件类型、原始内容块编号和结构化来源位置。

        合并时，块级元数据比文档级元数据更加具体，因此出现同名字段时，
        chunk.metadata 覆盖 document.metadata。

        source_type、block_ids 和 locations 属于转换过程维护的标准字段，
        必须根据当前 document 和 chunk 的真实数据生成，不能被原有元数据
        中的同名字段覆盖。

        本函数不得直接修改 document.metadata 或 chunk.metadata。

    参数：
        document:
            当前检索块所属的统一文档。用于取得文档级元数据和 source_type。

        chunk:
            准备转换成 RagChunk 的统一检索块。用于取得块级元数据、
            原始内容块编号和来源位置列表。

    返回值：
        dict[str, Any]:
            可以保存到 RagChunk.metadata 中的新字典，结构示例：

            {
                "parser_name": "text_pdf_parser",
                "chunk_strategy": "character_limit",
                "source_type": "pdf",
                "block_ids": ["BLOCK-1", "BLOCK-2"],
                "locations": [
                    {
                        "source": "docs/dogs.pdf",
                        "page_number": 1,
                    },
                    {
                        "source": "docs/dogs.pdf",
                        "page_number": 2,
                    },
                ],
            }
    """

    rag_metadata = dict(document.metadata)
    rag_metadata.update(chunk.metadata)
    location_metadata_list: list[dict[str, Any]] = []

    for location in chunk.locations:
        location_metadata_list.append(build_source_location_metadata(location))

    rag_metadata["source_type"] = document.source_type
    rag_metadata["block_ids"] = chunk.block_ids
    rag_metadata["locations"] = location_metadata_list
    return rag_metadata

def convert_document_chunk_to_rag_chunk(
    *,
    document: UnifiedDocument,
    chunk: DocumentChunk,
) -> RagChunk:
    """
    把一个 DocumentChunk 转换成项目正式使用的 RagChunk。

    功能：
        将统一文档检索块中的编号、正文和排列顺序映射到正式 RagChunk，
        同时从 UnifiedDocument 中取得文件来源和文档标题。

        转换前检查 document.doc_id 与 chunk.doc_id 是否一致，避免把一个文档
        的正文错误地绑定到另一个文档的来源、标题和元数据。

        RagChunk.metadata 通过 build_rag_chunk_metadata 函数生成，从而保留
        文件类型、原始内容块编号，以及 PDF 页码、Excel 工作表和表格行号等
        结构化来源信息。

    参数：
        document:
            当前 DocumentChunk 所属的统一文档，用于提供文档编号、文件来源、
            文档标题、文件类型和文档级元数据。

        chunk:
            准备转换的统一检索块，用于提供 Chunk 编号、所属文档编号、
            正文、排列顺序、原始 Block 编号和来源位置。

    返回值：
        RagChunk:
            项目正式 RAG Pipeline 使用的文本块模型。返回对象包含：
            chunk_id、doc_id、content、chunk_index、source、title 和 metadata。

    异常：
        ValueError:
            当 document.doc_id 与 chunk.doc_id 不一致时抛出，防止不同文档的
            数据在转换过程中被错误组合。
    """
    if document.doc_id != chunk.doc_id:
        raise ValueError(
            f"文档编号 {document.doc_id} 与 "
            f"Chunk 所属文档编号 {chunk.doc_id} 不一致"
        )

    rag_metadata = build_rag_chunk_metadata(
        document=document,
        chunk=chunk,
    )
    return RagChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        content=chunk.content,
        chunk_index=chunk.chunk_index,
        source=document.source,
        title=document.title,
        metadata=rag_metadata,
    )

def build_table_row_document_chunk(
    *,
    document: UnifiedDocument,
    block: DocumentBlock,
    chunk_id: str,
    chunk_index: int,
) -> DocumentChunk:
    """
    把一个 CSV 或 Excel 数据行内容块转换成最小 DocumentChunk。

    功能：
        将一个 table 类型的 DocumentBlock 作为独立检索块，
        复用已有的 build_document_chunk 函数生成 DocumentChunk。

        这种转换会保留数据行正文、原始 Block 编号，以及 CSV 或 Excel
        的文件来源、工作表名称和原始行号。

        当前 Day 13 的最小方案是一条有效表格数据行对应一个检索 Chunk，
        暂时不把多个数据行合并到同一个 Chunk 中。

    参数：
        document:
            当前表格内容块所属的统一文档。其 source_type 必须是 csv 或 excel，
            并用于提供新 Chunk 所属的 doc_id。

        block:
            准备转换的表格数据行内容块。block_type 必须是 table。

        chunk_id:
            新 DocumentChunk 的唯一编号。

        chunk_index:
            新 DocumentChunk 在当前文档中的排列顺序，从 0 开始。

    返回值：
        DocumentChunk:
            由一个表格数据行组成的统一检索块。返回对象会保存该行正文、
            原始 Block 编号和结构化来源位置。

    异常：
        ValueError:
            当文档类型不是 csv 或 excel，或者传入的内容块不是 table 类型时抛出。
    """
    supported_source_types: set[str] = {"csv", "excel"}

    if document.source_type not in supported_source_types:
        raise ValueError("暂不支持当前文档类型")

    if block.block_type != "table":
        raise ValueError("内容块不是表格类型")

    table_chunk = build_document_chunk(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        chunk_index=chunk_index,
        blocks=[block],
    )
    return table_chunk

def convert_table_document_to_rag_chunks(
    document: UnifiedDocument,
) -> list[RagChunk]:
    """
    把一份 CSV 或 Excel 统一文档批量转换成正式 RagChunk 列表。

    功能：
        按照 DocumentBlock.block_index 从小到大排列文档中的内容块，
        将每一个表格数据行转换成独立的 DocumentChunk，
        再继续转换成项目正式使用的 RagChunk。

        返回的 RagChunk.chunk_index 从 0 开始连续编号，不直接依赖原始
        block_index 是否连续。

        每个 Chunk 的唯一编号由文档编号和连续 chunk_index 组成，
        格式为“文档编号-CHUNK-顺序编号”。

        本函数只负责 CSV 和 Excel 表格文档，不处理 PDF 或 Markdown。

    参数：
        document:
            需要批量转换的统一表格文档。其 source_type 必须是 csv 或 excel，
            blocks 中的每个有效内容块都必须是 table 类型。

    返回值：
        list[RagChunk]:
            按照原始内容顺序生成的正式 RagChunk 列表。
            如果 document.blocks 为空，则返回空列表。

    异常：
        ValueError:
            当文档类型不是 csv 或 excel，或者某个内容块不是 table 类型时，
            由当前函数或其调用的表格行转换函数抛出。
    """
    supported_source_types: set[str] = {"csv", "excel"}
    if document.source_type not in supported_source_types:
        raise ValueError("暂不支持当前文档类型")

    ordered_blocks = sorted(document.blocks, key=lambda b: b.block_index)
    rag_chunks: list[RagChunk] = []

    for chunk_index,block in enumerate(ordered_blocks):
        chunk_id = f"{document.doc_id}-CHUNK-{chunk_index}"
        table_chunk = build_table_row_document_chunk(
            document=document,
            block=block,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
        )
        rag_chunk = convert_document_chunk_to_rag_chunk(
            document=document,
            chunk=table_chunk,
        )
        rag_chunks.append(rag_chunk)

    return rag_chunks

def convert_document_chunks_to_rag_chunks(
    *,
    document: UnifiedDocument,
    chunks: list[DocumentChunk],
) -> list[RagChunk]:
    """
    把一组已经切分完成的 DocumentChunk 批量转换成正式 RagChunk。

    功能：
        按照 DocumentChunk.chunk_index 从小到大排列传入的检索块，
        再逐个调用 convert_document_chunk_to_rag_chunk 完成正式模型转换。

        本函数不会重新生成 chunk_id，也不会重新分配 chunk_index，
        因为传入的 DocumentChunk 已经完成切块，并且已有自己的编号和顺序。

        该函数主要用于 PDF 等已经在前一步完成切块的文档，也可以用于其他
        已经提前生成 DocumentChunk 的文件格式。

    参数：
        document:
            这些 DocumentChunk 所属的统一文档，用于提供文件来源、标题、
            文件类型和文档级元数据。

        chunks:
            已经完成切块的 DocumentChunk 列表。列表可以为空；每个 Chunk
            的 doc_id 都必须与 document.doc_id 一致。

    返回值：
        list[RagChunk]:
            按照 chunk_index 排列的正式 RagChunk 列表。
            如果 chunks 为空，则返回空列表。

    异常：
        ValueError:
            当某个 DocumentChunk.doc_id 与 document.doc_id 不一致时，
            由 convert_document_chunk_to_rag_chunk 抛出。
    """
    ordered_chunks = sorted(chunks, key=lambda c: c.chunk_index)

    rag_chunks: list[RagChunk] = []

    for chunk in ordered_chunks:
        rag_chunk = convert_document_chunk_to_rag_chunk(
                    document=document,
                    chunk=chunk,
                )
        rag_chunks.append(rag_chunk)

    return rag_chunks

def build_rag_chunk_index_preview(
    chunks: list[RagChunk],
) -> dict[str, Any]:
    """
    构建 RagChunk 列表的入库前预览结果。

    功能：
        读取已经准备完成的 RagChunk 列表，整理出 Chunk 总数、索引编号列表，
        以及每个 Chunk 的核心字段和完整 metadata。

        本函数只生成普通 Python 字典，不调用 Embedding、不连接向量数据库，
        也不执行真实入库。

        预览结果用于确认 PDF 页码、Excel 工作表、表格行号和原始 Block 编号
        在进入正式 Indexer 之前仍然完整保存在 RagChunk.metadata 中。

        本函数不得修改传入的 RagChunk 对象或原 metadata 字典。

    参数：
        chunks:
            准备交给 Indexer 的正式 RagChunk 列表。列表可以为空。

    返回值：
        dict[str, Any]:
            入库前预览结果，格式如下：

            {
                "total_chunks": 1,
                "index_ids": ["DOC-1-CHUNK-0"],
                "chunks": [
                    {
                        "chunk_id": "DOC-1-CHUNK-0",
                        "doc_id": "DOC-1",
                        "chunk_index": 0,
                        "source": "docs/dogs.xlsx",
                        "title": "犬种资料",
                        "content": "犬种：柴犬 | 原产地：日本",
                        "metadata": {
                            "source_type": "excel",
                            "block_ids": ["BLOCK-1"],
                            "locations": [
                                {
                                    "source": "docs/dogs.xlsx",
                                    "sheet_name": "犬种资料",
                                    "row_start": 2,
                                    "row_end": 2,
                                }
                            ],
                        },
                    }
                ],
            }
    """
    index_ids: list[str] = []
    chunk_previews: list[dict[str, Any]] = []

    preview_result:dict[str, Any] = {}

    for chunk in chunks:
        index_ids.append(chunk.chunk_id)
        chunk_preview: dict[str, Any] = {}
        chunk_preview["chunk_id"] = chunk.chunk_id
        chunk_preview["chunk_index"] = chunk.chunk_index
        chunk_preview["doc_id"] = chunk.doc_id
        chunk_preview["source"] = chunk.source
        chunk_preview["title"] = chunk.title
        chunk_preview["content"] = chunk.content
        chunk_preview["metadata"] = dict(chunk.metadata)
        chunk_previews.append(chunk_preview)

    preview_result["total_chunks"] = len(index_ids)
    preview_result["index_ids"] = index_ids
    preview_result["chunks"] = chunk_previews

    return preview_result

def build_table_file_rag_index_preview(
    file_path: str,
) -> dict[str, Any]:
    """
    从一个 CSV 或 Excel 文件生成 RAG 入库前预览结果。

    功能：
        创建 Day 12 已完成的多格式解析器登记表，
        通过统一解析入口 parse_document 读取传入的文件，
        得到 UnifiedDocument。

        随后将统一文档中的每一条表格数据行转换成正式 RagChunk，
        最后生成只存在于内存中的入库前预览结果。

        本函数只支持 CSV 和 Excel。它不会调用 Embedding，
        不会连接向量数据库，也不会执行真实入库。

    参数：
        file_path:
            需要处理的 CSV 或 .xlsx 文件路径。
            文件类型由 parse_document 根据扩展名识别。

    返回值：
        dict[str, Any]:
            RagChunk 入库前预览字典，包含：
            total_chunks：
                转换得到的 Chunk 总数。
            index_ids：
                所有 Chunk 的唯一编号。
            chunks：
                每个 Chunk 的正文、来源、标题和完整 metadata。

    异常：
        ValueError:
            当文件格式不受支持、没有登记对应解析器，
            或解析结果不是 CSV/Excel 表格文档时抛出。

        TypeError:
            当解析器没有返回 UnifiedDocument 时，
            由 parse_document 抛出。
    """
    parser_registry = build_multiformat_parser_registry()
    parsed_document = parse_document(
        file_path=file_path,
        parser_registry=parser_registry,
    )

    if parsed_document.source_type not in {"csv", "excel"}:
        raise ValueError("此方法只能处理csv和excel文件")

    rag_chunks = convert_table_document_to_rag_chunks(
        document=parsed_document,
    )
    preview_result = build_rag_chunk_index_preview(
        chunks=rag_chunks,
    )
    return preview_result

def build_pdf_file_rag_index_preview(
    *,
    file_path: str,
    max_characters: int,
) -> dict[str, Any]:
    """
    从一个文字型 PDF 文件生成 RAG 入库前预览结果。

    功能：
        创建 Day 12 已完成的多格式解析器登记表，
        通过统一解析入口 parse_document 读取 PDF，
        得到包含页面内容块和真实页码位置的 UnifiedDocument。

        随后调用 Day 11 的 PDF 切块函数，按照最大字符数量将页面内容块
        组合成 DocumentChunk，再将这些检索块转换成正式 RagChunk，
        最后生成只存在于内存中的入库前预览结果。

        本函数不会调用 OCR、Embedding 或向量数据库，
        也不会执行真实入库。

    参数：
        file_path:
            需要处理的文字型 PDF 文件路径。

        max_characters:
            每个 PDF DocumentChunk 允许包含的最大字符数量。
            必须大于 0。当前按 Python 字符数量计算，不是 Token 数量。

    返回值：
        dict[str, Any]:
            PDF RagChunk 的入库前预览字典，包含：
            total_chunks：
                PDF 切分后生成的 Chunk 总数。
            index_ids：
                所有 Chunk 的唯一编号。
            chunks：
                每个 Chunk 的正文、来源、标题和完整 metadata。
                metadata 中应当保留原始 Block 编号和 PDF 页码位置。

    异常：
        FileNotFoundError:
            当 PDF 文件不存在时，由 PDF 解析器抛出。

        ValueError:
            当文件不是 PDF、max_characters 不合法，
            或单个有效 DocumentBlock 超过最大字符限制时抛出。

        TypeError:
            当解析器没有返回 UnifiedDocument 时，
            由 parse_document 抛出。
    """
    parser_registry = build_multiformat_parser_registry()
    parsed_document = parse_document(
        file_path=file_path,
        parser_registry=parser_registry,
    )

    if parsed_document.source_type.lower() != "pdf":
        raise ValueError("只能处理PDF")

    document_chunks = split_pdf_blocks_into_chunks(
        doc_id=parsed_document.doc_id,
        blocks=parsed_document.blocks,
        max_characters=max_characters,
    )

    rag_chunks = convert_document_chunks_to_rag_chunks(
        document=parsed_document,
        chunks=document_chunks,
    )
    preview_result = build_rag_chunk_index_preview(
        chunks=rag_chunks,
    )

    return preview_result

def build_multiformat_rag_index_preview(
    *,
    file_path: str,
    pdf_max_characters: int = 1000,
) -> dict[str, Any]:
    """
    根据文件格式生成统一的 RAG 入库前预览结果。

    功能：
        根据文件路径识别文档类型。

        CSV 和 Excel 文件交给 build_table_file_rag_index_preview，
        采用一条有效数据行对应一个 RagChunk 的策略。

        PDF 文件交给 build_pdf_file_rag_index_preview，
        按照 pdf_max_characters 将页面内容块组合成 RagChunk。

        不同格式最终都返回相同结构的入库前预览字典，
        从而为后续统一索引入口提供一致的数据形式。

        当前学习阶段只支持 PDF、CSV 和 Excel，
        尚未实现 Markdown Parser。

        本函数不会调用 Embedding、Chroma 或其他向量数据库，
        也不会执行真实入库。

    参数：
        file_path:
            需要处理的 PDF、CSV 或 .xlsx 文件路径。

        pdf_max_characters:
            PDF 每个 DocumentChunk 允许包含的最大字符数量。
            默认值为 1000。该参数只在处理 PDF 时使用，
            不影响 CSV 和 Excel 的一行一块策略。

    返回值：
        dict[str, Any]:
            统一的入库前预览字典，包含：
            total_chunks：
                转换得到的 Chunk 总数。
            index_ids：
                全部 Chunk 唯一编号。
            chunks：
                每个 Chunk 的正文、来源、标题和完整 metadata。

    异常：
        ValueError:
            当文件格式不是 PDF、CSV 或 Excel，
            PDF 最大字符数量不合法，或者下游解析和切块失败时抛出。

        FileNotFoundError:
            当文件不存在时，由具体解析器抛出。

        TypeError:
            当具体解析器没有返回 UnifiedDocument 时，
            由统一解析入口抛出。
    """
    source_type = resolve_document_source_type(file_path)

    if source_type.lower() == "pdf":
        preview_result = build_pdf_file_rag_index_preview(
            file_path=file_path,
            max_characters=pdf_max_characters,
        )
    elif source_type.lower() in {"csv", "excel"}:
        preview_result = build_table_file_rag_index_preview(
            file_path=file_path,
        )
    else:
        raise ValueError("暂不支持其他格式文件")

    return preview_result

def run_multiformat_rag_index_preview_demo(
    *,
    file_paths: list[str],
    pdf_max_characters: int = 1000,
) -> None:
    """
    演示 PDF、CSV 和 Excel 文件生成统一 RAG 入库前预览的过程。

    功能：
        按照传入顺序遍历多个文件路径，
        将每个文件交给 build_multiformat_rag_index_preview 统一入口。

        对每个文件打印 Chunk 总数，并逐个打印 Chunk 编号、文件类型
        和结构化来源位置，用于人工确认不同格式在转换成 RagChunk 后，
        是否仍然保留 PDF 页码、Excel 工作表和表格行号。

        本函数只打印内存中的预览结果，不调用 Embedding，
        不连接 Chroma，也不执行真实向量库入库。

    参数：
        file_paths:
            需要处理的文件路径列表，可以包含 PDF、CSV 和 .xlsx 文件。
            列表不能为空。

        pdf_max_characters:
            PDF 每个 DocumentChunk 允许包含的最大字符数量。
            默认值为 1000。CSV 和 Excel 不使用该参数。

    返回值：
        None:
            本函数只输出方便人工查看的预览信息，不返回业务数据。

    输出格式：
        每份文件先输出文件路径和 Chunk 总数。

        随后逐个输出：
        - Chunk 编号；
        - source_type 文件类型；
        - locations 结构化来源位置。

        不打印完整正文，避免较长 PDF 内容影响观察。

    异常：
        ValueError:
            当 file_paths 为空、文件格式不支持，
            或具体解析和切块过程失败时抛出。

        FileNotFoundError:
            当某个文件不存在时，由具体解析器抛出。
    """

    if not file_paths:
        raise ValueError("文档路径为空")

    for file_path in file_paths:
        preview_result = build_multiformat_rag_index_preview(
            file_path=file_path,
            pdf_max_characters=pdf_max_characters,
        )
        print(f"当前文件路径：{file_path}")
        print(f"总chunk数量：{preview_result['total_chunks']}")

        chunk_previews = preview_result["chunks"]
        for chunk_preview in chunk_previews:
            chunk_metadata = chunk_preview["metadata"]
            source_type = chunk_metadata.get("source_type")
            locations = chunk_metadata.get("locations", [])
            print(f"当前chunk_id：{chunk_preview['chunk_id']}")
            print(f"当前source_type：{source_type}")
            print(f"当前location：{locations}")
        print("------------------------------------------------------------")

def build_index_safe_rag_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """
    把 RagChunk metadata 转换成当前 Chroma Indexer 可以保留的格式。

    功能：
        遍历原始 metadata，将字符串、整数、浮点数和布尔值直接保留。

        对列表和字典等结构化值，使用 JSON 格式转换成字符串，
        使 block_ids、locations 等多格式来源信息不会在正式 Indexer
        清洗 metadata 时被过滤掉。

        值为 None 的字段不写入返回结果。

        本函数只创建并返回新字典，不修改传入的 metadata。

    参数：
        metadata:
            RagChunk 当前保存的原始元数据。可以同时包含基础类型、
            列表、字典和 None。

    返回值：
        dict[str, str | int | float | bool]:
            只包含当前 Chroma Indexer 支持的数据类型的新元数据字典。

            字符串、整数、浮点数和布尔值保持原值；
            列表和字典被转换成 JSON 字符串；
            None 字段被忽略。

    异常：
        TypeError:
            当某个元数据值既不是受支持的基础类型，
            也不是可以转换的列表或字典时抛出。
    """
    safe_metadata: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str|int|float|bool):
            safe_metadata[key] = value
        elif isinstance(value, dict|list):
            safe_metadata[key] = json.dumps(
                                    value,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
        else:
            raise TypeError(
                f"metadata 字段 {key!r} 的类型 "
                f"{type(value).__name__} 暂不支持"
            )

    return safe_metadata

def build_index_safe_rag_chunks(
    chunks: list[RagChunk],
) -> list[RagChunk]:
    """
    根据原始 RagChunk 列表创建适合交给当前 Chroma Indexer 的新列表。

    功能：
        按照传入顺序遍历 RagChunk。

        对每个 Chunk 调用 build_index_safe_rag_metadata，
        将列表和字典形式的复杂 metadata 转换成 JSON 字符串。

        使用转换后的 metadata 创建新的 RagChunk，
        同时原样保留 chunk_id、doc_id、content、chunk_index、
        source 和 title。

        本函数不会修改原始 RagChunk，也不会调用真实 Indexer、
        Embedding 或向量数据库。

    参数：
        chunks:
            已经完成多格式转换的原始 RagChunk 列表。
            列表可以为空。

    返回值：
        list[RagChunk]:
            按照原始顺序生成的索引安全 RagChunk 列表。

            返回对象的核心字段与原 Chunk 相同，
            但 metadata 中的列表和字典已经转换成 JSON 字符串。

            如果 chunks 为空，则返回空列表。

    异常：
        TypeError:
            当某个 Chunk 的 metadata 包含不受支持的数据类型时，
            由 build_index_safe_rag_metadata 抛出。
    """

    safe_chunks: list[RagChunk] = []
    for chunk in chunks:
        safe_metadata = build_index_safe_rag_metadata(chunk.metadata)
        safe_chunk = RagChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            title=chunk.title,
            metadata=safe_metadata,
        )
        safe_chunks.append(safe_chunk)
    return safe_chunks


if __name__ == "__main__":
    preview_result = build_multiformat_rag_index_preview(
        file_path="output/excel/day12_dog_tables.xlsx",
    )

    original_chunks = [
        RagChunk(**chunk_preview)
        for chunk_preview in preview_result["chunks"]
    ]

    safe_chunks = build_index_safe_rag_chunks(original_chunks)

    original_locations = original_chunks[0].metadata["locations"]
    safe_locations = safe_chunks[0].metadata["locations"]
    restored_locations = json.loads(safe_locations)

    print(f"原始 locations 类型：{type(original_locations).__name__}")
    print(f"索引安全 locations 类型：{type(safe_locations).__name__}")
    print(f"还原后 locations 类型：{type(restored_locations).__name__}")
    print(f"还原内容与原始内容一致：{restored_locations == original_locations}")
    print(f"索引安全 metadata：{safe_chunks[0].metadata}")