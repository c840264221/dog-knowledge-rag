from tests.learning.test_day08_unified_document import (
    DocumentBlock,
    DocumentChunk,
    build_document_chunk,
)

from tests.learning.test_day10_pdf_loader import load_text_pdf


def collect_pdf_chunk_page_numbers(
    chunk: DocumentChunk,
) -> list[int]:
    """
    收集一个 PDF 检索块涉及的真实页码。

    功能：
        遍历 DocumentChunk 中保存的来源位置，
        读取每个来源位置的 PDF 页码，
        对重复页码进行去重，
        最后按照从小到大的顺序返回。

        这个函数只负责收集页码，
        暂时不负责生成最终的引用文字。

    参数：
        chunk:
            需要收集页码的 PDF 检索块。
            一个检索块可能包含同一页或多个页面的内容。

    返回值：
        list[int]:
            去重并按照从小到大排列的 PDF 页码。

    异常：
        ValueError:
            当检索块没有任何来源位置，
            或者某个来源位置没有保存 PDF 页码时抛出。
    """
    # 请在这里补充代码
    if len(chunk.locations) == 0:
        raise ValueError('chunk涉及的所有原文位置为空')

    page_numbers :set[int] = set()
    for location in chunk.locations:
        if location.page_number is None:
            raise ValueError('页码为None')
        page_numbers.add(location.page_number)

    ordered_page_numbers = sorted(list(page_numbers))

    return ordered_page_numbers

def format_pdf_page_numbers(
    page_numbers: list[int],
) -> str:
    """
    将 PDF 页码整理成适合展示的中文页码范围。

    功能：
        对传入的页码进行去重和排序，
        把连续页码合并成一个范围，
        最后生成适合展示给用户的中文文字。

    参数：
        page_numbers:
            需要整理的 PDF 页码。
            页码可以顺序混乱，也可以包含重复值。

    返回值：
        str:
            整理后的中文页码范围。
            例如："第1-3、5、7-8页"

    异常：
        ValueError:
            当页码列表为空，
            或者其中包含小于 1 的无效页码时抛出。
    """
    if not page_numbers:
        raise ValueError("页码列表不能为空")

    if any(page_number < 1 for page_number in page_numbers):
        raise ValueError("PDF 页码不能小于 1")

    ordered_page_numbers = sorted(set(page_numbers))

    page_ranges: list[str] = []
    range_start = ordered_page_numbers[0]
    previous_page = ordered_page_numbers[0]

    for current_page in ordered_page_numbers[1:]:
        if current_page == previous_page + 1 :
            previous_page = current_page
            continue

        if range_start == previous_page:
            page_ranges.append(str(range_start))
        else:
            page_ranges.append(f"{range_start}-{previous_page}")

        range_start = current_page
        previous_page = current_page

    if range_start == previous_page:
        page_ranges.append(str(range_start))
    else:
        page_ranges.append(f"{range_start}-{previous_page}")

    formatted_ranges = '、'.join(page_ranges)
    return f"第{formatted_ranges}页"

def build_pdf_chunk_reference(
    chunk: DocumentChunk,
) -> str:
    """
    为一个 PDF 检索块生成完整的来源引用。

    功能：
        从 DocumentChunk 的来源位置中取得文件路径和页码，
        调用已有函数收集真实页码，
        再把页码整理成容易阅读的范围，
        最后生成“文件路径，第几页”的引用文字。

        同一个 PDF 检索块中的所有来源位置，
        必须属于同一个源文件。

    参数：
        chunk:
            需要生成来源引用的 PDF 检索块。

    返回值：
        str:
            完整的 PDF 来源引用。
            例如：
            "documents/dog_health.pdf，第2-4、6页"

    异常：
        ValueError:
            当来源文件为空，
            或者一个检索块包含多个不同源文件时抛出。
            页码不合法时，继续使用已有页码函数抛出的异常。
    """
    # 请在这里补充代码

    sources: set[str] = set()

    for location in chunk.locations:
        source = location.source.strip()

        if not source:
            raise ValueError("PDF 来源文件不能为空")

        sources.add(source)

    if not sources:
        raise ValueError("PDF 检索块没有来源位置")

    if len(sources) > 1:
        raise ValueError("一个 PDF 检索块不能包含多个不同的源文件")

    source = list(sources)[0]
    page_numbers = collect_pdf_chunk_page_numbers(chunk)
    page_reference = format_pdf_page_numbers(page_numbers)
    return f"{source}，{page_reference}"


def calculate_pdf_chunk_character_count(
    blocks: list[DocumentBlock],
) -> int:
    """
    计算一组 PDF 内容块合并后的真实字符数量。

    功能：
        读取每个 DocumentBlock 的文字内容，
        去掉内容前后的空格，
        跳过没有实际文字的内容块，
        使用换行符连接剩余内容，
        最后返回合并后文字的字符数量。

        换行符也属于最终内容的一部分，
        因此必须计算在字符数量中。

    参数：
        blocks:
            准备合并到同一个 DocumentChunk 中的内容块。

    返回值：
        int:
            所有有效内容合并后的真实字符数量。
            如果没有任何有效内容，则返回 0。
    """
    # 请在这里补充代码
    normalized_contents: list[str] = []
    for block in blocks:
        block_content = block.content.strip()
        if not block_content:
            continue

        normalized_contents.append(block_content)

    combined_content = '\n'.join(normalized_contents)
    return len(combined_content)

def can_append_pdf_block(
    current_blocks: list[DocumentBlock],
    next_block: DocumentBlock,
    max_characters: int,
) -> bool:
    """
    判断一个 PDF 内容块能否继续加入当前检索块。

    功能：
        把 current_blocks 和 next_block 临时组合起来，
        计算它们合并后的真实字符数量，
        判断是否超过允许的最大字符数量。

        这个函数只负责判断，
        不会直接修改 current_blocks。

    参数：
        current_blocks:
            当前检索块中已经包含的 PDF 内容块。

        next_block:
            准备加入当前检索块的新内容块。

        max_characters:
            一个检索块允许包含的最大字符数量。
            必须大于 0。

    返回值：
        bool:
            合并后没有超过最大字符数时返回 True；
            合并后超过最大字符数时返回 False。

            如果 next_block 没有实际文字，也返回 False。

    异常：
        ValueError:
            当 max_characters 小于或等于 0 时抛出。
    """
    if max_characters <= 0:
        raise ValueError("一个检索块允许包含的最大字符数量必须大于0")
    # 请在这里补充代码
    next_content = next_block.content.strip()
    if not next_content:
        return False
    candidate_blocks = [*current_blocks, next_block]

    candidate_character_count = calculate_pdf_chunk_character_count(candidate_blocks)
    if candidate_character_count > max_characters:
        return False
    return True

def split_pdf_blocks_into_chunks(
    *,
    doc_id: str,
    blocks: list[DocumentBlock],
    max_characters: int,
) -> list[DocumentChunk]:
    """
    按照最大字符数量将 PDF 内容块分成多个检索块。

    功能：
        首先按照 block_index 排列 PDF 内容块，
        然后依次尝试把每个有效 Block 放入当前 Chunk。

        如果加入新 Block 后没有超过最大字符数，
        就继续放入当前 Chunk。

        如果加入后会超过最大字符数，
        就先把当前已经收集的 Block 创建成 DocumentChunk，
        再使用这个新 Block 开始下一个 Chunk。

        循环结束后，还要把最后剩余的 Block 创建成 Chunk。

        当前版本只允许在不同 DocumentBlock 之间切分，
        暂时不负责拆分一个 DocumentBlock 内部的长文本。

    参数：
        doc_id:
            当前 PDF 对应的统一文档编号。
            不能为空。

        blocks:
            从 PDF 中解析出来的内容块列表。
            函数不会修改原列表。

        max_characters:
            每个 DocumentChunk 允许包含的最大字符数量。
            必须大于 0。

    返回值：
        list[DocumentChunk]:
            按照原文顺序生成的 PDF 检索块列表。
            如果没有任何有效文字，返回空列表。

    异常：
        ValueError:
            当 doc_id 为空、max_characters 小于或等于 0，
            或者单个有效 Block 已经超过最大字符数时抛出。
    """
    # 请在这里补充代码
    normalized_doc_id = doc_id.strip()
    if not normalized_doc_id:
        raise ValueError("文档ID不能为空")
    if max_characters <= 0:
        raise ValueError("最大字符数量必须大于0")

    chunks = []
    current_blocks = []
    chunk_index = 0

    ordered_blocks = sorted(
        blocks,
        key=lambda block: block.block_index,
    )

    for block in ordered_blocks:
        block_content = block.content.strip()

        if not block_content:
            continue

        if len(block_content) > max_characters:
            raise ValueError("单个 Block 内容长度大于最大允许长度")

        if can_append_pdf_block(
                current_blocks,
                block,
                max_characters
        ):
            current_blocks.append(block)
            continue
        else:
            current_document =build_document_chunk(
                chunk_id=f"{normalized_doc_id}-chunk-{chunk_index}",
                doc_id=normalized_doc_id,
                chunk_index=chunk_index,
                blocks=current_blocks,
            )
            chunks.append(current_document)
            chunk_index += 1
            current_blocks = [block]

    if len(current_blocks) > 0:
        last_document = build_document_chunk(
            chunk_id=f"{normalized_doc_id}-chunk-{chunk_index}",
            doc_id=normalized_doc_id,
            chunk_index=chunk_index,
            blocks=current_blocks,
        )
        chunks.append(last_document)
    return chunks
def load_and_split_text_pdf(
    *,
    file_path: str,
    max_characters: int,
) -> list[DocumentChunk]:
    """
    读取文字型 PDF，并按照最大字符数量生成检索块。

    功能：
        使用第 10 天完成的 PDF 读取函数加载真实 PDF，
        得到包含多个 DocumentBlock 的 UnifiedDocument，
        再使用本文件的切块函数把这些 Block 分成多个
        DocumentChunk。

        这个函数负责连接“PDF 读取”和“PDF 切块”，
        不负责生成最终回答，也不负责写入向量数据库。

    参数：
        file_path:
            需要读取的文字型 PDF 文件路径。

        max_characters:
            每个 DocumentChunk 允许包含的最大字符数量。
            必须大于 0。

    返回值：
        list[DocumentChunk]:
            从真实 PDF 中读取并生成的检索块列表。

    异常：
        FileNotFoundError:
            当 PDF 文件不存在时，由 PDF 读取函数抛出。

        ValueError:
            当文件不是 PDF、最大字符数不合法，
            或者单个 DocumentBlock 超过最大长度时抛出。
    """
    # 请在这里补充代码
    document = load_text_pdf(file_path)

    chunks = split_pdf_blocks_into_chunks(
        doc_id=document.doc_id,
        blocks=document.blocks,
        max_characters=max_characters,
    )

    return chunks

def build_pdf_chunk_references(
    chunks: list[DocumentChunk],
) -> list[str]:
    """
    为多个 PDF 检索块生成来源引用。

    功能：
        按照传入顺序遍历 DocumentChunk，
        调用 build_pdf_chunk_reference 为每个 Chunk
        生成包含源文件和真实页码的引用文字。

        这个函数只负责批量生成引用，
        不负责读取 PDF，也不负责切块。

    参数：
        chunks:
            已经完成切分的 PDF 检索块列表。

    返回值：
        list[str]:
            按照 Chunk 原有顺序生成的来源引用列表。
            如果 chunks 为空，则返回空列表。

            例如：
            [
                "documents/dog.pdf，第1-2页",
                "documents/dog.pdf，第3页",
            ]

    异常：
        ValueError:
            当某个 Chunk 的来源文件或页码不合法时，
            由 build_pdf_chunk_reference 抛出。
    """
    # 请在这里补充代码
    references: list[str] = []
    for chunk in chunks:
        reference = build_pdf_chunk_reference(chunk)
        references.append(reference)

    return references

def run_pdf_chunk_reference_demo(
    *,
    file_path: str,
    max_characters: int,
) -> None:
    """
    演示真实 PDF 从读取、切块到生成引用的完整过程。

    功能：
        读取指定的文字型 PDF，
        按照最大字符数量生成多个 DocumentChunk，
        为每个 Chunk 生成包含文件路径和页码的引用，
        最后依次打印 Chunk 编号、引用和正文。

        这个函数只用于学习和手动检查，
        不属于正式的 RAG 检索业务代码。

    参数：
        file_path:
            需要处理的文字型 PDF 文件路径。

        max_characters:
            每个 DocumentChunk 允许包含的最大字符数量。

    返回值：
        None:
            函数不返回业务数据，只把处理结果打印到控制台。

    输出格式：
        每个 Chunk 输出以下内容：

        Chunk：文档编号和 Chunk 编号
        引用：PDF 文件路径和真实页码
        正文：当前 Chunk 包含的文字
    """
    # 请在这里补充代码
    chunks = load_and_split_text_pdf(
        file_path=file_path,
        max_characters=max_characters,
    )
    references = build_pdf_chunk_references(
        chunks=chunks,
    )
    for chunk, reference in zip(chunks, references):
        print(f"Chunk：{chunk.chunk_id}")
        print(f"引用：{reference}")
        print(f"正文：{chunk.content}")
        print("-" * 50)


if __name__ == "__main__":
    run_pdf_chunk_reference_demo(
        file_path="output/pdf/day10_text_sample.pdf",
        max_characters=50,
    )