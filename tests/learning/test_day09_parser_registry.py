from pathlib import Path
from typing import Literal

from collections.abc import Mapping
from typing import Protocol

from tests.learning.test_day08_unified_document import UnifiedDocument
import pytest


DocumentSourceType = Literal[
    "markdown",
    "pdf",
    "excel",
    "csv",
]


DOCUMENT_SOURCE_TYPE_BY_EXTENSION: dict[str, DocumentSourceType] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xls": "excel",
    ".csv": "csv",
}

def normalize_document_extension(file_path: str) -> str:
    """
    提取并统一文档路径中的文件扩展名。

    功能：
        清理文件路径首尾空格，取得文件名最后一个扩展名，
        并把扩展名统一转换成小写形式。
        返回值保留开头的点，例如“.pdf”和“.xlsx”。

    参数：
        file_path:
            需要识别类型的文件路径或文件名，
            例如“docs/dogs.PDF”或者“data/health.xlsx”。

    返回值：
        str：
            统一成小写的文件扩展名，例如“.pdf”。

    异常：
        ValueError：
            当文件路径为空、只包含空格，或者没有扩展名时抛出。
    """

    # 具体逻辑由你完成
    normalized_path = file_path.strip()
    if not normalized_path:
        raise ValueError("文档路径不能为空")

    file_extension = Path(normalized_path).suffix

    if not file_extension:
        raise ValueError("文档路径缺少文件扩展名")

    normalized_extension = file_extension.lower()
    return normalized_extension

def resolve_document_source_type(
    file_path: str,
) -> DocumentSourceType:
    """
    根据文档路径识别统一的文档来源类型。

    功能：
        先取得经过统一处理的文件扩展名，
        再根据扩展名映射出 Markdown、PDF、Excel 或 CSV。
        多个扩展名可以对应同一种文档类型，
        例如“.xls”和“.xlsx”都属于 Excel。

    参数：
        file_path:
            需要识别类型的文件路径或文件名。

    返回值：
        DocumentSourceType：
            统一的文档来源类型，
            可能是“markdown”“pdf”“excel”或者“csv”。

    异常：
        ValueError：
            当路径为空、没有扩展名，或者扩展名暂不支持时抛出。
    """

    # 具体逻辑由你完成
    normalized_extension = normalize_document_extension(file_path)
    if normalized_extension not in DOCUMENT_SOURCE_TYPE_BY_EXTENSION:
        raise ValueError(f"暂不支持当前扩展名：{normalized_extension}")

    source_type = DOCUMENT_SOURCE_TYPE_BY_EXTENSION[normalized_extension]
    return source_type


class DocumentParser(Protocol):
    """
    约定所有文档解析器必须提供的统一能力。

    功能：
        要求 PDF、Excel、CSV 和 Markdown 解析器都提供 parse 方法，
        接收文件路径并返回统一的 UnifiedDocument。

    方法参数：
        file_path:
            需要解析的文件路径。

    方法返回值：
        UnifiedDocument：
            不同格式文件解析后得到的统一文档。
    """

    def parse(self, file_path: str) -> UnifiedDocument:
        """
        解析指定文件并返回统一文档。

        参数：
            file_path:
                需要解析的文件路径。

        返回值：
            UnifiedDocument：
                解析完成后的统一文档。
        """

        ...

def select_document_parser(
    *,
    file_path: str,
    parser_registry: Mapping[
        DocumentSourceType,
        DocumentParser,
    ],
) -> DocumentParser:
    """
    根据文件类型从解析器登记表中选择对应解析器。

    功能：
        先根据文件路径识别统一文档类型，
        再从 parser_registry 中查找已经登记的解析器。
        本函数只负责选择，不负责真正读取文件。

    参数：
        file_path:
            准备解析的文件路径。
        parser_registry:
            解析器登记表。
            键是统一文档类型，值是对应的解析器对象。

    返回值：
        DocumentParser：
            与当前文件类型对应的解析器对象。

    异常：
        ValueError：
            当文件格式不受支持，或者对应类型没有登记解析器时抛出。
    """

    # 具体逻辑由你完成
    source_type = resolve_document_source_type(file_path)

    selected_parser = parser_registry.get(source_type, None)

    if selected_parser is None:
        raise ValueError(f"解析器中暂时还未注册{source_type}类型的解析器")

    return selected_parser

def parse_document(
    *,
    file_path: str,
    parser_registry: Mapping[
        DocumentSourceType,
        DocumentParser,
    ],
) -> UnifiedDocument:
    """
    使用登记表中的对应解析器解析文档。

    功能：
        根据文件路径选择对应解析器，
        调用解析器的 parse 方法得到统一文档，
        并检查解析器返回的类型和文档来源类型是否正确。

    参数：
        file_path:
            需要解析的文件路径。
        parser_registry:
            解析器登记表。
            键是统一文档类型，值是对应的解析器对象。

    返回值：
        UnifiedDocument：
            解析器生成的统一文档。

    异常：
        ValueError：
            当文件格式不支持、没有登记对应解析器，
            或解析结果的 source_type 与文件类型不一致时抛出。
        TypeError：
            当解析器没有返回 UnifiedDocument 时抛出。
    """

    # 具体逻辑由你完成
    selected_parser = select_document_parser(
        file_path=file_path,
        parser_registry=parser_registry,
    )

    parsed_document = selected_parser.parse(file_path=file_path)

    expected_source_type = resolve_document_source_type(
        file_path=file_path,
    )

    if not isinstance(parsed_document, UnifiedDocument):
        raise TypeError("解析器必须返回 UnifiedDocument")

    if parsed_document.source_type != expected_source_type:
        raise ValueError("解析器的来源文档类型与收到的文档类型不一致")

    return parsed_document

class RecordingDocumentParser:
    """
    提供只用于学习测试的可记录解析器。

    功能：
        不读取真实文件，而是直接返回预先准备好的 UnifiedDocument。
        每次调用 parse 时记录收到的文件路径，
        方便检查解析入口是否调用了正确的解析器。

    初始化参数：
        document:
            调用 parse 时准备返回的统一文档。

    属性：
        document:
            预先准备好的统一文档。
        received_file_paths:
            parse 方法实际收到的所有文件路径。

    返回值：
        RecordingDocumentParser：
            可以放入 parser_registry 的测试解析器。
    """

    def __init__(self, document: UnifiedDocument) -> None:
        """
        保存预先准备的文档并初始化调用记录。

        参数：
            document:
                parse 方法需要返回的统一文档。

        返回值：
            None：
                初始化对象，不返回业务结果。
        """

        # 具体逻辑由你完成
        self.document = document
        self.received_file_paths : list[str] = []

    def parse(self, file_path: str) -> UnifiedDocument:
        """
        记录文件路径并返回预先准备的统一文档。

        参数：
            file_path:
                统一解析入口传入的文件路径。

        返回值：
            UnifiedDocument：
                初始化时保存的统一文档。
        """

        self.received_file_paths.append(file_path)
        return self.document


def test_parse_document_should_select_matching_parser() -> None:
    """
    验证统一解析入口会根据扩展名选择正确解析器。

    功能：
        分别准备 PDF 和 Excel 假解析器，
        使用大写“.PDF”路径调用统一解析入口，
        验证系统选择了 PDF 解析器、返回 PDF 统一文档，
        并且没有错误调用 Excel 解析器。

    参数：
        无。

    返回值：
        None：
            所有断言通过时测试结束。
    """

    # 测试逻辑由你完成
    pdf_document = UnifiedDocument(
        doc_id="DOC-PDF-1",
        source="docs/dogs.PDF",
        source_type="pdf",
        title="狗狗资料",
    )

    excel_document = UnifiedDocument(
        doc_id="DOC-EXCEL-1",
        source="docs/dogs.xlsx",
        source_type="excel",
        title="狗狗数据表",
    )
    pdf_parser = RecordingDocumentParser(pdf_document)
    excel_parser = RecordingDocumentParser(excel_document)
    parser_registry = {
        "pdf": pdf_parser,
        "excel": excel_parser,
    }
    parsed_document = parse_document(
        file_path="docs/dogs.PDF",
        parser_registry=parser_registry,
    )

    assert parsed_document is pdf_document
    assert pdf_parser.received_file_paths == ["docs/dogs.PDF"]
    assert len(pdf_parser.received_file_paths) == 1
    assert excel_parser.received_file_paths == []

def test_select_document_parser_should_reject_unregistered_parser() -> None:
    """
    验证已支持的文件类型没有登记解析器时会被明确拒绝。

    功能：
        使用 PDF 文件路径调用解析器选择函数，
        但传入一个没有任何解析器的登记表，
        验证系统说明 PDF 解析器尚未登记。

    参数：
        无。

    返回值：
        None：
            成功捕获预期异常时测试结束。
    """

    # 测试逻辑由你完成
    empty_parser_registry: dict[
        DocumentSourceType,
        DocumentParser,
    ] = {}

    with pytest.raises(ValueError, match="解析器中暂时还未注册pdf类型的解析器"):
        select_document_parser(
            file_path="docs/dogs.pdf",
            parser_registry=empty_parser_registry,
        )

def test_parse_document_should_reject_mismatched_source_type() -> None:
    """
    验证解析结果类型与文件扩展名不一致时会被拒绝。

    功能：
        把一个返回 Excel 文档的假解析器登记为 PDF 解析器，
        然后使用 PDF 文件路径调用统一解析入口，
        验证系统不会接受类型错误的统一文档。

    参数：
        无。

    返回值：
        None：
            成功捕获预期异常时测试结束。
    """

    # 测试逻辑由你完成
    excel_document = UnifiedDocument(
        doc_id="DOC-WRONG-1",
        source="docs/dogs.pdf",
        source_type="excel",
        title="错误类型文档",
    )
    wrong_pdf_parser = RecordingDocumentParser(excel_document)

    parser_registry = {
        "pdf": wrong_pdf_parser,
    }

    with pytest.raises(ValueError,match="解析器的来源文档类型与收到的文档类型不一致",):
        parse_document(
            file_path="docs/dogs.pdf",
            parser_registry=parser_registry,
        )
    assert wrong_pdf_parser.received_file_paths == ["docs/dogs.pdf"]

if __name__ == "__main__":
    file_extension = normalize_document_extension("data/health.PDF")
    print(file_extension)