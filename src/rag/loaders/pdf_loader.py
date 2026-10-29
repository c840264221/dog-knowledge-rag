"""PDF加载器：普通文字提取与显式选页，输出项目标准文档，不执行视觉识别。"""

import hashlib
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.rag.loaders.base_loader import BaseDocumentLoader
from src.rag.schemas import RagDocument


class PdfDocumentLoader(BaseDocumentLoader):
    """将一个PDF加载为一份文档；页码为从1开始的物理页码，不推断印刷页码。"""

    def __init__(
        self, input_path: str | Path, *, doc_id: str, title: str,
        included_pages: list[int] | None = None, metadata: dict[str, Any] | None = None,
    ):
        """保存加载配置，不立即读取文件。

        参数：input_path为PDF路径；doc_id为稳定文档编号；title为资料标题；
            included_pages为明确选用的物理页码，None表示全部；metadata为来源等元数据。
        返回值：无。空选页、重复页码和非法页码均拒绝，避免悄悄扩大入库范围。
        """
        if not doc_id.strip():
            raise ValueError("文档编号不能为空。")
        if included_pages is not None and (
            not included_pages
            or any(type(number) is not int or number < 1 for number in included_pages)
            or len(set(included_pages)) != len(included_pages)
        ):
            raise ValueError("选页必须为非空、不重复的正整数列表。")
        self.input_path = Path(input_path).resolve()
        self.doc_id = doc_id
        self.title = title
        self.included_pages = None if included_pages is None else sorted(included_pages)
        self.metadata = dict(metadata or {})

    def load(self) -> list[RagDocument]:
        """读取所有页的原始文字，并单独标记选用范围。

        参数：无，使用构造时的配置。
        返回值：只包含一份RagDocument的列表；正文为选用页文字，metadata中的
            pdf_pages保留全部页原始提取结果供追溯。不修复词间空格、顺序或跨页段落。
            文件无效、页码越界时抛出异常，不静默切换提取方式。
        """
        payload = self.input_path.read_bytes()
        if not payload.startswith(b"%PDF-"):
            raise ValueError("源文件不是有效PDF文件头。")
        with self.input_path.open("rb") as stream:
            reader = PdfReader(stream)
            selected = self.included_pages if self.included_pages is not None else list(range(1, len(reader.pages) + 1))
            if not selected or selected[-1] > len(reader.pages):
                raise ValueError("选页超出PDF实际页数，或PDF没有页面。")
            pages = [{"page_number": number, "text": page.extract_text() or ""}
                     for number, page in enumerate(reader.pages, start=1)]
        selected_set = set(selected)
        content = "\n\n".join(page["text"] for page in pages if page["page_number"] in selected_set)
        return [RagDocument(
            doc_id=self.doc_id, source=self.input_path.as_posix(), title=self.title,
            content=content,
            metadata={
                **self.metadata,
                "loader_type": "pdf_plain", "source_type": "local_file",
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "pdf_page_count": len(pages), "selected_page_numbers": selected,
                "pdf_pages": pages,
            },
        )]
