from abc import ABC, abstractmethod

from src.rag.schemas import RagDocument


class BaseDocumentLoader(ABC):
    """
    文档加载器抽象基类。

    功能：
        定义所有 Document Loader（文档加载器）必须遵守的统一接口。
        Loader（加载器）负责把外部数据源，例如 Markdown 文件、PDF 文件、
        网页内容、数据库记录，转换成 RAG 系统内部标准的 RagDocument。

    设计说明：
        这里使用 Abstract Base Class（抽象基类），是为了让不同类型的 Loader
        都实现相同的 load 方法，方便后续 Index Pipeline（索引管线）统一调用。

    参数：
        无。

    返回值：
        这是抽象基类，本身不会直接返回业务数据。
        子类需要实现 load 方法，并返回 list[RagDocument]。
    """

    @abstractmethod
    def load(self) -> list[RagDocument]:
        """
        加载文档。

        功能：
            从指定数据源读取原始内容，并转换成 RagDocument 列表。

        参数：
            无。

        返回值：
            list[RagDocument]:
                RAG 原始文档列表。
        """

        raise NotImplementedError