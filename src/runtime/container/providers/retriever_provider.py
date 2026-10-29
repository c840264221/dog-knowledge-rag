from src.logger import logger

from src.rag.query_parsers.dog_query_filter_parser import (
    DogQueryFilterParser
)

from src.rag.retrievers.metadata_filter_retriever import (
    MetadataFilterRetriever
)


class RetrieverProvider:
    """
    Retriever Provider（召回器提供者）。

    功能：
        统一管理 RAG 运行时检索相关能力。

        当前负责提供：
        1. MetadataFilterRetriever：
           元数据过滤召回器，用于基于 Chroma metadata filter 和向量检索召回 RagContext。

        2. DogQueryFilterParser：
           狗狗查询过滤解析器，用于把用户自然语言问题解析成 RagQuery。

    技术名词：
        Retriever：
            召回器 / 检索器，负责从向量数据库中取回和用户问题相关的文本块。

        Provider：
            提供者，负责统一创建、缓存、管理服务对象。

        VectorStore：
            向量数据库，用于保存 embedding 后的文本块，并支持相似度搜索。

    参数：
        vectorstore_provider:
            VectorStoreProvider 实例。
            中文释义：向量库提供者，用于提供默认 RAG 向量库 db。

        default_top_k:
            默认召回数量。
            当外部没有指定 top_k 时，MetadataFilterRetriever 会使用该值。

    返回值：
        无直接返回值。该类实例会被注册进 RuntimeContainer。
    """

    def __init__(
            self,
            vectorstore_provider,
            default_top_k: int = 5,
            collection_providers: dict | None = None,
            collection_descriptions: dict[str, str] | None = None,
    ):
        """
        初始化 RetrieverProvider。

        功能：
            保存 vectorstore_provider 和默认 top_k。
            Retriever 和 Parser 采用懒加载方式创建，避免容器初始化阶段过早创建对象。

        参数：
            vectorstore_provider:
                VectorStoreProvider 实例。

            default_top_k:
                默认召回数量。
            collection_providers:
                可选的可信集合标识到向量库提供者映射；查询时懒加载，由本提供者关闭。
            collection_descriptions:
                可选的同标识内容说明，供查询理解选择资料范围，不包含访问授权。

        返回值：
            None。
        """

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k 必须大于 0"
            )

        self.vectorstore_provider = vectorstore_provider
        self.default_top_k = default_top_k
        # 子提供者归此服务管理，按实际选中的集合懒加载，不新增Agent。
        self.collection_providers = dict(collection_providers or {})
        self.collection_descriptions = dict(collection_descriptions or {})

        self._metadata_filter_retriever = None
        self._dog_query_filter_parser = None

    def knowledge_catalog(self) -> dict[str, str]:
        """提供当前可用公共集合说明；无参数；返回标识到说明，不暴露路径或授予权限。"""
        return {key: self.collection_descriptions.get(key, key) for key in self.collection_providers}

    def get_collection_retriever(self, collection_id: str) -> MetadataFilterRetriever:
        """按可信目录解析检索器；参数为集合标识；返回禁止放宽条件的检索器，未知集合报错。"""
        if collection_id not in self.collection_providers:
            raise ValueError(f"不允许查询未知集合：{collection_id}")
        return MetadataFilterRetriever(self.collection_providers[collection_id].db,
            default_top_k=self.default_top_k, allow_filter_relaxation=False)

    @property
    def metadata_filter_retriever(self) -> MetadataFilterRetriever:
        """
        获取 MetadataFilterRetriever 实例。

        功能：
            通过懒加载方式创建元数据过滤召回器。
            内部使用 vectorstore_provider.db 作为默认 RAG 向量库。

        参数：
            无。

        返回值：
            MetadataFilterRetriever：
                元数据过滤召回器实例。
        """

        if self._metadata_filter_retriever is None:

            logger.info(
                "🚀 初始化 MetadataFilterRetriever..."
            )

            self._metadata_filter_retriever = MetadataFilterRetriever(
                vector_store=self.vectorstore_provider.db,
                default_top_k=self.default_top_k,
            )

        return self._metadata_filter_retriever

    @property
    def dog_query_filter_parser(self) -> DogQueryFilterParser:
        """
        获取 DogQueryFilterParser 实例。

        功能：
            通过懒加载方式创建狗狗查询过滤解析器。
            Parser 负责把用户自然语言问题转换成 RagQuery。

        参数：
            无。

        返回值：
            DogQueryFilterParser：
                狗狗查询过滤解析器实例。
        """

        if self._dog_query_filter_parser is None:

            logger.info(
                "🚀 初始化 DogQueryFilterParser..."
            )

            self._dog_query_filter_parser = DogQueryFilterParser()

        return self._dog_query_filter_parser

    async def startup(self):
        """
        启动 RetrieverProvider。

        功能：
            提前初始化 MetadataFilterRetriever 和 DogQueryFilterParser。
            这样可以在应用启动阶段提前暴露配置或依赖问题。

        参数：
            无。

        返回值：
            None。
        """

        _ = self.metadata_filter_retriever
        _ = self.dog_query_filter_parser

        logger.info(
            "RetrieverProvider 启动完成"
        )

    async def shutdown(self):
        """
        关闭 RetrieverProvider。

        功能：
            当前 RetrieverProvider 没有需要主动释放的资源。
            该方法主要用于保持 Provider 生命周期接口统一。

        参数：
            无。

        返回值：
            None。
        """

        for provider in self.collection_providers.values():
            await provider.shutdown()
        logger.info(
            "RetrieverProvider 已关闭"
        )
