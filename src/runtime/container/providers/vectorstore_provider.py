from langchain_chroma import Chroma
from pathlib import Path

from src.settings import settings

from src.logger import logger


class VectorStoreProvider:

    def __init__(
        self, embedding_provider, *, persist_directory: str | Path | None = None,
        collection_name: str | None = None, enable_memory: bool = True,
        create_collection_if_not_exists: bool = True,
        collection_configuration: dict | None = None,
    ):
        """配置向量库服务。

        参数：embedding_provider为向量模型服务；仅元数据审计时可传None，且必须
            关闭enable_memory，不提供向量化能力；persist_directory为可选存储目录；
            collection_name为可选集合名称，未指定时保持Chroma原默认集合；
            enable_memory表示启动时是否初始化记忆库，默认兼容旧行为。
            create_collection_if_not_exists表示集合不存在时是否允许自动创建；
            查询入口设为False以避免名称错误被误当成空库，默认仍兼容旧调用。
            collection_configuration为可选索引配置，用于维护时保持源集合距离参数。
        返回值：无；不在构造阶段连接数据库。
        """

        if embedding_provider is None and enable_memory:
            raise ValueError("无向量模型的审计模式必须关闭记忆初始化。")
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.enable_memory = enable_memory
        self.create_collection_if_not_exists = create_collection_if_not_exists
        self.collection_configuration = collection_configuration

        self._db = None

        self._memory_db = None

    @property
    def db(self):
        """
        默认 VectorStore

        中文释义：
        - VectorStore：向量数据库
        - db：默认向量库，主要给普通 RAG 使用
        """

        if self._db is None:

            logger.info(
                "🚀 初始化 VectorStore..."
            )

            collection_options = (
                {"collection_name": self.collection_name}
                if self.collection_name is not None else {}
            )
            if not self.create_collection_if_not_exists:
                collection_options["create_collection_if_not_exists"] = False
            if self.collection_configuration is not None:
                collection_options["collection_configuration"] = self.collection_configuration
            self._db = Chroma(

                persist_directory=str(
                    self.persist_directory or settings.path.CHROMA_DB_DIR
                ),

                embedding_function=(
                    self.embedding_provider.embedding if self.embedding_provider is not None else None
                ),

                **collection_options,
            )

        return self._db

    @property
    def native_collection(self):
        """获取原生集合供显式向量复制；无参数；返回Chroma集合，不自行读写记录。

        LangChain封装没有显式向量写入接口，兼容访问集中在提供者内。
        此属性使用包装库的私有_collection接口，升级依赖时需验证兼容。
        """
        return self.db._collection

    @property
    def memory_db(self):

        if self._memory_db is None:
            logger.info(
                "🚀 初始化 Memory VectorStore..."
            )

            memory_db_dir = settings.path.MEMORY_CHROMA_DB_DIR

            memory_db_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            logger.info(
                f"Memory Chroma DB Dir: {memory_db_dir}"
            )

            self._memory_db = Chroma(
                persist_directory=str(
                    memory_db_dir
                ),
                embedding_function=(
                    self.embedding_provider.memory_embedding
                ),
                collection_name="memory_vectors",
            )

        return self._memory_db

    async def startup(self):

        # 提前初始化默认向量库
        _ = self.db

        # 提前初始化 Memory 向量库
        if self.enable_memory:
            _ = self.memory_db

        logger.info(
            "VectorStoreProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "VectorStoreProvider 已关闭"
        )
