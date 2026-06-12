from langchain_chroma import Chroma

from src.settings import settings

from src.logger import logger


class VectorStoreProvider:

    def __init__(self,embedding_provider):

        self.embedding_provider = embedding_provider

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

            self._db = Chroma(

                persist_directory=str(
                    settings.path.CHROMA_DB_DIR
                ),

                embedding_function=(
                    self.embedding_provider.embedding
                ),

                # collection_name="default_documents",
            )

        return self._db

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
        _ = self.memory_db

        logger.info(
            "VectorStoreProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "VectorStoreProvider 已关闭"
        )