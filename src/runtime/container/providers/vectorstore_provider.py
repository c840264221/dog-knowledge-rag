from langchain_chroma import Chroma

from src.settings import settings

from src.logger import logger


class VectorStoreProvider:

    def __init__(self,embedding_provider):

        self.embedding_provider = embedding_provider

        self._db = None

    @property
    def db(self):

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
                )
            )

        return self._db

    async def startup(self):

        # 提前初始化
        _ = self.db

        logger.info(
            "VectorStoreProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "VectorStoreProvider 已关闭"
        )