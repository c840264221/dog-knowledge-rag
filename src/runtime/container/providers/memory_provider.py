from src.memory.memory_manager import (
    MemoryManager
)

from src.memory.memory_semantic_recall import (
    MemorySemanticRecallService
)

from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore
)

from src.logger import logger

from src.settings import settings

from src.memory.memory_ranker import (
    MemoryRanker
)



class MemoryProvider:


    def __init__(
            self,
            vectorstore_provider
    ):
        """
        初始化 MemoryProvider。

        功能：
        - 统一管理 SQLiteMemoryStore
        - 统一管理 MemoryManager
        - 统一管理 MemoryRanker
        - 统一管理 MemorySemanticRecallService
        - 将 Memory 模块接入 Container 生命周期

        参数：
        - vectorstore_provider:
          VectorStoreProvider 实例。
          中文释义：用于访问 Memory 专用向量数据库 memory_db。

        返回值：
        - None
          初始化函数不返回业务数据。
        """

        self.vectorstore_provider = vectorstore_provider

        self._store = None

        self._manager = None

        self._semantic_recall = None

        self._ranker = None

    @property
    def store(
            self
    ) -> SQLiteMemoryStore:
        """
        获取 SQLiteMemoryStore 实例。

        功能：
        - 懒加载 SQLiteMemoryStore
        - 统一管理 Memory 主数据库连接
        - 避免 memory 模块内部各自创建数据库连接

        参数：
        - 无

        返回值：
        - SQLiteMemoryStore
          Memory 主数据库存储服务。
        """

        if self._store is None:
            logger.info(
                "🚀 初始化 SQLiteMemoryStore..."
            )

            self._store = SQLiteMemoryStore()

        return self._store

    @property
    def manager(
            self
    ) -> MemoryManager:
        """
        获取 MemoryManager 实例。

        功能：
        - 懒加载 MemoryManager
        - 注入 SQLiteMemoryStore
        - 注入 VectorStoreProvider
        - 统一管理 Memory 保存、强化、冲突处理、向量同步流程

        参数：
        - 无

        返回值：
        - MemoryManager
          记忆管理器实例。
        """

        if self._manager is None:
            logger.info(
                "🚀 初始化 MemoryManager..."
            )

            self._manager = MemoryManager(
                store=self.store,
                vectorstore_provider=(
                    self.vectorstore_provider
                )
            )

        return self._manager

    @property
    def ranker(
            self
    ) -> MemoryRanker:
        """
        获取 MemoryRanker 实例。

        功能：
        - 懒加载 MemoryRanker
        - 从 settings.memory 读取排序权重
        - 统一管理 Memory 召回结果的精排逻辑

        参数：
        - 无

        返回值：
        - MemoryRanker
          记忆精排器实例。
        """

        if self._ranker is None:
            logger.info(
                "🚀 初始化 MemoryRanker..."
            )

            self._ranker = MemoryRanker(
                semantic_weight=(
                    settings.memory.semantic_weight
                ),
                memory_weight=(
                    settings.memory.memory_weight
                ),
                confidence_weight=(
                    settings.memory.confidence_weight
                ),
            )

        return self._ranker

    @property
    def semantic_recall(
            self
    ) -> MemorySemanticRecallService:
        """
        获取 MemorySemanticRecallService 实例。

        功能：
        - 懒加载 Memory 语义召回服务
        - 注入 SQLiteMemoryStore
        - 注入 VectorStoreProvider
        - 注入 MemoryRanker
        - 外部应通过 container.get("memory").semantic_recall 使用语义召回

        参数：
        - 无

        返回值：
        - MemorySemanticRecallService
          Memory 语义召回服务实例。
        """

        if self._semantic_recall is None:
            logger.info(
                "🚀 初始化 MemorySemanticRecallService..."
            )

            self._semantic_recall = MemorySemanticRecallService(
                store=self.store,
                vectorstore_provider=(
                    self.vectorstore_provider
                ),
                memory_ranker=(
                    self.ranker
                )
            )

        return self._semantic_recall

    async def startup(
            self
    ):
        """
        启动 MemoryProvider。

        功能：
        - 提前初始化 SQLiteMemoryStore
        - 提前初始化 MemoryManager
        - 提前初始化 MemoryRanker
        - 提前初始化 MemorySemanticRecallService
        - 接入 Container startup 生命周期

        参数：
        - 无

        返回值：
        - None
          只执行初始化逻辑。
        """

        _ = self.store

        _ = self.manager

        _ = self.ranker

        _ = self.semantic_recall

        logger.info(
            "MemoryProvider 启动完成"
        )

    async def shutdown(
            self
    ):
        """
        关闭 MemoryProvider。

        功能：
        - 关闭 SQLiteMemoryStore 数据库连接
        - 接入 Container shutdown 生命周期
        - 避免依赖 atexit 这种隐式关闭方式

        参数：
        - 无

        返回值：
        - None
          只执行关闭逻辑。
        """

        if self._store is not None:

            self._store.close()

            logger.info(
                "SQLiteMemoryStore 已关闭"
            )

        logger.info(
            "MemoryProvider 已关闭"
        )