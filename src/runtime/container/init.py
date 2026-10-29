from src.runtime.container.core import (
    RuntimeContainer
)

from src.runtime.container.providers.llm_provider import (
    LLMProvider
)

from src.runtime.container.providers.embedding_provider import (
    EmbeddingProvider
)

from src.runtime.container.providers.reranker_provider import (
    RerankerProvider
)

from src.runtime.container.providers.vectorstore_provider import (
    VectorStoreProvider
)

from src.runtime.services.graph_runtime_service import (
    GraphRuntimeService
)

from src.runtime.container.providers.runtime_persistence_provider import RuntimePersistenceProvider

from src.runtime.container.providers.checkpoint_provider import CheckpointProvider

from src.runtime.container.providers.memory_provider import (
    MemoryProvider
)

from src.runtime.container.providers.sqlite_mcp_provider import (
    SQLiteMcpProvider
)

from src.runtime.container.providers.retriever_provider import (
    RetrieverProvider
)
from src.runtime.container.providers.redis_provider import RedisProvider

from src.settings import settings
from src.runtime.container.providers.expense_mcp_provider import ExpenseMcpProvider
from src.mcp.tool_dispatcher import McpToolDispatcher

# 创建容器
container = RuntimeContainer()

# =========================
# 注册 Provider
# =========================

container.register(
    "redis",
    RedisProvider(settings.redis),
)

container.register(

    "llm",

    LLMProvider()
)

container.register(

    "embedding",

    EmbeddingProvider()
)

container.register(

    "reranker",

    RerankerProvider()
)

container.register(

    "vectorstore",

    VectorStoreProvider(
        embedding_provider=(
            container.get("embedding")
        )
    )
)

container.register(

    "retriever",

    RetrieverProvider(
        vectorstore_provider=(
            container.get("vectorstore")
        ),
        collection_providers={key: VectorStoreProvider(
            container.get("embedding"), collection_name=spec["collection_name"],
            enable_memory=False, create_collection_if_not_exists=False,
        ) for key, spec in settings.rag.knowledge_collections.items()},
        collection_descriptions={key: spec["description"]
                                 for key, spec in settings.rag.knowledge_collections.items()},
    )
)


# 注册memory provider
container.register(

    "memory",

    MemoryProvider(
        vectorstore_provider=(
            container.get("vectorstore")
        )
    )
)

persistence_provider = RuntimePersistenceProvider()
checkpoint_provider = (
    CheckpointProvider(
        persistence_provider.persistence
    )
)

container.register(
    "checkpoint",
    checkpoint_provider
)

container.register(

    "sqlite_mcp",

    SQLiteMcpProvider(
        allowed_databases=(
            settings.mcp.sqlite.allowed_databases
        ),
        default_limit=(
            settings.mcp.sqlite.default_limit
        ),
        max_limit=(
            settings.mcp.sqlite.max_limit
        ),
    )
)

container.register("expense_mcp", ExpenseMcpProvider(enabled=settings.mcp.expense.enabled))

# 具体服务只在装配层注册，GraphRuntimeService和ToolAgent只依赖统一入口。
container.register("mcp_tools", McpToolDispatcher(providers={
    "sqlite": container.get("sqlite_mcp") if settings.mcp.sqlite.enabled else None,
    "expenses": container.get("expense_mcp"),
}))

container.register(

    "graph_runtime",

    GraphRuntimeService(
        llm_provider=(
            container.get("llm")
        ),
        memory_provider=(
            container.get("memory")
        ),
        checkpoint_provider=(
            container.get("checkpoint")
        ),
        retriever_provider=(
            container.get("retriever")
        ),
        reranker_provider=(
            container.get("reranker")
        ),
        mcp_provider=container.get("mcp_tools"),
        redis_provider=(
            container.get("redis")
        ),
    )
)

# container.register(
#
#     "runtime_persistence",
#
#     RuntimePersistenceProvider()
# )

