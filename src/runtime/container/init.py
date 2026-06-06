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

# 创建容器
container = RuntimeContainer()

# =========================
# 注册 Provider
# =========================

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

    "graph_runtime",

    GraphRuntimeService()
)

# container.register(
#
#     "runtime_persistence",
#
#     RuntimePersistenceProvider()
# )


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