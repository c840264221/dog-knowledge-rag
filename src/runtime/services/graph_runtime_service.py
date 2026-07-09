from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)

from langgraph.graph import (
    StateGraph,
    END,
)

from langgraph.types import (
    interrupt,
)

from src.graph.states.dog_state import (
    DogState,
)

from src.logger import logger

from src.settings import settings

from src.graph.nodes.memory_extract_node import (
    memory_extract_node,
)

from src.graph.nodes.router_node import (
    semantic_router_node,
)

from src.agents.general_qa_agent.agent import (
    build_general_qa_agent,
)

from src.agents.dog_knowledge_agent.agent import (
    build_dog_knowledge_agent,
)

from src.agents.tool_agent.graph import (
    build_tool_agent_graph,
)

from src.graph.routes.route_after_semantic import (
    route_after_semantic,
)

# 导入主图语义路由映射对应的agent名字
from src.graph.routes.main_route_alias import (
    build_main_route_alias_map,
)

from src.agents.dog_knowledge_agent.adapters.entry_integration import (
    build_integrated_dog_knowledge_entry_node,
)


class GraphRuntimeService:
    """
    Graph 运行时服务。

    功能：
        负责主图 Main Graph 的构建、启动、关闭和运行时依赖注入。

        v1.5 当前改造重点：
        1. 主图仍然保留 semantic_router_node 的旧路由输出。
        2. semantic_router_node 可以继续输出 recommendation_agent / exact_agent。
        3. 主图通过 route alias 将 recommendation_agent / exact_agent 映射到 dog_knowledge_agent。
        4. dog_knowledge_agent 内部再决定进入 extract_model 或 recommendation_model。
        5. general_agent 暂时保持原逻辑不变。

    技术名词：
        Main Graph：
            主图。Dog Agent Framework 最外层 LangGraph。

        Runtime Service：
            运行时服务。负责管理 graph、checkpointer、provider 注入等运行能力。

        Route Alias：
            路由别名。表示旧路由名称映射到新节点，方便灰度迁移。

        Provider Injection：
            Provider 注入。把 LLM、Memory、Retriever、Reranker 等外部能力传入 Agent，
            避免节点内部直接 import container。
    """

    def __init__(
            self,
            llm_provider=None,
            memory_provider=None,
            checkpoint_provider=None,
            retriever_provider=None,
            reranker_provider=None,
    ):
        """
        初始化 GraphRuntimeService。

        功能：
            管理主 Graph 的构建与生命周期。
            接收外部 Provider，并在构建 Agent 时注入。

        参数：
            llm_provider:
                LLMProvider 实例。
                中文释义：统一管理主模型、备用模型、安全调用等 LLM 能力。

            memory_provider:
                MemoryProvider 实例。
                中文释义：统一管理用户长期记忆的保存与召回能力。

            checkpoint_provider:
                CheckpointProvider 实例。
                中文释义：统一管理运行时检查点保存能力。

            retriever_provider:
                RetrieverProvider 实例。
                中文释义：统一管理 RAG 检索相关能力。

            reranker_provider:
                RerankerProvider 实例。
                中文释义：统一管理 reranker 重排序模型能力。

        返回值：
            None:
                初始化函数不返回业务数据。
        """

        self.llm_provider = llm_provider

        self.memory_provider = memory_provider

        self.checkpoint_provider = checkpoint_provider

        self.retriever_provider = retriever_provider

        self.reranker_provider = reranker_provider

        self._graph = None

        self._checkpointer = None

        self._checkpointer_cm = None

    async def startup(self):
        """
        启动 GraphRuntimeService。

        功能：
            1. 初始化 AsyncSqliteSaver checkpointer。
            2. 构建主图。
            3. 将编译后的 graph 保存到 self._graph。

        参数：
            无。

        返回值：
            None。
        """

        logger.info(
            "🚀 GraphRuntime 启动中..."
        )

        self._checkpointer_cm = (
            AsyncSqliteSaver.from_conn_string(
                str(
                    settings.path.CHECKPOINTS_DB_PATH
                )
            )
        )

        self._checkpointer = (
            await self._checkpointer_cm.__aenter__()
        )

        logger.info(
            "✅ AsyncSqliteSaver 已初始化"
        )

        self._graph = await self._build_graph()

        logger.info(
            "✅ GraphRuntime 启动完成"
        )

    async def shutdown(self):
        """
        关闭 GraphRuntimeService。

        功能：
            关闭 AsyncSqliteSaver checkpointer 的异步上下文。

        参数：
            无。

        返回值：
            None。
        """

        logger.info(
            "🛑 GraphRuntime 关闭中..."
        )

        if self._checkpointer_cm:

            await self._checkpointer_cm.__aexit__(
                None,
                None,
                None,
            )

        logger.info(
            "✅ GraphRuntime 已关闭"
        )

    @property
    def graph(self):
        """
        获取已经构建完成的主图。

        功能：
            如果 graph 尚未初始化，则抛出 RuntimeError。

        参数：
            无。

        返回值：
            compiled graph:
                编译后的 LangGraph 主图。
        """

        if self._graph is None:

            raise RuntimeError(
                "Graph 尚未初始化"
            )

        return self._graph

    async def _build_graph(self):
        """
        构建主图。

        功能：
            构建 Dog Agent Framework 的 Main Graph。

            v1.5 当前主图结构：

                memory_extract
                    ↓
                semantic_router
                    ↓
                route_after_semantic
                    ├── recommendation_agent -> dog_knowledge_agent
                    ├── exact_agent          -> dog_knowledge_agent
                    ├── general_agent        -> general
                    ├── tool_agent           -> tool_agent
                    └── FINISH               -> END

            重要说明：
                semantic_router_node 暂时仍然输出旧路由：
                - recommendation_agent
                - exact_agent
                - general_agent

                主图通过 conditional_edges 映射把：
                - recommendation_agent
                - exact_agent

                都导向新的：
                - dog_knowledge_agent

                这样可以保留旧 route_decision 的语义，
                同时实际执行新 dog_knowledge_agent。

        参数：
            无。

        返回值：
            compiled graph:
                编译后的 LangGraph 主图。
        """

        logger.info(
            "构建主图中..."
        )

        dog_knowledge_agent = build_dog_knowledge_agent(
            llm_provider=(
                self.llm_provider
            ),
            memory_provider=(
                self.memory_provider
            ),
            checkpoint_provider=(
                self.checkpoint_provider
            ),
            retriever_provider=(
                self.retriever_provider
            ),
            reranker_provider=(
                self.reranker_provider
            ),
        )

        general_agent = (
            build_general_qa_agent(
                llm_provider=(
                    self.llm_provider
                ),
                memory_provider=(
                    self.memory_provider
                ),
                checkpoint_provider=(
                    self.checkpoint_provider
                ),
            )
        )

        # 构建新版 ToolAgent 子图，用来承接 RootAgent 路由出的工具请求。
        tool_agent = build_tool_agent_graph(
            llm_provider=(
                self.llm_provider
            ),
            checkpoint_manager=(
                self.checkpoint_provider.manager
                if self.checkpoint_provider is not None
                else None
            ),
            interrupt_func=interrupt,
        )

        graph = StateGraph(
            DogState
        )

        graph.add_node(
            "memory_extract",
            memory_extract_node,
        )

        graph.add_node(
            "semantic_router",
            semantic_router_node,
        )

        graph.add_node(
            "dog_knowledge_agent",
            build_integrated_dog_knowledge_entry_node(
                delegate_node=dog_knowledge_agent,
            ),
        )

        graph.add_node(
            "general",
            general_agent,
        )

        # 注册新版 ToolAgent 节点，主图路由 key 为 tool_agent 时会进入这里。
        graph.add_node(
            "tool_agent",
            tool_agent,
        )

        graph.set_entry_point(
            "memory_extract"
        )

        graph.add_edge(
            "memory_extract",
            "semantic_router",
        )

        graph.add_conditional_edges(
            "semantic_router",
            route_after_semantic,
            build_main_route_alias_map(
                end_node=END,
            ),
        )

        graph.add_edge(
            "dog_knowledge_agent",
            END,
        )

        graph.add_edge(
            "general",
            END,
        )

        graph.add_edge(
            "tool_agent",
            END,
        )

        logger.info(
            "✅ 主图构建完成"
        )

        return graph.compile(
            checkpointer=self._checkpointer
        )
