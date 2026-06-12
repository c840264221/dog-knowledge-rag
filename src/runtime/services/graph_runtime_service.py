from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver
)

from langgraph.graph import StateGraph

from src.graph.states.state import (
    DogState
)

from src.logger import logger

from src.settings import settings

from src.graph.nodes.memory_extract_node import (
    memory_extract_node
)

from src.graph.nodes.router_node import (
    semantic_router_node
)

from src.agents.general_qa_agent.agent import (
    build_general_qa_agent
)

from src.agents.recommendation_agent.agent import (
    build_recommendation_agent
)

from src.agents.exact_search_agent.agent import (
    build_exact_search_agent
)

from src.graph.routes.route_after_semantic import route_after_semantic

from langgraph.graph import END


class GraphRuntimeService:

    def __init__(
            self,
            llm_provider=None,
            memory_provider=None,
            checkpoint_provider=None,
    ):
        """
        初始化 GraphRuntimeService。

        功能：
        - 管理主 Graph 的构建与生命周期
        - 管理 LangGraph checkpointer
        - 接收 LLMProvider，用于注入生成节点
        - 接收 MemoryProvider，用于注入记忆召回能力
        - 接收 CheckpointProvider，用于注入 checkpoint 保存能力
        - 避免 Graph Node 内部直接 import container

        参数：
        - llm_provider:
          LLMProvider 实例。
          中文释义：统一管理主模型、备用模型、安全调用等 LLM 能力。

        - memory_provider:
          MemoryProvider 实例。
          中文释义：统一管理用户长期记忆的保存与召回能力。

        - checkpoint_provider:
          CheckpointProvider 实例。
          中文释义：统一管理运行时检查点保存能力。

        返回值：
        - None
          初始化函数不返回业务数据。
        """

        self.llm_provider = llm_provider

        self.memory_provider = memory_provider

        self.checkpoint_provider = checkpoint_provider

        self._graph = None

        self._checkpointer = None

        self._checkpointer_cm = None

    # startup
    async def startup(self):

        logger.info(
            "🚀 GraphRuntime 启动中..."
        )

        # 初始化 checkpointer
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

        # 构建 graph
        self._graph = await self._build_graph()

        logger.info(
            "✅ GraphRuntime 启动完成"
        )


    # shutdown
    async def shutdown(self):

        logger.info(
            "🛑 GraphRuntime 关闭中..."
        )

        if self._checkpointer_cm:

            await self._checkpointer_cm.__aexit__(
                None,
                None,
                None
            )

        logger.info(
            "✅ GraphRuntime 已关闭"
        )

    # =========================
    # graph property
    # =========================

    @property
    def graph(self):

        if self._graph is None:

            raise RuntimeError(
                "Graph 尚未初始化"
            )

        return self._graph

    # =========================
    # build graph
    # =========================

    async def _build_graph(self):

        logger.info(
            "构建主图中..."
        )

        recommendation_agent = (
            build_recommendation_agent(
                llm_provider=(
                    self.llm_provider
                ),
                memory_provider=(
                    self.memory_provider
                ),
                checkpoint_provider=(
                    self.checkpoint_provider
                )
            )
        )

        exact_agent = (
            build_exact_search_agent(
                llm_provider=(
                    self.llm_provider
                ),
                memory_provider=(
                    self.memory_provider
                ),
                checkpoint_provider=(
                    self.checkpoint_provider
                )
            )
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
                )
            )
        )

        graph = StateGraph(DogState)

        graph.add_node(
            "memory_extract",
            memory_extract_node
        )

        graph.add_node(
            "semantic_router",
            semantic_router_node
        )

        graph.add_node(
            "recommendation_agent",
            recommendation_agent
        )

        graph.add_node(
            "exact",
            exact_agent
        )

        graph.add_node(
            "general",
            general_agent
        )

        graph.set_entry_point(
            "memory_extract"
        )

        graph.add_edge(
            "memory_extract",
            "semantic_router"
        )

        graph.add_conditional_edges(

            "semantic_router",

            # lambda s: s["next_agent"],
            route_after_semantic,

            {

                "recommendation_agent":
                    "recommendation_agent",

                "exact_agent":
                    "exact",

                "general_agent":
                    "general",

                "FINISH":
                    END
            }
        )

        graph.add_edge(
            "recommendation_agent",
            END
        )

        graph.add_edge(
            "exact",
            END
        )

        graph.add_edge(
            "general",
            END
        )

        logger.info(
            "✅ 主图构建完成"
        )

        return graph.compile(
            checkpointer=self._checkpointer
        )