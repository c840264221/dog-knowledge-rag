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

    def __init__(self):

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
            build_recommendation_agent()
        )

        exact_agent = (
            build_exact_search_agent()
        )

        general_agent = (
            build_general_qa_agent()
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