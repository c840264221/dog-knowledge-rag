from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END


from src.graph.states.state import DogState

from src.graph.nodes.parse_node import parse_node
from src.graph.nodes.router_node import strategy_router_node

from src.graph.sub_graphs.recommend_subgraph import (
    build_recommendation_subgraph_with_human
)

from src.graph.sub_graphs.exact_search_subgraph import (
    build_exact_search_graph
)

from src.graph.sub_graphs.qa_subgraph import (
    build_general_qa_subgraph
)

from src.agents.general_qa_agent.agent import build_general_qa_agent

from src.agents.recommendation_agent.agent import build_recommendation_agent

from src.agents.exact_search_agent.agent import build_exact_search_agent

from src.agents.supervisor.supervisor_node import supervisor_node

from src.graph.nodes.router_node import semantic_router_node

from src.graph.nodes.memory_extract_node import memory_extract_node

from src.graph.routes.route_by_strategy import route_by_strategy
import sqlite3

from src.logger import logger

from src.config import CHECKPOINTS_DB_PATH

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import aiosqlite

import asyncio

import atexit


_app_2 = None
_saver = None
_saver_cm = None


async def init_checkpointer():
    """初始化 checkpointer，并设置退出清理"""
    global checkpointer
    # 直接获取异步上下文管理器，手动进入而不退出
    # AsyncSqliteSaver.from_conn_string 返回一个 AsyncContextManager
    cm = AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB_PATH)
    checkpointer = await cm.__aenter__()

    # 注册退出时的清理（使用已有的事件循环或新建）
    def cleanup():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(checkpointer.__aexit__(None, None, None))
            else:
                loop.run_until_complete(checkpointer.__aexit__(None, None, None))
        except Exception:
            pass

    atexit.register(cleanup)
    return checkpointer

async def build_main_graph():
    # 由于前端和后端部分代码均改为了异步 所以这里也要用AsyncSqliteSaver来作为checkpointer 原来的sqlite3不支持异步
    # async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB_PATH) as checkpointer:

    # 调用全局checkpointer 防止线程被重复启用
    # global compiled_graph
    # if checkpointer is None:
    #     await init_checkpointer()

    global _app_2, _saver, _saver_cm

    if _app_2 is not None:
        return _app_2

    _saver_cm = AsyncSqliteSaver.from_conn_string(
        CHECKPOINTS_DB_PATH
    )

    _saver = await _saver_cm.__aenter__()
    # 注册退出时清理
    # def cleanup():
    #     if _saver is not None:
    #         try:
    #             # 获取当前事件循环，如果正在运行则创建任务，否则同步运行
    #             loop = asyncio.get_event_loop()
    #             if loop.is_running():
    #                 loop.create_task(_saver.__aexit__(None, None, None))
    #             else:
    #                 loop.run_until_complete(_saver.__aexit__(None, None, None))
    #         except Exception:
    #             pass
    #
    # atexit.register(cleanup)


    logger.info("Async checkpointer 已就绪")
    logger.info(f"主图构建中......")
    # recommendation_graph = build_recommendation_subgraph()
    # recommendation_agent = build_recommendation_subgraph_with_human()
    recommendation_agent = build_recommendation_agent()

    exact_graph = build_exact_search_graph()
    exact_agent = build_exact_search_agent()

    # general_graph = build_general_qa_graph()
    # general_graph = build_general_qa_subgraph()
    general_qa_agent = build_general_qa_agent()
    #
    # def recommendation_node(state):
    #     return recommendation_graph.invoke(state)
    #
    # def exact_node(state):
    #     return exact_graph.invoke(state)
    #
    # def general_node(state):
    #     return general_graph.invoke(state)

    graph = StateGraph(DogState)

    # graph.add_node("supervisor", supervisor_node)

    graph.add_node(
        "memory_extract",
        memory_extract_node
    )

    graph.add_node(
        "semantic_router",
        semantic_router_node
    )

    # graph.add_node("parse", parse_node)

    # graph.add_node("router", strategy_router_node)

    # graph.add_node("recommendation", recommendation_node)
    graph.add_node("recommendation_agent", recommendation_agent)

    # graph.add_node("exact", exact_node)
    graph.add_node("exact", exact_agent)
    graph.add_node("general", general_qa_agent)

    # graph.set_entry_point("parse")
    # graph.set_entry_point("semantic_router")
    graph.set_entry_point("memory_extract")

    graph.add_edge("memory_extract", "semantic_router")

    # graph.add_edge("parse", "router")

    graph.add_conditional_edges(

        "semantic_router",

        lambda s: s["next_agent"],

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

    # graph.add_conditional_edges(
    #     "router",
    #     route_by_strategy,
    #     {
    #         # "filtered": "recommendation",
    #         "filtered": "recommendation_agent",
    #         "exact": "exact",
    #         "semantic": "general",
    #         "direct": "general"
    #     }
    # )
    logger.info(f"主图构建完成...")

    # from src.config import CHECKPOINTS_DB_PATH
    #
    # conn = sqlite3.connect(CHECKPOINTS_DB_PATH, check_same_thread=False)
    # checkpointer = SqliteSaver(conn)
    #
    # logger.info(f"checkpointer已就绪")

    # from langgraph.checkpoint.memory import MemorySaver
    # return graph.compile(checkpointer=MemorySaver())

    _app_2 = graph.compile(checkpointer=_saver)
    return _app_2


