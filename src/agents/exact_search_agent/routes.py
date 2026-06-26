from src.logger import logger


VALID_EXACT_WORKER_ROUTES = {
    "filter",
    "retrieve",
    "evaluate",
    "retry",
    "rerank",
    "generate",
    "finish",
}


def route_exact_worker(
        state,
) -> str:
    """
    exact_search_agent supervisor 后的路由函数。

    功能：
        从 state["next_worker"] 中读取 supervisor 决策，
        并返回 LangGraph conditional_edges 使用的路由 key。

        v1.5 兼容逻辑：
        1. 如果 next_worker 缺失，默认进入 retrieve。
        2. 如果 next_worker 是旧版 filter，允许返回 filter，
           由 agent.py 中的 conditional_edges 把 filter 重定向到 retrieve。
        3. 如果 next_worker 非法，默认进入 retrieve。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            路由 key。
    """

    next_worker = state.get(
        "next_worker",
        "retrieve",
    )

    if next_worker not in VALID_EXACT_WORKER_ROUTES:
        logger.warning(
            f"exact_search_agent 收到非法 next_worker={next_worker!r}，"
            "已兜底路由到 retrieve。"
        )

        return "retrieve"

    return next_worker