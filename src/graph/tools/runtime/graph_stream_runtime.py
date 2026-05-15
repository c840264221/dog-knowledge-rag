from src.logger import logger


def safe_stream_graph(graph,state,config,stream_mode="values",fallback=None):

    try:
        for chunk in graph.stream(
            state,
            config,
            stream_mode=stream_mode
        ):
            yield chunk

    except Exception as e:
        logger.exception(
            f"Graph stream执行失败: {e}"
        )

        if fallback:
            yield fallback(state,e)

        else:
            yield {
                "error": str(e),
                "graph_failed": True
            }