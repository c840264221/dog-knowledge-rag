from src.logger import logger
from src.core.errors.base import DogAgentError

def safe_stream_graph(graph,state,config,stream_mode="values",fallback=None):

    try:
        for chunk in graph.stream(
            state,
            config,
            stream_mode=stream_mode
        ):
            yield chunk

    except DogAgentError as e:

        logger.warning(
            f"Agent错误: {e.message}"
        )

        if e.recoverable:

            yield {
                "error": e.message,
                "recoverable": True
            }

        else:

            yield {
                "error": e.message,
                "recoverable": False,
                "graph_failed": True
            }