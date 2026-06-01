from src.retrieval.filter_by_tags import filter_by_tags
from src.logger import logger

from src.runtime.context import runtime_ctx

def tag_filter_node(state):

    runtime_ctx.get().state().set_node(
        "tag_filter_node"
    )

    logger.info(f"进入tag_filter_node节点，state："
        f"question:{state['question']}, "
        f"intent:{state['intent']}, "
        f"strategy:{state['strategy']}, "
        f"filters:{state['filters']}, "
        f"tags:{state['tags']}, "
        f"dog_name:{state['dog_name']}, "
        f"docs len:{len(state['docs'])}, ")
    docs = state.get("docs",[])
    tags = state.get("tags",[])

    if not tags:
        return {"docs": docs}

    filtered = filter_by_tags(docs, tags)
    logger.debug(f"tag_filter_node结束  过滤后的结果为：{filtered}")
    return {"docs": filtered}