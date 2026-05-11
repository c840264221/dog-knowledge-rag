from src.retrieval.filter_by_tags import filter_by_tags
from src.logger import logger

def tag_filter_node(state):
    logger.info(f"进入tag_filter_node节点，state：{state}")
    docs = state["docs"]
    tags = state["tags"]

    if not tags:
        return {"docs": docs}

    filtered = filter_by_tags(docs, tags)
    logger.debug(f"tag_filter_node结束  过滤后的结果为：{filtered}")
    return {"docs": filtered}