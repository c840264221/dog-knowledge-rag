from src.logger import logger


def retrieval_retry_node(state):
    print("retrieval_retry_node", state)
    logger.info(f"进入retrieval_retry_node重试节点，state：{state}")

    retry_count = state.get("retry_count", 0)
    logger.debug(f"retry_count次数为：{retry_count}")

    filters = state.get("filters", {}).copy()
    tags = state.get("tags", []).copy()
    top_k = state.get("top_k", 5)

    # 第一次失败：去掉tags
    if retry_count == 0:
        logger.info(f"第一次重试")
        tags = []

    # 第二次失败：扩大topK
    elif retry_count == 1:
        logger.info(f"第二次重试")
        top_k = 15

    # 第三次失败：放宽filter
    elif retry_count == 2:
        logger.info(f"第三次重试")
        filters.pop("barking", None)

    return {
        "retry_count": retry_count + 1,
        "filters": filters,
        "tags": tags,
        "top_k": top_k
    }