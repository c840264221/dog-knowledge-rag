from src.logger import logger


def evaluate_retrieval_node(state):
    logger.info(f"进入evaluate_retrieval_node节点，开始校验检索出的数据是否合格")
    docs = state.get("docs", [])

    enough = len(docs) >= 2
    logger.debug(f"evaluate_retrieval_node完成，校验的结果enough：{enough}")
    return {
        "retrieval_ok": enough
    }