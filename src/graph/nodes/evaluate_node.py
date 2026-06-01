from src.logger import logger

from src.runtime.context import runtime_ctx


def evaluate_retrieval_node(state):
    runtime_ctx.get().state().set_node(
        "evaluate_retrieval_node"
    )

    logger.info(f"进入evaluate_retrieval_node节点，开始校验检索出的数据是否合格")
    docs = state.get("docs", [])

    enough = len(docs) >= 2
    logger.debug(f"evaluate_retrieval_node完成，校验的结果enough：{enough}")
    return {
        "retrieval_ok": enough
    }