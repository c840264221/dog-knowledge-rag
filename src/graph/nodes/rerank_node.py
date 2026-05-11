from src.retrieval.retriever import rerank_docs
from src.logger import logger

def rerank_node(state):
    logger.info(f"进入rerank_node节点 state为{state}")
    docs = rerank_docs(state["question"], state["docs"], state["intent"], top_k=3)
    logger.debug(f"rerank结束，结果docs的数量为：{len(docs)}")
    return {"docs": docs}