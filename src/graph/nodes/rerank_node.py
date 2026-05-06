from src.retrieval.retriever import rerank_docs


def rerank_node(state):
    print("开始rerank_node......")
    print("当前state为：", state)
    docs = rerank_docs(state["question"], state["docs"], state["intent"], top_k=3)
    print("rerank结束，结果为：", docs)
    print("当前state为：", state)
    return {"docs": docs}