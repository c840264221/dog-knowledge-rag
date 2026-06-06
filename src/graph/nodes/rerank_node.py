from src.retrieval.retriever import rerank_docs
from src.logger import logger


from src.runtime.context import runtime_ctx

def rerank_node(state):

    runtime_ctx.get().state().set_node(
        "rerank_node"
    )

    logger.info(f"进入rerank_node节点 state为{state}"
        f"state: "
        f"question:{state['question']}"
        f"intent:{state['intent']}"
        f"strategy:{state['strategy']}"
        f"filters:{state['filters']}"
        f"tags:{state['tags']}"
        f"dog_name:{state['dog_name']}"
        f"docs len:{len(state['docs'])}"
                )
    docs = rerank_docs(state["question"], state["docs"], state["intent"], top_k=3)
    logger.debug(f"rerank结束，结果docs的数量为：{len(docs)}")

    from src.runtime.container.init import container

    container.get("checkpoint").manager.save_checkpoint()

    return {"docs": docs}