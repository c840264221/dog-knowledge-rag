from src.retrieval.retriever import handle_filters
from src.service.create_retriever import retriever_service
from src.logger import logger
from src.common.decorators.state_validation import validate_state

from src.runtime.context import runtime_ctx

from src.runtime.scopes.retrieval_scope import RetrievalScope



# @validate_state(["question","filters"])
async def retrieve_node(state):

    runtime_context = runtime_ctx.get()

    runtime_context.state().set_node(
        "retrieve_node"
    )

    logger.info(f"进入retrieve_node节点 state：{state}")
    question = state["question"]

    filter_dict = state.get("filters", {})
    # 如果有具体的狗狗名字 则放弃其他筛选条件 改为查询该狗狗的所有信息
    if "name" in filter_dict.keys():
        filter_dict = {"name": filter_dict["name"]}
    filter_dict = handle_filters(filter_dict)

    top_k = state.get("top_k", 5)
    logger.debug(f"filter_dict:{filter_dict}")

    docs = await retriever_service.retrieve(question, filter_dict, top_k)

    # 试运行runtime_ctx的作用域 往里面添加数据
    # runtime_ctx.get().retrieval.set_docs(docs)
    runtime_context.service(RetrievalScope).set_docs(docs)
    # retrieval_scope.set_docs(docs)

    # docs = retriever.invoke(question)
    logger.info(f"retrieve_node节点执行完毕 返回数据docs  docs长度为:{len(docs)}")

    return {"docs": docs}