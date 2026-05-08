from src.retrieval.retriever import handle_filters
from src.service.create_retriever import retriever_service


def retrieve_node(state):
    print("retrieve_node......")
    question = state["question"]

    filter_dict = state.get("filters", {})
    # 如果有具体的狗狗名字 则放弃其他筛选条件 改为查询该狗狗的所有信息
    if "name" in filter_dict.keys():
        filter_dict = {"name": filter_dict["name"]}
    filter_dict = handle_filters(filter_dict)

    top_k = state.get("top_k", 5)
    print("filter_dict为：", filter_dict)

    docs = retriever_service.retrieve(question, filter_dict, top_k)

    # docs = retriever.invoke(question)
    print("retriever返回的数据为：", docs)
    print("当前state为:", state)

    return {"docs": docs}