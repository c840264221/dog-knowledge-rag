from src.retrieval.retriever import handle_filters

class RetrieveNode:

    def __init__(self, db):
        print("初始化RetrieveNode,指定db......")
        self.db = db
    def __call__(self,state):
        print("调用RetrieveNode......")
        question = state["question"]
        filter_dict = state.get("filters", {})
        filter_dict = handle_filters(filter_dict)
        print("filter_dict为：", filter_dict)

        retriever = self.db.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": filter_dict if filter_dict else None
            }
        )

        docs = retriever.invoke(question)
        print("retriever返回的数据为：", docs)
        print("当前state为:", state)

        return {"docs": docs}