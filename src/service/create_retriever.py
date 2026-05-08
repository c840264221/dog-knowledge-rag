from src.vectorstore.vector_store import db

class RetrieverService:

    def __init__(self):
        self.db = db
    def retrieve(self, question, filters=None, top_k=5):
        retriever = self.db.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": filters if filters else None
            }
        )

        return retriever.invoke(question)


#  全局单例
retriever_service = RetrieverService()