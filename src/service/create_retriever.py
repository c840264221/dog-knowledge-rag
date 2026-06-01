# from src.vectorstore.vector_store import db

class RetrieverService:

    # def __init__(self):
    #     from src.runtime.container.init import container
    #
    #     self.db = container.get('vectorstore').db

    async def retrieve(self, question, filters=None, top_k=5):

        def get_vectorstore_provider():
            from src.runtime.container.init import container
            return container.get('vectorstore')

        vectorstore_provider = get_vectorstore_provider()

        self.db = vectorstore_provider.db

        retriever = self.db.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": filters if filters else None
            }
        )
        docs = await retriever.ainvoke(question)
        return docs


#  全局单例
retriever_service = RetrieverService()