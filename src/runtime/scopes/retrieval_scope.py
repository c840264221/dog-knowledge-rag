from src.runtime.context.request_scope import RequestScope

class RetrievalScope:

    KEY = "retrieval"

    def __init__(self, scope: RequestScope):

        self.scope = scope

    def set_docs(self,docs):

        self.scope.set(self.KEY,docs)

    def get_docs(self):

        return self.scope.get(
            self.KEY,
            []
        )

    def clear(self):

        self.scope.remove(
            self.KEY
        )

    async def startup(self):
        pass

    async def shutdown(self):
        self.clear()