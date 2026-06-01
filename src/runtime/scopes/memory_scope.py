from src.runtime.context.request_scope import RequestScope
from src.runtime.scopes.base_scope import BaseScope


class MemoryScope(BaseScope):

    KEY = "memory_snapshot"

    def __init__(self, scope:RequestScope):

        self.scope = scope

    def set_memories(self,memories):

        self.scope.set(
            self.KEY,
            memories
        )

    def get_memories(self):

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