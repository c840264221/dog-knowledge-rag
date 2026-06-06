from src.runtime.persistence.runtime_persistence_service import (
    RuntimePersistenceService
)


class RuntimePersistenceProvider:

    def __init__(self):

        self.persistence = (
            RuntimePersistenceService()
        )

    async def startup(self):
        pass

    async def shutdown(self):
        pass