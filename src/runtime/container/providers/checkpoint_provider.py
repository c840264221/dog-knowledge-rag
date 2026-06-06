from src.runtime.persistence.checkpoint_manager import (
    CheckpointManager
)


class CheckpointProvider:

    def __init__(self,persistence_service):

        self.manager = CheckpointManager(
            persistence_service
        )

    async def startup(self):
        pass

    async def shutdown(self):
        pass