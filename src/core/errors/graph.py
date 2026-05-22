from src.core.errors.base import DogAgentError


class GraphError(DogAgentError):
    pass


class GraphRecursionError(GraphError):
    def __init__(self, message):

        super().__init__(
            message,
            recoverable=False
        )