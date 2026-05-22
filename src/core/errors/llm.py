from src.core.errors.base import DogAgentError


class LLMError(DogAgentError):
    pass


class LLMOutputValidationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass