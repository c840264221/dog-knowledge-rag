from src.core.errors.base import DogAgentError


class ValidationError(
    DogAgentError
):
    pass


class StateValidationError(
    ValidationError
):
    pass


class UserInputValidationError(
    ValidationError
):
    pass