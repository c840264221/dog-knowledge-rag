from functools import wraps

class StateValidationError(Exception):
    pass


def validate_state(required_keys=None):

    required_keys = required_keys or []

    def decorator(func):

        @wraps(func)
        async def wrapper(state, *args, **kwargs):

            if not isinstance(state, dict):
                raise StateValidationError(
                    "state 必须是 dict"
                )

            for key in required_keys:

                if key not in state:

                    raise StateValidationError(
                        f"缺少必要字段: {key}"
                    )

            return await func(state, *args, **kwargs)

        return wrapper

    return decorator