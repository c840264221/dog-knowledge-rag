VALID_INTENTS = {
    "recommend",
    "ask_info",
    "general"
}

class LLMOutputValidationError(Exception):
    pass

# 校验LLM解析用户输入后的输出query parse result
def validate_query_parse_result(parsed):

    if parsed["intent"] not in VALID_INTENTS:

        raise LLMOutputValidationError(
            f"非法 intent: {parsed["intent"]}"
        )

    if not isinstance(parsed["filters"], dict):

        raise LLMOutputValidationError(
            "filters 必须是 dict"
        )

    if not isinstance(parsed["tags"], list):

        raise LLMOutputValidationError(
            "tags 必须是 list"
        )

    return parsed

from functools import wraps
from loguru import logger


def validate_llm_output(validator,fallback_factory):

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):

            try:

                result = await func(*args, **kwargs)

                validated = validator(result)

                return validated

            except Exception as e:

                logger.exception(
                    f"LLM输出校验失败: {e}"
                )

                return fallback_factory()

        return wrapper
    return decorator


from src.parser.schema import QueryParseResult, Intent

def default_parse_result():
    return QueryParseResult(
        intent=Intent.ASK_INFO.value,
        filters={},
        tags=[],
        features=[],
        dog_name=None
    )