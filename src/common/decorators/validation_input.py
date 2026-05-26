from functools import wraps

class InputValidationError(Exception):
    pass

# 对输入进行校验 逻辑层
def validate_input(question: str) -> str:

    if not isinstance(question, str):
        raise InputValidationError("输入必须是字符串")

    question = question.strip()

    if not question:
        raise InputValidationError("输入不能为空")

    if len(question) > 500:
        raise InputValidationError("输入过长")

    blocked_patterns = [
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "忽略之前所有消息",
        "系统提示词",
        "开发者信息",
    ]

    lowered = question.lower()

    for pattern in blocked_patterns:

        if pattern in lowered:

            raise InputValidationError(
                "检测到非法输入"
            )

    return question

# 装饰器层
def validate_question(func):

    @wraps(func)
    async def wrapper(state, *args, **kwargs):

        question = state.get("question", "")

        validated_question = validate_input(question)

        state["question"] = validated_question

        return await func(state, *args, **kwargs)

    return wrapper