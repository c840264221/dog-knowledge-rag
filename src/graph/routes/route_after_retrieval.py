def route_after_retrieval(state):
    """
    根据 retrieval_ok 判断检索结果是否可用。

    功能：
        如果 retrieval_ok 为 True，则返回 good。
        如果 retrieval_ok 为 False、缺失或其他假值，则返回 retry。

    参数：
        state：
            LangGraph 当前状态，通常是 DogState。

    返回值：
        str：
            good 表示检索结果可用。
            retry 表示需要重新检索或进入重试流程。
    """

    if state.get(
        "retrieval_ok",
        False,
    ):
        return "good"

    return "retry"