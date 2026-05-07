def route_after_retrieval(state):
    if state["retrieval_ok"]:
        return "good"
    else:
        return "retry"