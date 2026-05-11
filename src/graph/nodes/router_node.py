def strategy_router_node(state):
    intent = state.get("intent")
    dog_name = state.get("dog_name")
    filters = state.get("filters", {})
    top_k = state.get("top_k", 5)
    # 记忆性问题的关键词
    memory_keywords = ["喜欢什么", "上次说的", "之前推荐", "还记得", "我喜欢的", "我曾经", "最喜欢的"]
    # 精确查询（指定狗）
    if dog_name:
        strategy = "exact"

    # 推荐类（有筛选条件）
    elif intent == "recommend" or filters:
        strategy = "filtered"
        top_k = 10

    # 泛问
    elif intent == "ask_info":
        strategy = "semantic"

    else:
        strategy = "direct"
    strategy = (
        "semantic"
        if any(keyword in state["question"] for keyword in memory_keywords)
        else strategy
    )
    print("路由分流完成，stratege:", strategy)

    return {"strategy": strategy,
            "top_k": top_k
            }