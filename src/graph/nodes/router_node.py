def strategy_router_node(state):
    intent = state.get("intent")
    dog_name = state.get("dog_name")
    filters = state.get("filters", {})

    # 精确查询（指定狗）
    if dog_name:
        strategy = "exact"

    # 推荐类（有筛选条件）
    elif intent == "recommend" or filters:
        strategy = "filtered"

    # 泛问
    elif intent == "ask_info":
        strategy = "semantic"

    else:
        strategy = "direct"
    print("路由分流完成，stratege:", strategy)

    return {"strategy": strategy}