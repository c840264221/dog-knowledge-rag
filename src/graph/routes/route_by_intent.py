def route_by_intent(state):
    print("进入route_buy_intent分流......")
    intent = state.get("intent", "general")

    if intent == "recommendation":
        print("进入推荐流......")
        return "recommendation_flow"
    else:
        print("进入问答流......")
        return "qa_flow"