from src.graph.state import DogState

def route_after_ask_user(state: DogState) -> str:
    feedback = state.get("user_feedback", "").strip()
    if feedback == "1":
        return "retry"          # 重新走重试节点，会回到 retrieve
    elif feedback == "2":
        # 修改过滤条件 放宽限制来达到获取更多搜索结果的目的
        return "modify_filter"
    else:
        return "generate"       # 直接生成答案