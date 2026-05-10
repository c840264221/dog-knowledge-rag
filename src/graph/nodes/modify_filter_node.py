from src.graph.state import DogState

def modify_filter_node(state: DogState) -> dict:
    """根据用户选择 2，修改 filters，例如换一个品种"""
    # 测试阶段 采用简单策略对付一下 比如改为更为常见的金毛
    new_filters = state.get("filters", {}).copy()
    new_filters["name"] = "Golden Retriever"
    return {"filters": new_filters, "retry_count": state.get("retry_count", 0) + 1}