from src.graph.states.state import DogState
from src.logger import logger

def modify_filter_node(state: DogState) -> dict:
    """根据用户选择 2，修改 filters，例如换一个品种"""
    print("modify_filter_node", state)

    logger.info(f"进入modify_filter_node节点<UNK>state为：{state}")
    # 测试阶段 采用简单策略对付一下 比如改为更为常见的金毛
    new_filters = state.get("filters", {}).copy()
    new_filters["name"] = "Golden Retriever"
    logger.info(f"将filters修改为较常见的数据来增加符合条件的数据，filters为：<UNK>{new_filters}")
    return {"filters": new_filters, "retry_count": state.get("retry_count", 0) + 1}