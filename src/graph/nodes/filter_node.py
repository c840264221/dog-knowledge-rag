from src.logger import logger

def filter_node(state):
    """
    构建数据库可用的 filter_dict
    """
    logger.info(f"进入filter_node节点  state: {state}")
    filters = state.get("filters", {})
    dog_name = state.get("dog_name")

    filter_dict = {}

    # 数值过滤（直接透传）
    for k, v in filters.items():
        filter_dict[k] = v

    # 品种过滤（如果识别到了）
    if dog_name:
        filter_dict["name"] = dog_name

    logger.debug(f"filter_node节点执行完毕 filter_dict: {filter_dict}")
    return {
        "filters": filter_dict
    }