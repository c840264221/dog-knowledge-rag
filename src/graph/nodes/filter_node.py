def filter_node(state):
    """
    构建数据库可用的 filter_dict
    """
    print("filter_node开始......")
    print("当前state为:", state)
    filters = state.get("filters", {})
    dog_name = state.get("dog_name")

    filter_dict = {}

    # 数值过滤（直接透传）
    for k, v in filters.items():
        filter_dict[k] = v

    # 品种过滤（如果识别到了）
    if dog_name:
        filter_dict["name"] = dog_name

    print("filter_node结束，结果为：", filter_dict)
    print("当前state为:", state)
    return {
        "filters": filter_dict
    }