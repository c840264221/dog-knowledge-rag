def retrieval_retry_node(state):
    print("进入重试......")
    print("当前state为：", state)
    retry_count = state.get("retry_count", 0)

    filters = state.get("filter_dict", {}).copy()
    tags = state.get("tags", []).copy()

    # 第一次失败：去掉tags
    if retry_count == 0:
        print("第一次重试......")
        tags = []

    # 第二次失败：扩大topK
    elif retry_count == 1:
        print("第二次重试......")
        state["top_k"] = 15

    # 第三次失败：放宽filter
    elif retry_count == 2:
        print("第三次重试......")
        filters.pop("barking", None)
    print("重试阶段中state为：", state)

    return {
        "retry_count": retry_count + 1,
        "filters": filters,
        "tags": tags
    }