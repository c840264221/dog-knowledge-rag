from src.retrieval.filter_by_tags import filter_by_tags

def tag_filter_node(state):
    print("开始根据tags进行过滤......")
    docs = state["docs"]
    tags = state["tags"]

    if not tags:
        return {"docs": docs}

    filtered = filter_by_tags(docs, tags)
    print("过滤完成，结果为：", filtered)
    return {"docs": filtered}