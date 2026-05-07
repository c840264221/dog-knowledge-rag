def evaluate_retrieval_node(state):
    docs = state.get("docs", [])

    enough = len(docs) >= 2

    return {
        "retrieval_ok": enough
    }