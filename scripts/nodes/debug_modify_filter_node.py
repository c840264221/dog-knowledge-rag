from src.graph.nodes.modify_filter_node import execute_modify_filter_node

state = {
    "question": "推荐适合新手的小型犬",
    "filters": {
        "$and": [
            {
                "size": {
                    "$eq": "small"
                }
            }
        ]
    },
    "user_feedback": "换成金毛看看",
    "retry_count": 0,
    "rag_context": {
        "status": "success",
        "chunks": [
            {
                "mock": True
            }
        ],
        "context_text": "old context",
    },
    "docs": [
        "old doc"
    ],
}

result = execute_modify_filter_node(
    state=state,
    checkpoint_provider=None,
)

print(result)