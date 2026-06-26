from src.rag.query_builders import (
    build_rag_query_from_state,
)

from src.runtime.container.init import (
    container,
)


parser = container.get(
    "retriever"
).dog_query_filter_parser


state = {
    "question": "推荐适合新手的小型犬",
    "user_id": "test_user",
    "top_k": 5,
    "intent": "dog_recommendation",
    "filters": {
        "$and": [
            {
                "size": {
                    "$eq": "small"
                }
            },
            {
                "barking_level": {
                    "$lte": 3
                }
            }
        ]
    },
}


rag_query = build_rag_query_from_state(
    state=state,
    parser=parser,
)



import json

print(
    json.dumps(
        rag_query.model_dump(),
        indent=4,
        ensure_ascii=False
    )
)