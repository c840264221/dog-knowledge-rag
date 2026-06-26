from src.graph.nodes.retrieval_retry_node import relax_to_core_filters

filters = {
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
        },
        {
            "good_for_apartment": {
                "$eq": True
            }
        },
        {
            "good_for_beginner": {
                "$eq": True
            }
        }
    ]
}

print(
    relax_to_core_filters(
        filters=filters
    )
)

filters = {
    "$and": [
        {
            "dog_name": {
                "$eq": "Shih Tzu"
            }
        },
        {
            "section_title": {
                "$eq": "基本信息"
            }
        },
        {
            "barking_level": {
                "$lte": 3
            }
        }
    ]
}

print(
    relax_to_core_filters(
        filters=filters
    )
)