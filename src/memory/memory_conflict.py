class MemoryConflictResolver:

    CONFLICT_RULES = {
        "favorite_dog": ["dislike"],
        "dislike": ["favorite_dog"],
    }

    def get_conflict_types(self, memory_type: str) -> list[str]:

        return self.CONFLICT_RULES.get(
            memory_type,
            []
        )