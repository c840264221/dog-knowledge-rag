from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore
)

memory_store = SQLiteMemoryStore()


def retrieve_user_memory(user_id: str) -> str:

    memories = memory_store.get_memories(
        user_id
    )

    if not memories:
        return "暂无用户记忆"

    formatted = []

    for memory_type, content in memories:

        formatted.append(
            f"[{memory_type}] {content}"
        )

    return "\n".join(formatted)