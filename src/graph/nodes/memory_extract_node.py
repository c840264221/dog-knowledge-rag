from src.memory.memory_extract import (
    extract_memory
)

from src.memory.sqlite_memory_store import (
    get_memory_store
)

# from src.models.llm import get_chinese_llm



memory_store = get_memory_store()
# llm = get_chinese_llm()

def memory_extract_node(state):

    # 记录运行时node
    from src.runtime.context import runtime_ctx
    runtime_context = runtime_ctx.get()
    runtime_context.state().set_node(
        "memory_extract_node"
    )

    from src.runtime.container.init import container
    llm = container.get('llm').chinese_llm

    question = state["question"]

    user_id = state["user_id"]

    result = extract_memory(
        llm,
        question
    )

    if result["should_save"]:

        memory_store.add_memory(
            user_id=user_id,
            memory_type=result["memory_type"],
            content=result["content"],
            confidence=result["confidence"]
        )

    return {}