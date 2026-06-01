from src.graph.states.state import DogState
from langchain_core.messages import HumanMessage


async def answer_gen_node(state: DogState) -> dict:

    from src.runtime.context import runtime_ctx

    runtime_ctx.get().state().set_node(
        "answer_gen_node"
    )

    def get_llm_provider():
        from src.runtime.container.init import container
        return container.get("llm")

    llm_provider = get_llm_provider()

    main_llm = llm_provider.main_llm

    tool_results = state.get("tool_results", [])
    question = state["question"]
    history_text = "\n".join([f"用户: {m.content}" if isinstance(m, HumanMessage) else f"助手: {m.content}" for m in
                              state.get("messages", [])])

    # 构建强制指令
    base_prompt = f"""你是一个只能基于提供信息回答的助手。
    对话历史：
    {history_text}

    【重要规则】
    1. 如果用户的问题是关于他自己之前说过的话（例如“我最喜欢什么狗狗”），你必须仅从上面的对话历史中查找答案。如果历史中没有，回答“我不知道”。
    2. 不要利用你自己学到的知识（例如关于金毛的通用知识）来回答，除非问题明确要求百科知识。
    3. 用户问题：{question}

    请严格按照上述规则回答，只输出答案内容，不要额外解释。
    """
    if tool_results:
        base_prompt = f"工具结果：{tool_results}\n" + base_prompt

    # response = llm.invoke(base_prompt).content
    # 采用更安全版本的llm
    response = await llm_provider.safe_ainvoke(
        llm=main_llm,
        prompt=base_prompt,
        fallback_response="模型暂时不可用"
    )
    return {"answer": response.content}