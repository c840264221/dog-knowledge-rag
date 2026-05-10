from src.graph.state import DogState
from src.models.llm import get_llm


llm = get_llm()
def answer_gen_node(state: DogState) -> dict:
    # 如果有工具结果，让 LLM 结合结果生成回答
    tool_results = state.get("tool_results", [])
    question = state["question"]

    if tool_results:
        prompt = f"""用户问题: {question}
    工具执行结果: {chr(10).join(tool_results)}
    请根据工具结果，用自然语言回答用户。"""
    else:
        prompt = f"用户问题: {question}\n请直接回答。"

    response = llm.invoke(prompt).content
    return {"answer": response}