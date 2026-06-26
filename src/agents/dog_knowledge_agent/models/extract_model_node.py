from src.graph.states.dog_state import DogState
from src.logger import logger


def extract_model_node(
        state: DogState,
) -> DogState:
    """
    犬种知识提取模型节点。

    功能：
        作为 dog_knowledge_agent 内部的 extract_model 分支节点。
        用于处理普通犬种知识问答、精确信息查询、对比、护理建议等非推荐类知识问题。

        该节点不负责：
        1. 不调用 LLM。
        2. 不构建 Prompt。
        3. 不执行 RAG 检索。
        4. 不生成最终答案。

        该节点只负责：
        1. 标记当前执行分支为 extract_model。
        2. 保留现有 intent，不强行覆盖为 exact_info。
        3. 给后续日志、debug、trace 提供可观察字段。

    参数：
        state:
            DogState，当前 LangGraph 状态。
            其中可能包含 question、intent、rag_query、route_decision、messages 等字段。

    返回值：
        DogState:
            Partial DogState，只返回需要合并进主状态的字段。

    输出格式：
        {
            "current_agent": "dog_knowledge_agent",
            "strategy": "extract_model",
            "next_worker": "retrieve"
        }

    专业名词：
        Model：
            模型 / 内部分支。这里不是大模型，也不是 Pydantic Model，
            而是 dog_knowledge_agent 内部的业务处理分支。

        Partial State：
            局部状态。LangGraph 节点只返回自己修改的字段，
            然后由 LangGraph 合并进全局 DogState。

        Extract：
            提取。这里表示从犬种知识库中提取事实信息、对比信息、护理建议等。
    """

    logger.info(
        "[extract_model_node] 进入 dog_knowledge_agent.extract_model"
    )

    return {
        "current_agent": "dog_knowledge_agent",
        "strategy": "extract_model",
        "next_worker": "retrieve",
    }