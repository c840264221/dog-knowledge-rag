from src.graph.states.dog_state import DogState
from src.logger import logger


def recommendation_model_node(
        state: DogState,
) -> DogState:
    """
    犬种推荐模型节点。

    功能：
        作为 dog_knowledge_agent 内部的 recommendation_model 分支节点。
        用于处理“根据用户条件推荐犬种”的问题。

        该节点不负责：
        1. 不调用 LLM。
        2. 不构建推荐 Prompt。
        3. 不直接生成答案。
        4. 不执行 RAG 检索。

        该节点只负责：
        1. 标记当前执行分支为 recommendation_model。
        2. 设置 intent 为 recommend，帮助 generate_node 内部的 resolve_answer_strategy
           稳定识别 recommendation 回答策略。
        3. 给后续日志、debug、trace 提供可观察字段。

    参数：
        state:
            DogState，当前 LangGraph 状态。
            其中可能包含 question、intent、rag_query、filters、messages 等字段。

    返回值：
        DogState:
            Partial DogState，只返回需要合并进主状态的字段。

    输出格式：
        {
            "current_agent": "recommendation_agent",
            "intent": "recommend",
            "strategy": "recommendation_model",
            "next_worker": "retrieve"
        }

    专业名词：
        Recommendation：
            推荐。这里表示根据用户条件推荐合适犬种。

        Intent：
            意图。表示用户问题的业务目标，例如 recommend、ask_info、dog_info 等。

        Answer Strategy：
            回答策略。generate_node 会根据 intent、route_decision、question 等信息解析回答策略。
    """

    logger.info(
        "[recommendation_model_node] 进入 dog_knowledge_agent.recommendation_model"
    )

    return {
        "current_agent": "recommendation_agent",
        "intent": "recommend",
        "strategy": "recommendation_model",
        "next_worker": "retrieve",
    }