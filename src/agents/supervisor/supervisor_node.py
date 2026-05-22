from src.graph.states.main_state import MainState


def supervisor_node(state: MainState):

    question = state["question"]

    # 第一版先简单写死
    next_agent = "recommendation_agent"

    return {

        "current_agent": "supervisor",

        "next_agent": next_agent
    }