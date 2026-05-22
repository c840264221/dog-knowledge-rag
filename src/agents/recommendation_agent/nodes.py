from src.agents.recommendation_agent.agent import recommendation_graph


def recommendation_agent_node(state):

    result = recommendation_graph.invoke({

        "question": state["question"]
    })

    return {

        "current_agent":
            "recommendation_agent",

        "answer":
            result["answer"]
    }