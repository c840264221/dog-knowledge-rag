from src.logger import logger

def route_general_qa_worker(state):

    return state["next_worker"]

def route_after_executing_tool_worker(state):
    logger.info("进入qa_agent 的route_after_executing_tool_worker")
    logger.debug(f"tool_calls为：{state['tool_calls']}")

    if len(state["tool_calls"]) > 0:
        return "ask_confirm"
    else:
        return "answer_gen"