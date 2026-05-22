from src.graph.states.state import DogState
from src.logger import logger

def route_after_ask_user(state: DogState) -> str:
    logger.info(f"进入route_after_ask_user节点，state为：{state}")
    feedback = state.get("user_feedback", "").strip()
    logger.debug(f"用户输入feedback为：{feedback}")
    if feedback == "1":
        logger.info(f"用户输入为1，进入retry")
        return "retry"          # 重新走重试节点，会回到 retrieve
    elif feedback == "2":
        logger.info(f"用户输入为2，进入modify_filter")
        # 修改过滤条件 放宽限制来达到获取更多搜索结果的目的
        return "modify_filter"
    else:
        logger.info(f"用户输入3或其他，进入generate")
        return "generate"       # 直接生成答案