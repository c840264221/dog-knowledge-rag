from langgraph.graph import END
from src.logger import logger

def route_after_evaluate(state):
    logger.info(f"进入route_after_evaluate路由咯，检验retrieval_ok字段的值，state为：<UNK>{state}")
    logger.debug(f"重试次数retry_cnt为：{state.get("retry_count", 0)}")

    if state["retrieval_ok"]:
        logger.debug('state["retrieval_ok"]为True，数据合格，返回good')
        return "good"

    retry_cnt = state.get("retry_count", 0)

    if retry_cnt >= 3:
        logger.info(f"进入route_after_evaluate路由，retry_cnt重试次数大于等于3，放弃继续重试，返回give up")
        return "give_up"

    # 如果已经重试过至少一次，且还不是彻底放弃，就先去询问用户
    # 避免无限循环问用户，所以只问一次（可根据需求调整）
    logger.debug(f"检查has_asked_user字段{state.get("has_asked_user", False)}，避免无限循环询问用户。")

    if retry_cnt >= 1 and state.get("has_asked_user", False) is False:
        logger.info(f"满足询问用户条件，返回ask_user")
        return "ask_user"

    logger.info("数据不满足且不需要询问用户，直接返回retry")

    return "retry"


def route_after_ask_user(state) -> str:
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