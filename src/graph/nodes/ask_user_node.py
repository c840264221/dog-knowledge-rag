from src.graph.states.state import DogState
from langgraph.types import interrupt
from src.logger import logger

from src.runtime.context import runtime_ctx


def ask_user_node(state: DogState) -> dict:
    """
    节点功能：当检索结果不够好时，中断并询问用户，等待输入后再继续。
    返回：更新 state 中的 user_feedback。
    """

    runtime_ctx.get().state().set_node(
        "ask_user_node"
    )

    logger.info(f"进入ask_user_node节点，state为：{state}")
    # 构建给用户的消息
    question = (
        f"当前找到 {len(state.get('docs', []))} 条狗狗资料，数量较少。\n"
        "请选择希望如何处理：\n"
        "1 - 放宽过滤条件重试\n"
        "2 - 换一个常见品种试试\n"
        "3 - 直接基于现有资料回答\n"
        "请输入数字 1/2/3："
    )

    # 关键：interrupt() 会暂停整个图，并返回用户输入
    logger.info(f"ask_user_node中断，等待用户输入后恢复执行")
    user_input = interrupt(question)
    logger.info(f"ask_user_node恢复运行，user_input为：{user_input}")
    # 将用户输入存入状态，供后续节点判断

    from src.runtime.container.init import container

    container.get("checkpoint").manager.save_checkpoint()

    return {"user_feedback": user_input, "has_asked_user": True}