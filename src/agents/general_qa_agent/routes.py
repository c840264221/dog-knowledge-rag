from typing import Any

from src.logger import logger


def route_general_qa_worker(
    state: dict[str, Any],
) -> str:
    """
    根据 next_worker 路由到 general_qa_agent 的下一个 worker。

    功能：
        从 state 中读取 next_worker。
        如果 next_worker 缺失、为空或非法，则兜底到 answer_gen。
        返回值必须和 build_general_qa_agent 中 supervisor 的 conditional_edges key 保持一致。
        避免 LangGraph 因非法 route key 抛出 KeyError。

    参数：
        state：
            LangGraph 当前状态。
            主要读取 next_worker 字段。

    返回值：
        str：
            下一个 worker 的路由名称。
            只能是 tool_parse、ask_confirm、execute_tool、answer_gen、finish。

    专业名词：
        Route（路由）：
            根据当前 state 决定 Graph 下一步走向的函数。

        Conditional Edges（条件边）：
            LangGraph 中根据 route 函数返回值选择下一条边的机制。

        Worker（工作节点）：
            Graph 中负责具体任务的节点，例如 tool_parse、answer_gen。
    """

    allowed_workers = {
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
        "finish",
    }

    next_worker = str(
        state.get(
            "next_worker",
            "answer_gen",
        )
        or "answer_gen"
    ).strip()

    if next_worker not in allowed_workers:
        logger.warning(
            f"非法 next_worker，已兜底到 answer_gen: {next_worker!r}"
        )

        return "answer_gen"

    return next_worker


def route_after_executing_tool_worker(
    state: dict[str, Any],
) -> str:
    """
    根据剩余 tool_calls 判断工具链路是否继续。

    功能：
        execute_tool_node 每次只执行第一个工具。
        如果执行后 state 中还有剩余 tool_calls，则回到 ask_confirm。
        如果没有剩余 tool_calls，则进入 answer_gen 生成最终回答。

    参数：
        state：
            LangGraph 当前状态。
            主要读取 tool_calls 字段。

    返回值：
        str：
            ask_confirm：
                表示还有工具需要确认和执行。

            answer_gen：
                表示工具已经执行完，可以生成最终回答。

    专业名词：
        Tool Calls（工具调用列表）：
            等待执行的工具调用集合。

        Chain Tool Execution（链式工具执行）：
            每次只执行一个工具，剩余工具交给 Graph 下一轮继续处理。
    """

    logger.info(
        "进入qa_agent 的route_after_executing_tool_worker"
    )

    tool_calls = state.get(
        "tool_calls",
        []
    ) or []

    logger.debug(
        f"tool_calls为：{tool_calls}"
    )

    if not isinstance(
        tool_calls,
        list,
    ):
        logger.warning(
            f"tool_calls 类型非法，已兜底到 answer_gen: {type(tool_calls)}"
        )

        return "answer_gen"

    if len(
        tool_calls
    ) > 0:
        return "ask_confirm"

    return "answer_gen"