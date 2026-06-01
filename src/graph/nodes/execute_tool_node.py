from src.graph.states.state import DogState
from src.graph.tools.tools import TOOL_FUNCTIONS
from src.common.decorators.state_validation import validate_state
from src.graph.tools.runtime.tool_executor import safe_execute_tool
from src.logger import logger
from src.graph.tools.runtime.tool_executor import ToolExecutor

from src.runtime.context import runtime_ctx

executor = ToolExecutor()


# @validate_state(["tool_calls"])
async def execute_tool_node(state: DogState) -> dict:
    """

    将之前的全部工具遍历调用改为链式循环  这样可以将之前调用工具产生的结果作为参数传给下面的工具
    """

    runtime_ctx.get().state().set_node(
        "execute_tool_node"
    )

    logger.info(f"进入执行工具节点......")
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"need_tool": False, "tool_results": state.get("tool_results", [])}

    # 取第一个工具执行
    tc = tool_calls[0]
    name = tc["name"]
    args = tc["args"]

    logger.debug(f"tool_name: {name}...tool_args: {args}")

    # 运行时上下文中的runtime state 记录调用的工具名称
    runtime_ctx.get().state().set_tool(
        name
    )

    result = await executor.execute(name, args)

    # func = TOOL_FUNCTIONS.get(name)
    # if not func:
    #     result = f"未知工具: {name}"
    # else:
    #     # try:
    #     #     if asyncio.iscoroutinefunction(func):
    #     #         # 在同步环境中运行异步函数（注意：外部事件循环需已存在，这里简单用 asyncio.run）
    #     #         result = asyncio.run(func(args)) if args else asyncio.run(func())
    #     #     else:
    #     #         result = func(args) if args else func()
    #     # except Exception as e:
    #     #     result = f"{name} 执行失败: {str(e)}"
    #
    #     # 抽离执行层 让node内更干净 也方便以后对tools的扩展 例如重试 统计 日志 fallback等
    #     result = safe_execute_tool(
    #         func=func,
    #         args=args,
    #         timeout=5
    #     )
    new_results = state.get("tool_results", []) + [result.model_dump()]
    remaining_calls = tool_calls[1:]  # 剩余未执行的工具


    # 如果还有剩余，继续需要工具；否则结束
    return {
        "tool_results": new_results,
        "tool_calls": remaining_calls,
        "need_tool": bool(remaining_calls),
        "tool_round": state.get("tool_round", 0) + 1,
    }