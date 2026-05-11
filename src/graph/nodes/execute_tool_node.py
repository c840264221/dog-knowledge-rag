from src.graph.state import DogState
from src.graph.tools.tools import TOOL_FUNCTIONS
import asyncio


def execute_tool_node(state: DogState) -> dict:
    """

    将之前的全部工具遍历调用改为链式循环  这样可以将之前调用工具产生的结果作为参数传给下面的工具
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"need_tool": False, "tool_results": state.get("tool_results", [])}

    # 取第一个工具执行
    tc = tool_calls[0]
    name = tc["name"]
    args = tc["args"]

    func = TOOL_FUNCTIONS.get(name)
    if not func:
        result = f"未知工具: {name}"
    else:
        try:
            if asyncio.iscoroutinefunction(func):
                # 在同步环境中运行异步函数（注意：外部事件循环需已存在，这里简单用 asyncio.run）
                result = asyncio.run(func(args)) if args else asyncio.run(func())
            else:
                result = func(args) if args else func()
        except Exception as e:
            result = f"{name} 执行失败: {str(e)}"
    new_results = state.get("tool_results", []) + [f"{name}: {result}"]
    remaining_calls = tool_calls[1:]  # 剩余未执行的工具

    # 如果还有剩余，继续需要工具；否则结束
    return {
        "tool_results": new_results,
        "tool_calls": remaining_calls,
        "need_tool": bool(remaining_calls),
        "tool_round": state.get("tool_round", 0) + 1,
    }