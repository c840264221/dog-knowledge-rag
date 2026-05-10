from src.graph.state import DogState
from src.graph.tools.tools import TOOL_FUNCTIONS


def execute_tool_node(state: DogState) -> dict:
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"tool_results": ["没有需要执行的工具"]}

    results = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        if name in TOOL_FUNCTIONS:
            try:
                # 调用工具函数（注意：这里简单假设函数不需要参数或 args 为空）
                result = TOOL_FUNCTIONS[name]() if not args else TOOL_FUNCTIONS[name](args)
                results.append(f"{name} 执行结果: {result}")
            except Exception as e:
                results.append(f"{name} 执行失败: {str(e)}")
        else:
            results.append(f"未知工具: {name}")

    new_round = state.get("tool_round", 0) + 1
    return {
        "tool_results": results,
        "need_tool": False,  # 本次工具执行完后，不再立即需要更多工具
        "tool_round": new_round,
    }