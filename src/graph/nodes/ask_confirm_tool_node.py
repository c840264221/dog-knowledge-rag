from langgraph.types import interrupt

from src.graph.states.state import DogState
from src.logger import logger

from src.runtime.context import runtime_ctx


def ask_confirm_tool_node(state: DogState) -> dict:

    runtime_ctx.get().state().set_node(
        "ask_confirm_tool_node"
    )

    # 记录时间线
    runtime_ctx.get().timeline().add_event(

        event_type="node",

        name="ask_confirm_tool_node"
    )

    """在 execute_tool 之前询问用户是否真的要执行"""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"need_tool": False, "tool_results": "没有工具需要确认。"}

    tc = tool_calls[0]  # 简单场景只取第一个工具
    prompt = f"即将执行工具：【{tc['name']}】，参数：【{tc['args']}】。是否继续？(y/n)"

    from src.runtime.container.init import container

    container.get(
        "checkpoint"
    ).manager.save_checkpoint()


    user_input = interrupt(prompt)  # 中断，等待用户输入

    if user_input.strip().lower() == 'y':
        # 用户同意，保留 need_tool=True 和 tool_calls
        logger.debug(f"DEBUG: state = {state}------进入下一节点")


        container.get("checkpoint").manager.save_checkpoint()


        return {}  # 不修改状态，继续往下走
    else:
        # 用户拒绝

        from src.runtime.container.init import container

        container.get("checkpoint").manager.save_checkpoint()

        return {
            "need_tool": False,
            "tool_results": "用户取消了工具调用。",
            "tool_calls": []  # 清空，防止后续误执行
        }