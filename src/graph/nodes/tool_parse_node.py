from src.graph.state import DogState
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.llm import get_llm

llm = get_llm()
def tool_parse_node(state: DogState) -> dict:
    """
    调用 LLM，判断是否需要使用工具。
    输出格式为纯文本，如果包含 "TOOL: 工具名|参数" 则触发工具调用。
    """
    system_prompt = """你是一个帮助用户的助手。你可以使用以下工具：
- get_current_time: 获取当前时间（不需要参数）
- get_today_date: 获取今天的日期（不需要参数）

如果你需要使用工具，请严格按照以下格式输出一行：
TOOL: 工具名|参数
（如果工具不需要参数，参数部分留空，例如 "TOOL: get_current_time|"）

如果你不需要使用工具，请直接回答用户的问题。
注意：输出任何其他内容之前不要有多余解释。
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages).content
    print(f"[DEBUG] tool_parse_node 输出: {response[:200]}")  # 调试打印

    # 解析是否有 TOOL 标记
    if "TOOL:" in response:
        # 提取第一行 TOOL: 内容
        lines = response.split("\n")
        tool_line = None
        for line in lines:
            if line.strip().startswith("TOOL:"):
                tool_line = line.strip()
                break
        if tool_line:
            parts = tool_line.replace("TOOL:", "").strip().split("|")
            tool_name = parts[0].strip()
            tool_args = parts[1].strip() if len(parts) > 1 else ""
            # 存入 state
            return {
                "need_tool": True,
                "tool_calls": [{"name": tool_name, "args": tool_args}],
                "tool_results": [],
                "tool_round": 0,
            }

    # 没有工具调用，直接生成答案？
    # 注意：这里我们不直接生成，而是交给后续的 answer_gen_node 去做，这样统一出口
    return {"need_tool": False}
