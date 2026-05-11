from src.graph.state import DogState
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.llm import get_llm

llm = get_llm()
def tool_parse_node(state: DogState) -> dict:
    if state.get("tool_calls"):
        return {}
    """
    调用 LLM，判断是否需要使用工具。
    输出格式为纯文本，如果包含 "TOOL: 工具名|参数" 则触发工具调用。
    """
    system_prompt = """你是一个帮助用户的助手。你可以使用以下工具：
- get_current_time: 获取当前时间（不需要参数）
- get_today_date: 获取今天的日期（不需要参数）
- get_weather: 查询某地某天的天气。参数格式：地区|日期，例如 "北京|2025-03-15"

如果你需要使用工具，请严格按照以下格式输出一行：
TOOL: get_today_date|
TOOL: get_weather|北京|2025-03-15
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
    tool_calls = []
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("TOOL:"):
            parts = line[5:].strip().split("|")
            tool_name = parts[0].strip()
            tool_args = parts[1].strip() if len(parts) > 1 else ""
            tool_calls.append({"name": tool_name, "args": tool_args})

    if tool_calls:
        return {
            "need_tool": True,
            "tool_calls": tool_calls,
            "tool_results": [],
            "tool_round": state.get("tool_round", 0) + 1,
        }
    else:
        return {"need_tool": False}
