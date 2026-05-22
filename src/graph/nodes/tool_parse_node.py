from langchain_core.runnables import RunnableLambda

from src.graph.states.state import DogState
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.llm import get_instance_llm, safe_llm_invoke

from src.logger import logger

from langchain_core.output_parsers import (
    PydanticOutputParser
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from src.graph.tools.schemas.tool_call_schema import (
    ToolParseResult
)

from src.models.llm import get_backup_llm, get_chinese_llm



# llm = get_chinese_llm()
llm = get_backup_llm()

parser = PydanticOutputParser(
    pydantic_object=ToolParseResult
)

def tool_parse_node(state: DogState) -> dict:
    if state.get("tool_calls"):
        return {}

    system_prompt = """你是一个帮助用户的助手。你可以使用以下工具：
- date: 获取今天的日期（不需要参数）
- weather: 查询某地某天的天气。参数格式：地区|日期，例如 "北京|2025-03-15"

如果你需要使用工具，请严格按照以下格式输出，如果需要多个工具就输出多行：
TOOL: get_today_date|
TOOL: get_weather|北京|2025-03-15
（如果工具不需要参数，参数部分留空，例如 "TOOL: get_current_time|"）

如果你不需要使用工具，请直接回答用户的问题。
注意：输出任何其他内容之前不要有多余解释。
"""

    tool_parse_prompt = """
    你是一个工具调用分析助手。

    你可以使用以下工具：

    1. date
    功能：
    - 获取今天日期
    参数：
    {{}}
    -----------------------------------

    2. weather
    功能：
    - 查询天气
    参数：
    {{
      "city": "城市名称",
      "date": "日期"
    }}
    -----------------------------------

    规则：

    1. 如果问题需要工具：
    返回 need_tool=true

    2. 如果不需要工具：
    返回 need_tool=false

    3. 必须严格输出JSON

    4. 不允许输出Markdown

    5. 不允许输出解释

    6. args必须是JSON对象

    7. tool_calls必须是数组
    -----------------------------------

    用户问题：
    {question}
    """

    prompt = ChatPromptTemplate.from_template(

        tool_parse_prompt +
        "\n\n"
        "{format_instructions}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]
    # response = llm.invoke(messages).content

    # 采用更安全的llm
    safe_llm = RunnableLambda(
        lambda x: safe_llm_invoke(
            llm=llm,
            prompt=x,
            fallback_response="模型暂时不可用"
        )
    )

    chain = prompt | safe_llm | parser
    try:
        response = chain.invoke({
            "question": state["question"],
            "format_instructions": parser.get_format_instructions()
        })
        # response = safe_llm_invoke(llm=llm, prompt=messages, fallback_response="调用LLM失败").content
    except Exception as e:
        logger.exception(
            f"Tool Parse失败: {e}"
        )
        return {
            "need_tool": False
        }

    logger.debug(
        f"[DEBUG] tool_parse_node 输出: \n"
        f"{response.model_dump_json(indent=2)}"
    )

    # 解析是否有 TOOL 标记
    # tool_calls = []
    # for line in response.split("\n"):
    #     line = line.strip()
    #     if line.upper().startswith("TOOL:"):
    #         parts = line[5:].strip().split("|")
    #         tool_name = parts[0].strip()
    #         tool_args = parts[1].strip() if len(parts) > 1 else ""
    #         tool_calls.append({"name": tool_name, "args": tool_args})
    #
    # if tool_calls:
    #     return {
    #         "need_tool": True,
    #         "tool_calls": tool_calls,
    #         "tool_results": [],
    #         "tool_round": state.get("tool_round", 0) + 1,
    #     }
    # else:
    #     return {"need_tool": False}
    return {

        "need_tool":
            response.need_tool,

        "tool_calls": [

            tc.model_dump()

            for tc in response.tool_calls
        ],

        "tool_results": [],

        "tool_round":
            state.get(
                "tool_round",
                0
            ) + 1
    }
