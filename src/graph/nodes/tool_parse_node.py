from langchain_core.runnables import RunnableLambda

from langchain_core.output_parsers import (
    PydanticOutputParser
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from src.graph.states.state import DogState

from src.graph.tools.schemas.tool_call_schema import (
    ToolParseResult
)

from src.logger import logger

from src.runtime.context import runtime_ctx


parser = PydanticOutputParser(
    pydantic_object=ToolParseResult
)


def build_tool_parse_node(
    llm_provider,
    checkpoint_manager=None,
    runtime_context_getter=None,
):
    """
    构建 tool_parse_node 节点。

    功能：
        创建一个真正给 LangGraph 使用的工具解析节点。
        外层函数负责接收依赖，例如 llm_provider、checkpoint_manager。
        内层 async node 保持 LangGraph 需要的 state -> dict 调用格式。

    参数：
        llm_provider：
            LLM Provider（大语言模型提供者）。
            用于获取 backup_llm，并调用 safe_ainvoke。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于在工具解析成功后保存 checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        callable：
            返回一个 async node 函数。
            该函数接收 DogState，返回 dict，供 LangGraph 合并 state。

    专业名词：
        Closure（闭包）：
            内层函数可以记住外层函数传入的依赖变量。

        Dependency Injection，DI（依赖注入）：
            不在函数内部创建依赖，而是从外部传入依赖。

        Node（节点）：
            LangGraph 中的执行单元，接收 state，返回 state 更新数据。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def tool_parse_node(state: DogState) -> dict:
        """
        解析用户问题是否需要调用工具。

        功能：
            1. 写入当前 node 状态
            2. 记录 timeline 事件
            3. 如果 state 已有 tool_calls，则直接返回空 dict
            4. 调用 LLM 判断是否需要工具
            5. 使用 PydanticOutputParser 解析 LLM 输出
            6. 成功后返回 need_tool、tool_calls、tool_results、tool_round
            7. 解析失败时返回安全兜底结果

        参数：
            state：
                DogState，LangGraph 当前状态。
                必须包含 question 字段。

        返回值：
            dict：
                返回需要合并进 LangGraph state 的字段。
                例如 need_tool、tool_calls、tool_results、tool_round。

        专业名词：
            PydanticOutputParser（Pydantic 输出解析器）：
                LangChain 提供的结构化输出解析器，用于把 LLM 文本解析成 Pydantic 模型。

            Tool Call（工具调用）：
                表示模型判断需要调用某个工具，以及工具参数是什么。

            Fallback（兜底）：
                当 LLM 调用或解析失败时，返回安全默认结果，避免 Graph 中断。
        """

        ctx = runtime_context_getter()

        if ctx is not None:
            ctx.state().set_node(
                "tool_parse_node"
            )

            ctx.timeline().add_event(
                event_type="node",
                name="tool_parse_node"
            )

        if state.get("tool_calls"):
            return {}

        question = state.get(
            "question",
            ""
        )

        if not question:
            logger.warning(
                "tool_parse_node 缺少 question，已兜底为 need_tool=False"
            )

            return {
                "need_tool": False
            }

        backup_llm = llm_provider.backup_llm

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

        8. 如果用户问到天气则调用的工具中必须含有工具weather
        -----------------------------------

        用户问题：
        {question}
        """

        prompt = ChatPromptTemplate.from_template(
            tool_parse_prompt
            + "\n\n"
            + "{format_instructions}"
        )

        async def create_async_safe_llm_ainvoke(x):
            """
            安全调用 LLM。

            功能：
                调用 llm_provider.safe_ainvoke。
                使用 backup_llm 作为工具解析模型。
                当 LLM 调用失败时，safe_ainvoke 内部可以返回 fallback_response。

            参数：
                x：
                    ChatPromptTemplate 渲染后的 prompt 输入。

            返回值：
                str：
                    LLM 返回的文本结果。
            """

            return await llm_provider.safe_ainvoke(
                llm=backup_llm,
                prompt=x,
                fallback_response="调用LLM失败"
            )

        safe_llm = RunnableLambda(
            create_async_safe_llm_ainvoke
        )

        chain = prompt | safe_llm | parser

        try:
            response = await chain.ainvoke(
                {
                    "question": question,
                    "format_instructions": (
                        parser.get_format_instructions()
                    )
                }
            )

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

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return {
            "need_tool": response.need_tool,

            "tool_calls": [
                tc.model_dump()
                for tc in response.tool_calls
            ],

            "tool_results": [],

            "tool_round": (
                state.get(
                    "tool_round",
                    0
                )
                + 1
            )
        }

    return tool_parse_node