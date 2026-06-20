from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


def build_answer_gen_node(
    llm_provider,
    checkpoint_manager=None,
    runtime_context_getter=None,
):
    """
    构建 answer_gen_node 节点。

    功能：
        创建一个真正给 LangGraph 使用的最终回答生成节点。
        外层函数负责接收 llm_provider、checkpoint_manager、runtime_context_getter 等依赖。
        内层 answer_gen_node 保持 LangGraph 需要的 state -> dict 调用格式。

    参数：
        llm_provider：
            LLM Provider（大语言模型提供者）。
            用于获取 main_llm，并调用 safe_ainvoke 生成最终回答。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于回答生成后保存 checkpoint。
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
        Dependency Injection，DI（依赖注入）：
            不在节点内部直接 import container，而是从外部传入依赖。

        Prompt（提示词）：
            传给 LLM 的输入文本。

        Tool Results（工具结果）：
            工具执行后的结果集合。这里统一按 list 结构处理。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def answer_gen_node(
        state: DogState
    ) -> dict[str, str]:
        """
        生成最终回答。

        功能：
            1. 写入当前 node 状态
            2. 记录 timeline 事件
            3. 从 state 中读取 question
            4. 从 state 中读取 messages 并格式化历史对话
            5. 从 state 中读取 memory_context
            6. 从 state 中读取 tool_results 并格式化工具结果
            7. 组装 prompt
            8. 调用 LLM 生成最终回答
            9. 保存 checkpoint
            10. 返回 answer 字段

        参数：
            state：
                DogState，LangGraph 当前状态。
                包含用户问题、历史消息、工具结果、长期记忆等字段。

        返回值：
            dict[str, str]：
                返回 {"answer": answer_text}，供 LangGraph 合并进 state。
        """

        node_name = "answer_gen_node"

        ctx = runtime_context_getter()

        if ctx is not None:
            ctx.state().set_node(
                node_name
            )

            ctx.timeline().add_event(
                event_type="node",
                name=node_name
            )

        logger.info(
            "开始执行 answer_gen_node"
        )

        question = str(
            state.get(
                "question",
                ""
            )
        )

        memory_context = str(
            state.get(
                "memory_context",
                "暂无用户记忆"
            )
            or "暂无用户记忆"
        )

        logger.info(
            f"answer_gen_node 接收到 memory_context: {memory_context}"
        )

        history_text = format_history(
            state.get(
                "messages",
                []
            )
        )

        tool_results_text = format_tool_results(
            state.get(
                "tool_results",
                []
            )
        )

        main_llm = llm_provider.main_llm

        base_prompt = build_answer_prompt(
            question=question,
            memory_context=memory_context,
            history_text=history_text,
            tool_results_text=tool_results_text,
        )

        answer_text = "模型暂时不可用"

        try:
            response = await llm_provider.safe_ainvoke(
                llm=main_llm,
                prompt=base_prompt,
                fallback_response="模型暂时不可用"
            )

            answer_text = str(
                getattr(
                    response,
                    "content",
                    response
                )
            )

            logger.info(
                "answer_gen_node 回答生成完成"
            )

        except Exception as e:
            logger.warning(
                f"answer_gen_node 生成回答失败: {e}"
            )

        if checkpoint_manager is not None:
            try:
                checkpoint_manager.save_checkpoint()

            except Exception as checkpoint_error:
                logger.warning(
                    f"answer_gen_node 保存 checkpoint 失败: {checkpoint_error}"
                )

        return {
            "answer": answer_text
        }

    return answer_gen_node


def format_history(
    messages: list[Any]
) -> str:
    """
    格式化历史消息。

    功能：
        将 LangChain messages 转换成纯文本。
        HumanMessage 格式化为“用户: xxx”。
        其他消息格式化为“助手: xxx”。

    参数：
        messages：
            历史消息列表。

    返回值：
        str：
            格式化后的历史文本。

    专业名词：
        HumanMessage（用户消息）：
            LangChain 中表示用户输入的消息类型。
    """

    formatted = []

    for message in messages:

        content = str(
            getattr(
                message,
                "content",
                ""
            )
        )

        if isinstance(
            message,
            HumanMessage
        ):
            formatted.append(
                f"用户: {content}"
            )

        else:
            formatted.append(
                f"助手: {content}"
            )

    return "\n".join(
        formatted
    )


def normalize_tool_results(
    tool_results: Any
) -> list:
    """
    归一化工具结果。

    功能：
        将 tool_results 统一转换成 list。
        兼容历史遗留字符串、None、dict 等格式。
        后续统一按多工具结果列表处理。

    参数：
        tool_results：
            state 中的工具结果。
            可能是 list、str、dict、None 或其他类型。

    返回值：
        list：
            归一化后的工具结果列表。

    专业名词：
        Normalize（归一化）：
            把不同格式的数据统一转换成一种稳定格式。
    """

    if tool_results is None:
        return []

    if isinstance(
        tool_results,
        list
    ):
        return tool_results

    if isinstance(
        tool_results,
        str
    ):
        return [
            tool_results
        ]

    return [
        tool_results
    ]


def format_tool_results(
    tool_results: Any
) -> str:
    """
    格式化工具结果。

    功能：
        将 tool_results 统一格式化成适合注入 Prompt 的文本。
        支持 list[str]。
        支持 list[dict]。
        兼容历史遗留字符串。

    参数：
        tool_results：
            工具结果，可能是 list、str、dict、None。

    返回值：
        str：
            格式化后的工具结果文本。
            如果没有工具结果，则返回空字符串。

    专业名词：
        Tool Result Formatting（工具结果格式化）：
            将结构化或非结构化工具结果转换成 LLM 可读文本。
    """

    normalized_results = normalize_tool_results(
        tool_results
    )

    if not normalized_results:
        return ""

    formatted_results = []

    for index, result in enumerate(
        normalized_results,
        start=1,
    ):
        if isinstance(
            result,
            dict
        ):
            tool_name = result.get(
                "tool_name",
                result.get(
                    "name",
                    "unknown_tool"
                )
            )

            success = result.get(
                "success",
                None
            )

            content = result.get(
                "result",
                result.get(
                    "content",
                    result
                )
            )

            if success is None:
                formatted_results.append(
                    f"{index}. 工具：【{tool_name}】\n结果：{content}"
                )

            else:
                formatted_results.append(
                    f"{index}. 工具：【{tool_name}】\n"
                    f"是否成功：{success}\n"
                    f"结果：{content}"
                )

        else:
            formatted_results.append(
                f"{index}. {result}"
            )

    return "\n\n".join(
        formatted_results
    )


def build_answer_prompt(
    question: str,
    memory_context: str,
    history_text: str,
    tool_results_text: str = "",
) -> str:
    """
    构建最终回答 Prompt。

    功能：
        将长期记忆、对话历史、用户问题、工具结果组装成 LLM 输入文本。
        如果 tool_results_text 不为空，则把工具结果放在 Prompt 前部。

    参数：
        question：
            用户问题。

        memory_context：
            用户长期记忆上下文。

        history_text：
            对话历史文本。

        tool_results_text：
            工具结果文本。
            没有工具结果时为空字符串。

    返回值：
        str：
            最终传给 LLM 的 Prompt 文本。
    """

    base_prompt = f"""
你是一个只能基于提供信息回答的助手。

下面是你可以参考的信息。

【用户长期记忆】
{memory_context}

【对话历史】
{history_text}

【用户问题】
{question}

【重要规则】
1. 如果用户的问题是关于他自己之前说过的话，例如“我最喜欢什么狗狗”“我之前说过什么偏好”，你可以同时参考【用户长期记忆】和【对话历史】。
2. 如果【用户长期记忆】和【对话历史】都没有相关信息，回答“我不知道”。
3. 不要利用你自己学到的通用知识来猜用户个人信息。
4. 如果用户问题明确是在问百科知识、技术知识、代码问题，则可以正常回答。
5. 如果【用户长期记忆】和当前问题相关，可以自然参考它。
6. 如果【用户长期记忆】和当前问题无关，不要强行使用。
7. 技术名词尽量附带中文解释。
8. 只输出答案内容，不要额外解释你的推理过程。
"""

    if tool_results_text:
        return (
            f"【工具结果】\n{tool_results_text}\n\n"
            + base_prompt
        )

    return base_prompt