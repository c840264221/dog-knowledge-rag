from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


async def answer_gen_node(
        state: DogState
) -> dict[str, str]:
    """
    生成最终回答节点。

    功能：
    - 从 state 中读取用户问题 question
    - 从 state 中读取历史消息 messages
    - 从 state 中读取工具结果 tool_results
    - 从 state 中读取长期记忆 memory_context
    - 将历史消息、长期记忆、工具结果一起注入 Prompt
    - 调用 LLM 生成最终回答
    - 接入 Runtime Context、Timeline、Checkpoint、Logger
    - 返回 answer 字段，供 LangGraph 合并进状态

    参数：
    - state: DogState
      LangGraph 当前状态。
      中文释义：Graph 节点之间传递的数据结构，包含用户问题、历史消息、工具结果、记忆上下文等。

    返回值：
    - dict[str, str]
      返回需要合并进 state 的字段。
      当前返回 {"answer": "..."}。
    """

    node_name = "answer_gen_node"

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        node_name
    )

    runtime.timeline().add_event(
        event_type="node",
        name=node_name
    )

    logger.info(
        "开始执行 answer_gen_node"
    )

    def get_llm_provider():
        """
        获取 LLMProvider。

        功能：
        - 延迟导入 container，暂时避免循环导入
        - 从 RuntimeContainer 中获取 llm provider

        参数：
        - 无

        返回值：
        - LLMProvider
          大语言模型 Provider，用于获取 main_llm 和 safe_ainvoke。
        """

        from src.runtime.container.init import (
            container
        )

        return container.get(
            "llm"
        )

    def get_checkpoint_manager():
        """
        获取 CheckpointManager。

        功能：
        - 延迟导入 container，暂时避免循环导入
        - 从 RuntimeContainer 中获取 checkpoint manager

        参数：
        - 无

        返回值：
        - CheckpointManager
          检查点管理器，用于保存当前运行状态。
        """

        from src.runtime.container.init import (
            container
        )

        return container.get(
            "checkpoint"
        ).manager

    def format_history(
            messages: list[Any]
    ) -> str:
        """
        格式化历史消息。

        功能：
        - 将 LangChain messages 转换成纯文本
        - HumanMessage 格式化为“用户: xxx”
        - 其他消息格式化为“助手: xxx”
        - 用于注入 Prompt 作为短期对话历史

        参数：
        - messages: list[Any]
          历史消息列表。

        返回值：
        - str
          格式化后的历史文本。
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

    tool_results = state.get(
        "tool_results",
        []
    )

    history_text = format_history(
        state.get(
            "messages",
            []
        )
    )

    llm_provider = get_llm_provider()

    main_llm = llm_provider.main_llm

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

    if tool_results:
        base_prompt = (
            f"【工具结果】\n{tool_results}\n\n"
            + base_prompt
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

    try:
        checkpoint_manager = get_checkpoint_manager()

        checkpoint_manager.save_checkpoint()

    except Exception as checkpoint_error:
        logger.warning(
            f"answer_gen_node 保存 checkpoint 失败: {checkpoint_error}"
        )

    return {
        "answer": answer_text
    }