from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from src.graph.nodes.generate_prompt_builder_node import (
    build_generation_prompt,
)
from src.graph.nodes.generate_strategy_resolver_node import (
    resolve_answer_strategy,
)
from src.logger import logger
from src.memory.memory_retrieve import retrieve_user_memory
from src.rag.context_builders.generation_context_builder import (
    resolve_generation_context,
)
from src.runtime.context import runtime_ctx
from src.runtime.scopes.retrieval_scope import RetrievalScope


def build_history_text(
        messages: list[Any],
) -> str:
    """
    构建历史对话文本。

    功能：
        将 LangGraph state["messages"] 中的消息转换成可读字符串，
        供 Prompt 使用。

        当前会过滤掉 Supervisor 写入的调试消息，
        避免污染 Prompt。

    技术名词：
        Message：
            消息对象。LangChain 中常见消息类型包括 HumanMessage、AIMessage 等。

        History：
            历史记录。这里指用户与助手之前的对话内容。

        Prompt Pollution：
            Prompt 污染。指无关调试内容进入 Prompt，影响 LLM 回答质量。

    参数：
        messages:
            历史消息列表，通常包含 HumanMessage、AIMessage 等对象。

    返回值：
        str:
            格式化后的历史对话文本。
    """

    lines = []

    for message in messages:

        content = str(
            getattr(
                message,
                "content",
                "",
            )
            or ""
        )

        if not content:
            continue

        if is_debug_message(
                content=content,
        ):
            continue

        if isinstance(
                message,
                HumanMessage,
        ):
            lines.append(
                f"用户: {content}"
            )

        elif isinstance(
                message,
                AIMessage,
        ):
            lines.append(
                f"助手: {content}"
            )

        else:
            lines.append(
                f"消息: {content}"
            )

    return "\n".join(
        lines
    )


def is_debug_message(
        content: str,
) -> bool:
    """
    判断消息是否属于调试消息。

    功能：
        过滤 Supervisor 决策、路由日志等调试内容，
        避免这些内容进入 generate_node 的 Prompt。

    参数：
        content:
            消息内容。

    返回值：
        bool:
            True 表示是调试消息，需要过滤。
            False 表示可以进入历史对话。
    """

    debug_prefixes = [
        "Supervisor决策:",
        "RouteDecision:",
        "路由决策:",
        "next_worker:",
        "next_agent:",
    ]

    for prefix in debug_prefixes:

        if content.startswith(
                prefix,
        ):
            return True

    return False


def build_generate_node(
        llm_provider,
        memory_provider=None,
        checkpoint_provider=None,
):
    """
    构建 generate_node 节点函数。

    功能：
        使用闭包方式注入 LLMProvider、MemoryProvider、CheckpointProvider。

        v1.5 当前职责：
        1. 从 state 中读取 question、intent、user_id。
        2. 调用 resolve_generation_context 获取 Prompt 上下文。
        3. 召回用户长期记忆。
        4. 解析 answer_strategy 回答策略。
        5. 根据 answer_strategy 构建不同 Prompt。
        6. 调用 LLM 生成最终回答。
        7. 返回 answer、final_answer、memory_context、answer_strategy、messages。

    技术名词：
        Closure：
            闭包。内部函数可以访问外部函数传入的变量。

        Provider：
            提供者。用于统一创建、缓存和管理服务对象。

        Node：
            节点。LangGraph 中执行某个业务步骤的函数。

        Prompt：
            提示词。发送给 LLM 的输入模板。

        LLM：
            Large Language Model，大语言模型。

        Answer Strategy：
            回答策略。用于决定当前问题应该采用哪种回答模板。

    参数：
        llm_provider:
            LLMProvider 实例。
            中文释义：用于获取主模型 main_llm，并执行安全 LLM 调用。

        memory_provider:
            MemoryProvider 实例。
            中文释义：用于召回用户长期记忆。

        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint 检查点。

    返回值：
        callable:
            返回一个 async generate_node 函数，供 LangGraph 注册使用。
    """

    async def generate_node(
            state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        生成最终回答。

        功能：
            1. 设置 runtime 当前节点。
            2. 解析 RAG 上下文。
            3. 召回用户长期记忆。
            4. 构建历史对话文本。
            5. 解析回答策略 answer_strategy。
            6. 根据回答策略构建 Prompt。
            7. 调用 LLM。
            8. 保存 checkpoint。
            9. 返回状态更新。

        参数：
            state:
                LangGraph 当前状态。
                中文释义：包含 question、intent、rag_context、docs、messages、user_id 等字段。

        返回值：
            dict[str, Any]:
                返回需要合并进 DogState 的字段：
                - answer
                - final_answer
                - memory_context
                - answer_strategy
                - messages
        """

        runtime = runtime_ctx.get()

        runtime.state().set_node(
            "generate_node"
        )

        runtime.timeline().add_event(
            event_type="node",
            name="generate_node",
        )

        question = str(
            state.get(
                "question",
                "",
            )
            or ""
        ).strip()

        intent = str(
            state.get(
                "intent",
                "general",
            )
            or "general"
        )

        user_id = str(
            state.get(
                "user_id",
                "default",
            )
            or "default"
        )

        context, context_source = resolve_generation_context(
            state=state,
        )

        answer_strategy = resolve_answer_strategy(
            state=state,
        )

        logger.info(
            "进入 generate_node 节点，"
            f"question={question}, "
            f"intent={intent}, "
            f"user_id={user_id}, "
            f"context_source={context_source}, "
            f"context_length={len(context)}, "
            f"answer_strategy={answer_strategy.task_type}"
        )

        write_docs_to_retrieval_scope_safely(
            state=state,
            runtime=runtime,
        )

        memory_text = await resolve_memory_text(
            user_id=user_id,
            question=question,
            memory_provider=memory_provider,
        )

        history_text = build_history_text(
            messages=state.get(
                "messages",
                [],
            )
        )

        logger.debug(
            f"generate_node history_text={history_text}"
        )

        logger.debug(
            "generate_node 使用上下文，"
            f"context_source={context_source}, "
            f"context_preview={context[:1000]}"
        )

        prompt_text = build_generation_prompt(
            state=state,
            answer_strategy=answer_strategy,
            context=context,
            context_source=context_source,
            memory_text=memory_text,
            history_text=history_text,
        )

        main_llm = llm_provider.main_llm

        raw_response = await llm_provider.safe_ainvoke(
            llm=main_llm,
            prompt=prompt_text,
            fallback_response="调用 LLM 失败",
        )

        answer = normalize_llm_response_to_text(
            response=raw_response,
        )

        logger.info(
            f"generate_node 节点完成，answer={answer}"
        )

        logger.debug(
            f"Runtime State:{runtime.state().get_state()}"
        )

        save_checkpoint_safely(
            checkpoint_provider=checkpoint_provider,
        )

        output_state = {
            "answer": answer,
            "final_answer": answer,
            "memory_context": memory_text,
            "answer_strategy": answer_strategy.model_dump(),
            "messages": [
                AIMessage(
                    content=answer,
                )
            ],
        }

        logger.debug(
            "generate_node 即将返回 output_state，"
            f"keys={list(output_state.keys())}, "
            f"has_answer={bool(output_state.get('answer'))}, "
            f"has_final_answer={bool(output_state.get('final_answer'))}, "
            f"answer_strategy={answer_strategy.model_dump()}"
        )

        return output_state

    return generate_node


async def resolve_memory_text(
        user_id: str,
        question: str,
        memory_provider=None,
) -> str:
    """
    解析用户长期记忆文本。

    功能：
        如果 memory_provider 存在，则调用 retrieve_user_memory 召回用户长期记忆。
        如果 memory_provider 不存在，则返回默认文本。

    技术名词：
        Memory：
            记忆。这里指用户长期偏好、历史信息、个性化上下文。

        Memory Retrieval：
            记忆召回。从记忆存储中找出和当前问题相关的用户记忆。

    参数：
        user_id:
            用户 ID。

        question:
            当前用户问题。

        memory_provider:
            MemoryProvider 实例，可选。

    返回值：
        str:
            用户长期记忆文本。
    """

    if memory_provider is None:
        return "暂无用户记忆"

    return await retrieve_user_memory(
        user_id=user_id,
        question=question,
        memory_provider=memory_provider,
        limit=10,
    )


def write_docs_to_retrieval_scope_safely(
        state: dict[str, Any],
        runtime,
) -> None:
    """
    安全写入 RetrievalScope。

    功能：
        如果 state 中存在 docs，则写入 RetrievalScope，
        方便后续调试、观察或其他运行时能力使用。

        如果写入失败，只记录 debug，不中断主流程。

    技术名词：
        RetrievalScope：
            检索作用域。用于在运行时保存当前检索到的文档。

        Runtime Scope：
            运行时作用域。表示一次执行过程中的临时上下文容器。

    参数：
        state:
            当前 DogState。

        runtime:
            当前 RuntimeContext。

    返回值：
        None。
    """

    try:
        retrieval_scope = runtime.service(
            RetrievalScope,
        )

        docs = state.get(
            "docs",
            [],
        )

        if docs:
            retrieval_scope.set_docs(
                docs
            )

    except Exception as e:
        logger.debug(
            f"generate_node 写入 RetrievalScope 失败，可忽略: {e}"
        )


def normalize_llm_response_to_text(
        response: Any,
) -> str:
    """
    将 LLM 返回值转换成文本。

    功能：
        兼容不同 LLM 返回格式：
        1. str
        2. AIMessage / BaseMessage，带 content 字段
        3. 其他对象

    参数：
        response:
            LLM 原始返回值。

    返回值：
        str:
            解析后的文本答案。
    """

    if response is None:
        return ""

    if hasattr(
            response,
            "content",
    ):
        return str(
            response.content
            or ""
        ).strip()

    return str(
        response
        or ""
    ).strip()


def save_checkpoint_safely(
        checkpoint_provider=None,
) -> None:
    """
    安全保存 checkpoint。

    功能：
        如果 checkpoint_provider 存在，则调用 save_checkpoint。
        如果保存失败，只记录 warning，不中断主流程。

    技术名词：
        Checkpoint：
            检查点。用于保存图执行过程中的中间状态。

    参数：
        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        None。
    """

    if checkpoint_provider is None:
        return

    try:
        checkpoint_provider.manager.save_checkpoint()

    except Exception as e:
        logger.warning(
            f"generate_node 保存 checkpoint 失败: {e}"
        )