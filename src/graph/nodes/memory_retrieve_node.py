import inspect
from typing import Any, Awaitable, Callable

from src.logger import logger
from src.runtime.context import runtime_ctx


def build_memory_retrieve_node(
    semantic_recall: Any,
    checkpoint_manager: Any = None,
    runtime_context_getter=None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, str]]]:
    """
    构建 Memory 召回节点。

    功能：
        接收外部注入的 MemorySemanticRecallService。
        接收外部注入的 checkpoint_manager 检查点管理器。
        接收外部注入的 runtime_context_getter 运行时上下文获取函数。
        返回一个符合 LangGraph 节点签名的 async node 函数。
        node 执行时只使用注入的服务，不直接 import container。
        接入 Runtime Context、Timeline、Checkpoint、Logger。
        避免 Graph Node 与 RuntimeContainer 之间产生循环导入。

    参数：
        semantic_recall：
            MemorySemanticRecallService（记忆语义召回服务）。
            用于根据用户问题召回相关长期记忆。
            需要提供 retrieve(user_id, question, limit) 方法。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于在关键节点执行后保存运行状态。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        Callable[[dict[str, Any]], Awaitable[dict[str, str]]]：
            返回一个 LangGraph 可调用的异步节点函数。
            该函数接收 state，返回需要合并进 state 的 dict。

    专业名词：
        Semantic Recall（语义召回）：
            根据语义相似度查找相关长期记忆，而不是只做关键词匹配。

        Runtime Context（运行时上下文）：
            当前请求执行过程中的上下文对象，用于记录状态、时间线、trace 等信息。

        Checkpoint（检查点）：
            用于保存当前运行状态，方便恢复、追踪和调试。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def memory_retrieve_node(
        state: dict[str, Any]
    ) -> dict[str, str]:
        """
        执行 Memory 语义召回。

        功能：
            1. 设置当前运行节点名称
            2. 写入 Timeline 时间线事件
            3. 从 state 中读取 user_id
            4. 如果没有 user_id，则使用 session_id
            5. 如果 user_id 和 session_id 都没有，则使用 default_user
            6. 从 state 中读取用户问题 question
            7. 调用 MemorySemanticRecallService 召回相关长期记忆
            8. 将召回结果写入 memory_context 字段
            9. 保存 checkpoint 检查点
            10. 如果召回失败，则返回“暂无用户记忆”，不阻断主流程

        参数：
            state：
                LangGraph 当前状态。
                Graph 节点之间传递的数据字典。

        返回值：
            dict[str, str]：
                返回需要合并进 state 的字段。
                当前只返回 memory_context。
        """

        node_name = "memory_retrieve_node"
        memory_context = "暂无用户记忆"

        try:
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
                "开始执行 Memory 召回节点"
            )

            user_id = str(
                state.get(
                    "user_id"
                )
                or state.get(
                    "session_id"
                )
                or "default_user"
            )

            question = str(
                state.get(
                    "question"
                )
                or ""
            )

            logger.info(
                f"Memory Retrieve 输入: user_id={user_id}, question={question}"
            )

            retrieved_memory = semantic_recall.retrieve(
                user_id=user_id,
                question=question,
                limit=5
            )

            if inspect.isawaitable(
                retrieved_memory
            ):
                retrieved_memory = await retrieved_memory

            memory_context = _format_memory_context(
                retrieved_memory
            )

            logger.info(
                f"Memory Retrieve 结果: memory_context={memory_context}"
            )

        except Exception as e:
            logger.warning(
                f"Memory 召回失败，已降级为空记忆: {e}"
            )

        if checkpoint_manager is not None:
            try:
                checkpoint_manager.save_checkpoint()

            except Exception as checkpoint_error:
                logger.warning(
                    f"Memory 召回节点保存 checkpoint 失败: {checkpoint_error}"
                )

        return {
            "memory_context": memory_context
        }

    return memory_retrieve_node


def _format_memory_context(
    retrieved_memory: Any,
) -> str:
    """
    格式化 Memory 召回结果。

    功能：
        将 MemorySemanticRecallService 返回的结果统一转换成字符串。
        如果返回 None 或空值，则使用“暂无用户记忆”。
        如果返回 str，则直接使用。
        如果返回其他类型，则使用 str(...) 转换。

    参数：
        retrieved_memory：
            记忆召回服务返回的结果。
            可能是 str、list、dict、None 或其他对象。

    返回值：
        str：
            可注入 state["memory_context"] 的记忆上下文文本。
    """

    if not retrieved_memory:
        return "暂无用户记忆"

    if isinstance(
        retrieved_memory,
        str
    ):
        return retrieved_memory

    return str(
        retrieved_memory
    )