from typing import Any, Awaitable, Callable

from src.logger import logger
from src.runtime.context import runtime_ctx


def build_memory_retrieve_node(
        semantic_recall: Any,
        checkpoint_manager: Any
) -> Callable[[dict[str, Any]], Awaitable[dict[str, str]]]:
    """
    构建 Memory 召回节点。

    功能：
    - 接收外部注入的 MemorySemanticRecallService
    - 接收外部注入的 checkpoint_manager 检查点管理器
    - 返回一个符合 LangGraph 节点签名的 async node 函数
    - node 执行时只使用注入的服务，不直接 import container
    - 接入 Runtime Context、Timeline、Checkpoint、Logger
    - 避免 Graph Node 与 RuntimeContainer 之间产生循环导入

    参数：
    - semantic_recall:
      MemorySemanticRecallService 实例。
      中文释义：Memory 语义召回服务，用于根据用户问题召回相关长期记忆。

    - checkpoint_manager:
      检查点管理器。
      中文释义：用于在关键节点执行后保存运行状态，方便追踪、恢复或调试。

    返回值：
    - Callable[[dict[str, Any]], Awaitable[dict[str, str]]]
      返回一个 LangGraph 可调用的异步节点函数。
      该函数接收 state，返回需要合并进 state 的 dict。
    """

    async def memory_retrieve_node(
            state: dict[str, Any]
    ) -> dict[str, str]:
        """
        执行 Memory 语义召回。

        功能：
        - 设置当前运行节点名称
        - 写入 Timeline 时间线事件
        - 从 state 中读取 user_id
        - 从 state 中读取用户问题 question
        - 调用 MemorySemanticRecallService 召回相关长期记忆
        - 将召回结果写入 memory_context 字段
        - 保存 checkpoint 检查点
        - 如果召回失败，则返回“暂无用户记忆”，不阻断主流程

        参数：
        - state: dict[str, Any]
          LangGraph 当前状态。
          中文释义：Graph 节点之间传递的数据字典。

        返回值：
        - dict[str, str]
          返回需要合并进 state 的字段。
          当前只返回 memory_context。
        """

        node_name = "memory_retrieve_node"

        memory_context: str = "暂无用户记忆"

        try:
            runtime = runtime_ctx.get()

            runtime.state().set_node(
                node_name
            )

            runtime.timeline().add_event(
                event_type="node",
                name=node_name
            )

            logger.info(
                "开始执行 Memory 召回节点"
            )

            user_id = str(
                state.get("user_id")
                or state.get("session_id")
                or "default_user"
            )

            question = str(
                state.get("question")
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

            if isinstance(
                    retrieved_memory,
                    str
            ):
                memory_context = retrieved_memory

            elif retrieved_memory is not None:
                memory_context = str(
                    retrieved_memory
                )

            logger.info(
                f"Memory Retrieve 结果: memory_context={memory_context}"
            )

        except Exception as e:
            logger.warning(
                f"Memory 召回失败，已降级为空记忆: {e}"
            )

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