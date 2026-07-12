import inspect
from typing import Any, Awaitable, Callable

from src.graph.states.dog_state import (
    DogState
)

from src.logger import logger

from src.memory.memory_extract import (
    extract_memory
)

from src.runtime.context import (
    runtime_ctx
)


SUCCESSFUL_MEMORY_SAVE_ACTIONS = frozenset(
    {
        "created",
        "reinforced",
        "reactivated",
    }
)


def is_memory_save_success(
        save_result: dict[str, Any] | None
) -> bool:
    """
    判断记忆保存结果是否代表真实写入成功。

    功能：
    - 读取 save_result.action 保存动作
    - 只接受创建、强化和重新激活
    - 同时要求返回有效 memory_id，避免把数据库失败误报为已保存

    参数：
    - save_result: dict[str, Any] | None
      MemoryManager.save_memory 返回的保存结果。

    返回值：
    - bool
      True 表示记忆已真实保存或更新，False 表示跳过或失败。
    """

    if not isinstance(
        save_result,
        dict
    ):
        return False

    action = str(
        save_result.get(
            "action",
            ""
        )
        or ""
    )

    return (
        action in SUCCESSFUL_MEMORY_SAVE_ACTIONS
        and save_result.get("memory_id") is not None
    )


def build_memory_extract_node(
        llm_provider: Any,
        memory_provider: Any,
        checkpoint_manager: Any = None,
        runtime_context_getter: Callable[[], Any] | None = None,
        memory_extractor: Callable[..., Any] = extract_memory,
) -> Callable[[DogState], Awaitable[dict[str, Any]]]:
    """
    构建 Memory（记忆）抽取节点。

    功能：
        在主图构建阶段接收 LLMProvider（大语言模型服务提供者）、
        MemoryProvider（记忆服务提供者）和 CheckpointManager（检查点管理器），
        返回一个只需接收 DogState 的 LangGraph 异步节点。
        节点内部不再导入或查询全局 RuntimeContainer（运行时容器）。

    参数：
        llm_provider：
            用于 memory_extractor 调用 LLM 并抽取长期记忆。
        memory_provider：
            提供 manager.save_memory，负责 SQLite 保存和 Chroma 向量同步。
        checkpoint_manager：
            可选的检查点管理器，节点执行后保存检查点。
        runtime_context_getter：
            可选的 RuntimeContext（运行时上下文）获取函数。
            未传入时使用 runtime_ctx.get。
        memory_extractor：
            可选的记忆抽取函数，默认使用 extract_memory。
            保留该参数便于单元测试注入 Fake（测试假对象）。

    返回值：
        Callable[[DogState], Awaitable[dict[str, Any]]]：
            LangGraph 可调用的异步记忆抽取节点。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def memory_extract_node(
            state: DogState
    ) -> dict[str, Any]:
        """
        抽取当前输入中值得长期保存的记忆。

        功能：
            从 state 读取 question 和 user_id，调用注入的记忆抽取器；
            should_save=True 时通过 MemoryProvider.manager 保存记忆；
            抽取、保存或 checkpoint 失败时记录日志并保持主图继续。

        参数：
            state：LangGraph 当前 DogState（狗狗智能体状态）。

        返回值：
            dict[str, Any]：包含 memory_saved、memory_extract_result 和
            memory_save_result 的 state update（状态更新）。
        """

        node_name = "memory_extract_node"

        runtime_context = runtime_context_getter()

        if runtime_context is not None:
            runtime_context.state().set_node(
                node_name
            )

            runtime_context.timeline().add_event(
                event_type="node",
                name=node_name
            )

        logger.info(
            f"开始执行{node_name}"
        )

        question = str(
            state.get(
                "question",
                ""
            )
        )

        user_id = str(
            state.get("user_id")
            or state.get("session_id")
            or "default_user"
        )

        memory_saved = False

        memory_extract_result: dict[str, Any] = {
            "should_save": False,
            "memory_type": "preference",
            "content": "",
            "confidence": 0.0,
            "importance": 0.0,
            "reason": "未执行 Memory 抽取",
        }

        save_result: dict[str, Any] | None = None

        if not question:
            logger.warning(
                "memory_extract_node 未获取到 question，跳过 Memory 抽取"
            )

            return {
                "memory_saved": memory_saved,
                "memory_extract_result": memory_extract_result,
                "memory_save_result": save_result,
            }

        if user_id == "default_user":
            logger.warning(
                "memory_extract_node 未获取到真实 user_id，当前使用 default_user"
            )

        try:
            logger.info(
                f"Memory Extract 输入: user_id={user_id}, question={question}"
            )

            extracted_memory = memory_extractor(
                llm_provider=llm_provider,
                question=question
            )

            if inspect.isawaitable(extracted_memory):
                extracted_memory = await extracted_memory

            memory_extract_result = dict(
                extracted_memory
                or {}
            )

            logger.info(
                f"Memory 抽取结果: {memory_extract_result}"
            )

            if memory_extract_result.get(
                "should_save",
                False
            ):

                save_result = memory_provider.manager.save_memory(
                user_id=user_id,
                memory_type=str(
                    memory_extract_result.get(
                        "memory_type",
                        "preference"
                    )
                ),
                content=str(
                    memory_extract_result.get(
                        "content",
                        ""
                    )
                ),
                confidence=float(
                    memory_extract_result.get(
                        "confidence",
                        0.0
                    )
                ),
                importance=float(
                    memory_extract_result.get(
                        "importance",
                        0.5
                    )
                ),
                source="conversation",
            )

                memory_saved = is_memory_save_success(
                    save_result
                )

                logger.info(
                    f"Memory Save 结果: user_id={user_id}, result={save_result}"
                )

            else:
                logger.info(
                    f"当前输入不需要保存为 Memory: {memory_extract_result.get('reason')}"
                )

            if checkpoint_manager is not None:
                try:
                    checkpoint_manager.save_checkpoint()

                except Exception as checkpoint_error:
                    logger.warning(
                        f"memory_extract_node 保存 checkpoint 失败: {checkpoint_error}"
                    )

        except Exception as e:
            logger.warning(
                f"memory_extract_node 执行失败，已跳过 Memory 保存: {e}"
            )

        return {
            "memory_saved": memory_saved,
            "memory_extract_result": memory_extract_result,
            "memory_save_result": save_result,
        }

    return memory_extract_node
