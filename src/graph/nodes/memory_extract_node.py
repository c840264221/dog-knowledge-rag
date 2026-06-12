from typing import Any

from src.graph.states.state import (
    DogState
)

from src.logger import logger

from src.memory.memory_extract import (
    extract_memory
)

from src.runtime.context import (
    runtime_ctx
)


async def memory_extract_node(
        state: DogState
) -> dict[str, Any]:
    """
    Memory 抽取节点。

    功能：
    - 从 state 中读取用户问题 question
    - 从 state 中读取当前 user_id
    - 调用 extract_memory 抽取长期记忆
    - 如果抽取结果 should_save=True，则调用 MemoryManager 保存记忆
    - MemoryManager 来自 MemoryProvider，已经注入 VectorStoreProvider
    - 保存后会自动写入 SQLite，并同步到 Chroma memory_db
    - 接入 Runtime Context、Timeline、Checkpoint、Logger
    - 即使 Memory 抽取或保存失败，也不阻断主图后续流程

    参数：
    - state: DogState
      LangGraph 当前状态。
      中文释义：Graph 节点之间传递的数据结构。

    返回值：
    - dict[str, Any]
      返回 Memory 抽取状态。
      包含 memory_saved、memory_extract_result。
    """

    node_name = "memory_extract_node"

    runtime_context = runtime_ctx.get()

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
        state.get(
            "user_id"
        )
        or state.get(
            "session_id"
        )
        or "default_user"
    )

    memory_saved = False

    memory_extract_result: dict[str, Any] = {
        "should_save": False,
        "memory_type": "preference",
        "content": "",
        "confidence": 0.0,
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
        from src.runtime.container.init import (
            container
        )

        llm_provider = container.get(
            "llm"
        )

        memory_provider = container.get(
            "memory"
        )

        checkpoint_manager = container.get(
            "checkpoint"
        ).manager


        logger.info(
            f"Memory Extract 输入: user_id={user_id}, question={question}"
        )

        memory_extract_result = await extract_memory(
            llm_provider=llm_provider,
            question=question
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
                )
            )

            memory_saved = True

            logger.info(
                f"Memory Save 结果: user_id={user_id}, result={save_result}"
            )

        else:
            logger.info(
                f"当前输入不需要保存为 Memory: {memory_extract_result.get('reason')}"
            )

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