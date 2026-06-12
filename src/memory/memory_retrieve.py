from src.logger import logger


async def retrieve_user_memory(
        user_id: str,
        question: str,
        memory_provider,
        limit: int = 10
) -> str:
    """
    召回用户相关记忆。

    功能：
    - 作为 Memory 召回的统一入口
    - 接收外部注入的 MemoryProvider
    - 使用 MemorySemanticRecallService 执行语义召回
    - 返回适合注入 Prompt 的字符串
    - 避免在函数内部直接 import container

    参数：
    - user_id: str
      用户 ID，用于限定只召回当前用户的记忆。

    - question: str
      用户当前问题，用于语义检索相关 Memory。

    - memory_provider:
      MemoryProvider 实例。
      中文释义：由 Container 统一管理的 Memory 服务提供者。

    - limit: int
      最多返回多少条记忆。

    返回值：
    - str
      格式化后的用户记忆文本。
      如果没有召回结果，则返回“暂无用户记忆”。
    """

    try:
        semantic_recall = memory_provider.semantic_recall

        return semantic_recall.retrieve(
            user_id=user_id,
            question=question,
            limit=limit
        )

    except Exception as e:
        logger.error(
            f"语义召回用户记忆失败: {e}"
        )

        return "暂无用户记忆"