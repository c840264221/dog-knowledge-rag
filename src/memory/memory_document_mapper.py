from langchain_core.documents import Document

from src.memory.memory_schema import MemoryRecord

from src.logger import logger


def memory_record_to_document(
        memory: MemoryRecord
) -> Document:
    """
    将 MemoryRecord 转换成 LangChain Document。

    功能：
    - 把 SQLite 中的记忆实体转换成 Chroma 可以存储的 Document
    - memory.content 会作为 page_content，用于向量化检索
    - 其他字段会作为 metadata，用于后续筛选和排序

    参数：
    - memory: MemoryRecord
      SQLite 中的一条记忆实体对象

    返回值：
    - Document
      LangChain 的文档对象，可写入 Chroma 向量数据库
    """

    if memory.id is None:

        logger.error("MemoryRecord.id cannot be None when converting to Document.")

        raise ValueError(
            "MemoryRecord.id cannot be None when converting to Document."
        )

    return Document(
        page_content=memory.content,
        metadata={
            "memory_id": str(memory.id),
            "user_id": memory.user_id,
            "memory_type": memory.memory_type,
            "confidence": float(memory.confidence),
            "strength": float(memory.strength),
            "status": memory.status,
            "created_at": memory.created_at or "",
            "last_seen": memory.last_seen or "",
        }
    )


def memory_dict_to_record(
        memory: dict
) -> MemoryRecord:
    """
    将 dict 格式的记忆数据转换成 MemoryRecord。

    功能：
    - SQLite 查询结果目前多数是 dict
    - 该函数负责把 dict 转换成标准 MemoryRecord
    - 方便后续统一转换成 Document

    参数：
    - memory: dict
      SQLite 查询出来的一条记忆字典数据

    返回值：
    - MemoryRecord
      标准记忆实体对象
    """

    memory_id = memory.get("id")

    if memory_id is not None:
        memory_id = int(memory_id)

    return MemoryRecord(
        id=memory_id,
        user_id=str(
            memory.get("user_id", "")
        ),
        memory_type=str(
            memory.get("memory_type", "")
        ),
        content=str(
            memory.get("content", "")
        ),
        confidence=float(
            memory.get("confidence", 0.0)
        ),
        strength=float(
            memory.get("strength", 1.0)
        ),
        status=str(
            memory.get("status", "active")
        ),
        created_at=memory.get("created_at"),
        last_seen=memory.get("last_seen"),
    )


def memory_dict_to_document(
        memory: dict
) -> Document:
    """
    将 dict 格式的记忆数据转换成 LangChain Document。

    功能：
    - 先把 dict 转换成 MemoryRecord
    - 再把 MemoryRecord 转换成 Document
    - 给 SQLite 查询结果直接写入 Chroma 使用

    参数：
    - memory: dict
      SQLite 查询出来的一条记忆字典数据

    返回值：
    - Document
      LangChain 的文档对象，可写入 Chroma 向量数据库
    """

    record = memory_dict_to_record(
        memory
    )

    return memory_record_to_document(
        record
    )


def get_memory_chroma_id(
        memory_id: int | str
) -> str:
    """
    生成 Chroma 中使用的 Memory 向量 ID。

    功能：
    - 将 SQLite 的 memory_id 转换成 Chroma 的字符串 ID
    - 保证 SQLite 和 Chroma 可以通过同一个 ID 对齐
    - 后续 update / delete 都依赖这个 ID

    参数：
    - memory_id: int | str
      SQLite 中的记忆 ID

    返回值：
    - str
      Chroma 中使用的向量文档 ID
    """

    return f"memory_{memory_id}"