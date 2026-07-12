from datetime import datetime

from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore
)

from src.memory.memory_conflict import (
    MemoryConflictResolver
)

from src.memory.memory_document_mapper import (
    memory_dict_to_document,
    get_memory_chroma_id,
)

from src.memory.memory_content_normalizer import (
    normalize_memory_content
)
from src.memory.memory_schema import (
    VALID_MEMORY_SOURCES,
    VALID_MEMORY_TYPES,
)

from src.logger import logger


class MemoryManager:

    def __init__(
            self,
            store: SQLiteMemoryStore,
            vectorstore_provider=None
    ):
        """
           初始化 MemoryManager。

           功能：
           - 接收 MemoryProvider 注入的 SQLiteMemoryStore
           - 接收 VectorStoreProvider，用于同步 Chroma memory_db
           - 初始化冲突处理器
           - 统一管理 Memory 保存、强化、冲突失效、向量同步流程

           参数：
           - store: SQLiteMemoryStore
             Memory 主数据库存储服务。
             中文释义：负责把用户长期记忆写入 SQLite。

           - vectorstore_provider:
             向量数据库提供者。
             中文释义：负责提供 Chroma memory_db，用于语义召回。

           返回值：
           - None
             初始化函数不返回业务数据。
       """

        self.store = store

        self.conflict_resolver = MemoryConflictResolver()

        self.vectorstore_provider = vectorstore_provider

    def _sync_memory_to_vectorstore(
            self,
            memory_id: int
    ) -> None:
        """
        将单条 Memory 同步到 Chroma 向量数据库。

        功能：
        - 根据 memory_id 从 SQLite 查询最新记忆
        - 将记忆转换成 LangChain Document
        - 删除 Chroma 中旧的同 ID 向量
        - 重新写入最新 Document
        - 实现一种简单稳定的 upsert 效果

        参数：
        - memory_id: int
          SQLite 中 user_memory 表的主键 ID

        返回值：
        - None
          该函数只负责同步，不返回业务数据
        """

        if self.vectorstore_provider is None:
            logger.warning(
                "MemoryManager 未注入 vectorstore_provider，跳过向量同步"
            )
            return

        memory = self.store.get_memory_by_id(
            memory_id
        )

        if not memory:
            logger.warning(
                f"未找到 memory_id={memory_id} 的记忆，跳过向量同步"
            )
            return

        document = memory_dict_to_document(
            memory
        )

        chroma_id = get_memory_chroma_id(
            memory_id
        )

        memory_db = self.vectorstore_provider.memory_db

        try:
            memory_db.delete(
                ids=[
                    chroma_id
                ]
            )
        except Exception as e:
            logger.warning(
                f"删除旧 Memory 向量失败，可能是首次写入: {e}"
            )

        memory_db.add_documents(
            documents=[
                document
            ],
            ids=[
                chroma_id
            ]
        )

        logger.info(
            f"Memory 已同步到 Chroma: memory_id={memory_id}, chroma_id={chroma_id}"
        )

    def _deactivate_memory_vector(
            self,
            memory_id: int
    ) -> None:
        """
        删除 Chroma 中指定 Memory 的向量数据。

        功能：
        - 当 SQLite 中的记忆被设置为 inactive 后
        - 删除 Chroma 中对应的向量，避免召回失效记忆
        - 当前策略是物理删除向量，而不是只更新 metadata.status

        参数：
        - memory_id: int
          SQLite 中 user_memory 表的主键 ID

        返回值：
        - None
          该函数只负责删除向量，不返回业务数据
        """

        if self.vectorstore_provider is None:
            logger.warning(
                "MemoryManager 未注入 vectorstore_provider，跳过向量删除"
            )
            return

        chroma_id = get_memory_chroma_id(
            memory_id
        )

        memory_db = self.vectorstore_provider.memory_db

        try:
            memory_db.delete(
                ids=[
                    chroma_id
                ]
            )

            logger.info(
                f"Memory 向量已删除: memory_id={memory_id}, chroma_id={chroma_id}"
            )

        except Exception as e:
            logger.warning(
                f"删除 Memory 向量失败: memory_id={memory_id}, error={e}"
            )

    def save_memory(
            self,
            user_id: str,
            memory_type: str,
            content: str,
            confidence: float,
            importance: float = 0.5,
            source: str = "conversation",
            expires_at: str | None = None,
    ):
        """
        保存用户记忆。

        功能：
        - 在保存前对 memory_type 和 content 做兜底清洗
        - 在冲突检查之前归一化 content，保证冲突判断使用稳定内容
        - 检查是否存在冲突记忆
        - 将冲突记忆设置为 inactive
        - 删除冲突记忆在 Chroma memory_db 中的向量
        - 如果当前记忆已存在，则强化 strength 和 confidence
        - 如果当前记忆不存在，则创建新记忆
        - 将新增或更新后的记忆同步到 Chroma memory_db

        参数：
        - user_id: str
          用户 ID。
          中文释义：用于区分不同用户的记忆，避免不同用户的长期记忆互相污染。

        - memory_type: str
          记忆类型。
          中文释义：例如 favorite_dog、preference、dislike、hobby、profile。

        - content: str
          记忆内容。
          中文释义：保存前会被归一化，例如“用户喜欢金毛”会归一化为“金毛”。

        - confidence: float
          LLM 判断这条记忆可信度的分数。
          中文释义：取值范围建议为 0 到 1。

        - importance: float
          记忆重要程度。
          中文释义：取值范围为 0 到 1，越高表示越值得影响后续回答。

        - source: str
          记忆来源。
          中文释义：由系统填写，例如 conversation、tool、manual、system。

        - expires_at: str | None
          记忆过期时间。
          中文释义：None 表示长期有效，否则保存 ISO 格式时间字符串。

        返回值：
        - dict
          返回本次保存动作结果。
          包含 action、deactivated、memory_id、memory_type、content、strength 等字段。
        """

        clean_user_id = str(
            user_id
            or ""
        ).strip()

        clean_memory_type = str(
            memory_type
            or "preference"
        ).strip()

        if clean_memory_type not in VALID_MEMORY_TYPES:
            logger.warning(
                f"非法 memory_type，已兜底为 preference: {clean_memory_type!r}"
            )
            clean_memory_type = "preference"

        clean_source = str(
            source
            or "conversation"
        ).strip()

        if clean_source not in VALID_MEMORY_SOURCES:
            logger.warning(
                f"非法 memory source，已兜底为 conversation: {clean_source!r}"
            )
            clean_source = "conversation"

        clean_content = normalize_memory_content(
            memory_type=clean_memory_type,
            content=content
        )

        try:
            clean_confidence = float(
                confidence
            )

        except Exception:
            clean_confidence = 0.0

        clean_confidence = max(
            0.0,
            min(
                clean_confidence,
                1.0
            )
        )

        try:
            clean_importance = float(
                importance
            )

        except Exception:
            clean_importance = 0.5

        clean_importance = max(
            0.0,
            min(
                clean_importance,
                1.0
            )
        )

        if not clean_user_id:
            return {
                "action": "skipped",
                "reason": "user_id 为空",
                "memory_id": None,
                "memory_type": clean_memory_type,
                "content": clean_content,
            }

        if not clean_content:
            return {
                "action": "skipped",
                "reason": "memory content 为空",
                "memory_id": None,
                "memory_type": clean_memory_type,
                "content": clean_content,
            }

        # 1. 先检查冲突记忆
        #
        # 注意：
        # 这里必须使用归一化后的 clean_content。
        # 例如：
        # - 用户喜欢金毛
        # - 金毛
        # 都应该统一成“金毛”后再判断冲突。
        conflict_types = (
            self.conflict_resolver
            .get_conflict_types(
                clean_memory_type
            )
        )

        conflict_memories = (
            self.store.find_conflict_memories(
                user_id=clean_user_id,
                conflict_types=conflict_types,
                content=clean_content
            )
        )

        deactivated_count = 0

        for memory in conflict_memories:
            conflict_memory_id = int(
                memory["id"]
            )

            self.store.deactivate_memory(
                conflict_memory_id
            )

            self._deactivate_memory_vector(
                conflict_memory_id
            )

            deactivated_count += 1

        # 2. 再检查当前记忆是否已存在
        #
        # 这里同样必须使用归一化后的 clean_content。
        # 否则“用户喜欢金毛”和“金毛”会被当成两条不同记忆。
        existing = self.store.find_memory(
            user_id=clean_user_id,
            memory_type=clean_memory_type,
            content=clean_content,
            only_active=True,
        )

        # 3. 存在则强化
        if existing:
            old_strength = float(
                existing.get(
                    "strength",
                    1.0
                )
                or 1.0
            )

            old_confidence = float(
                existing.get(
                    "confidence",
                    0.0
                )
                or 0.0
            )

            old_importance = float(
                0.5
                if existing.get("importance") is None
                else existing["importance"]
            )

            new_strength = (
                    old_strength
                    + clean_confidence
            )

            new_confidence = max(
                old_confidence,
                clean_confidence
            )

            new_importance = max(
                old_importance,
                clean_importance
            )

            final_expires_at = (
                expires_at
                if expires_at is not None
                else existing.get(
                    "expires_at"
                )
            )

            memory_id = int(
                existing["id"]
            )

            self.store.update_memory(
                memory_id=memory_id,
                confidence=new_confidence,
                strength=new_strength,
                last_seen=datetime.now().isoformat(),
                status="active",
                source=clean_source,
                importance=new_importance,
                expires_at=final_expires_at,
            )

            self._sync_memory_to_vectorstore(
                memory_id
            )

            return {
                "memory_id": memory_id,
                "action": "reinforced",
                "deactivated": deactivated_count,
                "memory_type": clean_memory_type,
                "content": clean_content,
                "confidence": new_confidence,
                "strength": new_strength,
                "source": clean_source,
                "importance": new_importance,
                "expires_at": final_expires_at,
            }

        # 4. 同类型记忆曾经存在但已失效时，复用原记录并重新激活。
        # 这样既保留同一 memory_id，也不会触发 SQLite 唯一约束导致插入被忽略。
        inactive_existing = self.store.find_memory(
            user_id=clean_user_id,
            memory_type=clean_memory_type,
            content=clean_content,
            only_active=False,
        )

        if inactive_existing:
            memory_id = int(
                inactive_existing["id"]
            )
            old_strength = float(
                inactive_existing.get(
                    "strength",
                    1.0
                )
                or 1.0
            )
            old_confidence = float(
                inactive_existing.get(
                    "confidence",
                    0.0
                )
                or 0.0
            )
            old_importance = float(
                0.5
                if inactive_existing.get("importance") is None
                else inactive_existing["importance"]
            )
            reactivated_strength = (
                old_strength
                + clean_confidence
            )
            reactivated_confidence = max(
                old_confidence,
                clean_confidence
            )
            reactivated_importance = max(
                old_importance,
                clean_importance
            )

            self.store.update_memory(
                memory_id=memory_id,
                confidence=reactivated_confidence,
                strength=reactivated_strength,
                last_seen=datetime.now().isoformat(),
                status="active",
                source=clean_source,
                importance=reactivated_importance,
                expires_at=expires_at,
                update_expires_at=True,
            )

            self._sync_memory_to_vectorstore(
                memory_id
            )

            return {
                "action": "reactivated",
                "deactivated": deactivated_count,
                "memory_id": memory_id,
                "memory_type": clean_memory_type,
                "content": clean_content,
                "confidence": reactivated_confidence,
                "strength": reactivated_strength,
                "source": clean_source,
                "importance": reactivated_importance,
                "expires_at": expires_at,
            }

        # 5. 历史中也不存在时才创建新记录。
        initial_strength = max(
            clean_confidence,
            1.0
        )

        memory_id = self.store.add_memory(
            user_id=clean_user_id,
            memory_type=clean_memory_type,
            content=clean_content,
            confidence=clean_confidence,
            strength=initial_strength,
            status="active",
            source=clean_source,
            importance=clean_importance,
            expires_at=expires_at,
        )

        if memory_id is not None:
            self._sync_memory_to_vectorstore(
                memory_id
            )

        if memory_id is None:
            return {
                "action": "failed",
                "reason": "SQLite 未能创建或定位记忆记录",
                "deactivated": deactivated_count,
                "memory_id": None,
                "memory_type": clean_memory_type,
                "content": clean_content,
                "confidence": clean_confidence,
                "strength": initial_strength,
                "source": clean_source,
                "importance": clean_importance,
                "expires_at": expires_at,
            }

        return {
            "action": "created",
            "deactivated": deactivated_count,
            "memory_id": memory_id,
            "memory_type": clean_memory_type,
            "content": clean_content,
            "confidence": clean_confidence,
            "strength": initial_strength,
            "source": clean_source,
            "importance": clean_importance,
            "expires_at": expires_at,
        }
