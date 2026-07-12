from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore
)

from src.memory.memory_ranker import (
    MemoryRanker
)

from src.logger import logger
from src.settings import settings


class MemorySemanticRecallService:
    """
    MemorySemanticRecallService：Memory 语义召回服务。

    功能：
    - 使用 Chroma memory_db 对用户问题进行语义检索
    - 根据 Chroma 返回的 memory_id 回查 SQLite 主数据库
    - 结合语义分数、strength、时间衰减计算最终分数
    - 返回最相关的用户记忆列表
    """

    def __init__(
            self,
            store: SQLiteMemoryStore,
            vectorstore_provider,
            memory_ranker: MemoryRanker | None = None,
            minimum_semantic_score: float | None = None,
    ):
        """
        初始化 MemorySemanticRecallService。

        功能：
        - 接收 SQLiteMemoryStore，用于回查 Memory 主数据库
        - 接收 VectorStoreProvider，用于访问 Chroma memory_db
        - 接收 MemoryRanker，用于对召回结果精排
        - 避免在服务内部直接创建数据库连接

        参数：
        - store: SQLiteMemoryStore
          Memory 主数据库存储服务。
          中文释义：用于根据 memory_id 回查最新记忆内容。

        - vectorstore_provider:
          VectorStoreProvider 实例。
          中文释义：用于访问 Memory 专用 Chroma 向量库 memory_db。

        - memory_ranker: MemoryRanker | None
          记忆精排器。
          中文释义：用于结合语义分数、记忆强度、可信度等信息重新排序。

        - minimum_semantic_score: float | None
          最低语义相关分数。
          中文释义：候选低于该值时不会进入 SQLite 回查和后续精排。

        返回值：
        - None
          初始化函数不返回业务数据。
        """

        self.store = store

        self.vectorstore_provider = vectorstore_provider

        self.memory_ranker = (
            memory_ranker
            or MemoryRanker()
        )

        configured_minimum_score = (
            settings.memory.minimum_semantic_score
            if minimum_semantic_score is None
            else minimum_semantic_score
        )

        self.minimum_semantic_score = max(
            0.0,
            min(
                float(configured_minimum_score),
                1.0
            )
        )

    def _distance_to_score(
            self,
            distance: float
    ) -> float:
        """
        将 Chroma 返回的 distance 转换成 semantic_score。

        功能：
        - Chroma 返回的 distance 通常是距离，越小越相似
        - 业务排序更适合使用 score，越大越相关
        - 使用 1 / (1 + distance) 转换，避免 distance 大于 1 时出现负分

        参数：
        - distance: float
          Chroma 返回的距离分数

        返回值：
        - float
          语义相关分数，数值越大表示越相关
        """

        return 1.0 / (
            1.0 + float(distance)
        )

    def _extract_memory_id(
            self,
            metadata: dict
    ) -> int | None:
        """
        从 Chroma metadata 中提取 memory_id。

        功能：
        - 读取 metadata["memory_id"]
        - 将字符串 ID 转换成 int
        - 如果不存在或格式错误，则返回 None

        参数：
        - metadata: dict
          Chroma 文档中的 metadata

        返回值：
        - int | None
          成功时返回 SQLite memory_id；
          失败时返回 None。
        """

        raw_memory_id = metadata.get(
            "memory_id"
        )

        if raw_memory_id is None:
            return None

        try:
            return int(
                raw_memory_id
            )
        except ValueError:
            logger.warning(
                f"memory_id 转换失败: {raw_memory_id}"
            )
            return None

    def search(
            self,
            user_id: str,
            question: str,
            top_k: int = 5,
            candidate_k: int = 20
    ) -> list[dict]:
        """
        根据用户问题语义召回相关记忆。

        功能：
        - 使用 Chroma memory_db 按语义检索候选记忆
        - 使用 user_id 和 status 过滤结果
        - 从 Chroma metadata 中提取 memory_id
        - 回查 SQLite 获取最新记忆数据
        - 使用 MemoryScorer 计算时间衰减后的记忆分数
        - 合并 semantic_score 和 memory_score 得到 final_score
        - 返回最终 TopK 记忆列表

        参数：
        - user_id: str
          用户 ID，用于限定只召回当前用户的记忆

        - question: str
          用户当前问题，用于语义检索

        - top_k: int
          最终返回的记忆数量

        - candidate_k: int
          从 Chroma 中初步召回的候选数量。
          一般应该大于 top_k，方便后续重排。

        返回值：
        - list[dict]
          召回后的记忆列表。
          每条记录包含 SQLite 字段，以及 semantic_score、memory_score、final_score。
        """

        clean_question = question.strip()

        if not clean_question:
            return []

        memory_db = self.vectorstore_provider.memory_db

        chroma_results = memory_db.similarity_search_with_score(
            query=clean_question,
            k=candidate_k,
            filter={
                "$and": [
                    {
                        "user_id": {
                            "$eq": user_id
                        }
                    },
                    {
                        "status": {
                            "$eq": "active"
                        }
                    },
                ]
            }
        )

        if not chroma_results:
            return []

        memory_ids: list[int] = []

        semantic_score_map: dict[int, float] = {}

        distance_map: dict[int, float] = {}

        for document, distance in chroma_results:

            # 测试用  查看chroma召回结果
            logger.info(
                f"Memory Chroma召回: content={document.page_content}, "
                f"metadata={document.metadata}, "
                f"distance={distance}"
            )

            memory_id = self._extract_memory_id(
                document.metadata
            )

            if memory_id is None:
                continue

            if memory_id in semantic_score_map:
                continue

            distance_value = float(
                distance
            )

            semantic_score = self._distance_to_score(
                distance_value
            )

            if semantic_score < self.minimum_semantic_score:
                logger.info(
                    "Memory 候选未通过语义相关性门槛: "
                    f"memory_id={memory_id}, "
                    f"semantic_score={semantic_score:.4f}, "
                    f"minimum={self.minimum_semantic_score:.4f}"
                )
                continue

            memory_ids.append(
                memory_id
            )

            distance_map[memory_id] = distance_value

            semantic_score_map[memory_id] = semantic_score

        if not memory_ids:
            return []

        memories = self.store.get_memories_by_ids(
            memory_ids=memory_ids,
            only_active=True
        )

        ranked_memories = self.memory_ranker.rank(
            memories=memories,
            semantic_score_map=semantic_score_map,
            distance_map=distance_map,
            top_k=candidate_k
        )

        deduped_memories = self._deduplicate_memories(
            ranked_memories
        )

        return deduped_memories[:top_k]

    def format_memories(
            self,
            memories: list[dict]
    ) -> str:
        """
        将召回的 Memory 列表格式化为 Prompt 可用文本。

        功能：
        - 把结构化 Memory 转换成自然语言
        - 避免把 semantic_score、final_score 等调试分数暴露给 LLM
        - 根据 memory_type 转换成更容易理解的中文表达
        - 如果没有记忆，则返回“暂无用户记忆”

        参数：
        - memories: list[dict]
          search 方法返回的记忆列表。

        返回值：
        - str
          格式化后的用户记忆文本。
        """

        if not memories:
            return "暂无用户记忆"

        formatted = []

        for memory in memories:

            memory_type = str(
                memory.get(
                    "memory_type",
                    ""
                )
            )

            content = str(
                memory.get(
                    "content",
                    ""
                )
            ).strip()

            normalized_content = (
                self._normalize_memory_content_for_dedup(
                    memory_type=memory_type,
                    content=content
                )
            )

            if memory_type == "favorite_dog":
                formatted.append(
                    f"- 用户喜欢的狗狗：{normalized_content}"
                )

            elif memory_type == "dislike":
                formatted.append(
                    f"- 用户不喜欢：{normalized_content}"
                )

            elif memory_type == "preference":
                formatted.append(
                    f"- 用户偏好：{content}"
                )

            elif memory_type == "hobby":
                formatted.append(
                    f"- 用户兴趣：{content}"
                )

            elif memory_type == "profile":
                formatted.append(
                    f"- 用户信息：{content}"
                )

            else:
                formatted.append(
                    f"- {content}"
                )

        return "\n".join(
            formatted
        )

    def _normalize_memory_content_for_dedup(
            self,
            memory_type: str,
            content: str
    ) -> str:
        """
        为 Memory 去重归一化 content。

        功能：
        - 将不同表达形式的同一条记忆归一成相同 key
        - 例如“用户喜欢金毛”和“金毛”都归一成“金毛”
        - 主要用于召回结果去重，不会修改数据库原始内容
        - 避免 Prompt 中重复注入同一事实

        参数：
        - memory_type: str
          Memory 类型，例如 favorite_dog、dislike、preference。

        - content: str
          Memory 内容，例如“用户喜欢金毛”。

        返回值：
        - str
          归一化后的 content，用于构造去重 key。
        """

        clean_content = str(
            content
            or ""
        ).strip()

        if memory_type == "favorite_dog":
            clean_content = clean_content.replace(
                "用户喜欢",
                ""
            )

            clean_content = clean_content.replace(
                "用户最喜欢",
                ""
            )

            clean_content = clean_content.replace(
                "喜欢",
                ""
            )

        if memory_type == "dislike":
            clean_content = clean_content.replace(
                "用户不喜欢",
                ""
            )

            clean_content = clean_content.replace(
                "用户讨厌",
                ""
            )

            clean_content = clean_content.replace(
                "不喜欢",
                ""
            )

            clean_content = clean_content.replace(
                "讨厌",
                ""
            )

        return clean_content.strip()

    def _deduplicate_memories(
            self,
            memories: list[dict]
    ) -> list[dict]:
        """
        对召回到的 Memory 进行去重。

        功能：
        - 根据 memory_type 和归一化后的 content 去重
        - 如果多条 Memory 表示同一事实，只保留 final_score 最高的一条
        - 不修改 SQLite 和 Chroma，只优化当前召回结果
        - 避免 answer_gen_node 的 Prompt 中出现重复记忆

        参数：
        - memories: list[dict]
          MemoryRanker 排序后的记忆列表。

        返回值：
        - list[dict]
          去重后的记忆列表。
        """

        dedup_map: dict[str, dict] = {}

        for memory in memories:

            memory_type = str(
                memory.get(
                    "memory_type",
                    ""
                )
            )

            content = str(
                memory.get(
                    "content",
                    ""
                )
            )

            normalized_content = (
                self._normalize_memory_content_for_dedup(
                    memory_type=memory_type,
                    content=content
                )
            )

            dedup_key = (
                f"{memory_type}:{normalized_content}"
            )

            existing = dedup_map.get(
                dedup_key
            )

            if existing is None:
                dedup_map[dedup_key] = memory
                continue

            current_score = float(
                memory.get(
                    "final_score",
                    0.0
                )
            )

            existing_score = float(
                existing.get(
                    "final_score",
                    0.0
                )
            )

            if current_score > existing_score:
                dedup_map[dedup_key] = memory

        deduped_memories = list(
            dedup_map.values()
        )

        deduped_memories.sort(
            key=lambda memory: float(
                memory.get(
                    "final_score",
                    0.0
                )
            ),
            reverse=True
        )

        return deduped_memories

    def retrieve(
        self,
        user_id: str,
        question: str,
        limit: int | None = None
    ) -> str:
        """
        召回用户记忆并格式化成字符串。

        功能：
        - 根据用户问题执行语义召回
        - 获取最相关的用户记忆
        - 将结果格式化成适合 Prompt 注入的文本
        - 如果 limit 未传入，则使用 settings.memory.default_top_k

        参数：
        - user_id: str
          用户 ID。

        - question: str
          用户当前问题。

        - limit: int | None
          最终返回的记忆数量。
          如果为 None，则使用配置中的默认值。

        返回值：
        - str
          格式化后的用户记忆文本。
        """

        actual_limit = (
            limit
            or settings.memory.default_top_k
        )

        memories = self.search(
            user_id=user_id,
            question=question,
            top_k=actual_limit,
            candidate_k=max(
                actual_limit * 4,
                settings.memory.default_candidate_k
            )
        )

        return self.format_memories(
            memories
        )
