from src.memory.memory_scoring import (
    MemoryScorer
)


class MemoryRanker:
    """
    MemoryRanker：记忆精排器。

    功能：
    - 对 Chroma 初步召回的 Memory 进行二次排序
    - 综合 semantic_score、memory_score、confidence 等因素
    - 输出 final_score，供最终 TopK 排序使用

    中文释义：
    - Ranker：排序器 / 精排器
    - semantic_score：语义相似度分数
    - memory_score：记忆强度与时间衰减分数
    - confidence：记忆可信度
    - final_score：最终排序分数
    """

    def __init__(
            self,
            memory_scorer: MemoryScorer | None = None,
            semantic_weight: float = 3.0,
            memory_weight: float = 1.0,
            confidence_weight: float = 0.5,
            importance_weight: float = 0.5,
    ):
        """
        初始化 MemoryRanker。

        功能：
        - 初始化记忆评分器 MemoryScorer
        - 设置语义分数、记忆分数、可信度分数的权重
        - 后续可通过配置调整不同评分维度的重要性

        参数：
        - memory_scorer: MemoryScorer | None
          记忆评分器。
          如果外部没有传入，则默认创建 MemoryScorer。

        - semantic_weight: float
          语义分数权重。
          值越大，说明语义相关性在最终排序中越重要。

        - memory_weight: float
          记忆分数权重。
          值越大，说明 strength 和 last_seen 对排序影响越大。

        - confidence_weight: float
          可信度权重。
          值越大，说明 LLM 保存记忆时的 confidence 越重要。

        - importance_weight: float
          记忆重要程度权重。
          值越大，说明通过相关性门槛后，高重要程度记忆的排序越靠前。

        返回值：
        - None
          初始化函数不返回业务数据。
        """

        self.memory_scorer = (
            memory_scorer
            or MemoryScorer()
        )

        self.semantic_weight = semantic_weight

        self.memory_weight = memory_weight

        self.confidence_weight = confidence_weight

        self.importance_weight = importance_weight

    def score_memory(
            self,
            memory: dict,
            semantic_score: float,
            distance: float
    ) -> dict:
        """
        对单条 Memory 进行评分。

        功能：
        - 接收 SQLite 中的一条 memory 数据
        - 接收 Chroma 计算出的 semantic_score 和 distance
        - 使用 MemoryScorer 计算 memory_score
        - 综合多个分数计算 final_score
        - 返回带评分字段的新 memory dict

        参数：
        - memory: dict
          SQLite 查询出来的一条记忆数据。

        - semantic_score: float
          语义相似度分数。
          数值越大，说明和用户问题越相关。

        - distance: float
          Chroma 返回的距离。
          数值越小，说明语义越相似。

        返回值：
        - dict
          返回增强后的记忆数据。
          原始字段会保留，并额外增加：
          distance、semantic_score、memory_score、confidence_score、final_score。
        """

        strength = float(
            memory.get(
                "strength",
                1.0
            )
        )

        last_seen = memory.get(
            "last_seen"
        )

        confidence_score = float(
            memory.get(
                "confidence",
                0.0
            )
        )

        importance_score = float(
            0.5
            if memory.get("importance") is None
            else memory["importance"]
        )

        memory_score = self.memory_scorer.score(
            strength=strength,
            last_seen=last_seen
        )

        final_score = (
            semantic_score * self.semantic_weight
            + memory_score * self.memory_weight
            + confidence_score * self.confidence_weight
            + importance_score * self.importance_weight
        )

        return {
            **memory,
            "distance": distance,
            "semantic_score": semantic_score,
            "memory_score": memory_score,
            "confidence_score": confidence_score,
            "importance_score": importance_score,
            "final_score": final_score,
        }

    def rank(
            self,
            memories: list[dict],
            semantic_score_map: dict[int, float],
            distance_map: dict[int, float],
            top_k: int
    ) -> list[dict]:
        """
        对多条 Memory 进行精排。

        功能：
        - 遍历 SQLite 回查得到的 memories
        - 为每条 memory 计算 final_score
        - 按 final_score 从高到低排序
        - 返回最终 TopK 记忆

        参数：
        - memories: list[dict]
          SQLite 查询得到的记忆列表。

        - semantic_score_map: dict[int, float]
          memory_id 到 semantic_score 的映射。
          来自 Chroma 语义召回结果。

        - distance_map: dict[int, float]
          memory_id 到 distance 的映射。
          来自 Chroma 语义召回结果。

        - top_k: int
          最终返回的记忆数量。

        返回值：
        - list[dict]
          精排后的记忆列表。
          每条记忆都包含 final_score 等评分字段。
        """

        scored_memories = []

        for memory in memories:

            memory_id = int(
                memory["id"]
            )

            semantic_score = semantic_score_map.get(
                memory_id,
                0.0
            )

            distance = distance_map.get(
                memory_id,
                999999.0
            )

            scored_memory = self.score_memory(
                memory=memory,
                semantic_score=semantic_score,
                distance=distance
            )

            scored_memories.append(
                scored_memory
            )

        scored_memories.sort(
            key=lambda memory: memory["final_score"],
            reverse=True
        )

        return scored_memories[:top_k]
