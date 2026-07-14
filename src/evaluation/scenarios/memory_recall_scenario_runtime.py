from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from src.evaluation.schemas import AgentEvaluationCase
from src.graph.nodes.memory_retrieve_node import build_memory_retrieve_node
from src.memory.memory_semantic_recall import MemorySemanticRecallService
from src.runtime.context import RuntimeContext, runtime_ctx


class EvaluationMemoryVectorStore:
    """
    Memory 评估专用确定性向量数据库。

    功能：
        根据真实 MemorySemanticRecallService 传入的 user_id 和 status filter
        过滤预设记忆，再返回固定 Chroma distance（向量距离）。

    参数含义：
        memories:
            当前用例配置的记忆记录列表。
        should_fail:
            是否模拟向量检索异常。

    返回值含义：
        EvaluationMemoryVectorStore:
            支持 similarity_search_with_score 并记录调用轨迹的对象。
    """

    def __init__(
        self,
        memories: list[dict[str, Any]],
        should_fail: bool = False,
    ) -> None:
        self.memories = [dict(memory) for memory in memories]
        self.should_fail = should_fail
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        按用户与有效状态返回确定性向量候选。

        参数含义：
            query:
                当前用户问题。
            k:
                最多返回的候选数量。
            filter:
                MemorySemanticRecallService 构造的 Chroma 过滤条件。

        返回值含义：
            list[tuple[Document, float]]:
                文档与向量距离组成的候选列表。
        """

        normalized_filter = dict(filter or {})
        self.calls.append(
            {
                "query": query,
                "k": k,
                "filter": normalized_filter,
            }
        )
        if self.should_fail:
            raise RuntimeError("evaluation memory vector search failed")

        expected_user_id = _extract_filter_eq(
            normalized_filter,
            "user_id",
        )
        expected_status = _extract_filter_eq(
            normalized_filter,
            "status",
        )
        results: list[tuple[Document, float]] = []
        for memory in self.memories:
            if expected_user_id and memory.get("user_id") != expected_user_id:
                continue
            if expected_status and memory.get("status") != expected_status:
                continue
            results.append(
                (
                    Document(
                        page_content=str(memory.get("content", "")),
                        metadata={
                            "memory_id": str(memory.get("id", "")),
                            "user_id": str(memory.get("user_id", "")),
                            "status": str(memory.get("status", "")),
                            "memory_type": str(memory.get("memory_type", "")),
                        },
                    ),
                    float(memory.get("distance", 0.0)),
                )
            )
        return results[:k]


class EvaluationMemoryVectorStoreProvider:
    """
    向真实记忆召回服务提供确定性 memory_db。

    参数含义：
        memory_db:
            评估专用向量数据库。

    返回值含义：
        EvaluationMemoryVectorStoreProvider:
            具有 memory_db 属性的服务提供者。
    """

    def __init__(self, memory_db: EvaluationMemoryVectorStore) -> None:
        self.memory_db = memory_db


class EvaluationMemoryStore:
    """
    Memory 评估专用 SQLite Store（关系数据库存储替身）。

    功能：
        按 memory_id 回查预设记录，并执行 only_active（仅有效记忆）过滤。

    参数含义：
        memories:
            当前用例配置的完整记忆记录。

    返回值含义：
        EvaluationMemoryStore:
            支持 get_memories_by_ids 并记录调用轨迹的对象。
    """

    def __init__(self, memories: list[dict[str, Any]]) -> None:
        self.memories = {
            int(memory["id"]): dict(memory)
            for memory in memories
        }
        self.calls: list[dict[str, Any]] = []

    def get_memories_by_ids(
        self,
        memory_ids: list[int],
        only_active: bool = True,
    ) -> list[dict[str, Any]]:
        """
        根据 ID 列表返回当前评估场景的记忆记录。

        参数含义：
            memory_ids:
                从向量 metadata（元数据）中提取的 SQLite 记忆 ID。
            only_active:
                是否只返回 status=active 的有效记忆。

        返回值含义：
            list[dict[str, Any]]:
                符合 ID 和有效状态要求的记忆列表。
        """

        self.calls.append(
            {
                "memory_ids": list(memory_ids),
                "only_active": only_active,
            }
        )
        results: list[dict[str, Any]] = []
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory is None:
                continue
            if only_active and memory.get("status") != "active":
                continue
            results.append(dict(memory))
        return results


class EvaluationMemoryRanker:
    """
    Memory 评估专用确定性精排器。

    功能：
        保留真实语义分数并按分数降序排列，不引入时间衰减等环境时间变量。

    参数含义：
        无。

    返回值含义：
        EvaluationMemoryRanker:
            支持 rank 方法并记录调用轨迹的对象。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rank(
        self,
        memories: list[dict[str, Any]],
        semantic_score_map: dict[int, float],
        distance_map: dict[int, float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        为 SQLite 记忆补充语义分数并稳定排序。

        参数含义：
            memories:
                SQLite Store 回查出的有效记忆。
            semantic_score_map:
                memory_id 到语义相关分的映射。
            distance_map:
                memory_id 到原始向量距离的映射。
            top_k:
                最多保留的精排结果数量。

        返回值含义：
            list[dict[str, Any]]:
                包含 semantic_score、distance 和 final_score 的记忆列表。
        """

        self.calls.append(
            {
                "memory_count": len(memories),
                "top_k": top_k,
            }
        )
        ranked: list[dict[str, Any]] = []
        for memory in memories:
            memory_id = int(memory["id"])
            semantic_score = float(semantic_score_map.get(memory_id, 0.0))
            ranked.append(
                {
                    **memory,
                    "semantic_score": semantic_score,
                    "distance": float(distance_map.get(memory_id, 0.0)),
                    "memory_score": semantic_score,
                    "final_score": semantic_score,
                }
            )
        ranked.sort(
            key=lambda item: float(item.get("final_score", 0.0)),
            reverse=True,
        )
        return ranked[:top_k]


@dataclass
class MemoryRecallScenarioRuntime:
    """
    单条 Memory Recall（记忆召回）评估用例的运行环境。

    参数含义：
        node:
            注入真实 MemorySemanticRecallService 后构建的真实召回节点。
        initial_state:
            清除 evaluation_* 配置后的节点输入状态。
        vector_store、store、ranker:
            确定性外部依赖及调用轨迹。
        runtime_context:
            当前用例独享的真实运行时上下文。

    返回值含义：
        MemoryRecallScenarioRuntime:
            可以通过 invoke 执行真实记忆节点的场景对象。
    """

    node: Any
    initial_state: dict[str, Any]
    vector_store: EvaluationMemoryVectorStore
    store: EvaluationMemoryStore
    ranker: EvaluationMemoryRanker
    runtime_context: RuntimeContext = field(default_factory=RuntimeContext)

    async def invoke(self) -> dict[str, Any]:
        """
        在隔离运行时上下文中执行真实记忆召回节点。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                合并初始状态与节点更新后的最终状态。
        """

        previous_context = runtime_ctx.get()
        runtime_ctx.set(self.runtime_context)
        try:
            update = await self.node(dict(self.initial_state))
            return {
                **self.initial_state,
                **update,
            }
        finally:
            runtime_ctx.set(previous_context)


def build_memory_recall_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> MemoryRecallScenarioRuntime:
    """
    根据黄金用例构建真实 Memory 语义召回评估场景。

    功能：
        从 input_state 读取预设记忆、语义门槛和异常开关，构建真实
        MemorySemanticRecallService 与真实 memory_retrieve_node。

    参数含义：
        eval_case:
            当前统一 Agent 评估用例。

    返回值含义：
        MemoryRecallScenarioRuntime:
            包含真实业务逻辑和确定性存储依赖的评估运行环境。
    """

    raw_state = dict(eval_case.input_state)
    raw_memories = raw_state.pop("evaluation_memories", [])
    should_fail = bool(raw_state.pop("evaluation_vector_error", False))
    semantic_threshold = float(
        raw_state.pop("evaluation_semantic_threshold", 0.45)
    )
    if not isinstance(raw_memories, list):
        raise ValueError("evaluation_memories 必须是 list")
    memories = [
        dict(memory)
        for memory in raw_memories
        if isinstance(memory, dict)
    ]

    vector_store = EvaluationMemoryVectorStore(memories, should_fail)
    store = EvaluationMemoryStore(memories)
    ranker = EvaluationMemoryRanker()
    semantic_recall = MemorySemanticRecallService(
        store=store,
        vectorstore_provider=EvaluationMemoryVectorStoreProvider(vector_store),
        memory_ranker=ranker,
        minimum_semantic_score=semantic_threshold,
    )
    node = build_memory_retrieve_node(
        semantic_recall=semantic_recall,
        checkpoint_manager=None,
    )
    initial_state = {
        **raw_state,
        "question": eval_case.question,
        "user_id": str(raw_state.get("user_id", "evaluation_user")),
    }

    return MemoryRecallScenarioRuntime(
        node=node,
        initial_state=initial_state,
        vector_store=vector_store,
        store=store,
        ranker=ranker,
        runtime_context=RuntimeContext(
            trace_id=f"evaluation-{eval_case.case_id}",
            user_id=initial_state["user_id"],
            component="memory_recall_evaluation",
        ),
    )


def _extract_filter_eq(filters: dict[str, Any], field_name: str) -> Any:
    """
    从 Chroma $and / $eq 过滤条件中读取指定字段值。

    参数含义：
        filters:
            MemorySemanticRecallService 构造的 Chroma 过滤条件。
        field_name:
            需要读取的业务字段名称，例如 user_id 或 status。

    返回值含义：
        Any:
            找到的 $eq 比较值；没有找到时返回 None。
    """

    direct_condition = filters.get(field_name)
    if isinstance(direct_condition, dict) and "$eq" in direct_condition:
        return direct_condition["$eq"]

    conditions = filters.get("$and", [])
    if isinstance(conditions, list):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            value = _extract_filter_eq(condition, field_name)
            if value is not None:
                return value
    return None
