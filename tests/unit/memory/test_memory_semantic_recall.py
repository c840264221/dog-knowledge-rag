from langchain_core.documents import Document

from src.memory.memory_ranker import MemoryRanker
from src.memory.memory_semantic_recall import (
    MemorySemanticRecallService,
)
from src.memory.sqlite_memory_store import (
    SQLiteMemoryStore,
)


class FakeMemoryDb:
    """提供固定 Chroma 候选结果的测试向量库。"""

    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def similarity_search_with_score(
            self,
            query: str,
            k: int,
            filter: dict,
    ) -> list[tuple[Document, float]]:
        """
        返回预设的向量检索结果。

        功能：记录查询参数并模拟 Chroma 的带距离检索。
        参数：query 是查询文本，k 是候选数量，filter 是 metadata 过滤条件。
        返回值：Document 与 distance 距离组成的列表。
        """

        self.calls.append(
            {
                "query": query,
                "k": k,
                "filter": filter,
            }
        )
        return self.results


class FakeVectorStoreProvider:
    """向召回服务提供测试 memory_db 的向量库 Provider。"""

    def __init__(self, memory_db: FakeMemoryDb) -> None:
        self.memory_db = memory_db


class FakeMemoryStore:
    """根据 memory_id 返回固定记录的测试 SQLite Store。"""

    def __init__(self, memories: list[dict]) -> None:
        self.memories = memories
        self.calls: list[dict] = []

    def get_memories_by_ids(
            self,
            memory_ids: list[int],
            only_active: bool = True,
    ) -> list[dict]:
        """
        模拟根据编号批量回查记忆。

        功能：记录回查参数，并只返回编号匹配的预设记录。
        参数：memory_ids 是记忆编号列表，only_active 表示是否只取有效记忆。
        返回值：匹配到的记忆字典列表。
        """

        self.calls.append(
            {
                "memory_ids": memory_ids,
                "only_active": only_active,
            }
        )
        wanted_ids = set(memory_ids)
        return [
            memory
            for memory in self.memories
            if int(memory["id"]) in wanted_ids
        ]


def build_document(
        memory_id: int,
        content: str,
) -> Document:
    """
    创建测试用 Chroma 记忆文档。

    功能：把记忆编号和正文包装成 Document。
    参数：memory_id 是 SQLite 记忆编号，content 是向量化记忆文本。
    返回值：包含 memory_id metadata 的 Document。
    """

    return Document(
        page_content=content,
        metadata={
            "memory_id": str(memory_id),
            "user_id": "user_001",
            "status": "active",
        },
    )


def build_memory(
        memory_id: int,
        content: str,
        importance: float = 0.5,
        strength: float = 1.0,
) -> dict:
    """
    创建测试用 SQLite 记忆记录。

    功能：生成 MemoryRanker 可以直接评分的完整记忆字典。
    参数：memory_id 是编号，content 是内容，importance 是重要程度，strength 是强度。
    返回值：模拟 SQLite 查询结果的字典。
    """

    return {
        "id": memory_id,
        "user_id": "user_001",
        "memory_type": "preference",
        "content": content,
        "confidence": 0.9,
        "strength": strength,
        "status": "active",
        "created_at": "2026-07-01T00:00:00",
        "last_seen": "2026-07-01T00:00:00",
        "source": "conversation",
        "importance": importance,
        "updated_at": "2026-07-01T00:00:00",
        "expires_at": None,
    }


def test_search_should_reject_low_semantic_score_before_sqlite_lookup() -> None:
    """测试低相关候选在 SQLite 回查和高强度排序之前被拒绝。"""

    memory_db = FakeMemoryDb(
        [(build_document(1, "用户喜欢金毛"), 1.5)]
    )
    store = FakeMemoryStore(
        [build_memory(1, "用户喜欢金毛", strength=100.0)]
    )
    service = MemorySemanticRecallService(
        store=store,
        vectorstore_provider=FakeVectorStoreProvider(memory_db),
        minimum_semantic_score=0.45,
    )

    result = service.search(
        user_id="user_001",
        question="今天成都天气怎么样",
    )

    assert result == []
    assert store.calls == []


def test_search_should_return_candidate_above_semantic_threshold() -> None:
    """测试达到语义门槛的候选可以回查、精排并返回。"""

    memory_db = FakeMemoryDb(
        [(build_document(1, "用户喜欢金毛"), 0.5)]
    )
    store = FakeMemoryStore(
        [build_memory(1, "用户喜欢金毛")]
    )
    service = MemorySemanticRecallService(
        store=store,
        vectorstore_provider=FakeVectorStoreProvider(memory_db),
        minimum_semantic_score=0.45,
    )

    result = service.search(
        user_id="user_001",
        question="我喜欢什么狗",
    )

    assert [memory["id"] for memory in result] == [1]
    assert result[0]["semantic_score"] > 0.45
    assert store.calls[0]["memory_ids"] == [1]


def test_retrieve_with_details_should_return_observability_contract() -> None:
    """
    测试结构化召回入口返回真实诊断数据。

    功能：
        使用一条通过门槛和一条未通过门槛的 Chroma 候选，
        验证候选数、通过数、采用数和记忆 ID 都来自真实召回过程。

    参数：
        无。

    返回值：
        None。
    """

    memory_db = FakeMemoryDb(
        [
            (build_document(1, "用户喜欢金毛"), 0.5),
            (build_document(2, "用户喜欢边牧"), 2.0),
        ]
    )
    service = MemorySemanticRecallService(
        store=FakeMemoryStore(
            [
                build_memory(1, "用户喜欢金毛"),
                build_memory(2, "用户喜欢边牧"),
            ]
        ),
        vectorstore_provider=FakeVectorStoreProvider(memory_db),
        minimum_semantic_score=0.45,
    )

    result = service.retrieve_with_details(
        user_id="user_001",
        question="我喜欢什么狗？",
        limit=5,
    )

    assert result["status"] == "applied"
    assert result["candidate_count"] == 2
    assert result["threshold_passed_count"] == 1
    assert result["selected_count"] == 1
    assert result["semantic_threshold"] == 0.45
    assert result["max_semantic_score"] > 0.45
    assert result["selected_memory_ids"] == [1]
    assert "金毛" in result["memory_context"]


def test_ranker_should_use_importance_after_semantic_gate() -> None:
    """测试相关度和其他分数相同时，高重要程度记忆排在前面。"""

    ranker = MemoryRanker(
        importance_weight=1.0
    )
    memories = [
        build_memory(1, "普通偏好", importance=0.2),
        build_memory(2, "重要偏好", importance=0.9),
    ]

    result = ranker.rank(
        memories=memories,
        semantic_score_map={1: 0.8, 2: 0.8},
        distance_map={1: 0.25, 2: 0.25},
        top_k=2,
    )

    assert [memory["id"] for memory in result] == [2, 1]
    assert result[0]["importance_score"] == 0.9


def test_ranker_should_preserve_zero_importance() -> None:
    """测试 importance=0.0 时不会被错误替换为默认值 0.5。"""

    ranker = MemoryRanker()

    result = ranker.score_memory(
        memory=build_memory(1, "低重要程度记忆", importance=0.0),
        semantic_score=0.8,
        distance=0.25,
    )

    assert result["importance_score"] == 0.0


def test_retrieve_should_return_default_when_all_candidates_rejected() -> None:
    """测试所有候选都未通过门槛时返回无记忆提示。"""

    memory_db = FakeMemoryDb(
        [(build_document(1, "用户喜欢金毛"), 2.0)]
    )
    service = MemorySemanticRecallService(
        store=FakeMemoryStore([build_memory(1, "用户喜欢金毛")]),
        vectorstore_provider=FakeVectorStoreProvider(memory_db),
        minimum_semantic_score=0.45,
    )

    result = service.retrieve(
        user_id="user_001",
        question="查询数据库有哪些表",
    )

    assert result == "暂无用户记忆"


def test_sqlite_lookup_should_filter_expired_memory(tmp_path) -> None:
    """测试 SQLite 批量回查只返回未过期的有效记忆。"""

    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory_expiration.sqlite3"
    )

    try:
        expired_id = store.add_memory(
            user_id="user_001",
            memory_type="preference",
            content="已过期偏好",
            confidence=0.9,
            expires_at="2020-01-01T00:00:00",
        )
        active_id = store.add_memory(
            user_id="user_001",
            memory_type="preference",
            content="仍有效偏好",
            confidence=0.9,
            expires_at="2999-01-01T00:00:00",
        )

        assert expired_id is not None
        assert active_id is not None

        result = store.get_memories_by_ids(
            [expired_id, active_id],
            only_active=True,
        )

        assert [memory["id"] for memory in result] == [active_id]
    finally:
        store.close()
