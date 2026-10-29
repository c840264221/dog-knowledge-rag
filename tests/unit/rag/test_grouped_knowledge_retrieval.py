"""统一知识检索的关键检查：目录约束、条件继承、并发上限和跨问题覆盖。"""

import asyncio
import pytest

from src.rag.schemas import RagQuery, RagChunk, RagContext, RagRetrievedChunk
from src.rag.pipelines.grouped_knowledge_retrieval import (
    InjectedKnowledgeCatalog, retrieve_groups, combine_group_contexts, rank_and_select_groups,
)


def make_hit(key: str, content: str | None = None) -> RagRetrievedChunk:
    """构建候选；参数为编号及可选正文；返回标准块，不写库。"""
    return RagRetrievedChunk(chunk=RagChunk(doc_id="doc", chunk_id=key, content=content or key), retrieval_score=.2)


class Retriever:
    """记录参数及同时调用数量的检索替身。"""
    allow_filter_relaxation = False

    def __init__(self):
        """初始化计数；无参数；无返回值。"""
        self.calls, self.active, self.maximum = [], 0, 0

    async def async_retrieve(self, query):
        """模拟短暂等待；参数为标准请求；返回按本次问题标识的候选。"""
        self.calls.append(query.model_dump())
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(.001)
            return RagContext(question=query.question, chunks=[make_hit(query.question)], status="success")
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_catalog_conditions_and_concurrency():
    """验证分组路由；无参数；返回None，身份及条件不串组且实际查询不超过2个。"""
    retriever = Retriever()
    provider = InjectedKnowledgeCatalog({"breeds": retriever, "care": retriever},
                                         {"breeds": "犬种", "care": "养护"})
    groups = [{"query_id": "q1", "query": RagQuery(question="金毛运动", user_id="trusted", top_k=5,
        filters={"dog_name": {"$eq": "Golden Retriever"}}).model_dump(),
        "english_question": "Golden exercise", "collection_ids": ["breeds"]},
        {"query_id": "q2", "query": RagQuery(question="接猫用品", user_id="trusted", top_k=5,
        filters={}).model_dump(), "english_question": "Cat supplies", "collection_ids": ["care"]}]
    results = await retrieve_groups(groups, provider=provider, bilingual=True, concurrency=2, max_groups=4)
    assert len(results) == 2 and len(retriever.calls) == 4 and retriever.maximum <= 2
    for call in retriever.calls:
        assert call["user_id"] == "trusted"
        assert bool(call["filters"]) == (call["question"] in {"金毛运动", "Golden exercise"})
    assert len(combine_group_contexts("完整问题", results).chunks) == 4
    groups[1]["collection_ids"] = ["unknown"]
    before = len(retriever.calls)
    with pytest.raises(ValueError):
        await retrieve_groups(groups, provider=provider, bilingual=True, concurrency=2, max_groups=4)
    assert len(retriever.calls) == before


def test_group_rerank_selects_evidence_for_both_questions():
    """验证覆盖而非全局分数排序；无参数；返回None，两组低高分都能分到资料。"""
    class Ranker:
        """不同问题的评分尺度不同，禁止跨组直接比较。"""
        def predict(self, pairs, **kwargs):
            """记录语义配对；参数为问题正文对；返回按组不同尺度分值。"""
            return [100 - i if q == "运动" else .1 - i for i, (q, _) in enumerate(pairs)]
    groups = [{"query_id": "q1", "query": {"question": "运动"},
        "candidates": [make_hit("exercise1").model_dump(), make_hit("exercise2").model_dump()]},
        {"query_id": "q2", "query": {"question": "护理"},
        "candidates": [make_hit("grooming").model_dump()]}]
    result = rank_and_select_groups("运动与护理", groups, reranker=Ranker(), max_chunks=2, max_characters=5000)
    assert [h["chunk"]["chunk_id"] for h in result["rag_context"]["chunks"]] == ["exercise1", "grooming"]
    assert all(c["status"] == "selected_evidence" for c in result["coverage"])
    assert len(result["rag_context"]["context_text"]) <= 5000


def test_shared_chunk_is_deduplicated_and_empty_group_is_reported():
    """验证来源映射；无参数；返回None，共用块只出现一次、空组不冒充覆盖。"""
    class Ranker:
        """固定评分替身。"""
        def predict(self, pairs, **kwargs):
            """参数为文本对；返回固定有限评分。"""
            return [1.] * len(pairs)
    shared = make_hit("shared").model_dump()
    groups = [{"query_id": f"q{i}", "query": {"question": f"问题{i}"}, "candidates": hits}
              for i, hits in enumerate(([shared], [shared], []), 1)]
    result = rank_and_select_groups("全部问题", groups, reranker=Ranker(), max_chunks=5, max_characters=5000)
    assert len(result["rag_context"]["chunks"]) == 1
    assert result["coverage"][0]["citation_ids"] == result["coverage"][1]["citation_ids"] == ["S1"]
    assert result["coverage"][2]["status"] == "no_candidates"
