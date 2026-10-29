"""双集合融合单元测试：使用确定性候选，不连接向量库或加载模型。"""

import asyncio

import pytest

from src.rag.retrievers.multi_collection_retriever import MultiCollectionRetriever, fuse_collection_results
from src.rag.schemas import RagChunk, RagContext, RagQuery, RagRetrievedChunk


def make_hit(chunk_id: str, *, doc_id: str = "document", content: str = "text", score: float = 0.1):
    """构造召回替身；参数为块编号、文档编号、正文与原始分值；返回标准召回块。"""
    return RagRetrievedChunk(
        chunk=RagChunk(chunk_id=chunk_id, doc_id=doc_id, content=content),
        retrieval_score=score,
    )


class FakeRetriever:
    """记录请求并返回固定候选的测试替身，不模拟向量计算。"""

    def __init__(self, hits: list[RagRetrievedChunk], error: Exception | None = None):
        """保存候选和可选故障；参数hits为候选，error为故障；无返回值。"""
        self.hits = hits
        self.error = error
        self.queries = []

    async def async_retrieve(self, query: RagQuery) -> RagContext:
        """记录query请求后返回上下文或抛出配置故障；返回标准上下文。"""
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return RagContext(question=query.question, chunks=self.hits)


def test_shared_chunk_accumulates_rank_contributions():
    """验证跨集合重复块累计排名贡献、保留来源；无参数，无返回值。"""
    shared = make_hit("shared")
    result = fuse_collection_results({"breed": [make_hit("first"), shared], "care": [shared]})
    assert result[0].chunk.chunk_id == "shared"
    assert result[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert [part.collection_name for part in result[0].contributions] == ["breed", "care"]
    assert [part.rank for part in result[0].contributions] == [2, 1]


def test_disjoint_results_are_all_retained_with_stable_ties():
    """验证不同集合各自候选均保留，同分使用稳定顺序；无参数，无返回值。"""
    result = fuse_collection_results({
        "breed": [make_hit("b1"), make_hit("b2")],
        "care": [make_hit("c1"), make_hit("c2")],
    })
    assert [item.chunk.chunk_id for item in result] == ["b1", "c1", "b2", "c2"]
    assert result[0].rrf_score == result[1].rrf_score


def test_distance_values_are_not_compared_between_collections():
    """验证融合只使用排名，不拿不同集合的距离作全局排序；无参数，无返回值。"""
    result = fuse_collection_results({"a": [make_hit("a", score=999)], "b": [make_hit("b", score=0.001)]})
    assert result[0].rrf_score == result[1].rrf_score
    assert result[0].contributions[0].retrieval_score == 999


def test_duplicate_inside_one_collection_does_not_double_count():
    """验证同一路重复编号不重复加分；无参数，无返回值。"""
    hit = make_hit("same")
    result = fuse_collection_results({"a": [hit, hit]})
    assert len(result) == 1
    assert result[0].rrf_score == pytest.approx(1 / 61)
    assert len(result[0].contributions) == 1


def test_same_chunk_id_from_different_documents_is_not_merged():
    """验证去重键包含文档编号，防止跨文档误合并；无参数，无返回值。"""
    result = fuse_collection_results({
        "a": [make_hit("same", doc_id="doc_a")], "b": [make_hit("same", doc_id="doc_b")],
    })
    assert len(result) == 2


def test_conflicting_content_is_rejected():
    """验证相同身份却正文冲突时拒绝合并；无参数，无返回值。"""
    with pytest.raises(ValueError, match="不同正文"):
        fuse_collection_results({"a": [make_hit("same", content="old")], "b": [make_hit("same", content="new")]})


def test_fusion_does_not_mutate_original_hits():
    """验证融合候选与原始对象隔离，原始分数不被覆盖；无参数，无返回值。"""
    hit = make_hit("same")
    original = hit.model_dump()
    result = fuse_collection_results({"a": [hit]})
    result[0].chunk.metadata["added"] = True
    assert hit.model_dump() == original


@pytest.mark.parametrize("rrf_k", [0, -1, True])
def test_invalid_rrf_constant(rrf_k):
    """验证平滑常数不能为零、负数或布尔值；rrf_k为测试输入；无返回值。"""
    with pytest.raises(ValueError):
        fuse_collection_results({}, rrf_k=rrf_k)


def test_multi_retrieval_queries_both_collections():
    """验证各路收到相同问题及每路数量，并返回各自候选；无参数，无返回值。"""
    breed = FakeRetriever([make_hit("breed")])
    care = FakeRetriever([make_hit("care")])
    result = asyncio.run(MultiCollectionRetriever({"breed": breed, "care": care}).async_retrieve(
        RagQuery(question="pet question", top_k=3),
    ))
    assert [item.collection_name for item in result.collections] == ["breed", "care"]
    assert len(result.fused_candidates) == 2
    assert breed.queries[0].question == care.queries[0].question == "pet question"
    assert breed.queries[0].top_k == care.queries[0].top_k == 3


def test_filters_are_rejected_before_search():
    """验证未落实硬过滤之前拒绝过滤请求，不能悄悄放宽；无参数，无返回值。"""
    retriever = FakeRetriever([])
    with pytest.raises(ValueError, match="过滤"):
        asyncio.run(MultiCollectionRetriever({"a": retriever}).async_retrieve(
            RagQuery(question="question", filters={"species": "cat"}),
        ))
    assert not retriever.queries


def test_one_collection_failure_is_not_hidden():
    """验证单路故障不能当作完整成功，且各路查询已结束；无参数，无返回值。"""
    good = FakeRetriever([make_hit("good")])
    bad = FakeRetriever([], error=RuntimeError("unavailable"))
    with pytest.raises(RuntimeError, match="集合bad召回失败"):
        asyncio.run(MultiCollectionRetriever({"good": good, "bad": bad}).async_retrieve(
            RagQuery(question="question"),
        ))
    assert len(good.queries) == len(bad.queries) == 1


def test_empty_collection_result_is_not_a_service_failure():
    """验证查询成功但无命中可与其他路一起返回，不冒充服务故障；无参数，无返回值。"""
    result = asyncio.run(MultiCollectionRetriever({
        "a": FakeRetriever([]), "b": FakeRetriever([make_hit("b")]),
    }).async_retrieve(RagQuery(question="question")))
    assert result.collections[0].chunks == []
    assert len(result.fused_candidates) == 1


def test_subqueries_keep_separate_candidate_groups():
    """验证子问题各自查询和融合，相同候选不跨问题加分；无参数，无返回值。"""
    breed = FakeRetriever([make_hit("shared")])
    care = FakeRetriever([make_hit("care")])
    result = asyncio.run(MultiCollectionRetriever({"breed": breed, "care": care}).async_retrieve_subqueries({
        "q1": RagQuery(question="dog exercise", top_k=5),
        "q2": RagQuery(question="cat supplies", top_k=5),
    }))
    assert list(result) == ["q1", "q2"]
    assert result["q1"].question == "dog exercise"
    assert result["q2"].question == "cat supplies"
    assert [query.question for query in breed.queries] == ["dog exercise", "cat supplies"]
    assert [query.question for query in care.queries] == ["dog exercise", "cat supplies"]
    assert result["q1"].fused_candidates[0].rrf_score == pytest.approx(1 / 61)
    assert result["q2"].fused_candidates[0].rrf_score == pytest.approx(1 / 61)
    assert result["q1"].fused_candidates[0] is not result["q2"].fused_candidates[0]


def test_invalid_subquery_rejected_before_any_search():
    """验证先检查全部子问题，后续输入非法时不先执行前一个；无参数，无返回值。"""
    retriever = FakeRetriever([])
    with pytest.raises(ValueError):
        asyncio.run(MultiCollectionRetriever({"a": retriever}).async_retrieve_subqueries({
            "q1": RagQuery(question="valid"), "q2": RagQuery(question=" "),
        }))
    assert not retriever.queries
