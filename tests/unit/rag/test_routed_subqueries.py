"""严格条件与人工路由测试，不加载模型或连接数据库。"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.retrievers.multi_collection_retriever import MultiCollectionRetriever
from src.rag.schemas import RagContext, RagQuery


@pytest.mark.parametrize("is_async", [False, True])
def test_empty_strict_filter_never_relaxes(is_async):
    """验证同步与异步严格检索为空时只查一次；is_async选择接口；无返回值。"""
    retriever = MetadataFilterRetriever(object(), allow_filter_relaxation=False)
    condition = {"dog_name": {"$eq": "Missing"}}
    if is_async:
        retriever._async_search_documents = AsyncMock(return_value=[])
        result = asyncio.run(retriever._async_search_documents_with_fallback("问题", condition, 5))
        retriever._async_search_documents.assert_awaited_once_with(
            query_text="问题", metadata_filter=condition, top_k=5,
        )
    else:
        retriever._search_documents = Mock(return_value=[])
        result = retriever._search_documents_with_fallback("问题", condition, 5)
        retriever._search_documents.assert_called_once_with(
            query_text="问题", metadata_filter=condition, top_k=5,
        )
    assert result == ([], condition, "")


def test_legacy_default_still_relaxes():
    """验证旧调用默认保留放宽能力；无参数，无返回值。"""
    retriever = MetadataFilterRetriever(object())
    retriever._search_documents = Mock(side_effect=[[], [("document", 0.2)]])
    retriever._build_fallback_filter_candidates = Mock(return_value=[(None, "放宽")])
    assert retriever._search_documents_with_fallback("问题", {"dog_name": "Missing"}, 5)[2] == "放宽"
    assert retriever._search_documents.call_count == 2


def test_each_subquery_only_calls_its_own_collection():
    """验证条件只传犬种路，养护路无犬种条件，结果分组保留；无参数，无返回值。"""
    breed = Mock(allow_filter_relaxation=False)
    care = Mock(allow_filter_relaxation=False)
    breed.async_retrieve = AsyncMock(return_value=RagContext(question="金毛运动"))
    care.async_retrieve = AsyncMock(return_value=RagContext(question="新猫用品"))
    condition = {"dog_name": {"$eq": "Golden Retriever"}}
    result = asyncio.run(MultiCollectionRetriever({"breed": breed, "care": care}).async_retrieve_routed_subqueries({
        "q1": {"breed": RagQuery(question="金毛运动", filters=condition)},
        "q2": {"care": RagQuery(question="新猫用品")},
    }))
    assert list(result) == ["q1", "q2"]
    assert [item.collection_name for item in result["q1"].collections] == ["breed"]
    assert [item.collection_name for item in result["q2"].collections] == ["care"]
    assert breed.async_retrieve.await_args.args[0].filters == condition
    assert not care.async_retrieve.await_args.args[0].filters
    breed.async_retrieve.assert_awaited_once()
    care.async_retrieve.assert_awaited_once()


@pytest.mark.parametrize("unsafe", [True, False])
def test_all_routes_validated_before_calls(unsafe):
    """验证不安全条件或未知集合均在任何查询前拦截；unsafe选择异常类型；无返回值。"""
    retriever = Mock(allow_filter_relaxation=True)
    retriever.async_retrieve = AsyncMock()
    bad = {"a": RagQuery(question="问题", filters={"dog_name": "Golden Retriever"})} if unsafe else {
        "unknown": RagQuery(question="问题"),
    }
    with pytest.raises(ValueError):
        asyncio.run(MultiCollectionRetriever({"a": retriever}).async_retrieve_routed_subqueries({
            "q1": {"a": RagQuery(question="有效问题")}, "q2": bad,
        }))
    retriever.async_retrieve.assert_not_awaited()


def test_failure_is_not_empty_success():
    """验证数据库故障明确失败而不是当成无命中；无参数，无返回值。"""
    retriever = Mock(allow_filter_relaxation=False)
    retriever.async_retrieve = AsyncMock(side_effect=RuntimeError("故障"))
    with pytest.raises(RuntimeError, match="召回失败"):
        asyncio.run(MultiCollectionRetriever({"a": retriever}).async_retrieve_routed_subqueries({
            "q1": {"a": RagQuery(question="问题")},
        }))
