"""犬种主链路双语候选融合及条件保留检查。"""

from types import SimpleNamespace
import pytest

from src.rag.pipelines.query_variant_retrieval import retrieve_query_variants
from src.rag.schemas import RagQuery, RagContext, RagChunk, RagRetrievedChunk


class RecordingRetriever:
    """记录标准请求，按语言返回不同候选的确定性检索器。"""

    allow_filter_relaxation = False

    def __init__(self):
        """初始化调用列表；无参数；无返回值。"""
        self.calls = []

    async def async_retrieve(self, query):
        """记录请求；参数为RagQuery；返回单路候选，模拟英文补回新证据。"""
        self.calls.append(query.model_dump())
        key = "english" if query.question.startswith("Golden") else "chinese"
        hit = RagRetrievedChunk(chunk=RagChunk(chunk_id=key, doc_id="doc", content=key),
                                retrieval_score=0.2)
        return RagContext(question=query.question, chunks=[hit], status="success")


@pytest.mark.asyncio
async def test_same_filters_identity_and_full_pool():
    """检查双路继承；无参数；返回None，完整池含英文独有块且请求不变。"""
    retriever = RecordingRetriever()
    query = RagQuery(question="金毛运动", user_id="trusted", top_k=1,
                     filters={"dog_name": {"$eq": "Golden Retriever"}})
    context, audit = await retrieve_query_variants(query, "Golden Retriever exercise", retriever=retriever)
    assert len(context.chunks) == 2  # 不在重排前按单路top_k截断。
    assert audit["mode"] == "bilingual"
    assert all(q["filters"] == query.filters and q["user_id"] == "trusted" for q in retriever.calls)
    assert all(q["top_k"] == 1 for q in retriever.calls)
    assert query.question == "金毛运动"


@pytest.mark.asyncio
async def test_filter_relaxation_is_not_allowed():
    """检查过滤边界；无参数；返回None，条件可放宽的检索器不能执行双路。"""
    with pytest.raises(ValueError):
        await retrieve_query_variants(RagQuery(question="金毛", user_id="trusted", top_k=5,
            filters={"dog_name": {"$eq": "Golden Retriever"}}), "Golden Retriever",
            retriever=SimpleNamespace(allow_filter_relaxation=True))


@pytest.mark.asyncio
async def test_real_retrieve_node_consumes_translation(monkeypatch):
    """验证生产召回节点接线；参数为替换工具；返回None，翻译进入双路而非停留在状态中。"""
    import importlib
    module = importlib.import_module("src.graph.nodes.retrieve_node")
    noop = lambda **kwargs: None
    runtime = SimpleNamespace(trace_id="test", user_id="trusted", session_id="test", component="test",
        state=lambda: SimpleNamespace(set_node=lambda name: None),
        timeline=lambda: SimpleNamespace(add_event=noop),
        service=lambda cls: SimpleNamespace(set_docs=lambda docs: None))
    monkeypatch.setattr(module.runtime_ctx, "get", lambda: runtime)
    monkeypatch.setattr(module.settings.rag, "bilingual_enabled", True)
    monkeypatch.setattr(module, "build_retrieval_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(module, "build_retrieval_quality_log_summary", lambda **kwargs: {})
    retriever = RecordingRetriever()
    state = {"question": "原始问题", "retrieval_question": "金毛运动", "user_id": "trusted",
             "filters": {"dog_name": {"$eq": "Golden Retriever"}},
             "dog_query_understanding_result": {"done": True},
             "rag_query_translation": {"source_question": "金毛运动", "english_question": "Golden exercise"}}
    provider = SimpleNamespace(metadata_filter_retriever=retriever)
    result = await module.execute_new_rag_retrieve(state, provider)
    assert result["rag_bilingual_retrieval"]["mode"] == "bilingual"
    assert len(result["rag_context"]["chunks"]) == 2
    assert len(retriever.calls) == 2
    # 上游检索文本变化后旧翻译不可继续使用。
    state["retrieval_question"] = "柯基运动"
    result = await module.execute_new_rag_retrieve(state, provider)
    assert result["rag_bilingual_retrieval"]["reason"] == "translation_source_changed"
    assert len(retriever.calls) == 3
