"""自动翻译接入的关键边界检查，不替代真实检索评估。"""

import pytest

from src.rag.schemas import RagQuery
from src.rag.pipelines import bilingual_retrieval_pipeline as pipeline
from src.rag.query_builders.english_query_rewriter import translate_retrieval_question


@pytest.mark.asyncio
async def test_translated_query_is_passed_without_mutating_request(monkeypatch):
    """检查翻译接线；参数为替换工具；返回None，断言原请求不变且使用真实翻译返回值。"""
    query = RagQuery(question="未打完疫苗的3个月金毛注意事项", user_id="trusted", top_k=5, filters={})
    before = query.model_dump()

    async def translate(llm, question):
        """模拟翻译；参数为模型及问题；返回含英文的审计记录。"""
        assert llm == "model"
        assert question == query.question
        return {"english_question": "Precautions for a 3 month old Golden Retriever not fully vaccinated"}

    async def retrieve(actual_query, english_question, **kwargs):
        """模拟已存在检索管线；参数为请求、英文及服务；返回可追踪结果。"""
        assert actual_query.user_id == "trusted"
        assert "not fully vaccinated" in english_question
        return {"channels": {}}

    monkeypatch.setattr(pipeline, "translate_retrieval_question", translate)
    monkeypatch.setattr(pipeline, "retrieve_bilingual", retrieve)
    result = await pipeline.retrieve_auto_bilingual(query, llm="model", retriever=None, reranker=None)
    assert query.model_dump() == before
    assert result["translation_seconds"] >= 0


@pytest.mark.asyncio
async def test_translation_failure_does_not_claim_bilingual_success(monkeypatch):
    """检查失败传播；参数为替换工具；返回None，失败不得静默生成英文或执行双语检索。"""
    async def fail(llm, question):
        """模拟接口失败；参数为模型与问题；无返回值，抛超时异常。"""
        raise TimeoutError("translation unavailable")

    monkeypatch.setattr(pipeline, "translate_retrieval_question", fail)
    with pytest.raises(TimeoutError):
        await pipeline.retrieve_auto_bilingual(
            RagQuery(question="金毛运动", user_id="trusted", top_k=5, filters={}),
            llm=None, retriever=None, reranker=None)


@pytest.mark.asyncio
async def test_empty_question_rejected_before_model_call():
    """检查输入；无参数；返回None，空问题不发起模型请求。"""
    with pytest.raises(ValueError):
        await translate_retrieval_question(None, "  ")
