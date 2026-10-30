"""DogKnowledgeAgent 查询理解节点单元测试。"""

from __future__ import annotations

import json

import pytest

from src.agents.dog_knowledge_agent.nodes.query_understanding_node import (
    build_dog_query_understanding_node,
)
from src.rag.schemas import RagQuery


class FakeMessage:
    """保存测试 LLM 返回文本。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLMProvider:
    """返回固定查询理解 JSON 并记录调用的测试模型提供者。"""

    main_llm = object()

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def safe_ainvoke(self, **kwargs):
        """记录调用并返回固定 JSON 消息。"""

        self.calls.append(dict(kwargs))
        return FakeMessage(json.dumps(self.payload, ensure_ascii=False))


class FakeQueryFilterParser:
    """返回固定规则 RagQuery 的测试解析器。"""

    def parse(self, question, user_id, top_k, intent):
        """返回包含规则犬种条件的 RagQuery。"""

        return RagQuery(
            question=question,
            user_id=user_id,
            top_k=top_k,
            intent=intent,
            filters={"dog_name": {"$eq": "Golden Retriever"}},
        )


@pytest.mark.asyncio
async def test_rewrite_and_translation_share_one_model_call() -> None:
    """验证同次改写输出英文；无参数；返回None，断言身份、过滤及原始问题保留。"""
    provider = FakeLLMProvider({
        "retrieval_question": "3个月金毛未打完疫苗的外出注意事项",
        "english_retrieval_question": "Outdoor precautions for a 3 month old Golden Retriever not fully vaccinated",
    })
    node = build_dog_query_understanding_node(llm_provider=provider,
                                             query_filter_parser=FakeQueryFilterParser())
    state = {"question": "3个月金毛还没打完疫苗，出门注意什么？", "user_id": "trusted"}
    update = await node(state)
    assert len(provider.calls) == 1
    assert update["rag_query_translation"]["source_question"] == update["retrieval_question"]
    assert "not fully vaccinated" in update["rag_query_translation"]["english_question"]
    assert update["rag_query"]["user_id"] == "trusted"
    assert update["filters"] == {"dog_name": {"$eq": "Golden Retriever"}}
    assert "question" not in update


@pytest.mark.asyncio
async def test_collection_routing_and_per_question_subjects() -> None:
    """检查集合选择与主体隔离；无参数；返回None，猫的问题不能继承整体金毛条件。"""
    class Parser:
        """按当前子问题主体返回条件。"""
        def parse(self, question, user_id, top_k, intent):
            """参数为查询文本、身份、数量及意图；返回本题犬种条件。"""
            return RagQuery(question=question, user_id=user_id, top_k=top_k, intent=intent,
                filters={"dog_name": {"$eq": "Golden Retriever"}} if "金毛" in question else {})
    payload = {"retrieval_question": "金毛运动与接猫用品", "sub_queries": [
        {"question": "金毛运动需求", "english_question": "Golden exercise", "collection_ids": ["dog_breeds"]},
        {"question": "接猫用品", "english_question": "Cat supplies", "collection_ids": ["pet_care"]}]}
    provider = FakeLLMProvider(payload)
    node = build_dog_query_understanding_node(llm_provider=provider, query_filter_parser=Parser(),
        collection_catalog={"dog_breeds": "犬种", "pet_care": "养护"})
    result = await node({"question": "金毛运动与接猫用品", "user_id": "trusted"})
    groups = result["rag_sub_queries"]
    assert len(provider.calls) == 1
    assert groups[0]["query"]["filters"] == {"dog_name": {"$eq": "Golden Retriever"}}
    assert groups[1]["query"]["filters"] == {}
    assert groups[1]["query"]["user_id"] == "trusted"
    payload["sub_queries"][1]["collection_ids"] = ["invented"]
    with pytest.raises(ValueError):
        await node({"question": "金毛运动与接猫用品", "user_id": "trusted"})


@pytest.mark.asyncio
async def test_query_understanding_should_call_llm_and_keep_rule_priority() -> None:
    """验证始终调用 LLM，且冲突的犬种条件由确定性规则覆盖。"""

    llm_provider = FakeLLMProvider(
        {
            "rag_filters": {
                "dog_name": "Labrador Retriever",
                "trainability_level": {"$gte": 4},
            },
            "pet_profile_suggested_attributes": [
                "age_years",
                "breed",
                "unknown_field",
            ],
            "reason": "训练计划需要年龄和目标。",
        }
    )
    node = build_dog_query_understanding_node(
        llm_provider=llm_provider,
        query_filter_parser=FakeQueryFilterParser(),
    )

    update = await node(
        {
            "retrieval_question": "帮6岁的金毛制定训练计划",
            "user_id": "user_001",
            "intent": "training",
        }
    )

    assert len(llm_provider.calls) == 1
    assert update["filters"] == {
        "$and": [
            {"dog_name": {"$eq": "Golden Retriever"}},
            {"trainability_level": {"$gte": 4}},
        ]
    }
    assert "age_years" in update["pet_profile_suggested_attributes"]
    assert "breed" in update["pet_profile_suggested_attributes"]
    assert "unknown_field" not in update[
        "pet_profile_suggested_attributes"
    ]
    assert update["dog_query_understanding_result"][
        "invalid_llm_profile_attributes"
    ] == ["unknown_field"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rewritten", ["金毛日常运动需求与户外运动注意事项", None, "   "])
async def test_retrieval_rewrite_preserves_original_question(rewritten) -> None:
    """检查改写衔接和空值兼容，不验证模型真实语义能力。

    参数：rewritten 为模型返回的检索文本或空值。
    返回：无；断言检索字段同步、原问题不变且只调用模型一次。
    """
    from src.rag.query_builders.rag_query_builder import build_rag_query_from_state

    original = "成都现在这个天气，带金毛出去运动需要注意什么？"
    provider = FakeLLMProvider({"retrieval_question": rewritten})
    node = build_dog_query_understanding_node(
        llm_provider=provider, query_filter_parser=FakeQueryFilterParser(),
    )
    state = {"question": original, "user_id": "user_001"}
    update = await node(state)
    expected = (rewritten or "").strip() or original
    assert len(provider.calls) == 1
    assert "question" not in update
    assert state["question"] == original
    assert update["retrieval_question"] == expected
    assert update["rag_query"]["question"] == expected
    assert update["dog_query_understanding_result"]["question"] == original
    downstream = build_rag_query_from_state(
        state={**state, **update}, parser=FakeQueryFilterParser(),
    )
    assert downstream.question == expected


@pytest.mark.asyncio
async def test_invalid_rewrite_type_keeps_existing_retrieval_input() -> None:
    """检查非法结构走既有降级路径。

    参数：无，使用固定错误模型结果。
    返回：无；断言未把非法改写用于检索，也没有增加修复调用。
    """
    provider = FakeLLMProvider({"retrieval_question": ["不合法的列表"]})
    node = build_dog_query_understanding_node(
        llm_provider=provider, query_filter_parser=FakeQueryFilterParser(),
    )
    update = await node({"question": "金毛运动需求"})
    assert update["retrieval_question"] == "金毛运动需求"
    assert len(provider.calls) == 1
