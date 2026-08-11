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
