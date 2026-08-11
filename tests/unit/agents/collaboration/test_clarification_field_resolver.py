"""多智能体自然语言澄清字段解析与步骤分配测试。"""

from __future__ import annotations

import json

import pytest

from src.agents.collaboration.adapters import (
    MultiAgentClarificationFieldResolver,
    allocate_fields_to_steps,
    build_default_multi_agent_clarification_field_resolver,
)


class FakeMessage:
    """保存测试用 LLM 文本响应。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLMProvider:
    """记录调用并返回固定 JSON 的测试模型提供者。"""

    def __init__(self, response: dict[str, object]) -> None:
        self.main_llm = object()
        self.response = response
        self.prompts: list[str] = []

    async def safe_ainvoke(self, **kwargs: object) -> FakeMessage:
        """返回固定响应并保存提示词。"""

        self.prompts.append(str(kwargs.get("prompt") or ""))
        return FakeMessage(json.dumps(self.response, ensure_ascii=False))


class FailingLLMProvider:
    """模拟模型调用失败的测试提供者。"""

    main_llm = object()

    async def safe_ainvoke(self, **_: object) -> FakeMessage:
        """主动抛出异常以验证规则降级。"""

        raise RuntimeError("模拟 LLM 不可用")


def test_resolver_should_only_extract_requested_fields() -> None:
    """验证字段解析层只返回当前澄清申请的字段。"""

    resolver = build_default_multi_agent_clarification_field_resolver()

    result = resolver.extract(
        user_text="它是一只金毛，6岁，希望学习等待和召回。",
        requested_field_ids=["age", "training_goal"],
    )

    assert result.extracted_fields == {
        "age": "6岁",
        "training_goal": "学习等待和召回",
    }
    assert "breed" not in result.extracted_fields
    assert result.missing_field_ids == []
    assert result.coverage_ratio == 1.0


def test_resolver_should_merge_fields_across_turns() -> None:
    """验证前一轮年龄可以和本轮训练目标合并。"""

    resolver = build_default_multi_agent_clarification_field_resolver()

    result = resolver.extract(
        user_text="希望学习等待和召回。",
        requested_field_ids=["age", "training_goal"],
        existing_fields={"age": "6岁"},
    )

    assert result.extracted_fields == {
        "training_goal": "学习等待和召回"
    }
    assert result.resolved_fields == {
        "age": "6岁",
        "training_goal": "学习等待和召回",
    }
    assert result.missing_field_ids == []


def test_resolver_should_mark_conflicting_rule_values_as_ambiguous() -> None:
    """验证不同规则对同一字段给出冲突值时不会自动猜测。"""

    resolver = MultiAgentClarificationFieldResolver(
        {
            "first_rule": lambda _: {"age": "5岁"},
            "second_rule": lambda _: {"age": "6岁"},
        }
    )

    result = resolver.extract(
        user_text="年龄信息存在冲突",
        requested_field_ids=["age"],
    )

    assert result.extracted_fields == {}
    assert result.ambiguous_field_ids == ["age"]
    assert result.coverage_ratio == 0.0


def test_allocator_should_copy_shared_field_to_all_consumer_steps() -> None:
    """验证步骤分配层会把共享年龄复制给全部使用步骤。"""

    step_inputs = allocate_fields_to_steps(
        resolved_fields={
            "age": "6岁",
            "training_goal": "等待和召回",
        },
        field_consumers={
            "age": ["step_health", "step_training"],
            "training_goal": ["step_training"],
        },
    )

    assert step_inputs == {
        "step_health": {"age": "6岁"},
        "step_training": {
            "age": "6岁",
            "training_goal": "等待和召回",
        },
    }


@pytest.mark.asyncio
async def test_layered_resolver_should_only_use_llm_for_missing_fields(
) -> None:
    """验证规则结果优先，LLM 只能补充仍然缺失的白名单字段。"""

    provider = FakeLLMProvider(
        {
            "fields": {
                "age": "9岁",
                "training_goal": "学习召回",
                "unknown": "不应接收",
            },
            "confidences": {
                "age": 0.99,
                "training_goal": 0.92,
                "unknown": 0.99,
            },
            "reason": "识别到用户描述的训练目标。",
        }
    )
    resolver = MultiAgentClarificationFieldResolver(
        {"age_rule": lambda _: {"age": "6岁"}},
        llm_provider=provider,
    )

    result = await resolver.extract_layered(
        user_text="它已经六岁，我主要想训练叫回。",
        requested_field_ids=["age", "training_goal"],
        field_descriptions={"training_goal": "训练目标"},
    )

    assert result.resolved_fields == {
        "age": "6岁",
        "training_goal": "学习召回",
    }
    assert result.field_sources == {
        "age": ["age_rule"],
        "training_goal": ["llm_fallback"],
    }
    assert result.rejected_field_ids == ["age", "unknown"]
    assert result.llm_fallback_used is True
    assert len(provider.prompts) == 1
    assert '"training_goal"' in provider.prompts[0]


@pytest.mark.asyncio
async def test_layered_resolver_should_not_call_llm_when_rules_are_complete(
) -> None:
    """验证规则已经补全全部字段时不会产生额外 LLM 调用。"""

    provider = FakeLLMProvider(
        {"fields": {}, "confidences": {}, "reason": ""}
    )
    resolver = MultiAgentClarificationFieldResolver(
        {"age_rule": lambda _: {"age": "6岁"}},
        llm_provider=provider,
    )

    result = await resolver.extract_layered(
        user_text="它6岁。",
        requested_field_ids=["age"],
    )

    assert result.resolved_fields == {"age": "6岁"}
    assert result.llm_fallback_used is False
    assert provider.prompts == []


@pytest.mark.asyncio
async def test_layered_resolver_should_keep_rules_when_llm_fails() -> None:
    """验证 LLM 失败时保留规则结果并继续把剩余字段视为缺失。"""

    resolver = MultiAgentClarificationFieldResolver(
        {"age_rule": lambda _: {"age": "6岁"}},
        llm_provider=FailingLLMProvider(),
    )

    result = await resolver.extract_layered(
        user_text="它6岁，其他信息说得比较含糊。",
        requested_field_ids=["age", "training_goal"],
    )

    assert result.resolved_fields == {"age": "6岁"}
    assert result.missing_field_ids == ["training_goal"]
    assert result.llm_fallback_used is True
    assert "模拟 LLM 不可用" in result.llm_error_message


@pytest.mark.asyncio
async def test_layered_resolver_should_reclarify_low_confidence_field(
) -> None:
    """验证低可信度 LLM 候选不会直接进入步骤输入。"""

    provider = FakeLLMProvider(
        {
            "fields": {"training_goal": "可能是召回"},
            "confidences": {"training_goal": 0.4},
            "reason": "用户表达不够明确。",
        }
    )
    resolver = MultiAgentClarificationFieldResolver(
        {},
        llm_provider=provider,
        minimum_llm_confidence=0.75,
    )

    result = await resolver.extract_layered(
        user_text="大概练点平时会用到的。",
        requested_field_ids=["training_goal"],
    )

    assert result.resolved_fields == {}
    assert result.ambiguous_field_ids == ["training_goal"]
    assert result.missing_field_ids == []
