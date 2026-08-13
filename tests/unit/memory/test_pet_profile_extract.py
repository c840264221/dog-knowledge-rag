import json

import pytest

from src.memory.pet_profile_extract import (
    extract_pet_profile_facts,
    normalize_pet_profile_extraction_result,
)
from src.memory.pet_profile_value_normalizer import (
    normalize_age_years,
    normalize_age_years_for_skill,
)


class FakeMessage:
    """保存测试模型返回的文本内容。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakePetProfileLLMProvider:
    """为宠物档案抽取测试提供确定性的模型响应。"""

    def __init__(self, payload: dict) -> None:
        self.chinese_llm = object()
        self.payload = payload
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        *,
        llm,
        prompt: str,
        fallback_response: str,
        call_metadata=None,
    ) -> FakeMessage:
        """
        返回预设 JSON 并记录完整提示词。

        参数含义：
            llm：抽取器选择的模型占位对象。
            prompt：抽取器构造的完整提示词。
            fallback_response：真实调用失败时使用的兜底响应。

        返回值含义：
            FakeMessage：包含确定性 JSON 文本的测试消息。
        """

        _ = llm, fallback_response
        self.prompts.append(prompt)
        return FakeMessage(json.dumps(self.payload, ensure_ascii=False))


class FailingPetProfileLLMProvider:
    """模拟统一模型调用发生异常的服务提供者。"""

    def __init__(self) -> None:
        self.chinese_llm = object()

    async def safe_ainvoke(self, **kwargs):
        """
        抛出测试异常以验证安全降级。

        参数含义：
            **kwargs：抽取器传入的模型、提示词和兜底响应。

        返回值含义：
            无，始终抛出 RuntimeError。
        """

        _ = kwargs
        raise RuntimeError("test llm failure")


def test_normalizer_should_keep_valid_facts_and_reject_invalid_item() -> None:
    """
    验证一条非法候选不会导致同批合法候选全部丢失。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = normalize_pet_profile_extraction_result(
        {
            "facts": [
                {
                    "subject_reference": "豆豆",
                    "attribute": "breed",
                    "value": "金毛",
                    "confidence": 0.98,
                    "evidence_text": "豆豆是一只金毛",
                },
                {
                    "subject_reference": "豆豆",
                    "attribute": "temporary_mood",
                    "value": "开心",
                    "confidence": 0.90,
                    "evidence_text": "豆豆今天很开心",
                },
            ],
            "reason": "测试局部拒绝",
        }
    )

    assert len(result.facts) == 1
    assert result.facts[0].attribute == "breed"
    assert result.rejected_candidate_count == 1


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        (6, "6"),
        ("6", "6"),
        ("6.0", "6"),
        ("6岁", "6"),
        ("0.5年", "0.5"),
    ],
)
def test_age_years_normalizer_should_accept_supported_formats(
    raw_value,
    expected_value: str,
) -> None:
    """
    验证年龄数字和常见文本格式会统一成无单位数字字符串。

    参数含义：
        raw_value：待归一化的原始年龄。
        expected_value：期望得到的数据库标准值。

    返回值含义：
        None，pytest 根据归一化结果判断是否通过。
    """

    assert normalize_age_years(raw_value) == expected_value


def test_age_normalizers_should_be_idempotent() -> None:
    """
    验证档案年龄和 Skill 年龄重复归一化后结果不变。

    参数含义：无。
    返回值含义：None，pytest 根据两次处理结果判断是否通过。
    """

    profile_value = normalize_age_years("6岁")
    skill_value = normalize_age_years_for_skill("6岁")

    assert normalize_age_years(profile_value) == profile_value
    assert normalize_age_years_for_skill(skill_value) == skill_value


def test_numeric_age_candidate_should_be_normalized_before_validation() -> None:
    """
    验证 LLM 返回数字年龄时会在候选契约校验前转换成字符串。

    参数含义：无。
    返回值含义：None，pytest 根据候选数量和值判断是否通过。
    """

    result = normalize_pet_profile_extraction_result(
        {
            "facts": [
                {
                    "subject_reference": "豆豆",
                    "attribute": "age_years",
                    "value": 6,
                    "confidence": 0.98,
                    "evidence_text": "豆豆6岁",
                }
            ]
        }
    )

    assert result.rejected_candidate_count == 0
    assert result.facts[0].value == "6"


def test_invalid_age_candidate_should_be_rejected_independently() -> None:
    """
    验证无法识别的年龄值只会拒绝当前候选。

    参数含义：无。
    返回值含义：None，pytest 根据局部拒绝结果判断是否通过。
    """

    result = normalize_pet_profile_extraction_result(
        {
            "facts": [
                {
                    "subject_reference": "豆豆",
                    "attribute": "age_years",
                    "value": True,
                    "confidence": 0.98,
                    "evidence_text": "错误年龄",
                },
                {
                    "subject_reference": "豆豆",
                    "attribute": "breed",
                    "value": "金毛",
                    "confidence": 0.98,
                    "evidence_text": "豆豆是金毛",
                },
            ]
        }
    )

    assert [fact.attribute for fact in result.facts] == ["breed"]
    assert result.rejected_candidate_count == 1


@pytest.mark.asyncio
async def test_extractor_should_extract_multiple_profile_facts() -> None:
    """
    验证一句用户输入可以生成多条结构化宠物档案候选。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    provider = FakePetProfileLLMProvider(
        {
            "facts": [
                {
                    "subject_reference": "豆豆",
                    "attribute": "breed",
                    "value": "金毛",
                    "confidence": 0.98,
                    "evidence_text": "豆豆是一只金毛",
                },
                {
                    "subject_reference": "豆豆",
                    "attribute": "age_years",
                    "value": "6",
                    "confidence": 0.98,
                    "evidence_text": "6岁的金毛",
                },
                {
                    "subject_reference": "豆豆",
                    "attribute": "weight_kg",
                    "value": "32",
                    "confidence": 0.98,
                    "evidence_text": "体重32公斤",
                },
            ],
            "reason": "用户明确提供了多项档案",
        }
    )

    result = await extract_pet_profile_facts(
        llm_provider=provider,
        user_text="豆豆是一只6岁的金毛，体重32公斤。",
    )

    assert [fact.attribute for fact in result.facts] == [
        "breed",
        "age_years",
        "weight_kg",
    ]
    assert result.rejected_candidate_count == 0
    assert "豆豆是一只6岁的金毛" in provider.prompts[0]


@pytest.mark.asyncio
async def test_extractor_should_preserve_pronoun_for_entity_resolution() -> None:
    """
    验证抽取器保留“它”等代词，不擅自绑定数据库宠物实体。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    provider = FakePetProfileLLMProvider(
        {
            "facts": [
                {
                    "subject_reference": "它",
                    "attribute": "weight_kg",
                    "value": "32",
                    "confidence": 0.90,
                    "evidence_text": "它现在32公斤",
                }
            ],
            "reason": "提取到代词引用的体重事实",
        }
    )

    result = await extract_pet_profile_facts(
        llm_provider=provider,
        user_text="它现在32公斤。",
    )

    assert result.facts[0].subject_reference == "它"


@pytest.mark.asyncio
async def test_extractor_should_return_empty_result_for_blank_input() -> None:
    """
    验证空白输入不会调用模型并返回安全空结果。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    provider = FakePetProfileLLMProvider({"facts": []})
    result = await extract_pet_profile_facts(
        llm_provider=provider,
        user_text="   ",
    )

    assert result.facts == []
    assert provider.prompts == []


def test_normalizer_should_limit_candidates_per_request() -> None:
    """
    验证单轮候选数量受到上限保护，避免异常模型输出无限占用资源。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    raw_fact = {
        "subject_reference": "豆豆",
        "attribute": "weight_kg",
        "value": "32",
        "confidence": 0.98,
        "evidence_text": "豆豆体重32公斤",
    }
    result = normalize_pet_profile_extraction_result(
        {"facts": [dict(raw_fact) for _ in range(25)]}
    )

    assert len(result.facts) == 20
    assert result.rejected_candidate_count == 5


@pytest.mark.asyncio
async def test_extractor_should_degrade_when_llm_call_fails() -> None:
    """
    验证模型调用异常时返回空候选，不把异常抛给 Agent 主链路。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = await extract_pet_profile_facts(
        llm_provider=FailingPetProfileLLMProvider(),
        user_text="豆豆是一只金毛。",
    )

    assert result.facts == []
    assert result.reason == "宠物档案抽取失败"
