from datetime import datetime, timezone

from src.memory.memory_schema import (
    PetProfileFact,
    PetProfileFactCandidate,
)
from src.memory.pet_profile_entity_resolver import (
    resolve_pet_profile_facts,
)


OBSERVED_AT = datetime(2026, 8, 8, tzinfo=timezone.utc)


def _candidate(
    subject_reference: str,
    attribute: str = "weight_kg",
    value: str = "30",
) -> PetProfileFactCandidate:
    """
    构建单元测试使用的宠物档案候选事实。

    参数含义：
        subject_reference：候选事实中的宠物称呼。
        attribute：候选档案属性。
        value：候选档案值。

    返回值含义：
        PetProfileFactCandidate：通过契约校验的测试候选事实。
    """

    return PetProfileFactCandidate(
        subject_reference=subject_reference,
        attribute=attribute,
        value=value,
        confidence=0.95,
        evidence_text=f"{subject_reference}的{attribute}是{value}",
    )


def _existing_fact(
    pet_key: str,
    pet_name: str,
) -> PetProfileFact:
    """
    构建单元测试使用的已有宠物档案事实。

    参数含义：
        pet_key：已有宠物稳定标识。
        pet_name：已有宠物展示名称。

    返回值含义：
        PetProfileFact：可以提供给实体解析器的已有档案事实。
    """

    return PetProfileFact(
        user_id="user_001",
        pet_key=pet_key,
        pet_name=pet_name,
        attribute="breed",
        value="金毛",
        confidence=1.0,
        observed_at=OBSERVED_AT,
    )


def test_resolver_should_create_stable_key_for_explicit_chinese_name() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[
            _candidate("豆豆"),
            _candidate("豆豆", "age_years", "6"),
        ],
        observed_at=OBSERVED_AT,
    )

    assert len(result.facts) == 2
    assert result.facts[0].pet_key.startswith("pet_v1_")
    assert result.facts[0].pet_key == result.facts[1].pet_key
    assert result.facts[0].pet_name == "豆豆"
    assert result.unresolved_facts == []


def test_resolver_should_hash_explicit_ascii_name_with_same_format() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("Doudou")],
        observed_at=OBSERVED_AT,
    )

    assert result.facts[0].pet_key.startswith("pet_v1_")
    assert result.facts[0].pet_key != "doudou"


def test_resolver_should_generate_same_key_for_normalized_name() -> None:
    first_result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("Doudou")],
        observed_at=OBSERVED_AT,
    )
    second_result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate(" dou dou ")],
        observed_at=OBSERVED_AT,
    )

    assert first_result.facts[0].pet_key == second_result.facts[0].pet_key


def test_resolver_should_isolate_same_pet_name_between_users() -> None:
    first_result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("豆豆")],
        observed_at=OBSERVED_AT,
    )
    second_result = resolve_pet_profile_facts(
        user_id="user_002",
        candidates=[_candidate("豆豆")],
        observed_at=OBSERVED_AT,
    )

    assert first_result.facts[0].pet_key != second_result.facts[0].pet_key


def test_resolver_should_match_explicit_name_to_existing_pet() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("我家的豆豆")],
        observed_at=OBSERVED_AT,
        existing_facts=[_existing_fact("doudou", "豆豆")],
    )

    assert result.facts[0].pet_key == "doudou"
    assert result.facts[0].pet_name == "豆豆"


def test_resolver_should_bind_generic_reference_to_only_known_pet() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("它")],
        observed_at=OBSERVED_AT,
        existing_facts=[_existing_fact("doudou", "豆豆")],
    )

    assert result.facts[0].pet_key == "doudou"
    assert result.unresolved_facts == []


def test_resolver_should_not_guess_when_multiple_pets_exist() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("它")],
        observed_at=OBSERVED_AT,
        existing_facts=[
            _existing_fact("doudou", "豆豆"),
            _existing_fact("qiuqiu", "球球"),
        ],
    )

    assert result.facts == []
    assert result.unresolved_facts == [_candidate("它")]


def test_resolver_should_use_selected_pet_for_generic_reference() -> None:
    result = resolve_pet_profile_facts(
        user_id="user_001",
        candidates=[_candidate("我家狗狗")],
        observed_at=OBSERVED_AT,
        existing_facts=[
            _existing_fact("doudou", "豆豆"),
            _existing_fact("qiuqiu", "球球"),
        ],
        selected_pet_key="qiuqiu",
        selected_pet_name="球球",
    )

    assert result.facts[0].pet_key == "qiuqiu"
    assert result.facts[0].pet_name == "球球"
