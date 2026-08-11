from datetime import datetime, timezone

from src.memory.memory_schema import (
    PetProfileExtractionResult,
    PetProfileFactCandidate,
)
from src.memory.pet_profile_service import PetProfileService


OBSERVED_AT = datetime(2026, 8, 8, tzinfo=timezone.utc)


class FakePetProfileStore:
    """记录宠物档案服务查询和保存行为的测试存储器。"""

    def __init__(
        self,
        *,
        existing_facts: list[dict] | None = None,
        failed_attributes: set[str] | None = None,
    ) -> None:
        self.existing_facts = list(existing_facts or [])
        self.failed_attributes = set(failed_attributes or set())
        self.saved_facts = []

    def get_pet_profile_facts(
        self,
        user_id: str,
        pet_key: str | None = None,
        attributes: list[str] | None = None,
    ) -> list[dict]:
        """
        返回预设的已有宠物档案事实。

        参数含义：
            user_id：服务正在查询的用户标识。
            pet_key：可选的宠物稳定标识，本测试不使用筛选。
            attributes：可选的允许读取字段列表。

        返回值含义：
            list[dict]：初始化测试存储器时传入的已有事实。
        """

        _ = user_id
        matched_facts = [
            fact
            for fact in self.existing_facts
            if pet_key is None or fact.get("pet_key") == pet_key
        ]
        if attributes is None:
            return matched_facts
        return [
            fact
            for fact in matched_facts
            if fact.get("attribute") in attributes
        ]

    def list_pet_profile_identities(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        """
        从预设事实中返回去重后的宠物身份。

        参数含义：
            user_id：服务正在查询的用户标识。

        返回值含义：
            list[dict[str, str]]：测试数据中的宠物标识和名称。
        """

        _ = user_id
        identities: dict[str, str] = {}
        for fact in self.existing_facts:
            pet_key = str(fact.get("pet_key") or "")
            if pet_key:
                identities[pet_key] = str(fact.get("pet_name") or "")
        return [
            {"pet_key": pet_key, "pet_name": pet_name}
            for pet_key, pet_name in sorted(identities.items())
        ]

    def upsert_pet_profile_fact(self, fact) -> dict:
        """
        记录正式事实或按属性模拟数据库异常。

        参数含义：
            fact：宠物档案服务准备保存的正式宠物事实。

        返回值含义：
            dict：模拟 SQLite 存储器的 created 保存结果。
        """

        if fact.attribute in self.failed_attributes:
            raise RuntimeError(f"测试保存失败: {fact.attribute}")
        self.saved_facts.append(fact)
        return {
            "action": "created",
            "fact_id": len(self.saved_facts),
            "value": fact.value,
            "observed_at": fact.observed_at,
        }


def _extraction_result(
    *candidates: PetProfileFactCandidate,
) -> PetProfileExtractionResult:
    """
    构建宠物档案服务测试使用的批量抽取结果。

    参数含义：
        *candidates：需要放入批量结果的宠物档案候选事实。

    返回值含义：
        PetProfileExtractionResult：包含全部候选事实的测试抽取结果。
    """

    return PetProfileExtractionResult(facts=list(candidates))


def _candidate(
    subject_reference: str,
    attribute: str,
    value: str,
) -> PetProfileFactCandidate:
    """
    构建宠物档案服务测试使用的单条候选事实。

    参数含义：
        subject_reference：用户原话中的宠物称呼。
        attribute：候选宠物档案属性。
        value：候选档案属性值。

    返回值含义：
        PetProfileFactCandidate：经过契约校验的候选事实。
    """

    return PetProfileFactCandidate(
        subject_reference=subject_reference,
        attribute=attribute,
        value=value,
        confidence=0.95,
        evidence_text=f"{subject_reference}的{attribute}是{value}",
    )


def test_service_should_resolve_and_save_multiple_profile_facts() -> None:
    store = FakePetProfileStore()
    service = PetProfileService(store)

    result = service.save_extraction_result(
        user_id="user_001",
        extraction_result=_extraction_result(
            _candidate("豆豆", "age_years", "6"),
            _candidate("豆豆", "weight_kg", "30"),
        ),
        observed_at=OBSERVED_AT,
    )

    assert result.created_count == 2
    assert result.failed_count == 0
    assert len(store.saved_facts) == 2
    assert store.saved_facts[0].pet_key == store.saved_facts[1].pet_key


def test_service_should_not_save_ambiguous_generic_reference() -> None:
    store = FakePetProfileStore(
        existing_facts=[
            {"pet_key": "doudou", "pet_name": "豆豆"},
            {"pet_key": "qiuqiu", "pet_name": "球球"},
        ]
    )
    service = PetProfileService(store)

    result = service.save_extraction_result(
        user_id="user_001",
        extraction_result=_extraction_result(
            _candidate("它", "weight_kg", "30")
        ),
        observed_at=OBSERVED_AT,
    )

    assert result.persistence_records == []
    assert len(result.resolution.unresolved_facts) == 1
    assert store.saved_facts == []


def test_service_should_continue_after_single_fact_save_failure() -> None:
    store = FakePetProfileStore(failed_attributes={"weight_kg"})
    service = PetProfileService(store)

    result = service.save_extraction_result(
        user_id="user_001",
        extraction_result=_extraction_result(
            _candidate("豆豆", "weight_kg", "30"),
            _candidate("豆豆", "age_years", "6"),
        ),
        observed_at=OBSERVED_AT,
    )

    assert result.failed_count == 1
    assert result.created_count == 1
    assert result.persistence_records[0].action == "failed"
    assert "测试保存失败" in result.persistence_records[0].error
    assert result.persistence_records[1].action == "created"


def test_service_should_keep_existing_legacy_pet_key() -> None:
    store = FakePetProfileStore(
        existing_facts=[
            {"pet_key": "doudou", "pet_name": "豆豆"},
        ]
    )
    service = PetProfileService(store)

    result = service.save_extraction_result(
        user_id="user_001",
        extraction_result=_extraction_result(
            _candidate("豆豆", "weight_kg", "30")
        ),
        observed_at=OBSERVED_AT,
    )

    assert result.created_count == 1
    assert store.saved_facts[0].pet_key == "doudou"


def _stored_fact(
    *,
    pet_key: str,
    pet_name: str,
    attribute: str,
    value: str,
) -> dict:
    """
    构建召回测试使用的完整数据库事实。

    参数含义：
        pet_key：宠物稳定标识。
        pet_name：宠物展示名称。
        attribute：宠物档案属性。
        value：宠物档案属性值。

    返回值含义：
        dict：可以通过 PetProfileFact 契约校验的数据库记录。
    """

    return {
        "user_id": "user_001",
        "pet_key": pet_key,
        "pet_name": pet_name,
        "attribute": attribute,
        "value": value,
        "confidence": 0.98,
        "source": "conversation",
        "observed_at": OBSERVED_AT.isoformat(),
    }


def test_recall_profile_should_use_active_pet_identity() -> None:
    store = FakePetProfileStore(
        existing_facts=[
            _stored_fact(
                pet_key="pet_doudou",
                pet_name="豆豆",
                attribute="breed",
                value="金毛",
            ),
            _stored_fact(
                pet_key="pet_qiuqiu",
                pet_name="球球",
                attribute="breed",
                value="柯基",
            ),
        ]
    )

    result = PetProfileService(store).recall_profile(
        user_id="user_001",
        active_pet_key="pet_doudou",
        active_pet_name="豆豆",
    )

    assert result.status == "applied"
    assert result.selection_source == "active_pet"
    assert result.pet_name == "豆豆"
    assert result.facts == {"breed": "金毛"}


def test_recall_profile_should_select_only_pet_as_fallback() -> None:
    store = FakePetProfileStore(
        existing_facts=[
            _stored_fact(
                pet_key="pet_doudou",
                pet_name="豆豆",
                attribute="age_years",
                value="6",
            )
        ]
    )

    result = PetProfileService(store).recall_profile(user_id="user_001")

    assert result.status == "applied"
    assert result.selection_source == "single_pet_fallback"
    assert result.pet_key == "pet_doudou"
    assert result.facts == {"age_years": "6"}


def test_recall_profile_should_not_guess_between_multiple_pets() -> None:
    store = FakePetProfileStore(
        existing_facts=[
            _stored_fact(
                pet_key="pet_doudou",
                pet_name="豆豆",
                attribute="breed",
                value="金毛",
            ),
            _stored_fact(
                pet_key="pet_qiuqiu",
                pet_name="球球",
                attribute="breed",
                value="柯基",
            ),
        ]
    )

    result = PetProfileService(store).recall_profile(user_id="user_001")

    assert result.status == "ambiguous"
    assert result.facts == {}
    assert "没有明确当前宠物" in result.reason
