"""把宠物档案候选事实绑定到稳定宠物标识。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from src.memory.memory_schema import (
    MemorySource,
    PetProfileFact,
    PetProfileFactCandidate,
    PetProfileResolutionResult,
)


# 这些称呼只能说明“某只宠物”，不能唯一说明具体是哪一只。
GENERIC_PET_REFERENCES = frozenset(
    {
        "它",
        "他",
        "她",
        "狗",
        "狗狗",
        "宠物",
        "我家狗",
        "我家狗狗",
        "我的狗",
        "我的狗狗",
        "这只狗",
        "这只狗狗",
    }
)


def resolve_pet_profile_facts(
    *,
    user_id: str,
    candidates: Iterable[PetProfileFactCandidate],
    observed_at: datetime,
    existing_facts: Iterable[PetProfileFact | Mapping[str, Any]] | None = None,
    selected_pet_key: str | None = None,
    selected_pet_name: str | None = None,
    source: MemorySource = "conversation",
) -> PetProfileResolutionResult:
    """
    把一批宠物档案候选事实解析成可持久化事实。

    功能：
        明确名字优先匹配已有宠物；没有历史记录时为明确名字生成稳定
        pet_key。代词或“我家狗”这类泛指称呼，只在上游已指定宠物或
        当前用户只有一只已知宠物时绑定，否则保留为未解析候选。

    参数含义：
        user_id：当前用户编号，用于隔离不同用户并参与稳定标识生成。
        candidates：LLM 从本轮输入中提取出的宠物档案候选事实。
        observed_at：这些候选事实被用户表达或系统观察到的时间。
        existing_facts：数据库中该用户已有的宠物档案事实。
        selected_pet_key：上游已经明确选择的宠物稳定标识。
        selected_pet_name：与 selected_pet_key 对应的展示名称。
        source：档案事实来源，默认是 conversation（对话）。

    返回值含义：
        PetProfileResolutionResult：包含可保存事实和未解析候选的结构化结果。
    """

    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("宠物档案实体解析缺少 user_id")

    # 已有身份索引用来把“豆豆”等名称重新绑定到原来的 pet_key。
    identities_by_name, identities_by_key = _build_existing_identity_indexes(
        existing_facts or []
    )

    normalized_selected_key = str(selected_pet_key or "").strip()
    normalized_selected_name = str(selected_pet_name or "").strip()
    if normalized_selected_key:
        identities_by_key.setdefault(
            normalized_selected_key,
            normalized_selected_name,
        )

    # 合法事实和无法确定宠物对象的事实分开返回，避免整批数据一起失败。
    resolved_facts: list[PetProfileFact] = []
    unresolved_facts: list[PetProfileFactCandidate] = []

    for candidate in candidates:
        identity = _resolve_candidate_identity(
            user_id=normalized_user_id,
            subject_reference=candidate.subject_reference,
            identities_by_name=identities_by_name,
            identities_by_key=identities_by_key,
            selected_pet_key=normalized_selected_key,
            selected_pet_name=normalized_selected_name,
        )
        if identity is None:
            unresolved_facts.append(candidate)
            continue

        pet_key, pet_name = identity
        resolved_facts.append(
            PetProfileFact(
                user_id=normalized_user_id,
                pet_key=pet_key,
                pet_name=pet_name,
                attribute=candidate.attribute,
                value=candidate.value,
                confidence=candidate.confidence,
                source=source,
                observed_at=observed_at,
            )
        )

        # 同一轮后面的“它”可以引用本轮刚刚明确出现的唯一宠物。
        identities_by_key.setdefault(pet_key, pet_name)
        if pet_name:
            identities_by_name.setdefault(
                _normalize_identity_text(pet_name),
                (pet_key, pet_name),
            )

    return PetProfileResolutionResult(
        facts=resolved_facts,
        unresolved_facts=unresolved_facts,
        reason=(
            f"解析成功 {len(resolved_facts)} 条，"
            f"对象不明确 {len(unresolved_facts)} 条。"
        ),
    )


def _resolve_candidate_identity(
    *,
    user_id: str,
    subject_reference: str,
    identities_by_name: Mapping[str, tuple[str, str]],
    identities_by_key: Mapping[str, str],
    selected_pet_key: str,
    selected_pet_name: str,
) -> tuple[str, str] | None:
    """
    解析单条候选事实指向的宠物身份。

    参数含义：
        user_id：当前用户编号。
        subject_reference：用户原文中的宠物称呼。
        identities_by_name：归一化宠物名称到稳定身份的索引。
        identities_by_key：稳定 pet_key 到展示名称的索引。
        selected_pet_key：上游已经选择的宠物标识。
        selected_pet_name：上游已经选择的宠物名称。

    返回值含义：
        tuple[str, str] | None：成功时返回 pet_key 和 pet_name；对象不明确
        时返回 None。
    """

    cleaned_reference = _clean_subject_reference(subject_reference)
    normalized_reference = _normalize_identity_text(cleaned_reference)

    if normalized_reference not in GENERIC_PET_REFERENCES:
        existing_identity = identities_by_name.get(normalized_reference)
        if existing_identity is not None:
            return existing_identity
        return (
            _build_stable_pet_key(user_id, cleaned_reference),
            cleaned_reference,
        )

    if selected_pet_key:
        return selected_pet_key, selected_pet_name

    if len(identities_by_key) == 1:
        pet_key, pet_name = next(iter(identities_by_key.items()))
        return pet_key, pet_name

    return None


def _build_existing_identity_indexes(
    existing_facts: Iterable[PetProfileFact | Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """
    从已有档案事实构建名称索引和稳定标识索引。

    参数含义：
        existing_facts：数据库查询结果或 PetProfileFact 对象集合。

    返回值含义：
        tuple[dict, dict]：第一个字典按宠物名称查身份，第二个字典按
        pet_key 查展示名称。
    """

    identities_by_name: dict[str, tuple[str, str]] = {}
    identities_by_key: dict[str, str] = {}

    for raw_fact in existing_facts:
        if isinstance(raw_fact, PetProfileFact):
            pet_key = raw_fact.pet_key
            pet_name = raw_fact.pet_name
        elif isinstance(raw_fact, Mapping):
            pet_key = str(raw_fact.get("pet_key") or "").strip()
            pet_name = str(raw_fact.get("pet_name") or "").strip()
        else:
            continue

        if not pet_key:
            continue
        identities_by_key.setdefault(pet_key, pet_name)
        if pet_name:
            identities_by_name.setdefault(
                _normalize_identity_text(pet_name),
                (pet_key, pet_name),
            )

    return identities_by_name, identities_by_key


def _clean_subject_reference(value: str) -> str:
    """
    清理宠物称呼中不属于名字的口语前缀。

    参数含义：
        value：LLM 提取出的原始宠物称呼。

    返回值含义：
        str：可用于身份匹配和展示的宠物称呼。
    """

    cleaned = str(value or "").strip()
    cleaned = re.sub(r"^(?:我家的|我家|我的)", "", cleaned)
    cleaned = re.sub(r"^(?:名叫|叫做|叫)", "", cleaned)
    return cleaned.strip() or str(value or "").strip()


def _normalize_identity_text(value: str) -> str:
    """
    归一化用于身份比较的宠物名称或称呼。

    参数含义：
        value：待比较的宠物名称或称呼。

    返回值含义：
        str：去掉空白并转成小写后的比较文本。
    """

    return re.sub(r"\s+", "", str(value or "")).lower()


def _build_stable_pet_key(user_id: str, pet_name: str) -> str:
    """
    为首次出现的明确宠物名称生成稳定 pet_key。

    参数含义：
        user_id：当前用户编号，用于避免不同用户的同名宠物发生碰撞。
        pet_name：宠物展示名称。

    返回值含义：
        str：带 pet_v1 前缀的不可逆摘要标识。相同用户和归一化名称会
        稳定生成相同结果，不同用户的同名宠物会生成不同结果。
    """

    normalized_name = _normalize_identity_text(pet_name)
    digest = hashlib.sha256(
        f"{user_id}:{normalized_name}".encode("utf-8")
    ).hexdigest()[:16]
    return f"pet_v1_{digest}"
