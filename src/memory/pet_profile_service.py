"""串联宠物档案实体解析与数据库持久化。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

from src.logger import logger
from src.memory.memory_schema import (
    MemorySource,
    PetProfileExtractionResult,
    PetProfileAttribute,
    PetProfileFact,
    PetProfilePersistenceAction,
    PetProfilePersistenceRecord,
    PetProfileRecallResult,
    PetProfileSaveResult,
)
from src.memory.pet_profile_entity_resolver import (
    resolve_pet_profile_facts,
)


class PetProfileStore(Protocol):
    """宠物档案服务依赖的最小存储能力契约。"""

    def get_pet_profile_facts(
        self,
        user_id: str,
        pet_key: str | None = None,
        attributes: list[PetProfileAttribute] | None = None,
    ) -> list[dict]:
        """
        查询用户已有宠物档案事实。

        参数含义：
            user_id：需要查询的用户标识。
            pet_key：可选的宠物稳定标识；为空时查询用户全部宠物。
            attributes：可选的允许读取字段；为空时不限制字段。

        返回值含义：
            list[dict]：数据库中已有的宠物档案事实。
        """

        ...

    def list_pet_profile_identities(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        """
        查询用户已有的宠物身份。

        参数含义：
            user_id：需要查询的用户标识。

        返回值含义：
            list[dict[str, str]]：宠物稳定标识和展示名称列表。
        """

        ...

    def upsert_pet_profile_fact(
        self,
        fact: PetProfileFact,
    ) -> dict:
        """
        新建或更新一条宠物档案事实。

        参数含义：
            fact：已经完成实体解析的正式宠物档案事实。

        返回值含义：
            dict：包含保存动作、事实主键和最终值的存储结果。
        """

        ...


class PetProfileService:
    """
    Pet Profile Service（宠物档案服务）。

    功能：
        查询已有宠物身份，将 LLM 批量抽取结果交给实体解析器，再把解析
        成功的正式事实逐条写入数据库，并返回统一保存结果。

    参数含义：
        store：实现 PetProfileStore（宠物档案存储契约）的存储对象。

    返回值含义：
        PetProfileService：可执行宠物档案解析与保存流程的服务对象。
    """

    def __init__(self, store: PetProfileStore) -> None:
        self.store = store

    def save_extraction_result(
        self,
        *,
        user_id: str,
        extraction_result: PetProfileExtractionResult,
        observed_at: datetime,
        selected_pet_key: str | None = None,
        selected_pet_name: str | None = None,
        source: MemorySource = "conversation",
    ) -> PetProfileSaveResult:
        """
        解析并保存一轮宠物档案抽取结果。

        功能：
            先读取当前用户已有宠物档案，再解析每条候选事实所属的宠物，
            最后把身份明确的事实逐条写入数据库。单条写入失败时记录失败，
            但继续处理同批其他合法事实。

        参数含义：
            user_id：当前用户标识，用于用户数据隔离。
            extraction_result：LLM 生成并经过契约校验的批量抽取结果。
            observed_at：本轮档案事实的观测时间。
            selected_pet_key：上游已选宠物的稳定标识；没有时为空。
            selected_pet_name：上游已选宠物的展示名称；没有时为空。
            source：档案事实来源，默认是 conversation（对话）。

        返回值含义：
            PetProfileSaveResult：包含实体解析、逐条保存动作和数量统计。
        """

        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("宠物档案保存缺少 user_id（用户标识）")

        # 已有事实用于把“豆豆”或“它”绑定到数据库中的稳定宠物身份。
        existing_facts = self.store.get_pet_profile_facts(
            normalized_user_id
        )
        resolution = resolve_pet_profile_facts(
            user_id=normalized_user_id,
            candidates=extraction_result.facts,
            observed_at=observed_at,
            existing_facts=existing_facts,
            selected_pet_key=selected_pet_key,
            selected_pet_name=selected_pet_name,
            source=source,
        )

        # 每条事实独立记录保存结果，避免一条异常导致整批合法数据被丢弃。
        persistence_records: list[PetProfilePersistenceRecord] = []
        for fact in resolution.facts:
            persistence_records.append(
                self._persist_fact(fact)
            )

        action_counts = {
            action: sum(
                record.action == action
                for record in persistence_records
            )
            for action in (
                "created",
                "updated",
                "ignored_stale",
                "failed",
            )
        }
        logger.info(
            "宠物档案保存完成: "
            f"user_id={normalized_user_id}, "
            f"resolved={len(resolution.facts)}, "
            f"unresolved={len(resolution.unresolved_facts)}, "
            f"created={action_counts['created']}, "
            f"updated={action_counts['updated']}, "
            f"ignored_stale={action_counts['ignored_stale']}, "
            f"failed={action_counts['failed']}"
        )

        return PetProfileSaveResult(
            resolution=resolution,
            persistence_records=persistence_records,
            created_count=action_counts["created"],
            updated_count=action_counts["updated"],
            ignored_stale_count=action_counts["ignored_stale"],
            failed_count=action_counts["failed"],
            reason=(
                f"解析成功 {len(resolution.facts)} 条，"
                f"对象不明确 {len(resolution.unresolved_facts)} 条，"
                f"数据库失败 {action_counts['failed']} 条。"
            ),
        )

    def recall_profile(
        self,
        *,
        user_id: str,
        active_pet_key: str | None = None,
        active_pet_name: str | None = None,
        selected_attributes: list[PetProfileAttribute] | None = None,
    ) -> PetProfileRecallResult:
        """
        召回当前问题所指宠物的结构化档案。

        功能：
            优先使用上游恢复或确认的 active_pet_key（当前宠物稳定标识）。
            没有当前标识时，如果用户只有一只已建档宠物，则自动选择；
            如果存在多只宠物，则返回 ambiguous（对象不明确）而不猜测。

        参数含义：
            user_id：当前用户标识，用于隔离不同用户的数据。
            active_pet_key：当前会话已经确认的宠物稳定标识。
            active_pet_name：当前会话已经确认的宠物展示名称。
            selected_attributes：本轮经过访问策略允许从数据库读取的字段；
            为空列表时不读取档案事实，None 保留旧调用方的全字段兼容行为。

        返回值含义：
            PetProfileRecallResult：宠物选择依据和结构化档案事实。
        """

        # 用户标识是数据库隔离边界，空值不能继续查询。
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("宠物档案召回缺少 user_id（用户标识）")

        # 空列表代表访问策略明确不允许读取任何字段，与 None 的兼容语义不同。
        if selected_attributes == []:
            return PetProfileRecallResult(
                status="empty",
                reason="本轮没有经过访问策略允许读取的宠物档案字段。",
            )

        # 上游已经确认的宠物标识优先级最高，不再重新猜测当前宠物。
        resolved_pet_key = str(active_pet_key or "").strip()
        resolved_pet_name = str(active_pet_name or "").strip()
        selection_source = "active_pet"

        if not resolved_pet_key:
            # 这里只查询宠物身份，不提前读取所有宠物的完整档案。
            identities = self.store.list_pet_profile_identities(
                normalized_user_id
            )
            if not identities:
                return PetProfileRecallResult(
                    status="empty",
                    reason="当前用户还没有可用的结构化宠物档案。",
                )
            if len(identities) > 1:
                return PetProfileRecallResult(
                    status="ambiguous",
                    reason=(
                        "当前用户存在多只宠物，但本轮没有明确当前宠物，"
                        "因此未自动注入任何宠物档案。"
                    ),
                )

            # 用户只有一只宠物时，可以确定当前问题只能指向这只宠物。
            identity = identities[0]
            resolved_pet_key = str(identity.get("pet_key") or "").strip()
            resolved_pet_name = str(identity.get("pet_name") or "").strip()
            selection_source = "single_pet_fallback"

        # 定位宠物只依赖 user_id + pet_key；年龄、体重等是返回属性，不是条件。
        if selected_attributes is None:
            # 旧调用方没有启用字段访问策略时，继续兼容原存储器方法签名。
            raw_facts = self.store.get_pet_profile_facts(
                normalized_user_id,
                resolved_pet_key,
            )
        else:
            raw_facts = self.store.get_pet_profile_facts(
                normalized_user_id,
                resolved_pet_key,
                attributes=selected_attributes,
            )
        if not raw_facts:
            return PetProfileRecallResult(
                status="empty",
                pet_key=resolved_pet_key,
                pet_name=resolved_pet_name,
                selection_source=selection_source,
                reason="已经确定当前宠物，但数据库中没有可用档案事实。",
            )

        # 只接受契约白名单中的属性，避免数据库异常字段进入 LLM 上下文。
        facts: dict[str, str] = {}
        for raw_fact in raw_facts:
            attribute = str(raw_fact.get("attribute") or "").strip()
            value = str(raw_fact.get("value") or "").strip()
            if not attribute or not value:
                continue
            try:
                validated_fact = PetProfileFact.model_validate(raw_fact)
            except Exception:
                logger.warning(
                    "忽略不符合宠物档案契约的数据库记录: pet_key=%s, attribute=%s",
                    resolved_pet_key,
                    attribute,
                )
                continue
            facts[validated_fact.attribute] = validated_fact.value
            if not resolved_pet_name and validated_fact.pet_name:
                resolved_pet_name = validated_fact.pet_name

        if not facts:
            return PetProfileRecallResult(
                status="empty",
                pet_key=resolved_pet_key,
                pet_name=resolved_pet_name,
                selection_source=selection_source,
                reason="宠物档案记录存在，但没有通过契约校验的事实。",
            )

        return PetProfileRecallResult(
            status="applied",
            pet_key=resolved_pet_key,
            pet_name=resolved_pet_name,
            selection_source=selection_source,
            facts=facts,
            selected_attributes=list(facts),
            reason=f"成功召回 {len(facts)} 条当前宠物档案事实。",
        )

    def _persist_fact(
        self,
        fact: PetProfileFact,
    ) -> PetProfilePersistenceRecord:
        """
        保存单条正式宠物档案事实并转换存储结果。

        参数含义：
            fact：已经绑定 user_id（用户标识）和 pet_key（宠物稳定标识）
            的正式宠物档案事实。

        返回值含义：
            PetProfilePersistenceRecord：成功、忽略旧数据或失败的保存记录。
        """

        try:
            raw_result = self.store.upsert_pet_profile_fact(fact)
            action = self._normalize_persistence_action(raw_result)
            return PetProfilePersistenceRecord(
                action=action,
                fact_id=self._read_fact_id(raw_result),
                pet_key=fact.pet_key,
                pet_name=fact.pet_name,
                attribute=fact.attribute,
                value=str(raw_result.get("value") or fact.value),
                observed_at=raw_result.get("observed_at") or fact.observed_at,
            )
        except Exception as exc:
            logger.exception(
                "宠物档案事实保存失败: pet_key=%s, attribute=%s",
                fact.pet_key,
                fact.attribute,
            )
            return PetProfilePersistenceRecord(
                action="failed",
                pet_key=fact.pet_key,
                pet_name=fact.pet_name,
                attribute=fact.attribute,
                value=fact.value,
                observed_at=fact.observed_at,
                error=str(exc),
            )

    @staticmethod
    def _normalize_persistence_action(
        raw_result: Mapping[str, Any],
    ) -> PetProfilePersistenceAction:
        """
        校验存储层返回的宠物档案保存动作。

        参数含义：
            raw_result：SQLite 存储器返回的原始保存结果。

        返回值含义：
            PetProfilePersistenceAction：created、updated 或 ignored_stale。
        """

        action = str(raw_result.get("action") or "").strip()
        if action not in {"created", "updated", "ignored_stale"}:
            raise ValueError(
                f"宠物档案存储返回未知 action（保存动作）: {action!r}"
            )
        return action  # type: ignore[return-value]

    @staticmethod
    def _read_fact_id(raw_result: Mapping[str, Any]) -> int | None:
        """
        从存储结果中读取宠物事实记录主键。

        参数含义：
            raw_result：SQLite 存储器返回的原始保存结果。

        返回值含义：
            int | None：存在 fact_id（事实记录主键）时返回整数，否则为空。
        """

        fact_id = raw_result.get("fact_id")
        return int(fact_id) if fact_id is not None else None
