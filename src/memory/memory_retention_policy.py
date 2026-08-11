"""Memory（记忆）长期保存资格审查策略。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.memory.memory_schema import (
    MemoryRetentionDecision,
    VALID_MEMORY_TYPES,
)


@dataclass(frozen=True)
class MemoryRetentionThreshold:
    """
    单类记忆的最低保存门槛。

    功能：
        保存某类记忆要求达到的最低可信度和最低重要度。

    参数含义：
        minimum_confidence：允许长期保存的最低可信度。
        minimum_importance：允许长期保存的最低重要度。

    返回值含义：
        MemoryRetentionThreshold：不可变的类型门槛配置。
    """

    minimum_confidence: float
    minimum_importance: float


DEFAULT_RETENTION_THRESHOLDS = {
    "favorite_dog": MemoryRetentionThreshold(0.80, 0.60),
    "dislike": MemoryRetentionThreshold(0.80, 0.60),
    "preference": MemoryRetentionThreshold(0.80, 0.60),
    "hobby": MemoryRetentionThreshold(0.75, 0.50),
    # 用户或宠物档案会长期影响回答，错误保存的代价更高，因此门槛更严格。
    "profile": MemoryRetentionThreshold(0.85, 0.70),
}


class MemoryRetentionPolicy:
    """
    审查 LLM 提出的候选记忆是否有资格长期保存。

    功能：
        使用确定性规则检查保存意愿、类型、内容、可信度和重要度。该对象
        不负责理解自然语言，也不负责写数据库，只负责给出保存决策。

    参数含义：
        thresholds：各记忆类型的最低保存门槛；不传时使用项目默认值。

    返回值含义：
        MemoryRetentionPolicy：可以重复审查候选记忆的策略对象。
    """

    def __init__(
        self,
        thresholds: Mapping[str, MemoryRetentionThreshold] | None = None,
    ) -> None:
        # 先复制默认门槛，再用调用方明确提供的类型覆盖对应配置。
        self.thresholds = dict(DEFAULT_RETENTION_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(thresholds)

    def evaluate(
        self,
        candidate: Mapping[str, Any],
    ) -> MemoryRetentionDecision:
        """
        审查一条候选记忆。

        参数含义：
            candidate：LLM 抽取并归一化后的候选记忆数据。

        返回值含义：
            MemoryRetentionDecision：包含是否接受、分数、门槛和原因的决策。
        """

        memory_type = str(candidate.get("memory_type") or "").strip()
        confidence = self._bounded_score(candidate.get("confidence"))
        importance = self._bounded_score(candidate.get("importance"))
        threshold = self.thresholds.get(
            memory_type,
            MemoryRetentionThreshold(1.0, 1.0),
        )

        if not bool(candidate.get("should_save", False)):
            return self._reject(
                memory_type=memory_type,
                confidence=confidence,
                importance=importance,
                threshold=threshold,
                reason=(
                    str(candidate.get("reason") or "").strip()
                    or "LLM 未建议将当前输入保存为长期记忆。"
                ),
            )

        if memory_type not in VALID_MEMORY_TYPES:
            return self._reject(
                memory_type=memory_type,
                confidence=confidence,
                importance=importance,
                threshold=threshold,
                reason=f"记忆类型 {memory_type!r} 不在允许的类型白名单中。",
            )

        if not str(candidate.get("content") or "").strip():
            return self._reject(
                memory_type=memory_type,
                confidence=confidence,
                importance=importance,
                threshold=threshold,
                reason="候选记忆内容为空，不能长期保存。",
            )

        if confidence < threshold.minimum_confidence:
            return self._reject(
                memory_type=memory_type,
                confidence=confidence,
                importance=importance,
                threshold=threshold,
                reason=(
                    f"可信度 {confidence:.2f} 低于 {memory_type} 类型要求的 "
                    f"{threshold.minimum_confidence:.2f}。"
                ),
            )

        if importance < threshold.minimum_importance:
            return self._reject(
                memory_type=memory_type,
                confidence=confidence,
                importance=importance,
                threshold=threshold,
                reason=(
                    f"重要度 {importance:.2f} 低于 {memory_type} 类型要求的 "
                    f"{threshold.minimum_importance:.2f}。"
                ),
            )

        return MemoryRetentionDecision(
            action="accepted",
            memory_type=memory_type,
            confidence=confidence,
            importance=importance,
            minimum_confidence=threshold.minimum_confidence,
            minimum_importance=threshold.minimum_importance,
            reason="候选记忆通过确定性长期保存门槛。",
        )

    def _reject(
        self,
        *,
        memory_type: str,
        confidence: float,
        importance: float,
        threshold: MemoryRetentionThreshold,
        reason: str,
    ) -> MemoryRetentionDecision:
        """
        构建统一的拒绝决策。

        参数含义：
            memory_type：候选记忆类型。
            confidence：归一化后的可信度。
            importance：归一化后的重要度。
            threshold：当前类型使用的保存门槛。
            reason：拒绝长期保存的具体原因。

        返回值含义：
            MemoryRetentionDecision：action 固定为 rejected 的结构化结果。
        """

        return MemoryRetentionDecision(
            action="rejected",
            memory_type=memory_type,
            confidence=confidence,
            importance=importance,
            minimum_confidence=threshold.minimum_confidence,
            minimum_importance=threshold.minimum_importance,
            reason=reason,
        )

    @staticmethod
    def _bounded_score(value: Any) -> float:
        """
        把外部传入的分数限制在 0 到 1。

        参数含义：
            value：尚未确认类型和范围的原始分数。

        返回值含义：
            float：无法转换时为 0.0，否则为限制在 0 到 1 的分数。
        """

        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(score, 1.0))
