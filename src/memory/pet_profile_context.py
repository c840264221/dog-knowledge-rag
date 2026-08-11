"""把结构化宠物档案转换成受控的 LLM 上下文文本。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PET_PROFILE_ATTRIBUTE_LABELS = {
    "breed": "品种",
    "birth_date": "出生日期",
    "age_years": "年龄",
    "weight_kg": "体重",
    "sex": "性别",
    "neutered": "绝育情况",
    "health_condition": "健康状况",
    "allergy": "过敏情况",
    "diet_pattern": "饮食习惯",
    "activity_level": "活动水平",
    "training_goal": "训练目标",
}


def format_pet_profile_context(
    recall_result: Any,
    *,
    maximum_characters: int = 2000,
) -> str:
    """
    格式化成功召回的宠物档案。

    功能：
        只接受 status=applied（已经应用）的结构化召回结果，按照固定字段
        白名单和稳定顺序生成 Prompt 文本，并限制最大字符数。

    参数含义：
        recall_result：DogState 中的 pet_profile_recall_result（宠物档案
        召回结果），通常是普通字典。
        maximum_characters：允许进入 Prompt 的最大字符数。

    返回值含义：
        str：可注入回答 Prompt 的宠物档案文本；没有可用档案时返回空串。
    """

    if not isinstance(recall_result, Mapping):
        return ""
    if str(recall_result.get("status") or "").strip() != "applied":
        return ""

    # facts 是已经通过召回服务校验的结构化属性和值。
    facts = recall_result.get("facts")
    if not isinstance(facts, Mapping) or not facts:
        return ""

    # 先写宠物名称，再按固定白名单顺序写事实，保证输出稳定且便于测试。
    lines: list[str] = []
    pet_name = str(recall_result.get("pet_name") or "").strip()
    if pet_name:
        lines.append(f"- 宠物名称：{pet_name}")

    for attribute, label in PET_PROFILE_ATTRIBUTE_LABELS.items():
        value = str(facts.get(attribute) or "").strip()
        if value:
            lines.append(f"- {label}：{value}")

    if not lines:
        return ""

    # 长度限制是最后一道防线，避免异常数据无限扩大 LLM 上下文。
    return "\n".join(lines)[:maximum_characters]
