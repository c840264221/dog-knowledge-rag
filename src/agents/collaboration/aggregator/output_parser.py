"""
结果聚合器输出解析与完整性校验。

功能：
    从 LLM 文本中提取 ResultAggregationDraft，并用代码确认所有成功步骤
    都进入最终回答，部分失败时也提供了限制说明。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.agents.collaboration.aggregator.schemas import (
    ResultAggregationDraft,
)


def extract_aggregation_output_text(raw_output: Any) -> str:
    """
    从 LLM 返回值中提取普通文本。

    功能：
        同时兼容 LangChain 消息对象的 content 属性和测试中的普通字符串。

    参数含义：
        raw_output:
            LLM Provider 返回的消息对象或字符串。

    返回值含义：
        str:
            去除首尾空白后的 LLM 输出文本。
    """

    return str(getattr(raw_output, "content", raw_output) or "").strip()


def parse_result_aggregation_output(
    *,
    raw_output: Any,
    expected_used_step_ids: list[str],
    requires_limitations: bool,
) -> ResultAggregationDraft:
    """
    把 LLM 输出解析并校验成可使用的聚合回答草稿。

    功能：
        寻找包含 final_answer 的 JSON 对象，校验字段后再确认 used_step_ids
        与所有成功步骤完全一致；部分失败时 limitations 不能为空。

    参数含义：
        raw_output:
            LLM 返回的原始消息或文本。
        expected_used_step_ids:
            调度结果中所有 completed 步骤编号。
        requires_limitations:
            是否存在失败或跳过步骤，需要明确说明回答限制。

    返回值含义：
        ResultAggregationDraft:
            JSON 结构和业务完整性都通过校验的聚合草稿。
    """

    output_text = extract_aggregation_output_text(raw_output)
    decoder = json.JSONDecoder()
    candidate_errors: list[str] = []

    for index, character in enumerate(output_text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(output_text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, Mapping):
            continue
        if "final_answer" not in candidate:
            continue
        try:
            draft = ResultAggregationDraft.model_validate(candidate)
            _validate_aggregation_draft(
                draft=draft,
                expected_used_step_ids=expected_used_step_ids,
                requires_limitations=requires_limitations,
            )
            return draft
        except (TypeError, ValueError) as exc:
            candidate_errors.append(str(exc))

    if candidate_errors:
        raise ValueError(
            "LLM 聚合回答没有通过校验: " + candidate_errors[-1]
        )
    raise ValueError("LLM 输出中没有找到包含 final_answer 的 JSON 对象")


def _validate_aggregation_draft(
    *,
    draft: ResultAggregationDraft,
    expected_used_step_ids: list[str],
    requires_limitations: bool,
) -> None:
    """
    检查 LLM 是否使用了全部成功结果并说明失败影响。

    功能：
        使用集合比较允许 LLM 改变步骤编号排列顺序，但不允许遗漏成功步骤、
        加入失败步骤，或在部分失败时省略限制说明。

    参数含义：
        draft:
            已通过 Pydantic 字段校验的聚合回答草稿。
        expected_used_step_ids:
            程序根据真实结果计算出的成功步骤编号。
        requires_limitations:
            当前任务是否存在失败或跳过步骤。

    返回值含义：
        None:
            检查通过时不返回数据，违反完整性要求时抛出 ValueError。
    """

    expected_ids = set(expected_used_step_ids)
    actual_ids = set(draft.used_step_ids)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        unknown_ids = sorted(actual_ids - expected_ids)
        raise ValueError(
            "used_step_ids 与成功步骤不一致，"
            f"遗漏={missing_ids}，不应使用={unknown_ids}"
        )
    if requires_limitations and not draft.limitations:
        raise ValueError("任务存在失败或跳过步骤，limitations 不能为空")
