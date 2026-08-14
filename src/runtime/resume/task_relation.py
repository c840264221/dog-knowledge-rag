"""等待任务与本轮用户输入之间的关系判断。"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TaskRelation = Literal[
    "resume",
    "new_task",
    "cancel",
    "ambiguous",
]


class TaskRelationDecision(BaseModel):
    """
    保存本轮输入与暂停任务之间的关系判断。

    功能：
        用统一结构表达继续旧任务、开始新任务、取消旧任务或仍无法确定，
        供后续 Tool、Skill 和多智能体恢复入口共同使用。

    参数含义：
        relation:
            本轮输入与暂停任务的关系。
        normalized_input:
            去掉显式业务前缀后的用户文本。
        confidence:
            当前判断置信度，范围为 0 到 1。
        reason:
            说明为什么得到当前判断，方便日志、测试和用户确认。
        source:
            判断来源。explicit 表示用户明确说明，rule 表示确定性规则，
            fallback 表示现有信息不足。
        selected_task_id:
            已唯一确定的等待任务编号；尚未确定时为空。
        candidate_task_ids:
            本轮可能匹配的等待任务编号，供用户选择和审计使用。
        requires_task_selection:
            是否必须先让用户选择任务，True 时执行层不得恢复任何业务任务。

    返回值含义：
        TaskRelationDecision:
            可写入日志、状态或评估结果的任务关系决策。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    relation: TaskRelation
    normalized_input: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    source: Literal["explicit", "rule", "fallback"]
    selected_task_id: str | None = None
    candidate_task_ids: list[str] = Field(default_factory=list)
    requires_task_selection: bool = False


_CANCEL_INPUTS = {
    "取消",
    "全部取消",
    "取消全部",
    "全部停止",
    "cancel all",
    "算了",
    "不继续了",
    "停止任务",
    "cancel",
    "stop",
}

_NEW_TASK_PREFIXES = (
    "新问题:",
    "新问题：",
    "换个问题:",
    "换个问题：",
    "开始新任务:",
    "开始新任务：",
)

_RESUME_PREFIXES = (
    "继续任务:",
    "继续任务：",
    "恢复任务:",
    "恢复任务：",
    "补充信息:",
    "补充信息：",
)

_CONFIRMATION_INPUTS = {
    "y",
    "yes",
    "是",
    "好的",
    "可以",
    "允许",
    "确认",
    "继续",
}

_NEW_TASK_STARTERS = (
    "请",
    "帮我",
    "查询",
    "查一下",
    "介绍",
    "推荐",
    "制定",
    "生成",
    "分析",
    "告诉我",
    "为什么",
    "怎么",
    "如何",
    "什么",
    "是否",
    "能否",
)

# 这些形式通常是在回答年龄、体重、犬种、确认选项或训练现状，而不是提新问题。
_ANSWER_SHAPE_PATTERNS = (
    re.compile(r"\d+(?:\.\d+)?\s*(?:岁|个月|公斤|千克|kg|斤)", re.IGNORECASE),
    re.compile(r"^(?:它|他|她|狗狗|我家狗狗|是一只|品种是|年龄是|体重是)"),
    re.compile(r"^(?:允许|同意|确认|继续)(?:\S|\s)+$"),
)


def classify_pending_task_relation(
    user_input: str,
) -> TaskRelationDecision:
    """
    判断用户本轮文字是补充旧任务还是开始新任务。

    功能：
        优先识别用户明确写出的取消、新任务和继续前缀；随后识别 JSON、
        确认词和常见资料回答；完整请求句会判断为新任务；剩余输入返回
        ambiguous，交给后续语义分类或人工确认，而不是冒险恢复旧任务。

    参数含义：
        user_input:
            用户在存在暂停任务时提交的本轮原始文字。

    返回值含义：
        TaskRelationDecision:
            任务关系、归一化文本、置信度、原因和判断来源。
    """

    # 用户传来的文字还未经过判断，先清理首尾空白并拒绝空输入。
    normalized_input = str(user_input or "").strip()
    if not normalized_input:
        raise ValueError("存在暂停任务时，本轮用户输入不能为空")

    if normalized_input.casefold() in _CANCEL_INPUTS:
        return TaskRelationDecision(
            relation="cancel",
            normalized_input=normalized_input,
            confidence=1.0,
            reason="用户使用了明确的取消表达。",
            source="explicit",
        )

    explicit_new_task = _strip_first_prefix(
        normalized_input,
        _NEW_TASK_PREFIXES,
    )
    if explicit_new_task is not None:
        return TaskRelationDecision(
            relation="new_task",
            normalized_input=explicit_new_task,
            confidence=1.0,
            reason="用户使用了明确的新任务前缀。",
            source="explicit",
        )

    explicit_resume = _strip_first_prefix(
        normalized_input,
        _RESUME_PREFIXES,
    )
    if explicit_resume is not None:
        return TaskRelationDecision(
            relation="resume",
            normalized_input=explicit_resume,
            confidence=1.0,
            reason="用户使用了明确的继续任务前缀。",
            source="explicit",
        )

    if _looks_like_json_object(normalized_input):
        return TaskRelationDecision(
            relation="resume",
            normalized_input=normalized_input,
            confidence=0.98,
            reason="用户提供了适合多个等待步骤使用的 JSON 对象。",
            source="rule",
        )

    if normalized_input.casefold() in _CONFIRMATION_INPUTS:
        return TaskRelationDecision(
            relation="resume",
            normalized_input=normalized_input,
            confidence=0.98,
            reason="用户输入是明确的确认或继续表达。",
            source="rule",
        )

    if any(
        pattern.search(normalized_input)
        for pattern in _ANSWER_SHAPE_PATTERNS
    ):
        return TaskRelationDecision(
            relation="resume",
            normalized_input=normalized_input,
            confidence=0.90,
            reason="用户输入符合年龄、体重、档案或现状补充的常见形式。",
            source="rule",
        )

    if (
        normalized_input.startswith(_NEW_TASK_STARTERS)
        or "？" in normalized_input
        or "?" in normalized_input
    ):
        return TaskRelationDecision(
            relation="new_task",
            normalized_input=normalized_input,
            confidence=0.88,
            reason="用户输入具有完整请求或提问的形式。",
            source="rule",
        )

    return TaskRelationDecision(
        relation="ambiguous",
        normalized_input=normalized_input,
        confidence=0.50,
        reason="仅凭确定性规则无法确认是在补充旧任务还是开始新任务。",
        source="fallback",
    )


def _strip_first_prefix(
    text: str,
    prefixes: tuple[str, ...],
) -> str | None:
    """
    删除命中的第一个业务前缀。

    参数含义：
        text:
            已清理首尾空白的用户文字。
        prefixes:
            允许识别的前缀集合。

    返回值含义：
        str | None:
            命中时返回前缀后面的非空文字；未命中时返回 None。
    """

    for prefix in prefixes:
        if text.startswith(prefix):
            remaining_text = text[len(prefix):].strip()
            if not remaining_text:
                raise ValueError(f"{prefix} 后面必须提供具体内容")
            return remaining_text
    return None


def _looks_like_json_object(text: str) -> bool:
    """
    判断用户输入是否为非空 JSON 对象。

    参数含义：
        text:
            待检查的用户文字。

    返回值含义：
        bool:
            可以解析为非空 JSON 对象时返回 True，否则返回 False。
    """

    try:
        parsed_value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(parsed_value, dict) and bool(parsed_value)
