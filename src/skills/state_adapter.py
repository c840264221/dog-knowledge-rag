"""Skill 运行结果到主图状态的转换适配器。"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Mapping

from src.skills.schemas import SkillRuntimeResult


def build_skill_enhanced_question(
    *,
    question: str,
    skill_inputs: Mapping[str, Any],
    skill_context: str,
    execution_mode: str = "standard",
    ignored_input_ids: list[str] | None = None,
) -> str:
    """
    把已经准备完成的 Skill 输入和说明追加到业务问题。

    功能：
        恢复原有 Skill 执行方式，让下游 Agent 的答案生成过程继续看到完整
        Skill 说明。RAG 不读取这个增强问题，而是读取单独保存的
        retrieval_question。

    参数含义：
        question:
            不包含 Skill 说明的业务问题。
        skill_inputs:
            已经通过必需字段检查的结构化 Skill 输入。
        skill_context:
            SkillLoader 渲染出的完整技能执行说明。
        execution_mode:
            当前使用完整执行还是简化执行模式。
        ignored_input_ids:
            用户已经同意在简化执行中忽略的输入编号。

    返回值含义：
        str:
            包含业务问题、已校验输入和 Skill 说明的完整 Agent 问题。
    """

    normalized_question = str(question or "").strip()
    normalized_context = str(skill_context or "").strip()
    if not normalized_context:
        return normalized_question

    skill_input_text = json.dumps(
        dict(skill_inputs),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    execution_note = ""
    normalized_ignored_input_ids = [
        str(input_id).strip()
        for input_id in (ignored_input_ids or [])
        if str(input_id).strip()
    ]
    if execution_mode == "degraded":
        ignored_text = "、".join(normalized_ignored_input_ids) or "无"
        execution_note = (
            "\n\n本次 Skill 使用简化执行模式。\n"
            f"用户已同意忽略的缺失输入：{ignored_text}。\n"
            "生成最终答案时必须明确说明缺少了哪些资料、因此采用了哪些"
            "保守假设或通用方案，不得编造缺失信息。"
        )
    return (
        f"{normalized_question}\n\n"
        "以下是已经校验通过的 Skill 输入：\n"
        f"{skill_input_text}\n\n"
        "以下是当前步骤必须遵守的 Skill 执行说明：\n"
        f"{normalized_context}"
        f"{execution_note}"
    ).strip()


def build_skill_state_update(
    result: SkillRuntimeResult,
) -> dict[str, Any]:
    """
    把 SkillRuntime 标准结果转换成主图局部状态。

    功能：
        将 Pydantic 结果转换成适合 SQLite Checkpoint 保存的普通字典，并提取
        下一轮恢复所需的技能编号、已合并输入、提示和完整技能上下文。

    参数含义：
        result:
            SkillRuntime 返回的本轮标准准备结果。

    返回值含义：
        dict[str, Any]:
            可以由 LangGraph 合并进 DogState 的局部状态更新。
    """

    # model_dump 把内部 Pydantic 对象递归转换成 checkpoint 友好的普通数据。
    runtime_result_data = result.model_dump(mode="python")

    # 没有选中技能时使用空字符串，避免在状态中保存 Python 的 None。
    selected_skill_id = str(
        result.selection.selected_skill_id or ""
    )

    # merged_inputs 包含历史输入与本轮新输入，是下一轮恢复时需要继续保留的数据。
    merged_inputs = (
        dict(result.extraction.merged_inputs)
        if result.extraction is not None
        else {}
    )

    # 只有 awaiting_input 才存在面向用户的补充资料提示。
    pending_prompt = (
        result.input_check.clarification_prompt
        if result.status == "awaiting_input"
        and result.input_check is not None
        else ""
    )

    update: dict[str, Any] = {
        "skill_runtime_result": runtime_result_data,
        "skill_selected_id": selected_skill_id,
        "skill_inputs": merged_inputs,
        "skill_status": result.status,
        "skill_pending_prompt": pending_prompt,
        "skill_context": result.skill_context,
    }
    if result.status == "awaiting_input":
        update.update(
            {
                "pending_prompt": pending_prompt,
                "waiting_user_input": True,
                "has_asked_user": True,
                "final_answer": pending_prompt,
            }
        )
    elif result.status == "ready":
        update.update(
            {
                "pending_prompt": "",
                "waiting_user_input": False,
                "has_asked_user": False,
            }
        )
    return update
