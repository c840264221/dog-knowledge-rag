"""
PlannerAgent 任务计划提示词。

功能：
    把用户目标、可用 Agent 和输出 Schema 整理成明确提示词，要求 LLM
    只生成结构化任务计划，不直接执行任何步骤。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.agents.collaboration.contracts import AgentTaskPlan


def build_planner_prompt(
    *,
    objective: str,
    plan_id: str,
    available_agents: Mapping[str, str],
    context: Mapping[str, Any] | None = None,
) -> str:
    """
    构建 PlannerAgent 第一次生成任务计划时使用的提示词。

    功能：
        告诉 LLM 当前目标、允许使用的 Agent、已有上下文和标准输出结构。
        用户目标只作为待规划的数据，不允许覆盖提示词中的系统规则。

    参数含义：
        objective:
            需要拆解的用户目标。
        plan_id:
            由程序生成的计划编号，LLM 必须原样返回。
        available_agents:
            可以分配任务的 Agent 名称及其职责说明。
        context:
            PlannerAgent 可以参考的记忆、用户资料等补充数据。

    返回值含义：
        str:
            可以传给 LLM Provider 的完整文本提示词。
    """

    agents_text = json.dumps(
        dict(available_agents),
        ensure_ascii=False,
        indent=2,
    )
    context_text = json.dumps(
        dict(context or {}),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    schema_text = json.dumps(
        AgentTaskPlan.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    return f"""
你是 Dog Agent Framework 的 PlannerAgent（计划智能体）。

你的职责是把用户目标拆成一份可执行计划，不要回答用户问题，也不要执行任务。

必须遵守以下规则：
1. 只能从“可用 Agent”中选择 assigned_agent，禁止创造新 Agent。
2. 每个 step_id 必须唯一，depends_on 只能引用同一计划中的 step_id。
3. depends_on 表示真实执行依赖，不要只依靠 steps 数组的排列顺序。
4. 没有依赖关系的步骤可以并行，不要为了凑顺序添加虚假依赖。
5. 所有新步骤的 status 必须是 pending。
6. 信息充足时，计划 status 必须是 planned。
7. 缺少完成目标所必需的信息时，requires_user_input=true、status=awaiting_input，
   并在 clarification_prompt 中写出一个明确问题。
8. plan_id 必须原样返回为 {plan_id!r}。
9. objective 必须原样返回，不得改写。
10. 不要创建“汇总全部步骤”“整合最终方案”之类的最终汇总步骤。各 Worker
    只负责独立业务任务，全部步骤结果会由 ResultAggregator 统一生成最终回答。
11. input_data 只保存当前步骤的业务输入，不要写入 intent、route_decision、
    answer_strategy、current_agent、next_agent、strategy 等 Agent 内部控制字段。
12. 只输出一个 JSON 对象，不要输出 Markdown、解释或代码块。

可用 Agent：
{agents_text}

补充上下文：
{context_text}

用户目标开始：
{objective}
用户目标结束。

输出必须符合以下 JSON Schema：
{schema_text}
""".strip()


def build_planner_repair_prompt(
    *,
    original_prompt: str,
    previous_output: str,
    validation_error: str,
) -> str:
    """
    构建 PlannerAgent 修复错误结构化输出时使用的提示词。

    功能：
        保留原始计划要求，并把上一次输出和校验错误反馈给 LLM，让它只
        修复 JSON 结构或字段内容，不改变原始用户目标和可用 Agent 范围。

    参数含义：
        original_prompt:
            第一次生成计划时使用的完整提示词。
        previous_output:
            上一次 LLM 返回但未通过校验的文本。
        validation_error:
            Pydantic 或业务规则给出的失败原因。

    返回值含义：
        str:
            可以再次交给 LLM 的修复提示词。
    """

    return f"""
{original_prompt}

上一次输出没有通过程序校验。
上一次错误：{validation_error}

上一次输出开始：
{previous_output[:4000]}
上一次输出结束。

请根据错误修复输出，并严格遵守上方“输出必须符合以下 JSON Schema”中的
完整字段结构。仍然只返回一个 JSON 对象，不要输出解释、Markdown 或代码块。
""".strip()
