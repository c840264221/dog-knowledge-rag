"""
PlannerAgent 结构化输出解析与任务排序。

功能：
    把 LLM 文本转换成 AgentTaskPlan，并使用代码检查计划编号、Agent 白名单、
    初始状态和步骤数量。通过校验后，再按 depends_on 生成稳定展示顺序。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.agents.collaboration.contracts import AgentTaskPlan, AgentTaskStep


def extract_planner_output_text(raw_output: Any) -> str:
    """
    从 LLM 返回值中提取普通文本。

    功能：
        同时兼容 LangChain AIMessage 的 content 属性和测试中直接返回的字符串。

    参数含义：
        raw_output:
            LLM Provider 返回的消息对象或字符串。

    返回值含义：
        str:
            去除首尾空白后的 LLM 输出文本。
    """

    return str(getattr(raw_output, "content", raw_output) or "").strip()


def parse_planner_output(
    *,
    raw_output: Any,
    expected_plan_id: str,
    expected_objective: str,
    allowed_agent_names: set[str],
    maximum_steps: int,
) -> AgentTaskPlan:
    """
    把 LLM 输出解析并校验成正式任务计划。

    功能：
        从文本中寻找包含 plan_id 的 JSON 对象，交给 Pydantic 校验字段，
        再检查调用方固定的计划编号、用户目标、Agent 白名单和初始状态。

    参数含义：
        raw_output:
            LLM 返回的原始消息或文本。
        expected_plan_id:
            程序为本次计划生成的唯一编号。
        expected_objective:
            用户原始目标，LLM 不允许改写。
        allowed_agent_names:
            当前运行环境已注册的 Agent 名称。
        maximum_steps:
            一份计划最多允许包含的步骤数量。

    返回值含义：
        AgentTaskPlan:
            通过全部校验并按依赖关系排序的任务计划。
    """

    output_text = extract_planner_output_text(raw_output)
    decoder = json.JSONDecoder()
    candidate_errors: list[str] = []

    for index, character in enumerate(output_text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(output_text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, Mapping) or "plan_id" not in candidate:
            continue
        try:
            plan = AgentTaskPlan.model_validate(candidate)
            _validate_generated_plan(
                plan=plan,
                expected_plan_id=expected_plan_id,
                expected_objective=expected_objective,
                allowed_agent_names=allowed_agent_names,
                maximum_steps=maximum_steps,
            )
            return _order_plan_steps(plan)
        except (TypeError, ValueError) as exc:
            candidate_errors.append(str(exc))

    if candidate_errors:
        raise ValueError(
            "LLM 任务计划没有通过校验: " + candidate_errors[-1]
        )
    raise ValueError("LLM 输出中没有找到包含 plan_id 的 JSON 任务计划")


def _validate_generated_plan(
    *,
    plan: AgentTaskPlan,
    expected_plan_id: str,
    expected_objective: str,
    allowed_agent_names: set[str],
    maximum_steps: int,
) -> None:
    """
    检查 LLM 计划是否遵守本次调用的固定边界。

    功能：
        Pydantic 负责通用字段校验；本函数补充只能由调用方知道的规则，
        包括计划编号、原始目标、可用 Agent 和最大步骤数。

    参数含义：
        plan:
            已通过 AgentTaskPlan 基础校验的计划。
        expected_plan_id:
            调用方生成的计划编号。
        expected_objective:
            不允许 LLM 改写的用户目标。
        allowed_agent_names:
            当前允许分配任务的 Agent 名称集合。
        maximum_steps:
            当前 PlannerAgent 允许的最大步骤数。

    返回值含义：
        None:
            校验通过时不返回数据；违反边界时抛出 ValueError。
    """

    if plan.plan_id != expected_plan_id:
        raise ValueError("LLM 返回的 plan_id 与程序生成值不一致")
    if plan.objective != expected_objective:
        raise ValueError("LLM 改写了用户原始 objective")
    if len(plan.steps) > maximum_steps:
        raise ValueError(
            f"计划步骤数 {len(plan.steps)} 超过允许值 {maximum_steps}"
        )

    unknown_agents = sorted(
        {
            step.assigned_agent
            for step in plan.steps
            if step.assigned_agent not in allowed_agent_names
        }
    )
    if unknown_agents:
        raise ValueError(f"计划使用了未注册 Agent: {unknown_agents}")

    invalid_step_statuses = sorted(
        step.step_id
        for step in plan.steps
        if step.status != "pending"
    )
    if invalid_step_statuses:
        raise ValueError(
            "新计划中的步骤必须全部是 pending: "
            f"{invalid_step_statuses}"
        )

    expected_status = (
        "awaiting_input"
        if plan.requires_user_input
        else "planned"
    )
    if plan.status != expected_status:
        raise ValueError(
            "计划状态与 requires_user_input 不一致，"
            f"期望 {expected_status}，实际 {plan.status}"
        )


def _order_plan_steps(plan: AgentTaskPlan) -> AgentTaskPlan:
    """
    根据 depends_on 把任务步骤整理成稳定的拓扑顺序。

    功能：
        每轮选择所有前置步骤已经排好的任务。没有依赖关系的步骤保持 LLM
        原始相对顺序，因此既满足执行约束，也方便开发者阅读和复现。

    参数含义：
        plan:
            已确认不存在缺失依赖和循环依赖的任务计划。

    返回值含义：
        AgentTaskPlan:
            steps 已按依赖关系重新排列的新计划对象。
    """

    original_positions = {
        step.step_id: index
        for index, step in enumerate(plan.steps)
    }
    steps_by_id = {
        step.step_id: step
        for step in plan.steps
    }
    remaining_dependency_counts = {
        step.step_id: len(step.depends_on)
        for step in plan.steps
    }
    dependent_step_ids: dict[str, list[str]] = {
        step.step_id: []
        for step in plan.steps
    }
    for step in plan.steps:
        for dependency_id in step.depends_on:
            dependent_step_ids[dependency_id].append(step.step_id)

    ready_step_ids = [
        step.step_id
        for step in plan.steps
        if remaining_dependency_counts[step.step_id] == 0
    ]
    ordered_steps: list[AgentTaskStep] = []

    while ready_step_ids:
        ready_step_ids.sort(key=original_positions.__getitem__)
        current_step_id = ready_step_ids.pop(0)
        ordered_steps.append(steps_by_id[current_step_id])

        for dependent_id in dependent_step_ids[current_step_id]:
            remaining_dependency_counts[dependent_id] -= 1
            if remaining_dependency_counts[dependent_id] == 0:
                ready_step_ids.append(dependent_id)

    if len(ordered_steps) != len(plan.steps):
        raise ValueError("任务计划无法生成拓扑顺序")

    plan_data = plan.model_dump(mode="python")
    plan_data["steps"] = [
        step.model_dump(mode="python")
        for step in ordered_steps
    ]
    return AgentTaskPlan.model_validate(plan_data)
