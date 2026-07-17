"""
PlannerAgent 任务拆解服务。

功能：
    通过项目统一 LLM Provider 生成 AgentTaskPlan，并在结构化输出不合法时
    进行有限次数修复。该服务只负责生成计划，不执行 Worker Agent。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from src.agents.collaboration.contracts import AgentTaskPlan
from src.agents.collaboration.planner.output_parser import (
    extract_planner_output_text,
    parse_planner_output,
)
from src.agents.collaboration.planner.prompts import (
    build_planner_prompt,
    build_planner_repair_prompt,
)


class PlannerGenerationError(RuntimeError):
    """表示 PlannerAgent 多次尝试后仍无法生成合法任务计划。"""


class PlannerAgent:
    """
    使用 LLM 把复杂用户目标拆成结构化任务计划。

    功能：
        调用注入的 LLM Provider 生成计划，再通过 Pydantic 和业务规则做
        确定性校验。第一次输出不合法时，会反馈错误并要求 LLM 修复。

    参数含义：
        llm_provider:
            提供 main_llm 和 safe_ainvoke 的项目统一 LLM Provider。
        available_agents:
            当前允许 PlannerAgent 分配任务的 Agent 名称及职责。
        planning_llm:
            PlannerAgent 实际使用的模型对象。不传时默认读取
            llm_provider.main_llm；也可以传入备用模型或专用规划模型。
        maximum_steps:
            一份计划最多允许包含多少个步骤。
        maximum_plan_attempts:
            结构化计划最多生成或修复多少次。

    返回值含义：
        PlannerAgent:
            可以通过 create_plan 异步生成 AgentTaskPlan 的计划智能体。
    """

    def __init__(
        self,
        *,
        llm_provider: Any,
        available_agents: Mapping[str, str],
        planning_llm: Any | None = None,
        maximum_steps: int = 8,
        maximum_plan_attempts: int = 2,
    ) -> None:
        if llm_provider is None:
            raise ValueError("PlannerAgent 必须提供 llm_provider")
        if not available_agents:
            raise ValueError("PlannerAgent 必须至少注册一个可用 Agent")
        if maximum_steps < 1:
            raise ValueError("maximum_steps 必须大于 0")
        if maximum_plan_attempts < 1:
            raise ValueError("maximum_plan_attempts 必须大于 0")

        self.llm_provider = llm_provider
        self.available_agents = {
            str(name).strip(): str(description).strip()
            for name, description in available_agents.items()
            if str(name).strip()
        }
        if not self.available_agents:
            raise ValueError("可用 Agent 名称不能为空")
        self.maximum_steps = maximum_steps
        self.maximum_plan_attempts = maximum_plan_attempts
        self.planning_llm = planning_llm

    async def create_plan(
        self,
        objective: str,
        *,
        plan_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> AgentTaskPlan:
        """
        为一个复杂用户目标生成经过校验的任务计划。

        功能：
            构建计划提示词并调用主 LLM。若输出不是合法 AgentTaskPlan，
            下一次调用会附带校验错误要求修复；全部失败后抛出统一异常。

        参数含义：
            objective:
                需要拆解的用户原始目标。
            plan_id:
                可选计划编号；未提供时由程序生成，不交给 LLM 自由决定。
            context:
                可供规划参考的用户资料、记忆和运行时补充信息。

        返回值含义：
            AgentTaskPlan:
                通过 Schema、白名单和依赖顺序校验的正式任务计划。
        """

        normalized_objective = str(objective or "").strip()
        if not normalized_objective:
            raise ValueError("PlannerAgent 的 objective 不能为空")

        resolved_plan_id = str(
            plan_id or f"plan_{uuid4().hex}"
        ).strip()
        if not resolved_plan_id:
            raise ValueError("PlannerAgent 的 plan_id 不能为空")

        safe_ainvoke = getattr(
            self.llm_provider,
            "safe_ainvoke",
            None,
        )
        if not callable(safe_ainvoke):
            raise ValueError("llm_provider 缺少 safe_ainvoke 方法")
        resolved_llm = self.planning_llm
        if resolved_llm is None:
            resolved_llm = getattr(self.llm_provider, "main_llm", None)
        if resolved_llm is None:
            raise ValueError(
                "PlannerAgent 缺少 planning_llm，"
                "llm_provider 也没有提供 main_llm"
            )

        original_prompt = build_planner_prompt(
            objective=normalized_objective,
            plan_id=resolved_plan_id,
            available_agents=self.available_agents,
            context=context,
        )
        current_prompt = original_prompt
        previous_output = ""
        last_error: Exception | None = None

        for _ in range(self.maximum_plan_attempts):
            try:
                raw_output = await safe_ainvoke(
                    llm=resolved_llm,
                    prompt=current_prompt,
                    fallback_response=(
                        '{"planner_error":"LLM unavailable"}'
                    ),
                )
                previous_output = extract_planner_output_text(raw_output)
                return parse_planner_output(
                    raw_output=raw_output,
                    expected_plan_id=resolved_plan_id,
                    expected_objective=normalized_objective,
                    allowed_agent_names=set(self.available_agents),
                    maximum_steps=self.maximum_steps,
                )
            except Exception as exc:
                last_error = exc
                current_prompt = build_planner_repair_prompt(
                    original_prompt=original_prompt,
                    previous_output=previous_output,
                    validation_error=str(exc),
                )

        raise PlannerGenerationError(
            "PlannerAgent 无法生成合法任务计划: "
            f"{last_error}"
        ) from last_error
