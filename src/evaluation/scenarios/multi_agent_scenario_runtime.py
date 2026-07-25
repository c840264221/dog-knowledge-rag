from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.agents.collaboration.contracts import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)
from src.agents.collaboration.scheduler import (
    MultiAgentTaskCancellationToken,
    MultiAgentTaskScheduler,
)
from src.evaluation.schemas import AgentEvaluationCase


class EvaluationMultiAgentWorker:
    """
    为多 Agent 行为评估提供确定性的 Worker 执行结果。

    功能：
        根据黄金用例为每个 step_id 依次返回 completed、failed、
        awaiting_input，或主动抛出 error，用来稳定验证调度、重试和恢复行为。

    参数含义：
        behaviors_by_step_id:
            步骤编号到预设执行结果序列的映射。

    返回值含义：
        EvaluationMultiAgentWorker:
            可以注册到真实 MultiAgentTaskScheduler 的异步 Worker。
    """

    def __init__(
        self,
        behaviors_by_step_id: Mapping[str, Any],
    ) -> None:
        """
        初始化确定性多 Agent Worker。

        参数含义：
            behaviors_by_step_id:
                每个步骤需要按调用次数产生的 outcomes 配置。

        返回值含义：
            None。
        """

        self.behaviors_by_step_id = {
            str(step_id): dict(behavior)
            for step_id, behavior in behaviors_by_step_id.items()
            if isinstance(behavior, Mapping)
        }
        self.call_counts: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        执行一次确定性 Worker 调用。

        参数含义：
            step:
                Scheduler 当前交给 Worker 的完整任务步骤。
            dependency_results:
                当前步骤已经获得的前置步骤结果。

        返回值含义：
            AgentTaskResult:
                黄金场景指定的完成、失败或等待输入结果；error 会抛出异常。
        """

        call_number = self.call_counts.get(step.step_id, 0) + 1
        self.call_counts[step.step_id] = call_number
        self.calls.append(
            {
                "step_id": step.step_id,
                "call_number": call_number,
                "dependency_step_ids": list(dependency_results),
                "is_resuming": bool(
                    step.input_data.get("multi_agent_is_resuming")
                ),
            }
        )

        behavior = self.behaviors_by_step_id.get(step.step_id, {})
        outcomes = behavior.get("outcomes", ["completed"])
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError(
                f"步骤 {step.step_id} 的 outcomes 必须是非空列表"
            )
        outcome_index = min(call_number - 1, len(outcomes) - 1)
        outcome = str(outcomes[outcome_index])

        if outcome == "error":
            raise RuntimeError(
                str(
                    behavior.get("error_message")
                    or f"步骤 {step.step_id} 的评估异常"
                )
            )
        if outcome == "failed":
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="failed",
                error_message=str(
                    behavior.get("error_message")
                    or f"步骤 {step.step_id} 评估失败"
                ),
                metadata={
                    "evaluation_worker": True,
                    "call_number": call_number,
                },
            )
        if outcome == "awaiting_input":
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                summary="评估步骤正在等待用户输入。",
                output={
                    "call_number": call_number,
                },
                requires_user_input=True,
                clarification_prompt=str(
                    behavior.get("clarification_prompt")
                    or f"请补充步骤 {step.step_id} 所需信息。"
                ),
                metadata={
                    "evaluation_worker": True,
                },
            )
        if outcome != "completed":
            raise ValueError(
                f"步骤 {step.step_id} 使用了不支持的 outcome: {outcome}"
            )

        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary=str(
                behavior.get("summary")
                or f"{step.title}评估执行完成。"
            ),
            output={
                "call_number": call_number,
                "dependency_step_ids": list(dependency_results),
                "resume_input": step.input_data.get(
                    "multi_agent_resume_input"
                ),
            },
            evidence_ids=[
                str(value)
                for value in behavior.get("evidence_ids", [])
            ],
            metadata={
                "evaluation_worker": True,
            },
        )


@dataclass
class MultiAgentScenarioRuntime:
    """
    保存一条多 Agent 行为评估场景的真实调度器和确定性依赖。

    参数含义：
        scheduler:
            使用生产调度逻辑的 MultiAgentTaskScheduler。
        plan:
            黄金用例声明并经过 Schema 校验的任务计划。
        worker:
            记录步骤调用次数和依赖输入的确定性 Worker。
        multi_agent_task_id:
            当前评估场景使用的稳定任务编号。
        cancellation_token:
            可选取消令牌；取消用例会在执行前打开信号。
        resume_user_inputs:
            等待输入用例恢复时使用的 step_id 到回答映射。

    返回值含义：
        MultiAgentScenarioRuntime:
            可执行真实 Scheduler 并读取确定性调用轨迹的场景环境。
    """

    scheduler: MultiAgentTaskScheduler
    plan: AgentTaskPlan
    worker: EvaluationMultiAgentWorker
    multi_agent_task_id: str
    cancellation_token: MultiAgentTaskCancellationToken | None
    resume_user_inputs: dict[str, Any]

    async def run(self) -> MultiAgentTaskResult:
        """
        执行当前多 Agent 黄金场景，并在需要时恢复等待步骤。

        参数含义：
            无。

        返回值含义：
            MultiAgentTaskResult:
                首次调度或补充用户输入后得到的最新标准任务结果。
        """

        result = await self.scheduler.execute(
            self.plan,
            collaboration_id=self.multi_agent_task_id,
            cancellation_token=self.cancellation_token,
        )
        if (
            result.status == "awaiting_input"
            and self.resume_user_inputs
        ):
            result = await self.scheduler.resume(
                result,
                user_inputs=self.resume_user_inputs,
                cancellation_token=self.cancellation_token,
            )
        return result


def build_multi_agent_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> MultiAgentScenarioRuntime:
    """
    根据黄金用例构建真实 Scheduler 使用的确定性多 Agent 场景。

    功能：
        从 input_state 读取计划、Worker 行为和调度设置，使用真实契约完成
        校验，再把外部不稳定 Agent 替换为可重复的评估 Worker。

    参数含义：
        eval_case:
            category 为 multi_agent_behavior 的统一评估用例。

    返回值含义：
        MultiAgentScenarioRuntime:
            包含真实 Scheduler、标准计划和确定性 Worker 的场景运行环境。
    """

    raw_plan = eval_case.input_state.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("多 Agent 行为评估 input_state 缺少 plan")
    plan = AgentTaskPlan.model_validate(dict(raw_plan))

    raw_behaviors = eval_case.input_state.get("worker_behaviors", {})
    if not isinstance(raw_behaviors, Mapping):
        raise ValueError("worker_behaviors 必须是步骤编号到行为的映射")
    worker = EvaluationMultiAgentWorker(raw_behaviors)

    scheduler_settings = eval_case.input_state.get(
        "scheduler_settings",
        {},
    )
    if not isinstance(scheduler_settings, Mapping):
        raise ValueError("scheduler_settings 必须是映射")

    workers = {
        step.assigned_agent: worker
        for step in plan.steps
    }
    scheduler = MultiAgentTaskScheduler(
        workers=workers,
        maximum_parallel_steps=int(
            scheduler_settings.get("maximum_parallel_steps", 4)
        ),
        maximum_step_attempts=int(
            scheduler_settings.get("maximum_step_attempts", 1)
        ),
        step_timeout_seconds=scheduler_settings.get(
            "step_timeout_seconds"
        ),
    )

    cancellation_token: MultiAgentTaskCancellationToken | None = None
    if bool(eval_case.input_state.get("pre_cancelled")):
        cancellation_token = MultiAgentTaskCancellationToken()
        cancellation_token.cancel()

    raw_resume_inputs = eval_case.input_state.get(
        "resume_user_inputs",
        {},
    )
    if not isinstance(raw_resume_inputs, Mapping):
        raise ValueError("resume_user_inputs 必须是步骤编号到回答的映射")

    return MultiAgentScenarioRuntime(
        scheduler=scheduler,
        plan=plan,
        worker=worker,
        multi_agent_task_id=f"evaluation_multi_agent_{eval_case.case_id}",
        cancellation_token=cancellation_token,
        resume_user_inputs=dict(raw_resume_inputs),
    )
