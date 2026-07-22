"""
多 Agent 总编排器。

功能：
    按固定顺序串联 PlannerAgent、MultiAgentTaskScheduler 和
    ResultAggregator，为调用方提供一个完整的多 Agent 任务入口。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NoReturn
from uuid import uuid4

from src.agents.collaboration.aggregator import ResultAggregator
from src.agents.collaboration.contracts import (
    AgentTaskPlan,
    MultiAgentTaskResult,
)
from src.agents.collaboration.planner import PlannerAgent
from src.agents.collaboration.scheduler import (
    MultiAgentTaskCancellationToken,
    MultiAgentTaskScheduler,
)
from src.logger import logger


class MultiAgentOrchestrationError(RuntimeError):
    """
    表示多 Agent 任务在某个编排阶段发生异常。

    功能：
        保存失败阶段和原始异常，让调用方能够区分是规划、调度还是结果
        聚合失败，而不是只得到一条没有上下文的错误信息。

    参数含义：
        stage:
            发生异常的阶段，例如 planning、scheduling 或 aggregation。
        original_error:
            该阶段捕获到的原始异常。

    返回值含义：
        MultiAgentOrchestrationError:
            带失败阶段和原始异常的统一编排异常。
    """

    def __init__(self, stage: str, original_error: Exception) -> None:
        self.stage = stage
        self.original_error = original_error
        super().__init__(
            f"多 Agent 任务在 {stage} 阶段失败: {original_error}"
        )


class MultiAgentOrchestrator:
    """
    依次调用计划、调度和结果聚合组件。

    功能：
        PlannerAgent 负责拆任务，Scheduler 负责执行 Worker，Aggregator
        负责生成最终回答。总编排器只控制阶段顺序和提前返回条件。

    参数含义：
        planner:
            把用户目标拆成 AgentTaskPlan 的计划智能体。
        scheduler:
            按 depends_on 执行计划步骤的任务调度器。
        result_aggregator:
            把全部步骤结果整理成最终回答的结果聚合器。

    返回值含义：
        MultiAgentOrchestrator:
            可以通过 run 执行完整多 Agent 流程的总编排器。
    """

    def __init__(
        self,
        *,
        planner: PlannerAgent,
        scheduler: MultiAgentTaskScheduler,
        result_aggregator: ResultAggregator,
    ) -> None:
        if planner is None:
            raise ValueError("MultiAgentOrchestrator 必须提供 planner")
        if scheduler is None:
            raise ValueError("MultiAgentOrchestrator 必须提供 scheduler")
        if result_aggregator is None:
            raise ValueError(
                "MultiAgentOrchestrator 必须提供 result_aggregator"
            )
        self.planner = planner
        self.scheduler = scheduler
        self.result_aggregator = result_aggregator

    async def run(
        self,
        objective: str,
        *,
        context: Mapping[str, Any] | None = None,
        plan_id: str | None = None,
        multi_agent_task_id: str | None = None,
        cancellation_token: MultiAgentTaskCancellationToken | None = None,
    ) -> MultiAgentTaskResult:
        """
        执行一次完整的多 Agent 任务。

        功能：
            先生成计划，再执行步骤，最后聚合回答。计划等待用户输入或调度
            整体失败时提前返回，不会错误调用后面的结果聚合器。

        参数含义：
            objective:
                用户希望多 Agent 共同完成的原始目标。
            context:
                PlannerAgent 可以参考的用户资料、记忆和运行时补充信息。
            plan_id:
                可选计划编号；不传时由 PlannerAgent 自动生成。
            multi_agent_task_id:
                可选的整次多 Agent 任务编号；不传时由总编排器自动生成。
            cancellation_token:
                可选任务取消令牌，会传给 Scheduler 停止未完成步骤。

        返回值含义：
            MultiAgentTaskResult:
                完成时包含最终回答；等待输入、部分完成或失败时包含对应状态
                和已经产生的步骤结果。
        """

        normalized_objective = str(objective or "").strip()
        if not normalized_objective:
            raise ValueError("多 Agent 任务 objective 不能为空")
        resolved_task_id = str(
            multi_agent_task_id
            or f"multi_agent_task_{uuid4().hex}"
        ).strip()
        if not resolved_task_id:
            raise ValueError("multi_agent_task_id 不能为空")

        visited_stages: list[str] = []
        _log_orchestration_event(
            level="info",
            event="multi_agent_orchestration_started",
            payload={
                "multi_agent_task_id": resolved_task_id,
                "requested_plan_id": plan_id,
            },
        )

        try:
            plan = await self.planner.create_plan(
                normalized_objective,
                plan_id=plan_id,
                context=context,
            )
            plan = _attach_worker_runtime_context(
                plan=plan,
                context=context,
            )
            visited_stages.append("planning")
        except Exception as exc:
            _raise_orchestration_error(
                stage="planning",
                multi_agent_task_id=resolved_task_id,
                error=exc,
            )

        try:
            scheduled_result = await self.scheduler.execute(
                plan,
                collaboration_id=resolved_task_id,
                cancellation_token=cancellation_token,
            )
            visited_stages.append("scheduling")
        except Exception as exc:
            _raise_orchestration_error(
                stage="scheduling",
                multi_agent_task_id=resolved_task_id,
                error=exc,
            )

        if scheduled_result.status in {
            "awaiting_input",
            "failed",
            "cancelled",
        }:
            final_result = _attach_orchestration_metadata(
                task_result=scheduled_result,
                visited_stages=visited_stages,
            )
            _log_orchestration_finished(final_result)
            return final_result

        if scheduled_result.status not in {"running", "partial"}:
            _raise_orchestration_error(
                stage="scheduling",
                multi_agent_task_id=resolved_task_id,
                error=ValueError(
                    "调度器返回了无法继续处理的状态: "
                    f"{scheduled_result.status}"
                ),
            )

        try:
            aggregated_result = await self.result_aggregator.aggregate(
                scheduled_result
            )
            visited_stages.append("aggregation")
        except Exception as exc:
            _raise_orchestration_error(
                stage="aggregation",
                multi_agent_task_id=resolved_task_id,
                error=exc,
            )

        final_result = _attach_orchestration_metadata(
            task_result=aggregated_result,
            visited_stages=visited_stages,
        )
        _log_orchestration_finished(final_result)
        return final_result

    async def resume(
        self,
        task_result: MultiAgentTaskResult,
        *,
        user_inputs: Mapping[str, Any],
        cancellation_token: MultiAgentTaskCancellationToken | None = None,
    ) -> MultiAgentTaskResult:
        """
        根据用户回答恢复一份等待输入的多 Agent 任务。

        功能：
            不再调用 Planner 重新规划，而是让 Scheduler 保留已有结果并继续
            等待步骤；调度结束后按正常流程决定提前返回或调用 Aggregator。

        参数含义：
            task_result:
                上一次返回给用户的 awaiting_input 多 Agent 任务结果。
            user_inputs:
                等待步骤编号到用户回答的映射。
            cancellation_token:
                可选任务取消令牌，会传给 Scheduler 停止未完成步骤。

        返回值含义：
            MultiAgentTaskResult:
                恢复执行后的最新任务结果，可能再次等待、失败或生成最终回答。
        """

        orchestration_metadata = task_result.metadata.get(
            "orchestration",
            {},
        )
        visited_stages = list(
            orchestration_metadata.get("visited_stages", [])
        )
        try:
            scheduled_result = await self.scheduler.resume(
                task_result,
                user_inputs=user_inputs,
                cancellation_token=cancellation_token,
            )
            visited_stages.append("resume_scheduling")
        except Exception as exc:
            _raise_orchestration_error(
                stage="resume_scheduling",
                multi_agent_task_id=task_result.collaboration_id,
                error=exc,
            )

        if scheduled_result.status in {
            "awaiting_input",
            "failed",
            "cancelled",
        }:
            final_result = _attach_orchestration_metadata(
                task_result=scheduled_result,
                visited_stages=visited_stages,
            )
            _log_orchestration_finished(final_result)
            return final_result

        if scheduled_result.status not in {"running", "partial"}:
            _raise_orchestration_error(
                stage="resume_scheduling",
                multi_agent_task_id=task_result.collaboration_id,
                error=ValueError(
                    "恢复调度返回了无法继续处理的状态: "
                    f"{scheduled_result.status}"
                ),
            )

        try:
            aggregated_result = await self.result_aggregator.aggregate(
                scheduled_result
            )
            visited_stages.append("aggregation")
        except Exception as exc:
            _raise_orchestration_error(
                stage="aggregation",
                multi_agent_task_id=task_result.collaboration_id,
                error=exc,
            )

        final_result = _attach_orchestration_metadata(
            task_result=aggregated_result,
            visited_stages=visited_stages,
        )
        _log_orchestration_finished(final_result)
        return final_result


def _attach_worker_runtime_context(
    *,
    plan: AgentTaskPlan,
    context: Mapping[str, Any] | None,
) -> AgentTaskPlan:
    """
    把主图中的可信运行身份写入每个 Worker 步骤。

    功能：
        Planner 可以参考 context，但不能保证把 user_id 等运行身份原样写入
        每个步骤。本函数在计划通过校验后由程序统一补充身份，避免 Worker
        子图回退到 default_user，同时阻止 LLM 伪造这些字段。

    参数含义：
        plan:
            PlannerAgent 已经生成并通过校验的 AgentTaskPlan。
        context:
            主图传入的补充上下文，可能包含 user_id、session_id 和 trace_id。

    返回值含义：
        AgentTaskPlan:
            每个步骤 input_data 都已合并可信运行身份的新计划对象。
    """

    runtime_identity = {
        field_name: str((context or {}).get(field_name) or "").strip()
        for field_name in ("user_id", "session_id", "trace_id")
        if str((context or {}).get(field_name) or "").strip()
    }
    if not runtime_identity:
        return plan

    plan_data = plan.model_dump(mode="python")
    for step_data in plan_data["steps"]:
        step_data["input_data"] = {
            **step_data["input_data"],
            **runtime_identity,
        }
    return type(plan).model_validate(plan_data)


def _attach_orchestration_metadata(
    *,
    task_result: MultiAgentTaskResult,
    visited_stages: list[str],
) -> MultiAgentTaskResult:
    """
    给任务结果补充实际经过的编排阶段。

    功能：
        复制任务结果，在 metadata 中记录总编排器名称和已经走过的阶段，
        不直接修改 Planner、Scheduler 或 Aggregator 返回的原对象。

    参数含义：
        task_result:
            当前阶段产生的多 Agent 任务结果。
        visited_stages:
            本次任务实际完成的阶段名称列表。

    返回值含义：
        MultiAgentTaskResult:
            已附加编排记录并重新通过 Pydantic 校验的新结果。
    """

    result_data = task_result.model_dump(mode="python")
    result_data["metadata"] = {
        **task_result.metadata,
        "orchestration": {
            "orchestrator": "MultiAgentOrchestrator",
            "visited_stages": list(visited_stages),
        },
    }
    return MultiAgentTaskResult.model_validate(result_data)


def _raise_orchestration_error(
    *,
    stage: str,
    multi_agent_task_id: str,
    error: Exception,
) -> NoReturn:
    """
    记录失败阶段并抛出统一编排异常。

    功能：
        输出一条简短结构化错误日志，再把原始异常包装成带 stage 的
        MultiAgentOrchestrationError，方便上层统一处理。

    参数含义：
        stage:
            planning、scheduling 或 aggregation 失败阶段。
        multi_agent_task_id:
            发生失败的整次多 Agent 任务编号。
        error:
            当前阶段捕获到的原始异常。

    返回值含义：
        NoReturn:
            本函数总会抛出异常，不会正常返回。
    """

    _log_orchestration_event(
        level="error",
        event="multi_agent_orchestration_failed",
        payload={
            "multi_agent_task_id": multi_agent_task_id,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )
    raise MultiAgentOrchestrationError(stage, error) from error


def _log_orchestration_finished(
    task_result: MultiAgentTaskResult,
) -> None:
    """
    记录多 Agent 总编排结束事件。

    功能：
        只记录任务编号、计划编号、最终状态和经过的阶段，不输出最终回答或
        Worker 详细结果。

    参数含义：
        task_result:
            已附加总编排 metadata 的最终任务结果。

    返回值含义：
        None:
            只负责输出日志，不返回业务数据。
    """

    orchestration_metadata = task_result.metadata.get(
        "orchestration",
        {},
    )
    _log_orchestration_event(
        level=(
            "error"
            if task_result.status == "failed"
            else "info"
        ),
        event="multi_agent_orchestration_finished",
        payload={
            "multi_agent_task_id": task_result.collaboration_id,
            "plan_id": task_result.plan.plan_id,
            "final_status": task_result.status,
            "visited_stages": orchestration_metadata.get(
                "visited_stages",
                [],
            ),
        },
    )


def _log_orchestration_event(
    *,
    level: str,
    event: str,
    payload: Mapping[str, Any],
) -> None:
    """
    用四空格缩进 JSON 输出一条总编排关键事件日志。

    功能：
        统一输出开始、结束和阶段失败日志；日志构建失败不能打断多 Agent
        主流程。

    参数含义：
        level:
            日志级别，只使用 info 或 error。
        event:
            总编排开始、结束或失败事件名称。
        payload:
            任务编号、阶段和状态等少量关键字段。

    返回值含义：
        None:
            只负责输出日志，不返回业务数据。
    """

    try:
        formatted_payload = json.dumps(
            {
                "event": event,
                **dict(payload),
            },
            indent=4,
            ensure_ascii=False,
            default=str,
        )
        log_method = getattr(logger, level, logger.info)
        log_method(
            "MultiAgentOrchestrator event:\n"
            f"{formatted_payload}"
        )
    except Exception as exc:
        logger.info(f"多 Agent 总编排日志构建失败: {exc}")
