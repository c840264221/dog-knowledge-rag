"""
多 Agent 结果聚合服务。

功能：
    调用项目统一 LLM Provider 把多个 Worker 结果整理成最终回答，并在
    输出缺少步骤或限制说明时进行有限次数修复。
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.collaboration.aggregator.output_parser import (
    extract_aggregation_output_text,
    parse_result_aggregation_output,
)
from src.agents.collaboration.aggregator.prompts import (
    build_result_aggregation_prompt,
    build_result_aggregation_repair_prompt,
)
from src.agents.collaboration.contracts import MultiAgentTaskResult
from src.logger import logger


class ResultAggregationError(RuntimeError):
    """表示多次尝试后仍无法生成合法最终回答。"""


class ResultAggregator:
    """
    使用 LLM 把多个 Worker 结果整理成最终回答。

    功能：
        只接收调度完成的 running 或 partial 结果，调用指定模型生成结构化
        草稿，再用代码检查成功步骤是否全部被使用。

    参数含义：
        llm_provider:
            提供 safe_ainvoke 和默认 main_llm 的项目统一 LLM Provider。
        aggregation_llm:
            聚合回答实际使用的模型；不传时使用 llm_provider.main_llm。
        maximum_aggregation_attempts:
            第一次生成加后续修复总共最多允许调用 LLM 多少次。

    返回值含义：
        ResultAggregator:
            可以通过 aggregate 异步生成最终回答的结果聚合器。
    """

    def __init__(
        self,
        *,
        llm_provider: Any,
        aggregation_llm: Any | None = None,
        maximum_aggregation_attempts: int = 2,
    ) -> None:
        if llm_provider is None:
            raise ValueError("ResultAggregator 必须提供 llm_provider")
        if maximum_aggregation_attempts < 1:
            raise ValueError(
                "maximum_aggregation_attempts 必须大于 0"
            )
        self.llm_provider = llm_provider
        self.aggregation_llm = aggregation_llm
        self.maximum_aggregation_attempts = maximum_aggregation_attempts

    async def aggregate(
        self,
        task_result: MultiAgentTaskResult,
    ) -> MultiAgentTaskResult:
        """
        根据全部步骤结果生成最终多 Agent 回答。

        功能：
            调用 LLM 生成 ResultAggregationDraft。输出不合法时把错误反馈给
            LLM 修复；成功后复制原任务结果并写入 final_answer 和聚合记录。

        参数含义：
            task_result:
                调度器已经执行完成、等待聚合的多 Agent 任务结果。

        返回值含义：
            MultiAgentTaskResult:
                已写入 final_answer 的新结果；完整任务变为 completed，存在
                允许失败步骤的任务保持 partial。
        """

        if task_result.status not in {"running", "partial"}:
            raise ValueError(
                "结果聚合器只接受 status=running 或 partial 的调度结果，"
                f"实际为 {task_result.status}"
            )
        _validate_aggregation_input(task_result)

        safe_ainvoke = getattr(
            self.llm_provider,
            "safe_ainvoke",
            None,
        )
        if not callable(safe_ainvoke):
            raise ValueError("llm_provider 缺少 safe_ainvoke 方法")
        resolved_llm = self.aggregation_llm
        if resolved_llm is None:
            resolved_llm = getattr(self.llm_provider, "main_llm", None)
        if resolved_llm is None:
            raise ValueError(
                "ResultAggregator 缺少 aggregation_llm，"
                "llm_provider 也没有提供 main_llm"
            )

        completed_step_ids = [
            result.step_id
            for result in task_result.task_results
            if result.status == "completed"
        ]
        incomplete_step_ids = [
            result.step_id
            for result in task_result.task_results
            if result.status in {"failed", "skipped"}
        ]
        original_prompt = build_result_aggregation_prompt(task_result)
        current_prompt = original_prompt
        previous_output = ""
        last_error: Exception | None = None

        _log_aggregation_event(
            level="info",
            event="result_aggregation_started",
            payload={
                "multi_agent_task_id": task_result.collaboration_id,
                "plan_id": task_result.plan.plan_id,
                "completed_step_ids": completed_step_ids,
                "incomplete_step_ids": incomplete_step_ids,
            },
        )

        for attempt_number in range(
            1,
            self.maximum_aggregation_attempts + 1,
        ):
            try:
                raw_output = await safe_ainvoke(
                    llm=resolved_llm,
                    prompt=current_prompt,
                    fallback_response=(
                        '{"aggregation_error":"LLM unavailable"}'
                    ),
                )
                previous_output = extract_aggregation_output_text(raw_output)
                draft = parse_result_aggregation_output(
                    raw_output=raw_output,
                    expected_used_step_ids=completed_step_ids,
                    requires_limitations=bool(incomplete_step_ids),
                )
                aggregated_result = _build_aggregated_task_result(
                    task_result=task_result,
                    final_answer=draft.final_answer,
                    used_step_ids=draft.used_step_ids,
                    limitations=draft.limitations,
                    attempt_count=attempt_number,
                )
                _log_aggregation_event(
                    level="info",
                    event="result_aggregation_completed",
                    payload={
                        "multi_agent_task_id": (
                            task_result.collaboration_id
                        ),
                        "plan_id": task_result.plan.plan_id,
                        "final_status": aggregated_result.status,
                        "attempt_count": attempt_number,
                    },
                )
                return aggregated_result
            except Exception as exc:
                last_error = exc
                _log_aggregation_event(
                    level="warning",
                    event="result_aggregation_attempt_failed",
                    payload={
                        "multi_agent_task_id": (
                            task_result.collaboration_id
                        ),
                        "plan_id": task_result.plan.plan_id,
                        "attempt_number": attempt_number,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                current_prompt = build_result_aggregation_repair_prompt(
                    original_prompt=original_prompt,
                    previous_output=previous_output,
                    validation_error=str(exc),
                )

        raise ResultAggregationError(
            "结果聚合器无法生成合法最终回答: "
            f"{last_error}"
        ) from last_error


def _validate_aggregation_input(
    task_result: MultiAgentTaskResult,
) -> None:
    """
    检查调度结果是否已经完整到可以生成最终回答。

    功能：
        确认计划中的每个步骤都有终态结果；running 必须全部成功，partial
        必须确实包含失败或跳过，避免状态标错或结果缺失时调用 LLM。

    参数含义：
        task_result:
            准备交给结果聚合器的多 Agent 调度结果。

    返回值含义：
        None:
            输入完整且状态一致时不返回数据，否则抛出 ValueError。
    """

    planned_step_ids = {
        step.step_id
        for step in task_result.plan.steps
    }
    result_step_ids = {
        result.step_id
        for result in task_result.task_results
    }
    missing_step_ids = sorted(planned_step_ids - result_step_ids)
    if missing_step_ids:
        raise ValueError(
            "结果聚合前仍缺少步骤结果: "
            f"{missing_step_ids}"
        )

    incomplete_step_ids = [
        result.step_id
        for result in task_result.task_results
        if result.status in {"failed", "skipped"}
    ]
    if task_result.status == "running" and incomplete_step_ids:
        raise ValueError(
            "status=running 的聚合输入不能包含失败或跳过步骤: "
            f"{incomplete_step_ids}"
        )
    if task_result.status == "partial" and not incomplete_step_ids:
        raise ValueError(
            "status=partial 的聚合输入必须包含失败或跳过步骤"
        )


def _build_aggregated_task_result(
    *,
    task_result: MultiAgentTaskResult,
    final_answer: str,
    used_step_ids: list[str],
    limitations: list[str],
    attempt_count: int,
) -> MultiAgentTaskResult:
    """
    把聚合草稿写入一份新的多 Agent 任务结果。

    功能：
        保留原计划、步骤结果和已有 metadata，只更新最终回答、最终状态和
        聚合记录，不直接修改调用方传入的原对象。

    参数含义：
        task_result:
            调度器生成的原始任务结果。
        final_answer:
            LLM 生成并通过校验的最终回答。
        used_step_ids:
            最终回答使用的成功步骤编号。
        limitations:
            最终回答需要说明的失败影响或数据限制。
        attempt_count:
            本次聚合成功前实际调用 LLM 的次数。

    返回值含义：
        MultiAgentTaskResult:
            已写入最终回答并重新通过 Pydantic 校验的新结果。
    """

    result_data = task_result.model_dump(mode="python")
    result_data["status"] = (
        "completed"
        if task_result.status == "running"
        else "partial"
    )
    result_data["final_answer"] = final_answer
    result_data["metadata"] = {
        **task_result.metadata,
        "result_aggregation": {
            "used_step_ids": used_step_ids,
            "limitations": limitations,
            "attempt_count": attempt_count,
        },
    }
    return MultiAgentTaskResult.model_validate(result_data)


def _log_aggregation_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
) -> None:
    """
    用四空格缩进 JSON 输出一条结果聚合关键事件日志。

    功能：
        只记录任务编号、步骤编号、尝试次数和错误摘要，不输出完整 Worker
        数据或最终回答；日志失败不能影响聚合主流程。

    参数含义：
        level:
            日志级别，只使用 info 或 warning。
        event:
            聚合开始、尝试失败或聚合完成等事件名称。
        payload:
            当前事件需要记录的少量关键字段。

    返回值含义：
        None:
            只负责输出日志，不返回业务数据。
    """

    try:
        formatted_payload = json.dumps(
            {
                "event": event,
                **payload,
            },
            indent=4,
            ensure_ascii=False,
            default=str,
        )
        log_method = getattr(logger, level, logger.info)
        log_method(
            "ResultAggregator event:\n"
            f"{formatted_payload}"
        )
    except Exception as exc:
        logger.info(f"结果聚合日志构建失败: {exc}")
