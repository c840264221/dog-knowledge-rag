from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.evaluation.scenarios.multi_agent_orchestration_scenario_runtime import (
    MultiAgentOrchestrationScenarioRuntime,
    build_multi_agent_orchestration_scenario_runtime,
)


OrchestrationRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    MultiAgentOrchestrationScenarioRuntime,
]

SUPPORTED_EXPECTED_FIELDS = {
    "task_status",
    "plan_status",
    "step_statuses",
    "visited_stages",
    "stage_statuses",
    "stage_latency_recorded",
    "planner_call_count",
    "aggregator_call_count",
    "worker_call_counts",
    "final_answer",
    "resume_count",
}


class MultiAgentOrchestrationEvaluator:
    """
    评估多 Agent 总编排器的跨阶段调用行为。

    功能：
        使用真实 Planner、Scheduler、Aggregator 和 Orchestrator 执行确定性
        场景，检查阶段顺序、提前返回、恢复执行和最终聚合是否符合黄金期望。

    参数含义：
        runtime_builder:
            根据单条黄金用例构建总编排评估工具箱的函数。

    返回值含义：
        MultiAgentOrchestrationEvaluator:
            可执行单条或批量总编排行为评估的对象。
    """

    def __init__(
        self,
        runtime_builder: OrchestrationRuntimeBuilder = (
            build_multi_agent_orchestration_scenario_runtime
        ),
    ) -> None:
        self.runtime_builder = runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行并检查一条总编排黄金用例。

        参数含义：
            eval_case:
                包含计划、Worker 行为、聚合响应和预期输出的评估用例。

        返回值含义：
            AgentEvaluationResult:
                总编排实际输出和逐字段检查组成的统一评估结果。
        """

        started_at = time.perf_counter()
        try:
            self._validate_case(eval_case)
            runtime = self.runtime_builder(eval_case)
            task_result = await runtime.run()
            output = self._build_output(
                task_result=task_result,
                runtime=runtime,
            )
            checks = self._build_checks(
                expected=eval_case.expected,
                output=output,
            )
            return AgentEvaluationResult(
                case_id=eval_case.case_id,
                category=eval_case.category,
                checks=checks,
                latency_ms=self._elapsed_ms(started_at),
                output=output,
                metadata={
                    "evaluator": type(self).__name__,
                    "orchestrator": "MultiAgentOrchestrator",
                    "external_dependencies": "deterministic",
                },
            )
        except Exception as exc:
            return AgentEvaluationResult(
                case_id=eval_case.case_id,
                category=eval_case.category,
                checks=[],
                latency_ms=self._elapsed_ms(started_at),
                error_message=str(exc),
                metadata={
                    "evaluator": type(self).__name__,
                },
            )

    async def evaluate_many(
        self,
        eval_cases: list[AgentEvaluationCase],
    ) -> list[AgentEvaluationResult]:
        """
        按黄金集顺序执行多条总编排评估用例。

        参数含义：
            eval_cases:
                待执行的总编排黄金用例列表。

        返回值含义：
            list[AgentEvaluationResult]:
                与输入顺序一致的统一评估结果列表。
        """

        results: list[AgentEvaluationResult] = []
        for eval_case in eval_cases:
            results.append(await self.evaluate_case(eval_case))
        return results

    def _validate_case(self, eval_case: AgentEvaluationCase) -> None:
        """
        校验评估类别和允许比较的期望字段。

        参数含义：
            eval_case:
                当前准备执行的总编排评估用例。

        返回值含义：
            None:
                校验通过时不返回数据；字段不合法时抛出 ValueError。
        """

        if eval_case.category != "multi_agent_orchestration":
            raise ValueError(
                "MultiAgentOrchestrationEvaluator 只接受 "
                "category=multi_agent_orchestration"
            )
        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "多 Agent 总编排评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        *,
        task_result: Any,
        runtime: MultiAgentOrchestrationScenarioRuntime,
    ) -> dict[str, Any]:
        """
        提取一条总编排任务的阶段和调用摘要。

        参数含义：
            task_result:
                真实总编排器返回的 MultiAgentTaskResult。
            runtime:
                保存 Planner、Worker 和 Aggregator 调用记录的评估工具箱。

        返回值含义：
            dict[str, Any]:
                可以与黄金 expected 逐项比较的实际输出。
        """

        orchestration_metadata = task_result.metadata.get(
            "orchestration",
            {},
        )
        stage_metrics = orchestration_metadata.get(
            "stage_metrics",
            [],
        )
        normalized_stage_metrics = [
            dict(metric)
            for metric in stage_metrics
            if isinstance(metric, dict)
        ]
        return {
            "task_status": task_result.status,
            "plan_status": task_result.plan.status,
            "step_statuses": {
                result.step_id: result.status
                for result in task_result.task_results
            },
            "visited_stages": list(
                orchestration_metadata.get("visited_stages", [])
            ),
            "stage_statuses": [
                {
                    "stage": metric.get("stage"),
                    "status": metric.get("status"),
                }
                for metric in normalized_stage_metrics
            ],
            "stage_latency_recorded": (
                len(normalized_stage_metrics) > 0
                and all(
                    isinstance(metric.get("latency_ms"), (int, float))
                    and not isinstance(metric.get("latency_ms"), bool)
                    and metric["latency_ms"] >= 0.0
                    for metric in normalized_stage_metrics
                )
                and isinstance(
                    orchestration_metadata.get("active_latency_ms"),
                    (int, float),
                )
                and orchestration_metadata.get(
                    "active_latency_ms",
                    -1.0,
                ) >= 0.0
            ),
            "planner_call_count": len(
                runtime.planning_provider.prompts
            ),
            "aggregator_call_count": len(
                runtime.aggregation_provider.prompts
            ),
            "worker_call_counts": dict(runtime.worker.call_counts),
            "final_answer": task_result.final_answer,
            "resume_count": int(
                task_result.metadata.get("resume_count", 0)
            ),
        }

    def _build_checks(
        self,
        *,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        比较黄金期望和总编排实际输出。

        参数含义：
            expected:
                当前用例声明的期望字段和值。
            output:
                真实总编排链路产生的字段和值。

        返回值含义：
            list[EvaluationCheckResult]:
                每个期望字段对应的一项结构化检查结果。
        """

        return [
            EvaluationCheckResult(
                check_name=field_name,
                passed=output.get(field_name) == expected_value,
                expected=expected_value,
                actual=output.get(field_name),
                message=(
                    f"{field_name} 符合预期。"
                    if output.get(field_name) == expected_value
                    else f"{field_name} 不符合预期。"
                ),
            )
            for field_name, expected_value in expected.items()
        ]

    def _elapsed_ms(self, started_at: float) -> float:
        """
        计算当前评估用例耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(0.0, (time.perf_counter() - started_at) * 1000)
