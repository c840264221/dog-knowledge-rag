from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.evaluation.scenarios.multi_agent_scenario_runtime import (
    MultiAgentScenarioRuntime,
    build_multi_agent_scenario_runtime,
)


ScenarioRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    MultiAgentScenarioRuntime,
]

SUPPORTED_EXPECTED_FIELDS = {
    "task_status",
    "plan_status",
    "step_statuses",
    "ready_batches",
    "worker_call_counts",
    "worker_call_sequence",
    "completed_step_ids",
    "failed_step_ids",
    "skipped_step_ids",
    "awaiting_step_ids",
    "resume_count",
    "cancellation_requested",
    "awaiting_result_aggregation",
}


class MultiAgentBehaviorEvaluator:
    """
    使用真实 Scheduler 评估多 Agent 调度和韧性行为。

    功能：
        执行确定性任务计划，检查依赖批次、步骤状态、重试次数、等待恢复和
        取消结果，并复用项目统一 AgentEvaluationResult 输出评估成绩。

    参数含义：
        scenario_runtime_builder:
            根据评估用例构建真实 Scheduler 场景环境的函数。

    返回值含义：
        MultiAgentBehaviorEvaluator:
            可执行单条或批量多 Agent 行为评估的对象。
    """

    def __init__(
        self,
        scenario_runtime_builder: ScenarioRuntimeBuilder = (
            build_multi_agent_scenario_runtime
        ),
    ) -> None:
        """
        初始化多 Agent 行为评估器。

        参数含义：
            scenario_runtime_builder:
                构建确定性多 Agent 场景运行环境的函数。

        返回值含义：
            None。
        """

        self.scenario_runtime_builder = scenario_runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条多 Agent 行为评估用例。

        参数含义：
            eval_case:
                包含任务计划、Worker 行为和黄金期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                调度输出摘要和逐项检查结果组成的统一评估成绩。
        """

        started_at = time.perf_counter()
        try:
            self._validate_case(eval_case)
            runtime = self.scenario_runtime_builder(eval_case)
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
                    "scheduler": "MultiAgentTaskScheduler",
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
        按黄金集顺序批量执行多 Agent 行为评估。

        参数含义：
            eval_cases:
                待执行的多 Agent 行为评估用例。

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
        校验用例类别和多 Agent 评估支持的期望字段。

        参数含义：
            eval_case:
                当前准备执行的统一评估用例。

        返回值含义：
            None:
                类别和字段合法时不返回数据，否则抛出 ValueError。
        """

        if eval_case.category != "multi_agent_behavior":
            raise ValueError(
                "MultiAgentBehaviorEvaluator 只接受 "
                "category=multi_agent_behavior"
            )
        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "多 Agent 行为评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        *,
        task_result: Any,
        runtime: MultiAgentScenarioRuntime,
    ) -> dict[str, Any]:
        """
        从标准任务结果和 Worker 轨迹提取评估摘要。

        参数含义：
            task_result:
                真实 Scheduler 返回的 MultiAgentTaskResult。
            runtime:
                保存确定性 Worker 调用轨迹的场景运行环境。

        返回值含义：
            dict[str, Any]:
                可与黄金期望逐项比较的多 Agent 行为摘要。
        """

        step_statuses = {
            result.step_id: result.status
            for result in task_result.task_results
        }
        return {
            "task_status": task_result.status,
            "plan_status": task_result.plan.status,
            "step_statuses": step_statuses,
            "ready_batches": [
                list(batch)
                for batch in task_result.metadata.get(
                    "ready_batches",
                    [],
                )
            ],
            "worker_call_counts": dict(runtime.worker.call_counts),
            "worker_call_sequence": [
                call["step_id"]
                for call in runtime.worker.calls
            ],
            "completed_step_ids": [
                step_id
                for step_id, status in step_statuses.items()
                if status == "completed"
            ],
            "failed_step_ids": [
                step_id
                for step_id, status in step_statuses.items()
                if status == "failed"
            ],
            "skipped_step_ids": [
                step_id
                for step_id, status in step_statuses.items()
                if status == "skipped"
            ],
            "awaiting_step_ids": [
                step_id
                for step_id, status in step_statuses.items()
                if status == "awaiting_input"
            ],
            "resume_count": int(
                task_result.metadata.get("resume_count", 0)
            ),
            "cancellation_requested": bool(
                task_result.metadata.get(
                    "cancellation_requested",
                    False,
                )
            ),
            "awaiting_result_aggregation": bool(
                task_result.metadata.get(
                    "awaiting_result_aggregation",
                    False,
                )
            ),
        }

    def _build_checks(
        self,
        *,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将多 Agent 黄金期望转换成逐项结构化检查结果。

        参数含义：
            expected:
                黄金用例声明的任务状态、步骤状态和调用轨迹。
            output:
                真实 Scheduler 产生的行为摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                每个已声明期望字段对应的检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            actual_value = output.get(field_name)
            passed = actual_value == expected_value
            checks.append(
                EvaluationCheckResult(
                    check_name=field_name,
                    passed=passed,
                    expected=expected_value,
                    actual=actual_value,
                    message=(
                        f"{field_name} 符合预期。"
                        if passed
                        else f"{field_name} 不符合预期。"
                    ),
                )
            )
        return checks

    def _elapsed_ms(self, started_at: float) -> float:
        """
        计算当前多 Agent 评估用例耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(0.0, (time.perf_counter() - started_at) * 1000)
