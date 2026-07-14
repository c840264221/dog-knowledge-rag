from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.evaluation.scenarios import (
    MemoryRecallScenarioRuntime,
    build_memory_recall_scenario_runtime,
)


ScenarioRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    MemoryRecallScenarioRuntime,
]

SUPPORTED_EXPECTED_FIELDS = {
    "recall_status",
    "memory_context_contains",
    "memory_context_not_contains",
    "candidate_count",
    "threshold_passed_count",
    "selected_count",
    "selected_memory_ids",
    "minimum_max_semantic_score",
    "vector_call_count",
    "store_call_count",
    "requested_user_id",
    "requested_status",
}


class MemoryRecallBehaviorEvaluator:
    """
    使用真实记忆召回节点和语义召回服务评估 Memory 行为。

    功能：
        执行真实 MemorySemanticRecallService（记忆语义召回服务）与
        memory_retrieve_node（记忆召回节点），检查召回状态、语义门槛、
        用户隔离、有效状态过滤、最终记忆文本和依赖调用轨迹。

    参数含义：
        scenario_runtime_builder:
            根据黄金用例创建确定性记忆评估场景的函数。

    返回值含义：
        MemoryRecallBehaviorEvaluator:
            可执行单条或批量记忆召回评估的对象。
    """

    def __init__(
        self,
        scenario_runtime_builder: ScenarioRuntimeBuilder = (
            build_memory_recall_scenario_runtime
        ),
    ) -> None:
        self.scenario_runtime_builder = scenario_runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条 Memory 召回行为评估用例。

        参数含义：
            eval_case:
                包含预设记忆和黄金期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                记忆召回摘要及逐项检查结果组成的统一评估结果。
        """

        started_at = time.perf_counter()
        try:
            self._validate_case(eval_case)
            runtime = self.scenario_runtime_builder(eval_case)
            result_state = await runtime.invoke()
            output = self._build_output(result_state, runtime)
            checks = self._build_checks(eval_case.expected, output)
            return AgentEvaluationResult(
                case_id=eval_case.case_id,
                category=eval_case.category,
                checks=checks,
                latency_ms=self._elapsed_ms(started_at),
                output=output,
                metadata={
                    "evaluator": type(self).__name__,
                    "node": "memory_retrieve_node",
                    "service": "MemorySemanticRecallService",
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
        按黄金集顺序批量执行 Memory 召回评估。

        参数含义：
            eval_cases:
                待执行的记忆召回评估用例列表。

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
        校验评估类别和支持的黄金期望字段。

        参数含义：
            eval_case:
                当前 Memory 召回评估用例。

        返回值含义：
            None。
        """

        if eval_case.category != "memory_recall_behavior":
            raise ValueError(
                "MemoryRecallBehaviorEvaluator 只接受 "
                "category=memory_recall_behavior"
            )
        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "Memory 召回评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        result_state: Mapping[str, Any],
        runtime: MemoryRecallScenarioRuntime,
    ) -> dict[str, Any]:
        """
        从最终状态和存储调用轨迹提取标准评估摘要。

        参数含义：
            result_state:
                真实记忆节点执行并合并后的状态。
            runtime:
                保存向量库、SQLite Store 和 Ranker 调用轨迹的场景环境。

        返回值含义：
            dict[str, Any]:
                可以与黄金期望逐项比较的记忆召回摘要。
        """

        recall_result = result_state.get("memory_recall_result", {})
        normalized_result = (
            dict(recall_result)
            if isinstance(recall_result, Mapping)
            else {}
        )
        vector_call = (
            runtime.vector_store.calls[0]
            if runtime.vector_store.calls
            else {}
        )
        vector_filter = vector_call.get("filter", {})
        return {
            "recall_status": normalized_result.get("status"),
            "memory_context": str(
                result_state.get("memory_context", "") or ""
            ),
            "candidate_count": normalized_result.get("candidate_count", 0),
            "threshold_passed_count": normalized_result.get(
                "threshold_passed_count",
                0,
            ),
            "selected_count": normalized_result.get("selected_count", 0),
            "selected_memory_ids": list(
                normalized_result.get("selected_memory_ids", [])
            ),
            "max_semantic_score": normalized_result.get("max_semantic_score"),
            "reason": str(normalized_result.get("reason", "") or ""),
            "vector_call_count": len(runtime.vector_store.calls),
            "store_call_count": len(runtime.store.calls),
            "requested_user_id": _extract_filter_eq(vector_filter, "user_id"),
            "requested_status": _extract_filter_eq(vector_filter, "status"),
        }

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将 Memory 黄金期望转换成逐项检查结果。

        参数含义：
            expected:
                黄金用例声明的期望字段。
            output:
                真实召回链路产生的评估摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                所有已声明期望字段对应的结构化检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            if field_name == "memory_context_contains":
                actual = str(output.get("memory_context", ""))
                passed = str(expected_value) in actual
            elif field_name == "memory_context_not_contains":
                actual = str(output.get("memory_context", ""))
                passed = str(expected_value) not in actual
            elif field_name == "minimum_max_semantic_score":
                actual = output.get("max_semantic_score")
                passed = (
                    isinstance(actual, (int, float))
                    and float(actual) >= float(expected_value)
                )
            else:
                actual = output.get(field_name)
                passed = actual == expected_value

            checks.append(
                self._build_check(
                    field_name,
                    expected_value,
                    actual,
                    passed,
                )
            )
        return checks

    def _build_check(
        self,
        check_name: str,
        expected: Any,
        actual: Any,
        passed: bool,
    ) -> EvaluationCheckResult:
        """
        创建单个统一记忆评估检查结果。

        参数含义：
            check_name:
                当前检查项名称。
            expected:
                黄金期望值。
            actual:
                真实召回结果值。
            passed:
                当前比较是否通过。

        返回值含义：
            EvaluationCheckResult:
                包含期望、实际值和中文说明的检查对象。
        """

        return EvaluationCheckResult(
            check_name=check_name,
            passed=passed,
            expected=expected,
            actual=actual,
            message=(
                f"{check_name} 符合预期。"
                if passed
                else f"{check_name} 不符合预期。"
            ),
        )

    def _elapsed_ms(self, started_at: float) -> float:
        """
        计算当前评估用例的毫秒耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(0.0, (time.perf_counter() - started_at) * 1000)


def _extract_filter_eq(filters: Any, field_name: str) -> Any:
    """
    从 Chroma $and / $eq 过滤条件中读取指定字段值。

    参数含义：
        filters:
            向量检索调用使用的过滤条件。
        field_name:
            需要提取的字段名。

    返回值含义：
        Any:
            找到的等值条件；没有找到时返回 None。
    """

    if not isinstance(filters, Mapping):
        return None
    direct_condition = filters.get(field_name)
    if isinstance(direct_condition, Mapping) and "$eq" in direct_condition:
        return direct_condition["$eq"]
    conditions = filters.get("$and", [])
    if isinstance(conditions, list):
        for condition in conditions:
            value = _extract_filter_eq(condition, field_name)
            if value is not None:
                return value
    return None
