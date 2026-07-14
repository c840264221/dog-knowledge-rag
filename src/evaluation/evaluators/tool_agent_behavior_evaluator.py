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
    ToolAgentScenarioRuntime,
    build_tool_agent_scenario_runtime,
)


ScenarioRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    ToolAgentScenarioRuntime,
]

SUPPORTED_EXPECTED_FIELDS = {
    "response_status",
    "permission_status",
    "confirmation_required",
    "clarification_required",
    "validation_ok",
    "executed_tool_names",
    "tool_result_count",
    "final_answer_contains",
    "validation_error_codes",
    "missing_fields",
    "parser_call_count",
    "executor_call_count",
}


class ToolAgentBehaviorEvaluator:
    """
    使用真实 ToolAgent 子图评估工具编排行为。

    功能：
        为每条黄金用例构建确定性 ToolAgent 场景，执行编译后的真实子图，
        再将响应、校验、澄清、确认和执行轨迹转换成统一评估结果。

    参数含义：
        scenario_runtime_builder:
            根据评估用例创建 ToolAgent 确定性场景运行环境的函数。

    返回值含义：
        ToolAgentBehaviorEvaluator:
            可执行单条或批量 ToolAgent 行为评估的对象。
    """

    def __init__(
        self,
        scenario_runtime_builder: ScenarioRuntimeBuilder = (
            build_tool_agent_scenario_runtime
        ),
    ) -> None:
        """
        初始化 ToolAgent 行为评估器。

        参数含义：
            scenario_runtime_builder:
                构建真实子图和确定性依赖的函数。

        返回值含义：
            None。
        """

        self.scenario_runtime_builder = scenario_runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条 ToolAgent 行为评估用例。

        参数含义：
            eval_case:
                包含确定性解析输入和行为期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                ToolAgent 子图输出及逐项行为检查组成的统一评估结果。
        """

        started_at = time.perf_counter()

        try:
            self._validate_case(eval_case)
            runtime = self.scenario_runtime_builder(eval_case)

            # 执行真实编译 ToolAgent 子图，确定性依赖只替换外部不稳定服务。
            result_state = await runtime.graph.ainvoke(
                runtime.initial_state
            )
            output = self._build_output(
                result_state=result_state,
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
                    "graph": "build_tool_agent_graph",
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
        按黄金集顺序批量执行 ToolAgent 行为评估。

        参数含义：
            eval_cases:
                待执行的 ToolAgent 行为评估用例列表。

        返回值含义：
            list[AgentEvaluationResult]:
                与输入顺序一致的统一评估结果列表。
        """

        results: list[AgentEvaluationResult] = []
        for eval_case in eval_cases:
            results.append(await self.evaluate_case(eval_case))
        return results

    def _validate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> None:
        """
        校验用例类别和 ToolAgent 支持的期望字段。

        参数含义：
            eval_case:
                当前 ToolAgent 行为评估用例。

        返回值含义：
            None。
        """

        if eval_case.category != "tool_behavior":
            raise ValueError(
                "ToolAgentBehaviorEvaluator 只接受 category=tool_behavior"
            )

        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "ToolAgent 行为评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        result_state: Mapping[str, Any],
        runtime: ToolAgentScenarioRuntime,
    ) -> dict[str, Any]:
        """
        从 ToolAgent 最终 state 和场景轨迹提取标准评估输出。

        参数含义：
            result_state:
                真实 ToolAgent 子图执行完成后的最终 state。
            runtime:
                保存 Parser、Executor 和确认提示轨迹的场景运行环境。

        返回值含义：
            dict[str, Any]:
                可与黄金期望逐项比较的 ToolAgent 行为摘要。
        """

        response = result_state.get("tool_agent_response", {})
        normalized_response = (
            dict(response)
            if isinstance(response, Mapping)
            else {}
        )
        permission = normalized_response.get("permission", {})
        normalized_permission = (
            dict(permission)
            if isinstance(permission, Mapping)
            else {}
        )
        clarification_request = result_state.get(
            "tool_agent_clarification_request",
            {},
        )
        normalized_clarification = (
            dict(clarification_request)
            if isinstance(clarification_request, Mapping)
            else {}
        )
        raw_errors = result_state.get(
            "tool_call_validation_errors",
            [],
        )
        validation_error_codes = [
            str(error.get("code", ""))
            for error in raw_errors
            if isinstance(error, Mapping) and error.get("code")
        ]
        tool_results = result_state.get("tool_results", [])
        normalized_tool_results = (
            tool_results
            if isinstance(tool_results, list)
            else []
        )

        return {
            "response_status": normalized_response.get("status"),
            "permission_status": normalized_permission.get("status"),
            "confirmation_required": bool(
                result_state.get("tool_confirmation_required", False)
            ),
            "confirmation_prompt": str(
                result_state.get("tool_confirmation_prompt", "") or ""
            ),
            "confirmation_prompt_count": len(
                runtime.confirmation_prompts
            ),
            "clarification_required": bool(normalized_clarification),
            "missing_fields": list(
                normalized_clarification.get("missing_fields", [])
            ),
            "validation_ok": bool(
                result_state.get("tool_call_validation_ok", False)
            ),
            "validation_error_codes": validation_error_codes,
            "executed_tool_names": [
                str(call.get("tool_name", ""))
                for call in runtime.executor.calls
            ],
            "executor_call_count": len(runtime.executor.calls),
            "parser_call_count": len(runtime.parser.inputs),
            "tool_result_count": len(normalized_tool_results),
            "final_answer": str(
                result_state.get("final_answer", "") or ""
            ),
        }

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将 ToolAgent 行为期望转换成逐项检查结果。

        参数含义：
            expected:
                黄金用例声明的预期行为字段。
            output:
                真实 ToolAgent 子图产生的行为摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                所有已声明期望字段的结构化检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            if field_name == "final_answer_contains":
                actual_answer = output.get("final_answer", "")
                passed = str(expected_value) in str(actual_answer)
                checks.append(
                    self._build_check(
                        check_name=field_name,
                        expected=expected_value,
                        actual=actual_answer,
                        passed=passed,
                    )
                )
                continue

            actual_value = output.get(field_name)
            checks.append(
                self._build_check(
                    check_name=field_name,
                    expected=expected_value,
                    actual=actual_value,
                    passed=actual_value == expected_value,
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
        创建一项 ToolAgent 行为检查结果。

        参数含义：
            check_name:
                当前检查字段名称。
            expected:
                黄金用例声明的期望值。
            actual:
                ToolAgent 子图产生的实际值。
            passed:
                当前比较是否通过。

        返回值含义：
            EvaluationCheckResult:
                包含期望、实际和中文说明的检查结果。
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
        计算当前 ToolAgent 评估用例的毫秒耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的执行耗时，单位毫秒。
        """

        return max(
            0.0,
            (time.perf_counter() - started_at) * 1000,
        )
