from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.evaluation.scenarios.main_graph_scenario_runtime import (
    MainGraphScenarioRuntime,
    build_main_graph_scenario_runtime,
)


ScenarioRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    MainGraphScenarioRuntime | Awaitable[MainGraphScenarioRuntime],
]

SUPPORTED_EXPECTED_FIELDS = {
    "route",
    "query_type",
    "final_answer_contains",
    "final_answer_empty",
    "tool_response_status",
    "memory_extract_call_count",
    "general_supervisor_call_count",
    "general_answer_call_count",
    "dog_parser_call_count",
    "dog_retriever_call_count",
    "dog_reranker_call_count",
    "dog_answer_call_count",
    "tool_answer_call_count",
    "tool_parser_call_count",
    "tool_executor_call_count",
    "required_state_fields",
}


class MainGraphBehaviorEvaluator:
    """
    使用真实 Main Graph（主图）评估跨 Agent 编排行为。

    功能：
        执行 GraphRuntimeService 构建的真实主图，检查 RootAgent 路由、
        最终答案、ToolAgent 状态、关键 state 字段和外部依赖调用轨迹，
        用于发现各模块单独通过但组合后链路失效的问题。

    参数含义：
        scenario_runtime_builder:
            根据黄金用例构建真实主图与确定性外部依赖的函数。

    返回值含义：
        MainGraphBehaviorEvaluator:
            可执行单条或批量主图端到端行为评估的对象。
    """

    def __init__(
        self,
        scenario_runtime_builder: ScenarioRuntimeBuilder = (
            build_main_graph_scenario_runtime
        ),
    ) -> None:
        """
        初始化 Main Graph 行为评估器。

        参数含义：
            scenario_runtime_builder:
                构建真实主图和确定性外部依赖的函数。

        返回值含义：
            None。
        """

        self.scenario_runtime_builder = scenario_runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条 Main Graph 行为评估用例。

        参数含义：
            eval_case:
                包含主图确定性依赖配置和黄金期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                主图输出摘要和逐项检查结果组成的统一评估结果。
        """

        started_at = time.perf_counter()
        try:
            self._validate_case(eval_case)
            runtime = self.scenario_runtime_builder(eval_case)
            if inspect.isawaitable(runtime):
                runtime = await runtime

            result_state = await runtime.invoke()
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
                    "graph": "GraphRuntimeService._build_graph",
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
        按黄金数据集顺序批量执行 Main Graph 行为评估。

        参数含义：
            eval_cases:
                待执行的主图行为评估用例列表。

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
        校验评估类别和 Main Graph 支持的黄金期望字段。

        参数含义：
            eval_case:
                当前主图行为评估用例。

        返回值含义：
            None。
        """

        if eval_case.category != "main_graph_behavior":
            raise ValueError(
                "MainGraphBehaviorEvaluator 只接受 "
                "category=main_graph_behavior"
            )

        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "Main Graph 行为评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

        if "route" not in eval_case.expected:
            raise ValueError("Main Graph 行为评估必须声明 expected.route")

    def _build_output(
        self,
        result_state: Mapping[str, Any],
        runtime: MainGraphScenarioRuntime,
    ) -> dict[str, Any]:
        """
        从最终 DogState 和依赖轨迹提取主图评估摘要。

        参数含义：
            result_state:
                真实 Main Graph 完整执行后的最终状态。
            runtime:
                保存 LLM、RAG 和 Tool 调用轨迹的确定性场景环境。

        返回值含义：
            dict[str, Any]:
                可与黄金期望逐项比较的主图行为摘要。
        """

        route_decision = result_state.get("route_decision", {})
        normalized_route_decision = (
            dict(route_decision)
            if isinstance(route_decision, Mapping)
            else {}
        )
        tool_response = result_state.get("tool_agent_response", {})
        normalized_tool_response = (
            dict(tool_response)
            if isinstance(tool_response, Mapping)
            else {}
        )
        final_answer = str(
            result_state.get("final_answer")
            or result_state.get("answer")
            or ""
        )

        return {
            "route": normalized_route_decision.get("route"),
            "query_type": normalized_route_decision.get("query_type"),
            "next_agent": result_state.get("next_agent"),
            "final_answer": final_answer,
            "final_answer_empty": not bool(final_answer.strip()),
            "tool_response_status": normalized_tool_response.get("status"),
            "memory_extract_call_count": runtime.llm_provider.count_calls(
                "memory_extract"
            ),
            "general_supervisor_call_count": runtime.llm_provider.count_calls(
                "general_supervisor"
            ),
            "general_answer_call_count": runtime.llm_provider.count_calls(
                "general_answer"
            ),
            "dog_answer_call_count": runtime.llm_provider.count_calls(
                "dog_answer"
            ),
            "tool_answer_call_count": runtime.llm_provider.count_calls(
                "tool_answer"
            ),
            "dog_parser_call_count": len(runtime.dog_parser.inputs),
            "dog_retriever_call_count": len(runtime.dog_retriever.queries),
            "dog_reranker_call_count": len(runtime.dog_reranker.calls),
            "tool_parser_call_count": len(runtime.tool_parser.inputs),
            "tool_executor_call_count": len(runtime.tool_executor.calls),
            "state_fields_present": sorted(result_state.keys()),
        }

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将 Main Graph 黄金期望转换成逐项结构化检查结果。

        参数含义：
            expected:
                黄金用例声明的预期路由、答案、状态和调用次数。
            output:
                真实主图执行后的标准评估摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                每个已声明期望字段对应的检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            if field_name == "final_answer_contains":
                actual = str(output.get("final_answer", ""))
                passed = str(expected_value) in actual
            elif field_name == "required_state_fields":
                actual_fields = set(output.get("state_fields_present", []))
                expected_fields = set(expected_value)
                actual = sorted(actual_fields)
                expected_value = sorted(expected_fields)
                passed = expected_fields.issubset(actual_fields)
            else:
                actual = output.get(field_name)
                passed = actual == expected_value

            checks.append(
                self._build_check(
                    check_name=field_name,
                    expected=expected_value,
                    actual=actual,
                    passed=passed,
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
        创建一个统一的 Main Graph 评估检查结果。

        参数含义：
            check_name:
                当前检查项名称。
            expected:
                黄金用例声明的期望值。
            actual:
                真实主图产生的实际值。
            passed:
                当前比较是否通过。

        返回值含义：
            EvaluationCheckResult:
                包含期望值、实际值和中文说明的结构化检查对象。
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
