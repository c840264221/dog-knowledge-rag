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
    DogKnowledgeScenarioRuntime,
    build_dog_knowledge_scenario_runtime,
)


ScenarioRuntimeBuilder = Callable[
    [AgentEvaluationCase],
    DogKnowledgeScenarioRuntime,
]

SUPPORTED_EXPECTED_FIELDS = {
    "response_status",
    "query_type",
    "is_fallback",
    "fallback_reason_contains",
    "final_answer_contains",
    "expected_breed_names",
    "min_evidence_count",
    "required_layer_outputs",
    "parser_call_count",
    "retriever_call_count",
    "reranker_call_count",
    "llm_call_count",
}

LAYER_OUTPUT_STATE_KEYS = {
    "query": "dog_query_result",
    "retrieval": "dog_retrieval_result",
    "recommendation": "dog_recommendation_result",
    "generation": "dog_generation_result",
    "fallback": "dog_fallback_result",
    "pipeline": "dog_knowledge_pipeline_result",
    "answer": "dog_knowledge_answer",
    "public_answer": "dog_knowledge_answer_public",
}


class DogKnowledgeBehaviorEvaluator:
    """
    使用真实 DogKnowledgeAgent 子图评估领域行为和响应契约。

    功能：
        每条用例执行真实编译子图，外部 LLM、Retriever（检索器）和
        Reranker（重排序器）由确定性场景替身提供；随后检查最终答案、
        推荐结果、兜底状态、分层契约和依赖调用轨迹。

    参数含义：
        scenario_runtime_builder:
            根据黄金用例创建确定性场景运行环境的函数。

    返回值含义：
        DogKnowledgeBehaviorEvaluator:
            可执行单条或批量领域智能体评估的对象。
    """

    def __init__(
        self,
        scenario_runtime_builder: ScenarioRuntimeBuilder = (
            build_dog_knowledge_scenario_runtime
        ),
    ) -> None:
        self.scenario_runtime_builder = scenario_runtime_builder

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条 DogKnowledgeAgent 行为评估用例。

        参数含义：
            eval_case:
                包含确定性依赖配置和黄金期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                最终状态摘要和逐项检查结果组成的统一评估结果。
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
                    "graph": "build_dog_knowledge_agent",
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
        按黄金集顺序批量执行 DogKnowledgeAgent 评估。

        参数含义：
            eval_cases:
                待执行的领域智能体评估用例列表。

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
        校验用例类别和支持的黄金期望字段。

        参数含义：
            eval_case:
                当前 DogKnowledgeAgent 评估用例。

        返回值含义：
            None。
        """

        if eval_case.category != "dog_knowledge_behavior":
            raise ValueError(
                "DogKnowledgeBehaviorEvaluator 只接受 "
                "category=dog_knowledge_behavior"
            )

        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "DogKnowledgeAgent 评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        result_state: Mapping[str, Any],
        runtime: DogKnowledgeScenarioRuntime,
    ) -> dict[str, Any]:
        """
        从最终 DogState 和依赖轨迹提取标准评估摘要。

        参数含义：
            result_state:
                真实 DogKnowledgeAgent 子图执行后的最终状态。
            runtime:
                保存解析、检索、重排和 LLM 调用轨迹的场景环境。

        返回值含义：
            dict[str, Any]:
                可以与黄金期望逐项比较的领域行为摘要。
        """

        answer = self._as_dict(result_state.get("dog_knowledge_answer"))
        recommendations = answer.get("recommended_breeds", [])
        evidences = answer.get("evidences", [])
        layer_outputs_present = [
            layer_name
            for layer_name, state_key in LAYER_OUTPUT_STATE_KEYS.items()
            if isinstance(result_state.get(state_key), Mapping)
        ]

        return {
            "response_status": answer.get("status"),
            "query_type": answer.get("query_type"),
            "is_fallback": bool(answer.get("is_fallback", False)),
            "fallback_reason": str(answer.get("fallback_reason", "") or ""),
            "final_answer": str(result_state.get("final_answer", "") or ""),
            "recommended_breed_names": [
                str(item.get("breed_name", ""))
                for item in recommendations
                if isinstance(item, Mapping) and item.get("breed_name")
            ],
            "evidence_count": len(evidences) if isinstance(evidences, list) else 0,
            "layer_outputs_present": layer_outputs_present,
            "parser_call_count": len(runtime.parser.inputs),
            "retriever_call_count": len(runtime.retriever.queries),
            "reranker_call_count": len(runtime.reranker.calls),
            "llm_call_count": len(runtime.llm_provider.prompts),
        }

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将黄金期望转换成逐项结构化检查结果。

        参数含义：
            expected:
                黄金用例声明的期望字段。
            output:
                真实子图执行后的标准评估摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                每个已声明期望字段对应的检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            if field_name in {
                "final_answer_contains",
                "fallback_reason_contains",
            }:
                output_key = field_name.removesuffix("_contains")
                actual_value = str(output.get(output_key, ""))
                checks.append(
                    self._build_check(
                        field_name,
                        expected_value,
                        actual_value,
                        str(expected_value) in actual_value,
                    )
                )
                continue

            if field_name == "expected_breed_names":
                actual_names = set(output.get("recommended_breed_names", []))
                expected_names = set(expected_value)
                checks.append(
                    self._build_check(
                        field_name,
                        sorted(expected_names),
                        sorted(actual_names),
                        expected_names.issubset(actual_names),
                    )
                )
                continue

            if field_name == "required_layer_outputs":
                actual_layers = set(output.get("layer_outputs_present", []))
                expected_layers = set(expected_value)
                checks.append(
                    self._build_check(
                        field_name,
                        sorted(expected_layers),
                        sorted(actual_layers),
                        expected_layers.issubset(actual_layers),
                    )
                )
                continue

            if field_name == "min_evidence_count":
                actual_count = int(output.get("evidence_count", 0))
                checks.append(
                    self._build_check(
                        field_name,
                        expected_value,
                        actual_count,
                        actual_count >= int(expected_value),
                    )
                )
                continue

            actual_value = output.get(field_name)
            checks.append(
                self._build_check(
                    field_name,
                    expected_value,
                    actual_value,
                    actual_value == expected_value,
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
        创建单个统一评估检查结果。

        参数含义：
            check_name:
                当前检查项名称。
            expected:
                黄金用例声明的期望值。
            actual:
                真实子图产生的实际值。
            passed:
                当前比较是否通过。

        返回值含义：
            EvaluationCheckResult:
                包含期望值、实际值和说明的结构化检查对象。
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

    def _as_dict(self, value: Any) -> dict[str, Any]:
        """
        将 Mapping 或 Pydantic 对象转换成普通字典。

        参数含义：
            value:
                待转换的响应对象。

        返回值含义：
            dict[str, Any]:
                转换成功后的普通字典；无法转换时返回空字典。
        """

        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="python")
            if isinstance(dumped, dict):
                return dumped
        return {}

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
