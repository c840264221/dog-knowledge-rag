"""Skill（技能）选择、输入准备和恢复行为评估器。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from src.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.skills import SkillRuntimeResult, build_default_skill_runtime


SUPPORTED_EXPECTED_FIELDS = {
    "status",
    "selected_skill_id",
    "selection_source",
    "candidate_skill_ids",
    "matched_hints",
    "extracted_inputs",
    "merged_inputs",
    "available_input_ids",
    "missing_input_ids",
    "input_ready",
    "skill_context_loaded",
    "skill_context_contains",
    "clarification_prompt_contains",
}


class SkillBehaviorEvaluator:
    """
    使用真实 SkillRuntime 评估技能选择、缺参和恢复行为。

    功能：
        从统一黄金用例读取用户文本、可选历史输入和已选技能编号，运行项目
        默认 SkillRuntime，再把实际选择、提取、检查和上下文加载结果与
        expected 逐项比较。

    参数含义：
        无。

    返回值含义：
        SkillBehaviorEvaluator:
            可执行单条或批量确定性 Skill 行为评估的对象。
    """

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行并检查一条 Skill 黄金评估用例。

        参数含义：
            eval_case:
                包含用户问题、历史 Skill 状态和黄金期望的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                实际 Skill 输出和逐字段检查组成的统一评估结果。
        """

        started_at = time.perf_counter()
        try:
            self._validate_case(eval_case)

            # 上一轮已经保存、但本轮尚未重新校验的 Skill 输入。
            existing_inputs = eval_case.input_state.get(
                "existing_inputs"
            )
            if existing_inputs is not None and not isinstance(
                existing_inputs,
                Mapping,
            ):
                raise ValueError("existing_inputs 必须是字段到输入值的映射")

            # 上一轮已经选中的技能编号；首轮没有该值时由选择器自动判断。
            selected_skill_id = eval_case.input_state.get(
                "selected_skill_id"
            )
            if selected_skill_id is not None:
                selected_skill_id = str(selected_skill_id).strip()
                if not selected_skill_id:
                    raise ValueError("selected_skill_id 不能为空字符串")

            runtime_result = build_default_skill_runtime().prepare(
                user_text=eval_case.question,
                existing_inputs=existing_inputs,
                selected_skill_id=selected_skill_id,
            )
            output = self._build_output(runtime_result)
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
                    "runtime": "SkillRuntime",
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
        按黄金集顺序执行多条 Skill 行为评估。

        参数含义：
            eval_cases:
                待执行的 Skill 黄金用例列表。

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
                当前准备执行的 Skill 评估用例。

        返回值含义：
            None:
                校验通过时不返回数据；类别或字段不合法时抛出 ValueError。
        """

        if eval_case.category != "skill_behavior":
            raise ValueError(
                "SkillBehaviorEvaluator 只接受 category=skill_behavior"
            )
        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "Skill 行为评估包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

    def _build_output(
        self,
        runtime_result: SkillRuntimeResult,
    ) -> dict[str, Any]:
        """
        从 SkillRuntimeResult 提取可与黄金期望比较的摘要。

        参数含义：
            runtime_result:
                真实 SkillRuntime 返回的统一准备结果。

        返回值含义：
            dict[str, Any]:
                包含选择、输入提取、缺参和上下文状态的实际输出。
        """

        extraction = runtime_result.extraction
        input_check = runtime_result.input_check
        return {
            "status": runtime_result.status,
            "selected_skill_id": (
                runtime_result.selection.selected_skill_id
            ),
            "selection_source": runtime_result.selection.source,
            "candidate_skill_ids": list(
                runtime_result.selection.candidate_skill_ids
            ),
            "matched_hints": list(
                runtime_result.selection.matched_hints
            ),
            "extracted_inputs": (
                dict(extraction.extracted_inputs)
                if extraction is not None
                else {}
            ),
            "merged_inputs": (
                dict(extraction.merged_inputs)
                if extraction is not None
                else {}
            ),
            "available_input_ids": (
                list(input_check.available_input_ids)
                if input_check is not None
                else []
            ),
            "missing_input_ids": (
                list(input_check.missing_input_ids)
                if input_check is not None
                else []
            ),
            "input_ready": bool(
                input_check is not None and input_check.is_ready
            ),
            "skill_context_loaded": bool(runtime_result.skill_context),
            "skill_context": runtime_result.skill_context,
            "clarification_prompt": (
                input_check.clarification_prompt
                if input_check is not None
                else ""
            ),
        }

    def _build_checks(
        self,
        *,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        将 Skill 黄金期望转换成逐项结构化检查结果。

        参数含义：
            expected:
                黄金用例声明的预期字段和值。
            output:
                真实 SkillRuntime 产生的输出摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                每个已声明期望字段对应的一项检查结果。
        """

        checks: list[EvaluationCheckResult] = []
        for field_name, expected_value in expected.items():
            if field_name == "skill_context_contains":
                actual_value = str(output.get("skill_context", ""))
                passed = self._contains_all(actual_value, expected_value)
            elif field_name == "clarification_prompt_contains":
                actual_value = str(
                    output.get("clarification_prompt", "")
                )
                passed = self._contains_all(actual_value, expected_value)
            else:
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

    def _contains_all(self, text: str, expected_value: Any) -> bool:
        """
        判断实际文本是否包含黄金用例要求的全部片段。

        参数含义：
            text:
                Skill 上下文或澄清提示的实际完整文本。
            expected_value:
                一个必含字符串，或由多个必含字符串组成的列表。

        返回值含义：
            bool:
                所有要求片段都存在时返回 True，否则返回 False。
        """

        required_fragments = (
            list(expected_value)
            if isinstance(expected_value, list)
            else [expected_value]
        )
        return all(str(fragment) in text for fragment in required_fragments)

    def _elapsed_ms(self, started_at: float) -> float:
        """
        计算单条 Skill 评估用例的执行耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(0.0, (time.perf_counter() - started_at) * 1000)
