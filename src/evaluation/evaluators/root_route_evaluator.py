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
from src.graph.nodes.router_node import semantic_router_node
from src.graph.routes.route_after_semantic import route_after_semantic


RouteNode = Callable[
    [dict[str, Any]],
    dict[str, Any] | Awaitable[dict[str, Any]],
]
RouteResolver = Callable[[dict[str, Any]], str]

SUPPORTED_EXPECTED_FIELDS = {
    "route",
    "query_type",
    "requires_rag",
    "requires_tool",
    "requires_memory",
    "min_confidence",
}


class RootRouteEvaluator:
    """
    评估 RootAgent（根智能体）的主图路由结果。

    功能：
        默认执行主图真实注册的 semantic_router_node，再通过
        route_after_semantic 解析条件边路由，最后把各项比较转换成统一评估结果。

    参数含义：
        route_node:
            主图路由节点。默认使用 semantic_router_node；测试时可注入 Mock 节点。
        route_resolver:
            主图条件边路由函数。默认使用 route_after_semantic。

    返回值含义：
        RootRouteEvaluator:
            可执行单条或批量 RootAgent 路由评估的对象。
    """

    def __init__(
        self,
        route_node: RouteNode = semantic_router_node,
        route_resolver: RouteResolver = route_after_semantic,
    ) -> None:
        """
        初始化 RootAgent 路由评估器。

        参数含义：
            route_node:
                被评估的主图路由节点。
            route_resolver:
                根据路由节点输出决定条件边目标的函数。

        返回值含义：
            None。
        """

        self.route_node = route_node
        self.route_resolver = route_resolver

    async def evaluate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        """
        执行一条 RootAgent 路由评估用例。

        参数含义：
            eval_case:
                包含问题、初始 state 和预期路由字段的统一评估用例。

        返回值含义：
            AgentEvaluationResult:
                路由字段、主图条件边和执行异常组成的统一评估结果。
        """

        started_at = time.perf_counter()

        try:
            self._validate_case(eval_case)
            initial_state = {
                **dict(eval_case.input_state),
                "question": eval_case.question,
            }

            # 调用主图真实路由入口，覆盖兼容适配器和 RootAgent 决策逻辑。
            route_update = self.route_node(initial_state)

            if inspect.isawaitable(route_update):
                route_update = await route_update

            if not isinstance(route_update, Mapping):
                raise TypeError("RootAgent 路由节点必须返回 Mapping 类型")

            normalized_update = dict(route_update)
            evaluated_state = {
                **initial_state,
                **normalized_update,
            }

            # 继续执行主图条件边函数，确认图真正采用的路由与决策一致。
            graph_route = self.route_resolver(evaluated_state)
            output = self._build_output(
                route_update=normalized_update,
                graph_route=graph_route,
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
                    "entry_node": "semantic_router_node",
                    "route_resolver": "route_after_semantic",
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
        按数据集顺序批量执行 RootAgent 路由评估。

        参数含义：
            eval_cases:
                待执行的统一评估用例列表。

        返回值含义：
            list[AgentEvaluationResult]:
                与输入顺序一致的评估结果列表。
        """

        results: list[AgentEvaluationResult] = []

        for eval_case in eval_cases:
            results.append(
                await self.evaluate_case(eval_case)
            )

        return results

    def _validate_case(
        self,
        eval_case: AgentEvaluationCase,
    ) -> None:
        """
        校验用例类别和 RootAgent 支持的期望字段。

        参数含义：
            eval_case:
                当前待执行的评估用例。

        返回值含义：
            None。
        """

        if eval_case.category != "root_route":
            raise ValueError(
                "RootRouteEvaluator 只接受 category=root_route 的评估用例"
            )

        unsupported_fields = (
            set(eval_case.expected) - SUPPORTED_EXPECTED_FIELDS
        )
        if unsupported_fields:
            raise ValueError(
                "RootAgent 评估用例包含不支持的 expected 字段: "
                f"{sorted(unsupported_fields)}"
            )

        if "route" not in eval_case.expected:
            raise ValueError("RootAgent 评估用例必须声明 expected.route")

    def _build_output(
        self,
        route_update: dict[str, Any],
        graph_route: str,
    ) -> dict[str, Any]:
        """
        从路由节点输出中提取可评估的普通字段。

        参数含义：
            route_update:
                semantic_router_node 返回的局部 state 更新。
            graph_route:
                route_after_semantic 返回的主图条件边目标。

        返回值含义：
            dict[str, Any]:
                仅包含路由评估所需字段的输出摘要。
        """

        route_decision = route_update.get("route_decision")
        if not isinstance(route_decision, Mapping):
            raise ValueError("RootAgent 输出缺少有效的 route_decision")

        return {
            "route": route_decision.get("route"),
            "graph_route": graph_route,
            "next_agent": route_update.get("next_agent"),
            "query_type": route_decision.get("query_type"),
            "confidence": route_decision.get("confidence"),
            "requires_rag": route_decision.get("requires_rag"),
            "requires_tool": route_decision.get("requires_tool"),
            "requires_memory": route_decision.get("requires_memory"),
        }

    def _build_checks(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> list[EvaluationCheckResult]:
        """
        根据期望字段创建 RootAgent 路由检查结果。

        参数含义：
            expected:
                黄金用例声明的预期路由字段。
            output:
                主图路由入口和条件边产生的实际输出摘要。

        返回值含义：
            list[EvaluationCheckResult]:
                路由决策、条件边和可选字段的逐项检查结果。
        """

        expected_route = expected["route"]
        checks = [
            self._build_equal_check(
                check_name="route",
                expected=expected_route,
                actual=output.get("route"),
            ),
            self._build_equal_check(
                check_name="graph_route",
                expected=expected_route,
                actual=output.get("graph_route"),
            ),
            self._build_equal_check(
                check_name="next_agent",
                expected=expected_route,
                actual=output.get("next_agent"),
            ),
        ]

        for field_name in (
            "query_type",
            "requires_rag",
            "requires_tool",
            "requires_memory",
        ):
            if field_name not in expected:
                continue

            checks.append(
                self._build_equal_check(
                    check_name=field_name,
                    expected=expected[field_name],
                    actual=output.get(field_name),
                )
            )

        if "min_confidence" in expected:
            minimum = expected["min_confidence"]
            actual_confidence = output.get("confidence")
            passed = (
                isinstance(minimum, (int, float))
                and isinstance(actual_confidence, (int, float))
                and actual_confidence >= minimum
            )
            checks.append(
                EvaluationCheckResult(
                    check_name="min_confidence",
                    passed=passed,
                    expected=minimum,
                    actual=actual_confidence,
                    message=(
                        "路由置信度达到最低要求。"
                        if passed
                        else "路由置信度低于最低要求。"
                    ),
                )
            )

        return checks

    def _build_equal_check(
        self,
        check_name: str,
        expected: Any,
        actual: Any,
    ) -> EvaluationCheckResult:
        """
        创建一个使用相等关系判断的检查结果。

        参数含义：
            check_name:
                当前检查项名称。
            expected:
                黄金用例声明的期望值。
            actual:
                主图路由链路产生的实际值。

        返回值含义：
            EvaluationCheckResult:
                包含期望值、实际值和判断结果的检查对象。
        """

        passed = actual == expected
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
        计算从指定开始时间到当前时刻的毫秒耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(
            0.0,
            (time.perf_counter() - started_at) * 1000,
        )
