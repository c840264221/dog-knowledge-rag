from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.dataset_loader import load_agent_evaluation_cases
from src.evaluation.evaluators import (
    DogKnowledgeBehaviorEvaluator,
    MainGraphBehaviorEvaluator,
    MemoryRecallBehaviorEvaluator,
    RagRetrievalBehaviorEvaluator,
    RootRouteEvaluator,
    ToolAgentBehaviorEvaluator,
)
from src.evaluation.schemas import AgentEvaluationResult


EvaluatorFactory = Callable[[], Any]
DatasetLoader = Callable[[str | Path], Any]


@dataclass(frozen=True)
class EvaluationTarget:
    """
    定义统一评估运行器需要执行的一门评估科目。

    参数含义：
        category:
            评估类别名称。
        dataset_path:
            当前类别黄金数据集路径。
        evaluator_factory:
            创建对应 Evaluator（评估器）的函数或类。

    返回值含义：
        EvaluationTarget:
            一项可由统一运行器加载和执行的评估目标。
    """

    category: str
    dataset_path: Path
    evaluator_factory: EvaluatorFactory


@dataclass
class EvaluationCategoryRun:
    """
    保存一门评估科目的原始执行结果。

    参数含义：
        target:
            当前科目的配置。
        results:
            当前科目产生的单条评估结果。
        duration_ms:
            当前科目执行耗时，单位毫秒。
        error_message:
            数据集加载或评估器执行产生的类别级异常。

    返回值含义：
        EvaluationCategoryRun:
            报告构建器可以继续汇总的类别运行记录。
    """

    # EvaluationTarget里面包含category（类别）比如root_route（主图路由）、测试用例的路径在哪、执行器是哪个
    target: EvaluationTarget
    results: list[AgentEvaluationResult] = field(default_factory=list)
    duration_ms: float = 0.0
    error_message: str | None = None


@dataclass
class EvaluationSuiteRun:
    """
    保存统一评估套件的原始执行记录。

    参数含义：
        category_runs:
            所有评估类别的运行记录。
        duration_ms:
            整套评估执行耗时，单位毫秒。

    返回值含义：
        EvaluationSuiteRun:
            可交给报告构建器生成完整成绩单的原始数据。
    """

    category_runs: list[EvaluationCategoryRun] = field(default_factory=list)
    duration_ms: float = 0.0


CORE_EVALUATION_TARGETS = (
    EvaluationTarget(
        category="root_route",
        dataset_path=Path("evaluation/datasets/root_agent_route_cases.json"),
        evaluator_factory=RootRouteEvaluator,
    ),
    EvaluationTarget(
        category="tool_behavior",
        dataset_path=Path("evaluation/datasets/tool_agent_behavior_cases.json"),
        evaluator_factory=ToolAgentBehaviorEvaluator,
    ),
    EvaluationTarget(
        category="dog_knowledge_behavior",
        dataset_path=Path(
            "evaluation/datasets/dog_knowledge_behavior_cases.json"
        ),
        evaluator_factory=DogKnowledgeBehaviorEvaluator,
    ),
    EvaluationTarget(
        category="memory_recall_behavior",
        dataset_path=Path(
            "evaluation/datasets/memory_recall_behavior_cases.json"
        ),
        evaluator_factory=MemoryRecallBehaviorEvaluator,
    ),
    EvaluationTarget(
        category="main_graph_behavior",
        dataset_path=Path("evaluation/datasets/main_graph_behavior_cases.json"),
        evaluator_factory=MainGraphBehaviorEvaluator,
    ),
)

RAG_RETRIEVAL_EVALUATION_TARGET = EvaluationTarget(
    category="rag_retrieval_behavior",
    dataset_path=Path(
        "evaluation/datasets/rag_retrieval_behavior_cases.json"
    ),
    evaluator_factory=RagRetrievalBehaviorEvaluator,
)

FULL_EVALUATION_TARGETS = (
    *CORE_EVALUATION_TARGETS,
    RAG_RETRIEVAL_EVALUATION_TARGET,
)

# 保持现有默认命令继续执行包含真实 RAG 的完整评估。
DEFAULT_EVALUATION_TARGETS = FULL_EVALUATION_TARGETS


class EvaluationSuiteRunner:
    """
    依次执行所有注册的 Agent Evaluation（智能体评估）科目。

    功能：
        加载每个目标的黄金数据集、创建对应评估器并执行批量评估；
        单个类别失败时记录错误并继续执行其他类别，避免整套报告被提前中断。

    参数含义：
        targets:
            需要执行的评估目标列表。
        dataset_loader:
            黄金数据集加载函数，测试时可注入替身。

    返回值含义：
        EvaluationSuiteRunner:
            可异步运行整套评估并返回原始运行记录的对象。
    """

    def __init__(
        self,
        targets: tuple[EvaluationTarget, ...] = DEFAULT_EVALUATION_TARGETS,
        dataset_loader: DatasetLoader = load_agent_evaluation_cases,
    ) -> None:
        """
        初始化统一评估运行器。

        参数含义：
            targets:
                按执行顺序排列的评估目标。
            dataset_loader:
                将数据集路径转换成评估用例列表的函数。

        返回值含义：
            None。
        """

        self.targets = tuple(targets)
        self.dataset_loader = dataset_loader

    async def run(self) -> EvaluationSuiteRun:
        """
        顺序执行整套评估目标。

        参数含义：
            无。

        返回值含义：
            EvaluationSuiteRun:
                所有类别的结果、异常和总耗时。
        """

        suite_started_at = time.perf_counter()
        category_runs: list[EvaluationCategoryRun] = []

        for target in self.targets:
            category_runs.append(
                await self._run_target(target)
            )

        return EvaluationSuiteRun(
            category_runs=category_runs,
            duration_ms=self._elapsed_ms(suite_started_at),
        )

    async def _run_target(
        self,
        target: EvaluationTarget,
    ) -> EvaluationCategoryRun:
        """
        执行单个评估类别并隔离类别级异常。

        参数含义：
            target:
                当前需要执行的数据集和评估器配置。

        返回值含义：
            EvaluationCategoryRun:
                当前类别的结果、耗时和可选异常。
        """

        started_at = time.perf_counter()
        try:
            eval_cases = list(
                self.dataset_loader(target.dataset_path)
            )
            invalid_categories = sorted(
                {
                    eval_case.category
                    for eval_case in eval_cases
                    if eval_case.category != target.category
                }
            )
            if invalid_categories:
                raise ValueError(
                    f"数据集类别与目标 {target.category} 不一致: "
                    f"{invalid_categories}"
                )

            evaluator = target.evaluator_factory()
            results = await evaluator.evaluate_many(eval_cases)
            normalized_results = list(results)
            self._validate_results(
                target=target,
                eval_cases=eval_cases,
                results=normalized_results,
            )
            return EvaluationCategoryRun(
                target=target,
                results=normalized_results,
                duration_ms=self._elapsed_ms(started_at),
            )
        except Exception as exc:
            return EvaluationCategoryRun(
                target=target,
                duration_ms=self._elapsed_ms(started_at),
                error_message=str(exc),
            )

    def _validate_results(
        self,
        target: EvaluationTarget,
        eval_cases: list[Any],
        results: list[AgentEvaluationResult],
    ) -> None:
        """
        校验评估器完整返回当前黄金数据集的所有成绩。

        参数含义：
            target:
                当前评估类别配置。
            eval_cases:
                数据集加载出的黄金用例。
            results:
                Evaluator 返回的单条评估结果。

        返回值含义：
            None；发现漏评、重复结果或类别错误时抛出 ValueError。
        """

        expected_case_ids = [eval_case.case_id for eval_case in eval_cases]
        actual_case_ids = [result.case_id for result in results]
        if len(actual_case_ids) != len(set(actual_case_ids)):
            raise ValueError(
                f"{target.category} 评估结果存在重复 case_id"
            )
        if set(actual_case_ids) != set(expected_case_ids):
            raise ValueError(
                f"{target.category} 评估结果与黄金用例不完整匹配，"
                f"expected={sorted(expected_case_ids)}, "
                f"actual={sorted(actual_case_ids)}"
            )

        invalid_result_categories = sorted(
            {
                result.category
                for result in results
                if result.category != target.category
            }
        )
        if invalid_result_categories:
            raise ValueError(
                f"{target.category} 评估结果返回了错误类别: "
                f"{invalid_result_categories}"
            )

    def _elapsed_ms(self, started_at: float) -> float:
        """
        计算指定开始时间到当前时刻的毫秒耗时。

        参数含义：
            started_at:
                time.perf_counter 返回的高精度开始时间。

        返回值含义：
            float:
                非负的毫秒耗时。
        """

        return max(0.0, (time.perf_counter() - started_at) * 1000)
