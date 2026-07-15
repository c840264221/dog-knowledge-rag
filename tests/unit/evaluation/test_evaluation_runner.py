from pathlib import Path
from typing import Any

import pytest

from src.evaluation import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    EvaluationCheckResult,
)
from src.evaluation.runner import (
    CORE_EVALUATION_TARGETS,
    DEFAULT_EVALUATION_TARGETS,
    EvaluationSuiteRunner,
    EvaluationTarget,
    FULL_EVALUATION_TARGETS,
)


class FakeEvaluator:
    """
    为统一运行器测试返回固定评估结果。

    参数含义：
        无。

    返回值含义：
        FakeEvaluator:
            支持 evaluate_many 的确定性评估器。
    """

    async def evaluate_many(
        self,
        eval_cases: list[AgentEvaluationCase],
    ) -> list[AgentEvaluationResult]:
        """
        为每条输入用例生成固定通过结果。

        参数含义：
            eval_cases:
                统一运行器加载的评估用例。

        返回值含义：
            list[AgentEvaluationResult]:
                与输入顺序一致的固定通过结果。
        """

        return [
            AgentEvaluationResult(
                case_id=eval_case.case_id,
                category=eval_case.category,
                checks=[
                    EvaluationCheckResult(
                        check_name="route",
                        passed=True,
                        expected="general_agent",
                        actual="general_agent",
                    )
                ],
            )
            for eval_case in eval_cases
        ]


class EmptyEvaluator:
    """
    模拟漏返回全部评估结果的错误评估器。

    参数含义：
        无。

    返回值含义：
        EmptyEvaluator:
            始终返回空结果列表的测试评估器。
    """

    async def evaluate_many(
        self,
        eval_cases: list[AgentEvaluationCase],
    ) -> list[AgentEvaluationResult]:
        """
        忽略输入并返回空结果，用于验证 Runner 完整性检查。

        参数含义：
            eval_cases:
                Runner 传入的黄金用例列表。

        返回值含义：
            list[AgentEvaluationResult]:
                空列表。
        """

        _ = eval_cases
        return []


def build_fake_case(category: str = "root_route") -> AgentEvaluationCase:
    """
    构建统一运行器测试使用的最小黄金用例。

    参数含义：
        category:
            测试用例所属评估类别。

    返回值含义：
        AgentEvaluationCase:
            包含 route 黄金期望的最小用例。
    """

    return AgentEvaluationCase(
        case_id=f"{category}_001",
        category=category,
        question="测试问题",
        expected={
            "route": "general_agent",
        },
    )


@pytest.mark.asyncio
async def test_evaluation_suite_runner_should_execute_registered_target() -> None:
    """
    测试统一运行器会加载数据集并执行对应评估器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    loaded_paths: list[Path] = []

    def fake_loader(dataset_path: str | Path) -> list[AgentEvaluationCase]:
        """
        记录数据集路径并返回固定用例。

        参数含义：
            dataset_path:
                Runner 传入的黄金数据集路径。

        返回值含义：
            list[AgentEvaluationCase]:
                单条固定 RootAgent 评估用例。
        """

        loaded_paths.append(Path(dataset_path))
        return [build_fake_case()]

    target = EvaluationTarget(
        category="root_route",
        dataset_path=Path("evaluation/root.json"),
        evaluator_factory=FakeEvaluator,
    )
    suite_run = await EvaluationSuiteRunner(
        targets=(target,),
        dataset_loader=fake_loader,
    ).run()

    assert loaded_paths == [Path("evaluation/root.json")]
    assert len(suite_run.category_runs) == 1
    assert suite_run.category_runs[0].error_message is None
    assert suite_run.category_runs[0].results[0].passed is True


@pytest.mark.asyncio
async def test_evaluation_suite_runner_should_continue_after_category_error() -> None:
    """
    测试某个类别加载失败后统一运行器仍会继续执行下一类别。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def fake_loader(dataset_path: str | Path) -> list[AgentEvaluationCase]:
        """
        为第一项模拟异常，为第二项返回合法用例。

        参数含义：
            dataset_path:
                当前评估目标的数据集路径。

        返回值含义：
            list[AgentEvaluationCase]:
                第二项对应的固定合法用例。
        """

        if Path(dataset_path).name == "broken.json":
            raise ValueError("模拟数据集损坏")
        return [build_fake_case(category="tool_behavior")]

    targets = (
        EvaluationTarget(
            category="root_route",
            dataset_path=Path("evaluation/broken.json"),
            evaluator_factory=FakeEvaluator,
        ),
        EvaluationTarget(
            category="tool_behavior",
            dataset_path=Path("evaluation/tool.json"),
            evaluator_factory=FakeEvaluator,
        ),
    )
    suite_run = await EvaluationSuiteRunner(
        targets=targets,
        dataset_loader=fake_loader,
    ).run()

    assert len(suite_run.category_runs) == 2
    assert "模拟数据集损坏" in str(
        suite_run.category_runs[0].error_message
    )
    assert suite_run.category_runs[1].results[0].passed is True


@pytest.mark.asyncio
async def test_evaluation_suite_runner_should_reject_missing_results() -> None:
    """
    测试评估器漏返回黄金用例成绩时 Runner 会记录类别异常。

    参数含义：
        无。

    返回值含义：
        None。
    """

    target = EvaluationTarget(
        category="root_route",
        dataset_path=Path("evaluation/root.json"),
        evaluator_factory=EmptyEvaluator,
    )
    suite_run = await EvaluationSuiteRunner(
        targets=(target,),
        dataset_loader=lambda path: [build_fake_case()],
    ).run()

    category_run = suite_run.category_runs[0]
    assert category_run.results == []
    assert "不完整匹配" in str(category_run.error_message)


def test_default_targets_should_include_real_rag_retrieval() -> None:
    """
    测试默认统一评估套件已经注册真实 RAG 检索类别。

    参数含义：
        无。

    返回值含义：
        None。
    """

    targets_by_category = {
        target.category: target
        for target in DEFAULT_EVALUATION_TARGETS
    }

    assert "rag_retrieval_behavior" in targets_by_category
    assert targets_by_category[
        "rag_retrieval_behavior"
    ].dataset_path == Path(
        "evaluation/datasets/rag_retrieval_behavior_cases.json"
    )


def test_core_and_full_targets_should_have_clear_dependency_boundary() -> None:
    """
    测试 core 档位不包含真实 RAG，而 full 档位包含真实 RAG。

    参数含义：
        无。

    返回值含义：
        None。
    """

    core_categories = {
        target.category
        for target in CORE_EVALUATION_TARGETS
    }
    full_categories = {
        target.category
        for target in FULL_EVALUATION_TARGETS
    }

    assert "rag_retrieval_behavior" not in core_categories
    assert full_categories == core_categories | {"rag_retrieval_behavior"}
    assert DEFAULT_EVALUATION_TARGETS == FULL_EVALUATION_TARGETS
