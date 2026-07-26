from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.scenarios import (
    build_multi_agent_orchestration_scenario_runtime,
)


DATASET_PATH = Path(
    "evaluation/datasets/multi_agent_orchestration_cases.json"
)


def test_orchestration_dataset_should_define_core_stage_scenarios() -> None:
    """
    检查总编排黄金集是否覆盖成功、等待、失败和恢复四条核心链路。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)

    assert len(eval_cases) == 4
    tags = {
        tag
        for eval_case in eval_cases
        for tag in eval_case.tags
    }
    assert {
        "completed",
        "awaiting_input",
        "blocking_failure",
        "resume",
    }.issubset(tags)


@pytest.mark.asyncio
async def test_orchestration_runtime_should_use_real_stage_chain() -> None:
    """
    检查评估工具箱是否通过真实总编排器完成三个阶段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = load_agent_evaluation_cases(DATASET_PATH)[0]
    runtime = build_multi_agent_orchestration_scenario_runtime(eval_case)

    result = await runtime.run()

    assert result.status == "completed"
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
        "aggregation",
    ]
    assert len(runtime.planning_provider.prompts) == 1
    assert len(runtime.aggregation_provider.prompts) == 1


@pytest.mark.asyncio
async def test_awaiting_orchestration_should_not_call_aggregator() -> None:
    """
    检查 Worker 等待输入时总编排器是否停止且不调用聚合器。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = load_agent_evaluation_cases(DATASET_PATH)[1]
    runtime = build_multi_agent_orchestration_scenario_runtime(eval_case)

    result = await runtime.run()

    assert result.status == "awaiting_input"
    assert result.metadata["orchestration"]["visited_stages"] == [
        "planning",
        "scheduling",
    ]
    assert runtime.aggregation_provider.prompts == []
