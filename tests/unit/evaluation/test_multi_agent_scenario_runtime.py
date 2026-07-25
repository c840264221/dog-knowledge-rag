from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import MultiAgentBehaviorEvaluator


DATASET_PATH = Path(
    "evaluation/datasets/multi_agent_behavior_cases.json"
)


def test_multi_agent_behavior_dataset_should_define_core_scenarios() -> None:
    """
    测试多 Agent 黄金集覆盖 V1.16 第一阶段的核心调度场景。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)

    assert len(eval_cases) == 6
    assert {
        tag
        for eval_case in eval_cases
        for tag in eval_case.tags
    }.issuperset(
        {
            "dependency",
            "retry",
            "partial",
            "blocking_failure",
            "awaiting_input",
            "cancelled",
        }
    )
    assert {
        eval_case.category
        for eval_case in eval_cases
    } == {
        "multi_agent_behavior",
    }


@pytest.mark.asyncio
async def test_multi_agent_behavior_dataset_should_pass_real_scheduler() -> None:
    """
    测试全部黄金用例通过真实 MultiAgentTaskScheduler 行为评估。

    功能：
        使用确定性 Worker 隔离外部 Agent 和 LLM 波动，同时真实执行生产
        Scheduler 的依赖批次、重试、失败阻断、恢复和取消逻辑。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)
    results = await MultiAgentBehaviorEvaluator().evaluate_many(eval_cases)

    assert len(results) == len(eval_cases)
    assert {
        result.case_id
        for result in results
        if not result.passed
    } == set()
