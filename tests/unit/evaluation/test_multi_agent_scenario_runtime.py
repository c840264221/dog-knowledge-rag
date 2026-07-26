from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import MultiAgentBehaviorEvaluator


DATASET_PATH = Path(
    "evaluation/datasets/multi_agent_behavior_cases.json"
)


def test_multi_agent_behavior_dataset_should_define_core_scenarios() -> None:
    """
    测试多 Agent 黄金集覆盖 V1.16 基础场景和 V1.17 韧性边界。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)

    assert len(eval_cases) == 12
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
            "runtime_cancellation",
            "timeout",
            "retry_exhausted",
            "resume_validation",
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


@pytest.mark.asyncio
async def test_runtime_cancellation_should_stop_worker_without_retry() -> None:
    """
    测试 Worker 真正启动后取消时不会进入重试。

    功能：
        通过 Worker 启动事件建立确定同步点，再验证共享取消令牌终止当前
        Worker、跳过后续步骤并保持单次调用。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)
    eval_case = next(
        case
        for case in eval_cases
        if case.case_id == "multi_agent_runtime_cancelled_001"
    )

    result = await MultiAgentBehaviorEvaluator().evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["worker_call_counts"] == {
        "step_running": 1,
    }
    assert result.output["cancelled_worker_step_ids"] == [
        "step_running",
    ]
    assert result.output["task_status"] == "cancelled"
    assert result.output["cancellation_response_latency_recorded"] is True
    assert result.output["worker_step_trace_statuses"] == {
        "step_running": "skipped",
        "step_after": "skipped",
    }
    assert result.output["worker_step_trace_batch_numbers"] == {
        "step_running": [1],
        "step_after": [],
    }


@pytest.mark.asyncio
async def test_timeout_should_retry_until_attempts_are_exhausted() -> None:
    """
    测试异步 Worker 连续超时后生成结构化失败结果。

    功能：
        使用永不主动完成的确定性 Worker 触发真实 Scheduler 超时，验证
        最大尝试次数、timed_out 元数据和最终 failed 状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_cases = load_agent_evaluation_cases(DATASET_PATH)
    eval_case = next(
        case
        for case in eval_cases
        if case.case_id == "multi_agent_timeout_exhausted_001"
    )

    result = await MultiAgentBehaviorEvaluator().evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["worker_call_counts"] == {
        "step_timeout": 2,
    }
    assert result.output["step_attempt_counts"] == {
        "step_timeout": 2,
    }
    assert result.output["timed_out_step_ids"] == [
        "step_timeout",
    ]
