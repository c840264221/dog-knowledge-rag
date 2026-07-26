import pytest

from src.agents.collaboration.contracts import (
    AgentTaskPlan,
    AgentTaskResult,
    MultiAgentTaskResult,
)
from src.evaluation import AgentEvaluationCase
from src.evaluation.evaluators import MultiAgentBehaviorEvaluator
from src.evaluation.scenarios import MultiAgentScenarioRuntime


class FakeMultiAgentRuntime:
    """
    为多 Agent 评估器单元测试提供固定任务结果和 Worker 轨迹。

    参数含义：
        task_result:
            run 方法需要返回的固定多 Agent 任务结果。

    返回值含义：
        FakeMultiAgentRuntime:
            具有异步 run 方法和 Worker 轨迹的模拟场景。
    """

    def __init__(self, task_result: MultiAgentTaskResult) -> None:
        """
        初始化模拟多 Agent 场景。

        参数含义：
            task_result:
                评估器需要读取的固定任务结果。

        返回值含义：
            None。
        """

        self.task_result = task_result
        self.worker = type(
            "FakeWorkerTrace",
            (),
            {
                "call_counts": {"step_one": 1},
                "calls": [{"step_id": "step_one"}],
            },
        )()

    async def run(self) -> MultiAgentTaskResult:
        """
        返回固定多 Agent 任务结果。

        参数含义：
            无。

        返回值含义：
            MultiAgentTaskResult:
                初始化时保存的标准任务结果。
        """

        return self.task_result


class FakeFailedMultiAgentRuntime:
    """
    为预期运行时错误评估提供固定异常。

    参数含义：
        error_message:
            run 方法需要抛出的固定错误信息。

    返回值含义：
        FakeFailedMultiAgentRuntime:
            具有异步 run 方法的失败场景对象。
    """

    def __init__(self, error_message: str) -> None:
        """
        初始化固定失败场景。

        参数含义：
            error_message:
                run 方法抛出的错误信息。

        返回值含义：
            None。
        """

        self.error_message = error_message

    async def run(self) -> MultiAgentTaskResult:
        """
        抛出预设运行时错误。

        参数含义：
            无。

        返回值含义：
            MultiAgentTaskResult:
                本方法不会正常返回，而是抛出 ValueError。
        """

        raise ValueError(self.error_message)


def build_completed_task_result() -> MultiAgentTaskResult:
    """
    构建评估器单元测试使用的调度完成结果。

    参数含义：
        无。

    返回值含义：
        MultiAgentTaskResult:
            包含一个 completed 步骤且等待聚合的标准结果。
    """

    plan = AgentTaskPlan(
        plan_id="plan_test",
        objective="测试多 Agent 评估器",
        status="completed",
        steps=[
            {
                "step_id": "step_one",
                "title": "步骤一",
                "assigned_agent": "worker_agent",
                "status": "completed",
            }
        ],
    )
    return MultiAgentTaskResult(
        collaboration_id="multi_agent_test",
        plan=plan,
        status="running",
        task_results=[
            AgentTaskResult(
                step_id="step_one",
                assigned_agent="worker_agent",
                status="completed",
            )
        ],
        metadata={
            "ready_batches": [["step_one"]],
            "worker_step_trace": [
                {
                    "step_id": "step_one",
                    "step_title": "步骤一",
                    "assigned_agent": "worker_agent",
                    "depends_on": [],
                    "batch_numbers": [1],
                    "status": "completed",
                    "attempt_count": 1,
                    "latency_ms": 1.0,
                    "timed_out": False,
                    "cancelled": False,
                }
            ],
            "awaiting_result_aggregation": True,
        },
    )


@pytest.mark.asyncio
async def test_multi_agent_evaluator_should_build_passed_result() -> None:
    """
    测试多 Agent 调度行为符合黄金期望时生成通过结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = FakeMultiAgentRuntime(build_completed_task_result())
    evaluator = MultiAgentBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="multi_agent_eval_001",
        category="multi_agent_behavior",
        question="执行测试计划",
        expected={
            "task_status": "running",
            "plan_status": "completed",
            "ready_batches": [["step_one"]],
            "worker_call_counts": {"step_one": 1},
            "worker_step_trace_statuses": {
                "step_one": "completed",
            },
            "worker_step_trace_batch_numbers": {
                "step_one": [1],
            },
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["step_statuses"] == {
        "step_one": "completed",
    }
    assert result.output["awaiting_result_aggregation"] is True


@pytest.mark.asyncio
async def test_multi_agent_evaluator_should_expose_mismatch() -> None:
    """
    测试步骤状态与黄金期望不一致时暴露具体失败项。

    参数含义：
        无。

    返回值含义：
        None。
    """

    runtime = FakeMultiAgentRuntime(build_completed_task_result())
    evaluator = MultiAgentBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="multi_agent_eval_002",
        category="multi_agent_behavior",
        question="执行测试计划",
        expected={
            "task_status": "failed",
            "completed_step_ids": [],
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert {
        check.check_name
        for check in result.failed_checks()
    } == {
        "task_status",
        "completed_step_ids",
    }


@pytest.mark.asyncio
async def test_multi_agent_evaluator_should_reject_unknown_field() -> None:
    """
    测试评估器不会静默忽略未知期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    evaluator = MultiAgentBehaviorEvaluator()
    eval_case = AgentEvaluationCase(
        case_id="multi_agent_eval_invalid_001",
        category="multi_agent_behavior",
        question="执行测试计划",
        expected={
            "unknown_field": True,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)


@pytest.mark.asyncio
async def test_multi_agent_evaluator_should_compare_expected_runtime_error() -> None:
    """
    测试恢复校验类预期异常可以转换成结构化检查结果。

    功能：
        只有黄金用例明确声明 runtime_error_message 时，评估器才把运行异常
        当作可比较行为；未声明的异常仍然是评估执行错误。

    参数含义：
        无。

    返回值含义：
        None。
    """

    expected_error = "恢复任务仍缺少等待步骤的用户回答"
    runtime = FakeFailedMultiAgentRuntime(expected_error)
    evaluator = MultiAgentBehaviorEvaluator(
        scenario_runtime_builder=lambda eval_case: runtime,
    )
    eval_case = AgentEvaluationCase(
        case_id="multi_agent_eval_expected_error_001",
        category="multi_agent_behavior",
        question="尝试恢复缺少回答的任务",
        expected={
            "runtime_error_message": expected_error,
        },
    )

    result = await evaluator.evaluate_case(eval_case)

    assert result.passed is True
    assert result.error_message is None
    assert result.output == {
        "runtime_error_message": expected_error,
    }


def test_multi_agent_runtime_type_is_exported() -> None:
    """
    测试多 Agent 场景运行时可以从统一 scenarios 包导入。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert MultiAgentScenarioRuntime.__name__ == "MultiAgentScenarioRuntime"
