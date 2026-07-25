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


def test_multi_agent_runtime_type_is_exported() -> None:
    """
    测试多 Agent 场景运行时可以从统一 scenarios 包导入。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert MultiAgentScenarioRuntime.__name__ == "MultiAgentScenarioRuntime"
