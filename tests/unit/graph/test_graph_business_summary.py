import pytest

from src.graph.graph_run import build_graph_business_summary


def test_normal_graph_result_should_default_to_completed_business() -> None:
    """测试普通 Agent 主图结果默认视为业务完成。"""

    summary = build_graph_business_summary(
        {
            "final_answer": "金毛通常很友好。",
            "multi_agent_task_result": {},
        }
    )

    assert summary["business_status"] == "completed"
    assert summary["business_error"] is None


@pytest.mark.parametrize(
    ("task_status", "expected_error_code"),
    [
        ("completed", None),
        ("partial", None),
        ("cancelled", "MULTI_AGENT_TASK_CANCELLED"),
    ],
)
def test_multi_agent_terminal_status_should_be_preserved(
    task_status: str,
    expected_error_code: str | None,
) -> None:
    """测试多 Agent 完成、部分成功和取消状态会进入主图摘要。"""

    summary = build_graph_business_summary(
        {
            "multi_agent_task_result": {
                "status": task_status,
                "final_answer": (
                    "多 Agent 任务已取消。"
                    if task_status == "cancelled"
                    else "任务已有结果。"
                ),
                "task_results": [],
            }
        }
    )

    assert summary["business_status"] == task_status
    if expected_error_code is None:
        assert summary["business_error"] is None
    else:
        assert summary["business_error"]["code"] == expected_error_code


def test_multi_agent_timeout_should_build_structured_error() -> None:
    """测试 Worker 超时会生成步骤级结构化业务错误。"""

    summary = build_graph_business_summary(
        {
            "multi_agent_task_result": {
                "status": "failed",
                "error_message": "关键步骤执行失败。",
                "task_results": [
                    {
                        "step_id": "step_health",
                        "status": "failed",
                        "metadata": {
                            "timed_out": True,
                            "timeout_seconds": 120,
                            "scheduler_attempt_count": 2,
                        },
                    }
                ],
            }
        }
    )

    assert summary["business_status"] == "failed"
    assert summary["business_error"]["code"] == (
        "MULTI_AGENT_STEP_TIMEOUT"
    )
    assert summary["business_error"]["details"]["timed_out_steps"] == [
        {
            "step_id": "step_health",
            "timeout_seconds": 120,
            "attempt_count": 2,
        }
    ]


def test_multi_agent_non_timeout_failure_should_use_generic_error() -> None:
    """测试普通多 Agent 失败使用通用结构化错误码。"""

    summary = build_graph_business_summary(
        {
            "multi_agent_task_result": {
                "status": "failed",
                "error_message": "关键步骤返回失败。",
                "task_results": [
                    {
                        "step_id": "step_plan",
                        "status": "failed",
                        "metadata": {
                            "scheduler_attempt_count": 1,
                        },
                    }
                ],
            }
        }
    )

    assert summary["business_status"] == "failed"
    assert summary["business_error"] == {
        "code": "MULTI_AGENT_TASK_FAILED",
        "message": "关键步骤返回失败。",
        "details": {},
    }
