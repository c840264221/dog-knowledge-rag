"""V1.10 多轮澄清 Checkpoint 冒烟检查测试。"""

import pytest

from src.agents.tool_agent.smoke.v110_clarification_checkpoint_smoke_checks import (
    assert_clarification_checkpoint_smoke,
    validate_clarification_checkpoint_smoke,
)


def build_valid_states() -> tuple[dict, dict]:
    """构建通过冒烟校验所需的两轮测试 state。"""

    return (
        {
            "tool_agent_clarification_request": {
                "status": "pending",
            },
            "tool_agent_response": {
                "status": "awaiting_clarification",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_list_tables",
                "args": {},
            },
        },
        {
            "tool_agent_clarification_resolution": {
                "action": "resumed",
            },
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "sqlite_list_tables",
                }
            ],
            "tool_agent_clarification_request": None,
            "tool_agent_pending_tool_call": None,
            "tool_agent_clarification_resume_ready": False,
        },
    )


def test_validate_smoke_should_pass_valid_two_turn_state() -> None:
    """测试合法的两轮状态会动态生成 PASS 结果。"""

    first_state, second_state = build_valid_states()
    result = validate_clarification_checkpoint_smoke(
        first_state=first_state,
        second_state=second_state,
        first_thread_id="conversation-1",
        second_thread_id="conversation-1",
        first_trace_id="trace-1",
        second_trace_id="trace-2",
        final_answer="查询完成。",
    )

    assert result.passed is True
    assert result.errors == []
    assert result.same_thread_id is True
    assert result.different_trace_ids is True


def test_validate_smoke_should_report_invalid_state() -> None:
    """测试缺少澄清和工具结果时返回清晰错误。"""

    result = validate_clarification_checkpoint_smoke(
        first_state={},
        second_state={},
        first_thread_id="conversation-1",
        second_thread_id="conversation-2",
        first_trace_id="trace-1",
        second_trace_id="trace-1",
        final_answer="",
    )

    assert result.passed is False
    assert result.errors
    assert result.clarification_saved is False
    assert result.tool_executed is False


def test_validate_smoke_should_include_runtime_error() -> None:
    """测试真实主图异常会写入动态失败报告。"""

    first_state, second_state = build_valid_states()
    result = validate_clarification_checkpoint_smoke(
        first_state=first_state,
        second_state=second_state,
        first_thread_id="conversation-1",
        second_thread_id="conversation-1",
        first_trace_id="trace-1",
        second_trace_id="trace-2",
        final_answer="查询完成。",
        runtime_error="LLM 调用失败",
    )

    assert result.passed is False
    assert "真实主图执行异常" in result.errors[-1]


def test_assert_smoke_should_raise_when_failed() -> None:
    """测试失败报告传入断言函数时抛出 AssertionError。"""

    result = validate_clarification_checkpoint_smoke(
        first_state={},
        second_state={},
        first_thread_id="",
        second_thread_id="",
        first_trace_id="",
        second_trace_id="",
    )

    with pytest.raises(
        AssertionError,
    ):
        assert_clarification_checkpoint_smoke(
            result=result,
        )
