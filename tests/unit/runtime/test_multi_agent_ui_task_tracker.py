"""多 Agent UI 运行中任务追踪器测试。"""

from __future__ import annotations

import pytest

from src.runtime.services.multi_agent_ui_task_tracker import (
    MultiAgentUiTaskTracker,
)


def test_ui_task_tracker_should_register_get_and_unregister() -> None:
    """
    检查 UI 会话任务编号可以登记、读取和按原编号清理。

    参数含义：无。
    返回值含义：None。
    """

    tracker = MultiAgentUiTaskTracker()
    tracker.register("session_001", "multi_agent_task_trace_001")

    assert tracker.get("session_001") == "multi_agent_task_trace_001"
    assert tracker.unregister(
        "session_001",
        "multi_agent_task_trace_001",
    ) is True
    assert tracker.get("session_001") is None


def test_ui_task_tracker_should_not_remove_newer_task() -> None:
    """
    检查旧请求清理动作不能误删编号不一致的任务。

    参数含义：无。
    返回值含义：None。
    """

    tracker = MultiAgentUiTaskTracker()
    tracker.register("session_001", "multi_agent_task_new")

    assert tracker.unregister(
        "session_001",
        "multi_agent_task_old",
    ) is False
    assert tracker.get("session_001") == "multi_agent_task_new"


def test_ui_task_tracker_should_reject_concurrent_session_request() -> None:
    """
    检查同一 UI 会话不能覆盖仍在运行的请求编号。

    参数含义：无。
    返回值含义：None。
    """

    tracker = MultiAgentUiTaskTracker()
    tracker.register("session_001", "multi_agent_task_first")

    with pytest.raises(ValueError, match="已经有正在运行的请求"):
        tracker.register("session_001", "multi_agent_task_second")
