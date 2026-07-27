from src.api.task_registry import ApiTaskRegistry


def test_task_registry_should_store_immutable_status_snapshots() -> None:
    """测试任务登记表会保存并更新不可变状态快照。"""

    registry = ApiTaskRegistry()
    started = registry.start(
        multi_agent_task_id="multi_agent_task_trace_001",
        trace_id="trace_001",
        session_id="session_001",
    )
    updated = registry.update(
        "multi_agent_task_trace_001",
        status="cancel_requested",
    )

    assert started.status == "running"
    assert updated is not None
    assert updated.status == "cancel_requested"
    assert updated.created_at == started.created_at
    assert registry.get(
        "multi_agent_task_trace_001"
    ) == updated


def test_task_registry_should_return_none_for_unknown_task() -> None:
    """测试更新或查询未知任务时安全返回 None。"""

    registry = ApiTaskRegistry()

    assert registry.get("missing") is None
    assert registry.update("missing", status="failed") is None
