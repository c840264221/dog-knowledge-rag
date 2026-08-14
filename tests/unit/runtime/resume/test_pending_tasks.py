"""统一等待任务模型与活动任务集合测试。"""

import pytest
from pydantic import ValidationError

from src.runtime.resume.pending_tasks import (
    DuplicatePendingTaskError,
    InvalidPendingTaskRegistrationError,
    InvalidPendingTaskTransitionError,
    PendingInputContract,
    PendingTaskCollection,
    PendingTaskNotFoundError,
    PendingTaskSnapshot,
    PendingTaskVersionConflictError,
    SkillPendingPayload,
    ToolPendingPayload,
    build_pending_task_id,
)


def _build_tool_task(
    *,
    task_id: str = "pending_tool_task_001",
    status: str = "awaiting_input",
) -> PendingTaskSnapshot:
    """
    构建等待数据库别名的统一 Tool 测试任务。

    参数含义：
        task_id:
            测试任务编号。
        status:
            初始任务状态。

    返回值含义：
        PendingTaskSnapshot:
            字段完整且时间固定的测试任务快照。
    """

    return PendingTaskSnapshot(
        task_id=task_id,
        task_kind="tool",
        status=status,
        user_id="test_user",
        thread_id="test_thread",
        title="查询数据库表",
        pending_prompt="请选择数据库别名。",
        input_contracts=[
            PendingInputContract(
                field_id="database_name",
                value_type="string",
                enum_values=["memory", "rag"],
            )
        ],
        payload=ToolPendingPayload(
            tool_name="sqlite_list_tables",
            arguments={},
            missing_fields=["database_name"],
        ),
        created_at="2026-08-14T00:00:00+00:00",
        updated_at="2026-08-14T00:00:00+00:00",
    )


def test_build_pending_task_id_should_use_kind_and_unique_token() -> None:
    """
    测试任务编号由系统生成，且业务名称不参与唯一性判断。

    参数含义：
        无。

    返回值含义：
        None。
    """

    first_id = build_pending_task_id(
        "tool",
        token_factory=lambda: "fixed-token-001",
    )
    second_id = build_pending_task_id(
        "tool",
        token_factory=lambda: "fixed-token-002",
    )

    assert first_id == "pending_tool_fixedtoken001"
    assert second_id == "pending_tool_fixedtoken002"
    assert first_id != second_id


def test_snapshot_should_reject_payload_kind_mismatch() -> None:
    """
    测试 Tool 公共类型不能携带 Skill 专属载荷。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="payload_kind"):
        PendingTaskSnapshot(
            task_id="pending_tool_mismatch",
            task_kind="tool",
            user_id="test_user",
            thread_id="test_thread",
            title="错误任务",
            pending_prompt="请补充信息。",
            input_contracts=[
                PendingInputContract(
                    field_id="current_behavior",
                    value_type="string",
                )
            ],
            payload=SkillPendingPayload(
                skill_id="dog-training-plan",
                target_agent="dog_knowledge_agent",
            ),
        )


def test_collection_should_reject_duplicate_task_id() -> None:
    """
    测试活动集合不会让后注册任务静默覆盖同编号任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])

    with pytest.raises(DuplicatePendingTaskError):
        collection.register(task)


def test_collection_state_should_round_trip_plain_dict() -> None:
    """
    测试活动集合可以转换为普通字典并从 Checkpoint 数据恢复。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    raw_state = PendingTaskCollection([task]).to_state()

    assert isinstance(raw_state[task.task_id], dict)
    assert isinstance(raw_state[task.task_id]["payload"], dict)
    assert raw_state[task.task_id]["payload"]["payload_kind"] == "tool"

    restored_collection = PendingTaskCollection.from_state(raw_state)
    restored_task = restored_collection.require(task.task_id)

    assert restored_task == task
    assert restored_task.payload.tool_name == "sqlite_list_tables"


def test_transition_should_increment_version_and_update_status() -> None:
    """
    测试合法状态迁移会更新状态并递增乐观锁版本号。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])

    running_task = collection.transition(
        task_id=task.task_id,
        target_status="running",
        expected_version=1,
    )

    assert running_task.status == "running"
    assert running_task.version == 2
    assert collection.require(task.task_id) == running_task


def test_refresh_waiting_task_should_update_payload_and_version() -> None:
    """
    测试部分补参后可以刷新恢复 Payload，并保持等待输入状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])
    refreshed_payload = task.payload.model_copy(
        update={
            "arguments": {"database_name": "memory"},
            "missing_fields": ["table_name"],
            "resume_state": {
                "tool_agent_pending_tool_call": {
                    "name": "sqlite_describe_table",
                    "args": {"database_name": "memory"},
                }
            },
        }
    )
    latest_snapshot = task.model_copy(
        update={
            "pending_prompt": "请继续补充表名。",
            "payload": refreshed_payload,
        }
    )

    refreshed_task = collection.refresh_waiting_task(
        latest_snapshot,
        expected_version=1,
    )

    assert refreshed_task.status == "awaiting_input"
    assert refreshed_task.version == 2
    assert refreshed_task.created_at == task.created_at
    assert refreshed_task.payload.arguments == {
        "database_name": "memory"
    }
    assert collection.require(task.task_id) == refreshed_task


def test_transition_should_reject_stale_expected_version() -> None:
    """
    测试旧版本调用方不能覆盖已经变化的任务状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])
    collection.transition(
        task_id=task.task_id,
        target_status="running",
        expected_version=1,
    )

    with pytest.raises(PendingTaskVersionConflictError):
        collection.transition(
            task_id=task.task_id,
            target_status="cancelled",
            expected_version=1,
        )


def test_terminal_transition_should_remove_task_from_active_collection() -> None:
    """
    测试进入终态后返回审计快照，并从活动集合自动移除任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])

    cancelled_task = collection.transition(
        task_id=task.task_id,
        target_status="cancelled",
        expected_version=1,
    )

    assert cancelled_task.status == "cancelled"
    assert cancelled_task.version == 2
    assert collection.get(task.task_id) is None
    assert collection.to_state() == {}
    with pytest.raises(PendingTaskNotFoundError):
        collection.require(task.task_id)


def test_transition_should_reject_invalid_state_change() -> None:
    """
    测试状态机拒绝从等待输入直接跳到已完成。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = _build_tool_task()
    collection = PendingTaskCollection([task])

    with pytest.raises(InvalidPendingTaskTransitionError):
        collection.transition(
            task_id=task.task_id,
            target_status="completed",
            expected_version=1,
        )


def test_register_should_reject_non_waiting_task() -> None:
    """
    测试首次注册只接受等待用户输入的任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    running_task = _build_tool_task(status="running")

    with pytest.raises(InvalidPendingTaskRegistrationError):
        PendingTaskCollection([running_task])


def test_from_state_should_restore_active_and_prune_terminal_tasks() -> None:
    """
    测试 Checkpoint 恢复保留活动任务并裁剪历史终态任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    running_task = _build_tool_task(
        task_id="pending_tool_running",
        status="running",
    )
    cancelled_task = _build_tool_task(
        task_id="pending_tool_cancelled",
        status="cancelled",
    )
    raw_state = {
        running_task.task_id: running_task.model_dump(mode="json"),
        cancelled_task.task_id: cancelled_task.model_dump(mode="json"),
    }

    restored_collection = PendingTaskCollection.from_state(raw_state)

    assert restored_collection.require(running_task.task_id) == running_task
    assert restored_collection.get(cancelled_task.task_id) is None
    assert list(restored_collection.to_state()) == [running_task.task_id]


def test_list_tasks_should_filter_status_with_stable_order() -> None:
    """
    测试任务列表可以按状态过滤并保持稳定顺序。

    参数含义：
        无。

    返回值含义：
        None。
    """

    later_task = _build_tool_task(task_id="pending_tool_task_002")
    earlier_task = _build_tool_task(task_id="pending_tool_task_001")
    running_task = _build_tool_task(
        task_id="pending_tool_task_003",
    )
    collection = PendingTaskCollection(
        [later_task, running_task, earlier_task]
    )
    collection.transition(
        task_id=running_task.task_id,
        target_status="running",
        expected_version=1,
    )

    waiting_tasks = collection.list_tasks(status="awaiting_input")

    assert [task.task_id for task in waiting_tasks] == [
        "pending_tool_task_001",
        "pending_tool_task_002",
    ]
