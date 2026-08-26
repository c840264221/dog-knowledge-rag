"""长任务应用服务与 Store 编排单元测试。"""

from __future__ import annotations

import pytest

from src.runtime.long_tasks import (
    LongTask,
    LongTaskAlreadyExistsError,
    LongTaskApplicationService,
    LongTaskGoal,
    LongTaskNotFoundError,
    LongTaskStep,
    LongTaskVersionConflictError,
)
from src.runtime.long_tasks.interaction_service import (
    LongTaskInteractionService,
)


class FakeQueuePublisher:
    """记录应用服务保存后发布的长任务恢复通知。"""

    def __init__(
        self,
        *,
        operation_log: list[str] | None = None,
        fail_on_publish: bool = False,
    ) -> None:
        """
        初始化测试发布器并配置可选失败行为。

        参数含义：
            operation_log:
                用于验证 Store 与 Publisher 调用顺序的共享记录。
            fail_on_publish:
                是否模拟 Redis Stream 发布失败。

        返回值含义：
            None。
        """

        self.messages = []
        self.operation_log = operation_log
        self.fail_on_publish = fail_on_publish

    async def publish(self, message) -> str:
        """
        保存测试消息并返回固定 Redis Stream 编号。

        参数含义：
            message:
                应用服务准备发布的轻量长任务消息。

        返回值含义：
            str:
                固定测试消息编号；启用失败模拟时抛出 RuntimeError。
        """

        if self.operation_log is not None:
            self.operation_log.append("publish")
        if self.fail_on_publish:
            raise RuntimeError("测试发布失败")
        self.messages.append(message)
        return "1000-0"


class InMemoryLongTaskStore:
    """为应用服务测试提供带乐观锁语义的内存 Store。"""

    def __init__(
        self,
        *,
        operation_log: list[str] | None = None,
    ) -> None:
        """
        初始化内存任务集合和可选操作顺序记录。

        参数含义：
            operation_log:
                用于验证创建与发布先后顺序的共享记录。

        返回值含义：
            None。
        """

        self.tasks: dict[str, LongTask] = {}
        self.saved_expected_versions: list[int] = []
        self.force_version_conflict = False
        self.operation_log = operation_log

    async def create(self, task: LongTask) -> LongTask:
        """
        创建不存在的测试任务。

        参数含义：
            task:
                版本 1 的测试任务快照。

        返回值含义：
            LongTask:
                创建成功后的原任务。
        """

        if task.task_id in self.tasks:
            raise LongTaskAlreadyExistsError("长任务已经存在")
        if self.operation_log is not None:
            self.operation_log.append("create")
        self.tasks[task.task_id] = task
        return task

    async def load(self, task_id: str) -> LongTask | None:
        """
        读取内存中的最新任务快照。

        参数含义：
            task_id:
                测试任务唯一编号。

        返回值含义：
            LongTask | None:
                任务存在时返回快照，否则返回 None。
        """

        return self.tasks.get(task_id)

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        校验当前版本后保存测试任务。

        参数含义：
            task:
                版本已经递增的新任务快照。
            expected_version:
                应用服务读取任务时看到的旧版本号。

        返回值含义：
            LongTask:
                乐观锁保存成功后的任务快照。
        """

        self.saved_expected_versions.append(expected_version)
        current_task = self.tasks.get(task.task_id)
        actual_version = current_task.version if current_task else -1
        if self.force_version_conflict or actual_version != expected_version:
            raise LongTaskVersionConflictError(
                "测试任务版本冲突: "
                f"expected={expected_version}, actual={actual_version}"
            )
        self.tasks[task.task_id] = task
        return task


def build_boundary_task() -> LongTask:
    """
    构建 Step 1 完成、Step 2 Ready 的运行中任务。

    返回值含义：
        LongTask:
            可以进入人工批准边界的版本 1 任务。
    """

    return LongTask(
        task_id="task_001",
        user_id="user_001",
        thread_id="thread_001",
        goal=LongTaskGoal(
            original_request="分步骤生成健康计划",
            objective="生成健康计划",
        ),
        steps=[
            LongTaskStep(
                step_id="step_1",
                task_id="task_001",
                title="读取档案",
                assigned_agent="profile_agent",
                status="completed",
            ),
            LongTaskStep(
                step_id="step_2",
                task_id="task_001",
                title="生成计划",
                assigned_agent="general_agent",
                depends_on=["step_1"],
                status="ready",
            ),
        ],
        status="running",
        progression_mode="guided",
    )


def build_durable_handoff_task() -> LongTask:
    """
    构建包含请求内历史和后台 Ready Step 的首次交接快照。

    返回值含义：
        LongTask:
            可由内部应用服务保存并发布 submitted 通知的版本 1 任务。
    """

    task = build_boundary_task()
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "execution_mode": "durable",
            "metadata": {
                "source_type": "collaboration_budget_handoff",
                "source_collaboration_id": "collaboration_001",
            },
        }
    )
    completed_step_data = task.steps[0].model_dump(mode="python")
    completed_step_data["metadata"] = {
        "migrated_from_inline_result": True,
    }
    task_data["steps"] = [
        LongTaskStep.model_validate(completed_step_data),
        task.steps[1],
    ]
    return LongTask.model_validate(task_data)


@pytest.mark.asyncio
async def test_handoff_should_create_then_publish_all_ready_steps() -> None:
    """
    验证内部交接先落库，再发布版本和全部 Ready Step 引用。

    返回值含义：
        None。
    """

    operation_log: list[str] = []
    store = InMemoryLongTaskStore(operation_log=operation_log)
    publisher = FakeQueuePublisher(operation_log=operation_log)
    service = LongTaskApplicationService(
        store,
        queue_publisher=publisher,
    )

    saved_task = await service.handoff_paused_collaboration_task(
        build_durable_handoff_task()
    )

    assert operation_log == ["create", "publish"]
    assert store.tasks[saved_task.task_id] == saved_task
    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message.task_id == saved_task.task_id
    assert message.task_version == 1
    assert message.reason == "submitted"
    assert message.ready_step_ids == ["step_2"]
    assert message.correlation_id == "collaboration_001"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_update", "error_message"),
    [
        ({"execution_mode": "inline"}, "durable"),
        ({"status": "created"}, "running"),
        ({"version": 2}, "版本 1"),
        ({"metadata": {}}, "预算暂停"),
    ],
)
async def test_handoff_should_reject_invalid_snapshot_before_create(
    task_update: dict,
    error_message: str,
) -> None:
    """
    验证非法协作快照不会写入 Store 或发布消息。

    参数含义：
        task_update:
            用于破坏合法交接快照的字段更新。
        error_message:
            预期业务异常中包含的提示文本。

    返回值含义：
        None。
    """

    task = build_durable_handoff_task()
    task_data = task.model_dump(mode="python")
    task_data.update(task_update)
    invalid_task = LongTask.model_validate(task_data)
    store = InMemoryLongTaskStore()
    publisher = FakeQueuePublisher()
    service = LongTaskApplicationService(
        store,
        queue_publisher=publisher,
    )

    with pytest.raises(ValueError, match=error_message):
        await service.handoff_paused_collaboration_task(invalid_task)

    assert store.tasks == {}
    assert publisher.messages == []


@pytest.mark.asyncio
async def test_handoff_should_require_ready_step_before_create() -> None:
    """
    验证没有可调度 Step 的快照不会形成无法消费的队列任务。

    返回值含义：
        None。
    """

    task = build_durable_handoff_task()
    task_data = task.model_dump(mode="python")
    pending_step_data = task.steps[1].model_dump(mode="python")
    pending_step_data["status"] = "pending"
    task_data["steps"] = [
        task.steps[0],
        LongTaskStep.model_validate(pending_step_data),
    ]
    invalid_task = LongTask.model_validate(task_data)
    store = InMemoryLongTaskStore()
    publisher = FakeQueuePublisher()
    service = LongTaskApplicationService(
        store,
        queue_publisher=publisher,
    )

    with pytest.raises(ValueError, match="Ready Step"):
        await service.handoff_paused_collaboration_task(invalid_task)

    assert store.tasks == {}
    assert publisher.messages == []


@pytest.mark.asyncio
async def test_handoff_publish_failure_should_keep_created_snapshot() -> None:
    """
    记录当前 MVP 在发布失败时保留已创建快照的一致性边界。

    返回值含义：
        None。
    """

    store = InMemoryLongTaskStore()
    publisher = FakeQueuePublisher(fail_on_publish=True)
    service = LongTaskApplicationService(
        store,
        queue_publisher=publisher,
    )
    task = build_durable_handoff_task()

    with pytest.raises(RuntimeError, match="测试发布失败"):
        await service.handoff_paused_collaboration_task(task)

    assert store.tasks[task.task_id] == task


@pytest.mark.asyncio
async def test_handoff_should_require_publisher_before_create() -> None:
    """
    验证队列未配置时不会创建无法唤醒的持久化任务。

    返回值含义：
        None。
    """

    store = InMemoryLongTaskStore()
    service = LongTaskApplicationService(store)

    with pytest.raises(RuntimeError, match="提交队列尚未配置"):
        await service.handoff_paused_collaboration_task(
            build_durable_handoff_task()
        )

    assert store.tasks == {}


def build_application_service(
    store: InMemoryLongTaskStore,
) -> LongTaskApplicationService:
    """
    构建使用固定交互编号的应用服务。

    参数含义：
        store:
            测试使用的内存任务 Store。

    返回值含义：
        LongTaskApplicationService:
            可预测 interaction_id 的应用服务。
    """

    interaction_service = LongTaskInteractionService(
        interaction_id_factory=lambda: "interaction_001"
    )
    return LongTaskApplicationService(
        store,
        interaction_service=interaction_service,
    )


@pytest.mark.asyncio
async def test_wait_for_approval_should_load_transition_and_save() -> None:
    """验证调用方只提供 task_id，应用服务完成加载、迁移和保存。"""

    store = InMemoryLongTaskStore()
    service = build_application_service(store)
    await service.create_task(build_boundary_task())

    awaiting_task = await service.wait_for_approval(
        task_id="task_001",
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="是否继续 Step 2？",
    )

    assert awaiting_task.status == "awaiting_input"
    assert awaiting_task.version == 2
    assert awaiting_task.pending_interaction is not None
    assert store.tasks["task_001"] == awaiting_task
    assert store.saved_expected_versions == [1]


@pytest.mark.asyncio
async def test_full_guided_lifecycle_should_create_wait_and_resume() -> None:
    """验证创建、等待批准、恢复和持久化组成一条完整生命周期。"""

    store = InMemoryLongTaskStore()
    service = build_application_service(store)
    created_task = await service.create_task(build_boundary_task())
    awaiting_task = await service.wait_for_approval(
        task_id="task_001",
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="是否继续 Step 2？",
    )

    running_task = await service.resume_after_approval(
        task_id="task_001",
        interaction_id="interaction_001",
        action="continue",
    )

    assert created_task.version == 1
    assert awaiting_task.status == "awaiting_input"
    assert awaiting_task.version == 2
    assert running_task.status == "running"
    assert running_task.active_step_ids == ["step_2"]
    assert running_task.pending_interaction is None
    assert running_task.version == 3
    assert store.saved_expected_versions == [1, 2]


@pytest.mark.asyncio
async def test_cancel_should_persist_terminal_snapshot() -> None:
    """验证取消操作也通过相同加载和乐观锁保存链路。"""

    store = InMemoryLongTaskStore()
    service = build_application_service(store)
    await service.create_task(build_boundary_task())
    await service.wait_for_approval(
        task_id="task_001",
        source_step_ids=["step_1"],
        target_step_ids=["step_2"],
        prompt="是否继续 Step 2？",
    )

    cancelled_task = await service.resume_after_approval(
        task_id="task_001",
        interaction_id="interaction_001",
        action="cancel",
    )

    assert cancelled_task.status == "cancelled"
    assert store.tasks["task_001"] == cancelled_task


@pytest.mark.asyncio
async def test_missing_task_should_raise_not_found_error() -> None:
    """验证不存在的 task_id 不会被静默创建。"""

    with pytest.raises(LongTaskNotFoundError, match="不存在"):
        await build_application_service(
            InMemoryLongTaskStore()
        ).wait_for_approval(
            task_id="task_missing",
            source_step_ids=["step_1"],
            target_step_ids=["step_2"],
            prompt="是否继续？",
        )


@pytest.mark.asyncio
async def test_version_conflict_should_not_be_blindly_retried() -> None:
    """验证应用服务把并发冲突交给上层重新加载和决策。"""

    store = InMemoryLongTaskStore()
    service = build_application_service(store)
    original_task = build_boundary_task()
    await service.create_task(original_task)
    store.force_version_conflict = True

    with pytest.raises(LongTaskVersionConflictError, match="版本冲突"):
        await service.wait_for_approval(
            task_id="task_001",
            source_step_ids=["step_1"],
            target_step_ids=["step_2"],
            prompt="是否继续？",
        )

    assert store.tasks["task_001"] == original_task


@pytest.mark.asyncio
async def test_missing_input_response_should_save_then_publish_resume() -> None:
    """验证回答经乐观锁保存后发布包含新版本和 Ready Step 的通知。"""

    store = InMemoryLongTaskStore()
    publisher = FakeQueuePublisher()
    interaction_service = LongTaskInteractionService(
        interaction_id_factory=lambda: "interaction_001"
    )
    task = build_boundary_task()
    step_data = task.steps[1].model_dump(mode="python")
    step_data.update(
        {
            "status": "awaiting_input",
            "waiting_reason": "missing_input",
        }
    )
    task_data = task.model_dump(mode="python")
    task_data.update(
        {
            "execution_mode": "durable",
            "steps": [
                task.steps[0],
                LongTaskStep.model_validate(step_data),
            ],
        }
    )
    waiting_task = interaction_service.wait_for_step_input(
        LongTask.model_validate(task_data),
        waiting_step_ids=["step_2"],
        interaction_type="missing_input",
        prompt="请补充狗狗年龄。",
    )
    store.tasks[waiting_task.task_id] = waiting_task
    service = LongTaskApplicationService(
        store,
        interaction_service=interaction_service,
        queue_publisher=publisher,
    )

    resumed_task = await service.respond_to_missing_input(
        task_id="task_001",
        user_id="user_001",
        interaction_id="interaction_001",
        action="submit_input",
        answers={"step_2": "6岁"},
    )

    assert resumed_task.status == "running"
    assert resumed_task.steps[1].status == "ready"
    assert resumed_task.active_step_ids == []
    assert store.saved_expected_versions == [waiting_task.version]
    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message.task_version == resumed_task.version
    assert message.reason == "resumed"
    assert message.ready_step_ids == ["step_2"]
