"""长任务人工交互边界的纯业务服务。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
from uuid import uuid4

from src.runtime.long_tasks.contracts import (
    LongTask,
    LongTaskPendingInteraction,
    LongTaskStep,
    LongTaskWaitingReason,
)
from src.runtime.long_tasks.state_machine import (
    transition_long_task,
    transition_long_task_step,
)


LongTaskApprovalAction = Literal["continue", "cancel"]
LongTaskMissingInputAction = Literal["submit_input", "cancel"]
InteractionIdFactory = Callable[[], str]


class InvalidLongTaskInteractionError(ValueError):
    """表示等待或恢复操作不符合当前长任务交互边界。"""


def _default_interaction_id_factory() -> str:
    """
    生成默认的等待交互唯一编号。

    返回值含义：
        str:
            带 ``interaction_`` 前缀的随机 UUID 字符串。
    """

    return f"interaction_{uuid4()}"


class LongTaskInteractionService:
    """
    使用业务动作封装长任务多个关联字段的同步变化。

    功能：
        负责创建人工批准边界，并根据用户动作恢复或取消任务。当前 MVP
        只操作不可变任务快照，不负责数据库持久化、Event 追加或 Redis
        入队；这些副作用将在 TaskStore 接入后由更外层用例协调。

    参数含义：
        interaction_id_factory:
            可选交互编号工厂。生产环境默认生成 UUID，测试可以注入固定值。

    返回值含义：
        LongTaskInteractionService:
            可重复使用、且不保存任务可变状态的长任务应用服务。
    """

    def __init__(
        self,
        *,
        interaction_id_factory: InteractionIdFactory | None = None,
    ) -> None:
        self._interaction_id_factory = (
            interaction_id_factory or _default_interaction_id_factory
        )

    def wait_for_approval(
        self,
        task: LongTask,
        *,
        source_step_ids: Sequence[str],
        target_step_ids: Sequence[str],
        prompt: str,
    ) -> LongTask:
        """
        在步骤边界暂停任务，并保存用户批准后应继续的目标步骤。

        参数含义：
            task:
                当前处于 running、且活动批次已经结束的任务快照。
            source_step_ids:
                导致本次批准请求产生的已完成步骤编号。
            target_step_ids:
                用户批准后准备进入下一个执行批次的 Ready 步骤编号。
            prompt:
                展示给用户的明确批准问题。

        返回值含义：
            LongTask:
                已进入 awaiting_input、活动步骤为空、并携带结构化等待交互
                的新任务快照。
        """

        if task.status != "running":
            raise InvalidLongTaskInteractionError(
                "只有 running 任务可以进入人工批准边界"
            )
        if task.active_step_ids:
            raise InvalidLongTaskInteractionError(
                "进入人工批准边界前必须先结束当前活动批次"
            )

        steps_by_id = {step.step_id: step for step in task.steps}
        source_ids = list(source_step_ids)
        target_ids = list(target_step_ids)
        self._require_step_statuses(
            steps_by_id=steps_by_id,
            step_ids=source_ids,
            allowed_statuses={"completed"},
            role="来源",
        )
        self._require_step_statuses(
            steps_by_id=steps_by_id,
            step_ids=target_ids,
            allowed_statuses={"ready"},
            role="目标",
        )

        interaction = LongTaskPendingInteraction(
            interaction_id=self._interaction_id_factory(),
            interaction_type="approval",
            source_step_ids=source_ids,
            target_step_ids=target_ids,
            prompt=prompt,
            allowed_actions=["continue", "cancel"],
        )
        return transition_long_task(
            task,
            target_status="awaiting_input",
            pending_interaction=interaction,
            active_step_ids=[],
        )

    def wait_for_step_input(
        self,
        task: LongTask,
        *,
        waiting_step_ids: Sequence[str],
        interaction_type: LongTaskWaitingReason,
        prompt: str,
        input_contract: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LongTask:
        """
        暂停当前批次并保存一个或多个 Step 的用户输入要求。

        参数含义：
            task:
                已应用批次步骤结果、活动批次已清空的 running 任务草稿。
            waiting_step_ids:
                当前状态已经变为 awaiting_input 的步骤编号。
            interaction_type:
                聚合后的缺少输入、确认或批准类型。
            prompt:
                API 可以直接展示给用户的聚合提示。
            input_contract:
                前端渲染输入控件时使用的可选结构化要求。
            metadata:
                批次编号和逐 Step 提示等非核心诊断信息。

        返回值含义：
            LongTask:
                Task 已进入 awaiting_input、等待 Step 保持活动语义，并携带
                一个结构化 pending_interaction 的新快照。
        """

        if task.status != "running":
            raise InvalidLongTaskInteractionError(
                "只有 running 任务可以等待 Step 输入"
            )
        if task.active_step_ids:
            raise InvalidLongTaskInteractionError(
                "等待 Step 输入前必须先结束当前活动批次"
            )
        steps_by_id = {step.step_id: step for step in task.steps}
        waiting_ids = list(waiting_step_ids)
        self._require_step_statuses(
            steps_by_id=steps_by_id,
            step_ids=waiting_ids,
            allowed_statuses={"awaiting_input"},
            role="等待",
        )
        allowed_actions = (
            ["submit_input", "cancel"]
            if interaction_type == "missing_input"
            else ["continue", "cancel"]
        )
        interaction = LongTaskPendingInteraction(
            interaction_id=self._interaction_id_factory(),
            interaction_type=interaction_type,
            source_step_ids=waiting_ids,
            target_step_ids=waiting_ids,
            prompt=prompt,
            allowed_actions=allowed_actions,
            input_contract=dict(input_contract or {}),
            metadata=dict(metadata or {}),
        )
        return transition_long_task(
            task,
            target_status="awaiting_input",
            pending_interaction=interaction,
            active_step_ids=waiting_ids,
        )

    def resume_after_approval(
        self,
        task: LongTask,
        *,
        interaction_id: str,
        action: LongTaskApprovalAction,
    ) -> LongTask:
        """
        校验用户响应属于当前等待交互，并继续或取消任务。

        参数含义：
            task:
                当前处于 awaiting_input 的任务快照。
            interaction_id:
                用户响应所对应的交互编号，用于拒绝迟到或重复响应。
            action:
                ``continue`` 表示启动目标批次，``cancel`` 表示取消任务。

        返回值含义：
            LongTask:
                继续时返回 running 快照并激活目标步骤；取消时返回 cancelled
                快照。两种结果都会清除已经消费的等待交互。
        """

        interaction = task.pending_interaction
        if task.status != "awaiting_input" or interaction is None:
            raise InvalidLongTaskInteractionError(
                "当前任务没有可处理的等待交互"
            )
        if interaction.interaction_id != interaction_id:
            raise InvalidLongTaskInteractionError(
                "交互编号已过期或不属于当前任务等待项"
            )
        if action not in interaction.allowed_actions:
            raise InvalidLongTaskInteractionError(
                f"当前等待交互不允许动作: {action}"
            )

        if action == "cancel":
            return transition_long_task(
                task,
                target_status="cancelled",
                active_step_ids=[],
            )
        if action != "continue":
            raise InvalidLongTaskInteractionError(
                f"长任务 MVP 尚不支持动作: {action}"
            )

        return transition_long_task(
            task,
            target_status="running",
            active_step_ids=interaction.target_step_ids,
        )

    def respond_to_missing_input(
        self,
        task: LongTask,
        *,
        interaction_id: str,
        action: LongTaskMissingInputAction,
        answers: Mapping[str, Any],
    ) -> LongTask:
        """
        消费缺少输入交互，把逐 Step 回答写入恢复参数并继续或取消任务。

        参数含义：
            task:
                当前处于 awaiting_input 的最新长任务快照。
            interaction_id:
                用户响应对应的交互编号，用于拒绝迟到或重复响应。
            action:
                submit_input 表示提交回答并恢复，cancel 表示取消整份任务。
            answers:
                以等待 Step 编号为键的用户回答；必须完整覆盖目标步骤且不能
                包含其他步骤，系统不会猜测一份回答属于哪个 Step。

        返回值含义：
            LongTask:
                提交回答时返回等待 Step 已变为 ready、Task 已恢复 running
                且活动集合为空的新快照；取消时返回 cancelled 快照。
        """

        interaction = task.pending_interaction
        if task.status != "awaiting_input" or interaction is None:
            raise InvalidLongTaskInteractionError(
                "当前任务没有可处理的等待交互"
            )
        if interaction.interaction_id != interaction_id:
            raise InvalidLongTaskInteractionError(
                "交互编号已过期或不属于当前任务等待项"
            )
        if interaction.interaction_type != "missing_input":
            raise InvalidLongTaskInteractionError(
                "当前交互不是缺少输入类型"
            )
        if action not in interaction.allowed_actions:
            raise InvalidLongTaskInteractionError(
                f"当前等待交互不允许动作: {action}"
            )
        if action == "cancel":
            return transition_long_task(
                task,
                target_status="cancelled",
                active_step_ids=[],
            )
        if action != "submit_input":
            raise InvalidLongTaskInteractionError(
                f"缺少输入交互不支持动作: {action}"
            )

        target_ids = list(interaction.target_step_ids)
        normalized_answers = dict(answers)
        expected_ids = set(target_ids)
        provided_ids = set(normalized_answers)
        if provided_ids != expected_ids:
            raise InvalidLongTaskInteractionError(
                "用户回答必须与等待 Step 完整对应: "
                f"expected={sorted(expected_ids)}, "
                f"provided={sorted(provided_ids)}"
            )
        empty_answer_ids = [
            step_id
            for step_id, answer in normalized_answers.items()
            if not _has_meaningful_answer(answer)
        ]
        if empty_answer_ids:
            raise InvalidLongTaskInteractionError(
                "等待 Step 的用户回答不能为空: "
                f"{sorted(empty_answer_ids)}"
            )

        steps_by_id = {step.step_id: step for step in task.steps}
        self._require_step_statuses(
            steps_by_id=steps_by_id,
            step_ids=target_ids,
            allowed_statuses={"awaiting_input"},
            role="等待",
        )
        updated_steps = [
            _build_resumable_step(
                step,
                answer=normalized_answers[step.step_id],
            )
            if step.step_id in expected_ids
            else step
            for step in task.steps
        ]
        task_data = task.model_dump(mode="python")
        task_data["steps"] = updated_steps
        transition_base = LongTask.model_validate(task_data)
        return transition_long_task(
            transition_base,
            target_status="running",
            active_step_ids=[],
        )

    @staticmethod
    def _require_step_statuses(
        *,
        steps_by_id: dict[str, LongTaskStep],
        step_ids: Sequence[str],
        allowed_statuses: set[str],
        role: str,
    ) -> None:
        """
        检查交互来源或目标步骤存在，并处于指定状态集合。

        参数含义：
            steps_by_id:
                以 step_id 为键的任务步骤索引。
            step_ids:
                需要校验的来源或目标步骤编号。
            allowed_statuses:
                当前业务角色允许的步骤状态。
            role:
                错误信息中使用的“来源”或“目标”角色名称。

        返回值含义：
            None。步骤不存在、列表为空或状态错误时抛出业务异常。
        """

        if not step_ids:
            raise InvalidLongTaskInteractionError(
                f"批准交互必须至少包含一个{role}步骤"
            )
        for step_id in step_ids:
            step = steps_by_id.get(step_id)
            if step is None:
                raise InvalidLongTaskInteractionError(
                    f"{role}步骤不存在: {step_id}"
                )
            status = step.status
            if status not in allowed_statuses:
                expected = ", ".join(sorted(allowed_statuses))
                raise InvalidLongTaskInteractionError(
                    f"{role}步骤状态必须是 {expected}: "
                    f"{step_id}={status}"
                )


def _build_resumable_step(
    step: LongTaskStep,
    *,
    answer: Any,
) -> LongTaskStep:
    """
    把一个等待 Step 转为 ready，并写入现有 Agent Worker 认识的恢复字段。

    参数含义：
        step:
            当前状态为 awaiting_input 的长任务步骤。
        answer:
            用户为当前步骤提交的自然语言或结构化回答。

    返回值含义：
        LongTaskStep:
            版本递增、等待原因清除且携带恢复上下文的 ready 步骤快照。
    """

    ready_step = transition_long_task_step(
        step,
        target_status="ready",
    )
    step_data = ready_step.model_dump(mode="python")
    step_data["input_data"] = {
        **ready_step.input_data,
        "multi_agent_resume_input": answer,
        "multi_agent_previous_worker_output": {
            "status": "awaiting_input",
            "summary": step.output_summary,
            "output_ref": step.output_ref,
        },
        "multi_agent_is_resuming": True,
    }
    return LongTaskStep.model_validate(step_data)


def _has_meaningful_answer(answer: Any) -> bool:
    """
    判断用户回答是否包含可交给 Worker 的有效内容。

    参数含义：
        answer:
            自然语言、结构化字典、列表、数字或其他 JSON 兼容值。

    返回值含义：
        bool:
            None、空白字符串和空容器返回 False，其他值返回 True。
    """

    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    if isinstance(answer, (Mapping, Sequence)) and not isinstance(
        answer,
        (str, bytes, bytearray),
    ):
        return bool(answer)
    return True
