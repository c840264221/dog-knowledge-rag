"""多智能体请求自动升级为后台长任务的跨层集成测试。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentOrchestrator,
    MultiAgentTaskScheduler,
    PlannerAgent,
    ResultAggregator,
    build_multi_agent_entry_node,
)
from src.agents.collaboration.adapters import LongTaskStepExecutorAdapter
from src.api.services import AgentApiService
from src.graph.graph_run import build_graph_business_summary
from src.runtime.long_tasks import (
    LongTask,
    LongTaskAlreadyExistsError,
    LongTaskApplicationService,
    LongTaskQueueMessage,
    LongTaskQueueMessageHandler,
    LongTaskVersionConflictError,
)
from src.runtime.resume.contracts import GraphFinalResult


class FixedMessage:
    """保存测试 LLM 返回的一段固定文本。"""

    def __init__(self, content: str) -> None:
        """
        初始化固定消息。

        参数含义：
            content:
                PlannerAgent 本次读取的结构化计划文本。

        返回值含义：
            None。
        """

        self.content = content


class RecordingLLMProvider:
    """返回固定计划并记录真实 Planner 调用次数的测试 Provider。"""

    def __init__(self) -> None:
        """
        初始化提示词记录。

        返回值含义：
            None。
        """

        self.main_llm = object()
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
        call_metadata: Any | None = None,
    ) -> FixedMessage:
        """
        记录真实 Planner 提示词并返回下一条固定响应。

        参数含义：
            llm:
                Planner 或 Aggregator 选择的模型对象。
            prompt:
                本次准备发送给模型的提示词。
            fallback_response:
                真实 Provider 调用失败时使用的兜底文本。
            call_metadata:
                本次 LLM 调用的可观测元数据。

        返回值含义：
            FixedMessage:
                包含固定结构化文本的测试消息。
        """

        _ = llm, fallback_response, call_metadata
        self.prompts.append(prompt)
        plan_id_match = re.search(
            r"plan_id 必须原样返回为 '([^']+)'",
            prompt,
        )
        if plan_id_match is None:
            raise AssertionError("Planner 提示词缺少程序生成的 plan_id")
        objective_start_marker = "用户目标开始：\n"
        objective_end_marker = "\n用户目标结束。"
        objective_start = prompt.index(objective_start_marker) + len(
            objective_start_marker
        )
        objective_end = prompt.index(
            objective_end_marker,
            objective_start,
        )
        objective = prompt[objective_start:objective_end].strip()
        return FixedMessage(
            build_three_step_plan_json(
                plan_id=plan_id_match.group(1),
                objective=objective,
            )
        )


class InMemoryHandoffStore:
    """保存自动升级结果并保留最小乐观锁语义的内存 Store。"""

    def __init__(self) -> None:
        """
        初始化空任务集合。

        返回值含义：
            None。
        """

        self.tasks: dict[str, LongTask] = {}

    async def create(self, task: LongTask) -> LongTask:
        """
        创建一份尚不存在的版本 1 长任务。

        参数含义：
            task:
                入口节点适配出的 durable LongTask。

        返回值含义：
            LongTask:
                保存成功后的原任务快照。
        """

        if task.task_id in self.tasks:
            raise LongTaskAlreadyExistsError("集成测试任务已经存在")
        self.tasks[task.task_id] = task
        return task

    async def load(self, task_id: str) -> LongTask | None:
        """
        根据任务编号读取最新内存快照。

        参数含义：
            task_id:
                需要读取的长任务编号。

        返回值含义：
            LongTask | None:
                找到时返回任务，否则返回 None。
        """

        return self.tasks.get(task_id)

    async def save(
        self,
        task: LongTask,
        *,
        expected_version: int,
    ) -> LongTask:
        """
        校验旧版本后保存新快照。

        参数含义：
            task:
                版本已经递增的新任务快照。
            expected_version:
                调用方修改前读取到的旧版本号。

        返回值含义：
            LongTask:
                乐观锁校验和保存成功后的任务。
        """

        current = self.tasks.get(task.task_id)
        if current is None or current.version != expected_version:
            raise LongTaskVersionConflictError("集成测试任务版本冲突")
        self.tasks[task.task_id] = task
        return task


class RecordingQueuePublisher:
    """记录应用服务准备发布到 Redis Stream 的轻量消息。"""

    def __init__(self) -> None:
        """
        初始化空消息列表。

        返回值含义：
            None。
        """

        self.messages: list[LongTaskQueueMessage] = []

    async def publish(self, message: LongTaskQueueMessage) -> str:
        """
        记录一条后台执行通知。

        参数含义：
            message:
                应用服务生成的长任务轻量队列消息。

        返回值含义：
            str:
                模拟 Redis Stream 生成的消息编号。
        """

        self.messages.append(message)
        return "1000-0"


class FakeGraphRuntime:
    """满足 AgentApiService 构造要求的最小运行时替身。"""

    def cancel_multi_agent_task(self, multi_agent_task_id: str) -> bool:
        """
        表示当前测试没有可取消的请求内任务。

        参数含义：
            multi_agent_task_id:
                API 希望取消的多智能体任务编号。

        返回值含义：
            bool:
                始终返回 False。
        """

        _ = multi_agent_task_id
        return False


def build_three_step_plan_json(
    *,
    plan_id: str,
    objective: str,
) -> str:
    """
    构建三个步骤依次执行的固定计划。

    参数含义：
        plan_id:
            Planner 提示词中由程序生成的计划编号。
        objective:
            Planner 必须原样返回的用户目标。

    返回值含义：
        str:
            PlannerAgent 可以校验的三步骤 JSON 文本。
    """

    return json.dumps(
        {
            "plan_id": plan_id,
            "objective": objective,
            "steps": [
                {
                    "step_id": "load_profile",
                    "title": "读取狗狗档案",
                    "assigned_agent": "profile_agent",
                    "depends_on": [],
                    "status": "pending",
                },
                {
                    "step_id": "build_advice",
                    "title": "生成长期照护建议",
                    "assigned_agent": "general_agent",
                    "depends_on": ["load_profile"],
                    "status": "pending",
                },
                {
                    "step_id": "finalize_plan",
                    "title": "整理最终照护计划",
                    "assigned_agent": "summary_agent",
                    "depends_on": ["build_advice"],
                    "status": "pending",
                },
            ],
            "status": "planned",
            "requires_user_input": False,
            "clarification_prompt": "",
        },
        ensure_ascii=False,
    )


def build_success_worker(calls: list[str]):
    """
    构建记录真实 Scheduler 执行步骤的成功 Worker。

    参数含义：
        calls:
            按执行顺序保存步骤编号的列表。

    返回值含义：
        Callable:
            返回 completed AgentTaskResult 的异步 Worker。
    """

    async def worker(
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        记录步骤及依赖并返回确定性成功结果。

        参数含义：
            step:
                Scheduler 当前执行的完整计划步骤。
            dependency_results:
                当前步骤已经完成的前置结果。

        返回值含义：
            AgentTaskResult:
                与当前步骤对应的成功结果。
        """

        _ = dependency_results
        calls.append(step.step_id)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary=f"{step.title}完成",
            output={"step_id": step.step_id},
        )

    return worker


@pytest.mark.asyncio
async def test_chat_should_auto_handoff_budget_paused_task() -> None:
    """
    验证普通聊天任务在批次预算耗尽后完成持久化交接和公开响应。

    功能：
        串联真实 Planner、Scheduler、Orchestrator、入口节点、LongTask
        应用服务、业务摘要、API 响应转换、后台 Handler 和 Step Executor，
        证明请求只执行第一批，后台随后通过 continued 消息推进两批。

    返回值含义：
        None。
    """

    provider = RecordingLLMProvider()
    worker_calls: list[str] = []
    worker = build_success_worker(worker_calls)
    clock_values = iter([0.0, 2.0])
    scheduler = MultiAgentTaskScheduler(
        workers={
            "profile_agent": worker,
            "general_agent": worker,
            "summary_agent": worker,
        },
        inline_budget_seconds=1.0,
        clock=lambda: next(clock_values),
    )
    orchestrator = MultiAgentOrchestrator(
        planner=PlannerAgent(
            llm_provider=provider,
            available_agents={
                "profile_agent": "读取狗狗档案。",
                "general_agent": "生成综合建议。",
                "summary_agent": "整理最终计划。",
            },
            maximum_plan_attempts=1,
        ),
        scheduler=scheduler,
        result_aggregator=ResultAggregator(
            llm_provider=provider,
            maximum_aggregation_attempts=1,
        ),
    )
    store = InMemoryHandoffStore()
    publisher = RecordingQueuePublisher()
    application_service = LongTaskApplicationService(
        store,
        queue_publisher=publisher,
    )
    entry_node = build_multi_agent_entry_node(
        orchestrator=orchestrator,
        long_task_application_service=application_service,
    )

    async def graph_runner(
        question: str,
        **kwargs: Any,
    ) -> GraphFinalResult:
        """
        用真实入口节点模拟主图本轮执行并构建标准图结果。

        参数含义：
            question:
                API 收到的普通用户问题。
            kwargs:
                API 传入的 thread_id、trace_id 和恢复值。

        返回值含义：
            GraphFinalResult:
                携带长任务业务摘要的主图完成结果。
        """

        update = await entry_node(
            {
                "question": question,
                "user_id": "user_auto_handoff_001",
                "session_id": kwargs["thread_id"],
                "trace_id": kwargs["trace_id"],
                "multi_agent_resume_action": "none",
            }
        )
        return GraphFinalResult(
            answer=update["final_answer"],
            thread_id=kwargs["thread_id"],
            trace_id=kwargs["trace_id"],
            metadata=build_graph_business_summary(update),
        )

    service = AgentApiService(
        graph_runtime=FakeGraphRuntime(),
        graph_runner=graph_runner,
    )

    response = await service.chat(
        question="根据档案生成长期照护建议",
        session_id="session_auto_handoff_001",
        trace_id="trace_auto_handoff_001",
    )

    task_id = "multi_agent_task_trace_auto_handoff_001"
    assert worker_calls == ["load_profile"]
    assert len(provider.prompts) == 1
    saved_task = store.tasks[task_id]
    assert saved_task.version == 1
    assert saved_task.status == "running"
    assert saved_task.execution_mode == "durable"
    assert saved_task.user_id == "user_auto_handoff_001"
    assert saved_task.thread_id == "session_auto_handoff_001"
    assert [step.status for step in saved_task.steps] == [
        "completed",
        "ready",
        "pending",
    ]
    assert [step.assigned_agent for step in saved_task.steps] == [
        "profile_agent",
        "general_agent",
        "summary_agent",
    ]

    assert len(publisher.messages) == 1
    message = publisher.messages[0]
    assert message.task_id == task_id
    assert message.task_version == 1
    assert message.reason == "submitted"
    assert message.ready_step_ids == [f"{task_id}:build_advice"]
    assert message.correlation_id == task_id

    assert response.status == "completed"
    assert response.business_status == "running"
    assert response.long_task is not None
    assert response.long_task.task_id == task_id
    assert response.long_task.task_version == 1
    assert response.long_task.status == "running"
    assert response.long_task.execution_mode == "durable"
    assert response.long_task.status_url == (
        f"/v1/long-tasks/{task_id}?user_id=user_auto_handoff_001"
    )
    assert response.long_task.events_url == (
        f"/v1/long-tasks/{task_id}/events"
        "?user_id=user_auto_handoff_001"
    )
    assert "long_task_handoff" not in response.metadata

    claim_ids = iter(["claim-build-advice", "claim-finalize-plan"])
    batch_ids = iter(["batch-build-advice", "batch-finalize-plan"])
    handler = LongTaskQueueMessageHandler(
        store=store,
        step_executor=LongTaskStepExecutorAdapter(
            {
                "profile_agent": worker,
                "general_agent": worker,
                "summary_agent": worker,
            }
        ),
        worker_name="integration-worker-1",
        queue_publisher=publisher,
        claim_id_factory=lambda: next(claim_ids),
        batch_id_factory=lambda: next(batch_ids),
    )

    await handler(publisher.messages[0])

    assert worker_calls == ["load_profile", f"{task_id}:build_advice"]
    assert len(publisher.messages) == 2
    continued_message = publisher.messages[1]
    assert continued_message.reason == "continued"
    assert continued_message.ready_step_ids == [
        f"{task_id}:finalize_plan"
    ]

    await handler(continued_message)

    final_task = store.tasks[task_id]
    assert worker_calls == [
        "load_profile",
        f"{task_id}:build_advice",
        f"{task_id}:finalize_plan",
    ]
    assert final_task.status == "completed"
    assert [step.status for step in final_task.steps] == [
        "completed",
        "completed",
        "completed",
    ]
    assert len(publisher.messages) == 2

    request_status = service.get_task_status(task_id)
    assert request_status is not None
    assert request_status.status == "completed"
    assert request_status.business_status == "running"
