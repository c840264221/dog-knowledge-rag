"""真实计时、Redis 和独立后台执行者的预算交接演示。

手动运行：REDIS_ENABLED=true python -m tests.learning.plus_real_budget_handoff
A 为确定性的教学步骤；B 使用已启动后台进程中的真实 general_agent。
不经过 HTTP 主图入口，不验证真实断网；记录保留，不自动删除。
"""

import asyncio
import json
from collections.abc import Mapping
from uuid import uuid4

from src.agents.collaboration.adapters.long_task_plan_adapter import (
    adapt_paused_collaboration_result_to_long_task,
)
from src.agents.collaboration.contracts import AgentTaskPlan, AgentTaskResult, AgentTaskStep
from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
from src.runtime.container.providers.redis_provider import RedisProvider
from src.runtime.long_tasks.application_service import LongTaskApplicationService
from src.runtime.long_tasks.store import RedisLongTaskStore
from src.runtime.long_tasks.stream import RedisLongTaskStream
from src.settings import settings


async def run_budget_handoff_demo() -> None:
    """执行 A 后因预算不足交接 B，观察真实后台结果。

    参数：无；Redis 读取项目配置，后台执行程序需由用户提前启动。
    返回：None；打印交接前快照及最终检查报告，异常保留调用栈。
    限制：A 人为等待两秒以稳定触发一秒预算；B 使用真实模型，可能收费。
    """
    provider = RedisProvider(settings.redis)
    await provider.startup()
    try:
        run_id = uuid4().hex
        evidence_marker = f"handoff-{run_id}"
        inline_calls: list[str] = []

        async def inline_worker(
            step: AgentTaskStep,
            dependency_results: Mapping[str, AgentTaskResult],
        ) -> AgentTaskResult:
            """生成 A 的教学资料，并拦截意外在请求内执行的 B。

            参数：step 为当前步骤；dependency_results 为调度器传入的依赖结果。
            返回：包含随机标记的 A 完成结果；B 被调用则报错。
            """
            inline_calls.append(step.step_id)
            if step.step_id != "prepare_material":
                raise RuntimeError("B 不应在请求内执行，请检查预算暂停逻辑")
            if dependency_results:
                raise RuntimeError("A 不应有前置依赖")
            await asyncio.sleep(2)
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="completed",
                summary=f"教学样例资料：主题为任务交接验证；核验标记为 {evidence_marker}。",
                output={"evidence_marker": evidence_marker},
            )

        plan = AgentTaskPlan(
            plan_id=f"budget-demo-{run_id}",
            objective="整理教学样例资料，再由后台复述前置资料中的核验标记。",
            steps=[
                AgentTaskStep(
                    step_id="prepare_material",
                    title="A：准备教学样例资料",
                    assigned_agent="general_agent",
                    input_data={"question": "准备教学样例资料。"},
                ),
                AgentTaskStep(
                    step_id="report_material",
                    title="B：读取前置资料并复述标记",
                    assigned_agent="general_agent",
                    depends_on=["prepare_material"],
                    input_data={"question": (
                        "这是任务交接教学验证。请从已完成前置步骤的摘要中读取核验标记，"
                        "原样输出完整标记，再用一句话说明资料主题。"
                        "不要调用工具。缺少前置资料时明确说明缺失，不要猜测标记。"
                    )},
                ),
            ],
        )
        scheduler = MultiAgentTaskScheduler(
            workers={"general_agent": inline_worker},
            maximum_parallel_steps=1,
            maximum_step_attempts=1,
            inline_budget_seconds=1.0,
        )
        paused_result = await scheduler.execute(plan)
        task = adapt_paused_collaboration_result_to_long_task(
            task_result=paused_result,
            user_id="plus_budget_demo_user",
            thread_id=f"budget-session-{run_id}",
            original_request=plan.objective,
        )
        print("交接前快照（尚未写入 Redis）：", flush=True)
        print(json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2), flush=True)
        store = RedisLongTaskStore(provider.client)
        service = LongTaskApplicationService(
            store, queue_publisher=RedisLongTaskStream(provider.client),
        )
        # 先保存成功，再告知后台已接收；失败时不宣称交接完成。
        saved_task = await service.handoff_paused_collaboration_task(task)
        print(f"已可靠交接，任务编号：{saved_task.task_id}", flush=True)
        print("接下来只是查询状态；等待超时不代表任务取消，请勿立即反复提交。", flush=True)

        deadline = asyncio.get_running_loop().time() + 240
        final_task = saved_task
        while asyncio.get_running_loop().time() < deadline:
            current = await store.load(saved_task.task_id)
            if current is None:
                raise RuntimeError("已交接任务的快照不存在")
            final_task = current
            if current.status != "running":
                break
            await asyncio.sleep(1)

        before_a = task.steps[0]
        after_a, after_b = final_task.steps
        events = await store.load_events(task.task_id)
        checks = {
            "inline_only_executed_a": inline_calls == ["prepare_material"],
            "a_snapshot_unchanged": after_a.model_dump() == before_a.model_dump(),
            "no_background_start_event_for_a": not any(
                event.event_type == "step_started" and event.step_id == after_a.step_id
                for event in events
            ),
            "b_completed": after_b.status == "completed",
            "b_summary_contains_a_marker": evidence_marker in after_b.output_summary,
            "task_completed": final_task.status == "completed",
        }
        print(json.dumps({
            "task_id": task.task_id,
            "inline_calls": inline_calls,
            "expected_marker": evidence_marker,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "task_status": final_task.status,
            "task_version": final_task.version,
            "b_output_summary": after_b.output_summary,
            "steps": [step.model_dump(mode="json") for step in final_task.steps],
            "data_retained": True,
        }, ensure_ascii=False, indent=2), flush=True)
    finally:
        await provider.shutdown()


if __name__ == "__main__":
    asyncio.run(run_budget_handoff_demo())
