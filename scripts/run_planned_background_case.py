"""真实模型规划后直接提交后台；提交端不运行步骤，按任务编号查询不重新提交。"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4


DEFAULT_QUESTION = (
    "请在后台整理我的接猫准备计划：先整理我已提供的条件和待确认信息，"
    "再根据整理结果生成按先后顺序排列的待办清单。"
    "我的预算300元，家里已有猫砂盆和食水碗，还没有确定接猫日期。"
    "只整理这些已知信息，缺少的内容列为待确认，不补造价格、购买记录或养护事实，不调用工具。"
)
TEACHING_USER = "planned_background_learning_user"


def save_report(path: Path, report: dict) -> None:
    """保存阶段报告；path为本任务文件、report为记录；无返回值，报告不是执行授权。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def validate_background_plan(plan) -> None:
    """检查本案例分工；plan为模型正式计划；通过无返回值，否则拒绝，不替模型生成步骤。

    本案例验证两个文本步骤的后台依赖推进，不把这个步数限制加到通用规划器。
    """
    if len(plan.steps) != 2:
        raise ValueError("本案例要求两个文本整理步骤，请检查模型是否正确拆分。")
    if any(step.assigned_agent != "general_agent" or step.execution_mode != "capability" for step in plan.steps):
        raise ValueError("本案例只允许通用智能体完成文本整理。")
    roots = [step for step in plan.steps if not step.depends_on]
    children = [step for step in plan.steps if step.depends_on]
    if len(roots) != 1 or len(children) != 1 or children[0].depends_on != [roots[0].step_id]:
        raise ValueError("待办清单步骤需要依赖已知条件整理结果，不自动补依赖。")


def build_initial_background_task(plan, task_id: str, user_id: str):
    """适配未执行计划并确定初始后台状态。

    参数：plan为已审核计划，task_id为宿主动作身份，user_id为可信教学身份。
    返回：初次保存前的版本1、running状态后台任务；根步骤ready，后续步骤pending，全部未执行。
    """
    from src.agents.collaboration.adapters.long_task_plan_adapter import adapt_collaboration_plan_to_long_task
    from src.runtime.long_tasks.contracts import LongTask

    task = adapt_collaboration_plan_to_long_task(plan=plan, user_id=user_id,
        thread_id=task_id, original_request=plan.objective, task_id=task_id, execution_mode="durable")
    # 尚未持久化的初始状态；不是将已保存任务越过状态机改写。
    task = LongTask.model_validate({**task.model_dump(mode="python"), "status": "running"})
    if task.version != 1 or any(step.attempt_count or step.output_summary for step in task.steps):
        raise ValueError("初始后台快照不能包含已执行步骤或已有结果。")
    return task


async def run(question: str, task_id: str | None, wait_seconds: int) -> None:
    """提交新计划或只读查询已有任务。

    参数：question为新需求，task_id非空表示仅查询原任务，wait_seconds为查询等待上限。
    返回：无，保存UTF-8报告并显示状态；查询等待结束不会取消任务。
    """
    from src.settings import settings
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.redis_provider import RedisProvider
    from src.runtime.container.providers.llm_provider import LLMProvider
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.agents.collaboration.planner import PlannerAgent
    from src.runtime.long_tasks.store import RedisLongTaskStore
    from src.runtime.long_tasks.contracts import LongTaskQueueMessage
    from src.runtime.long_tasks.commit_service import build_commit_request

    is_query = task_id is not None
    task_id = task_id or "planned-background-" + uuid4().hex
    path = Path(__file__).resolve().parents[1] / "output/learning/long_tasks" / task_id / "report.json"
    context = RuntimeContext(trace_id=uuid4().hex, session_id=task_id, user_id=TEACHING_USER,
                             component="planned_background_case")
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    container = RuntimeContainer()
    container.register("redis", RedisProvider(settings.redis))
    report = {"task_id": task_id, "status": "starting", "runtime_environment": context.time_context.as_dict(),
              "notice": "教学入口选择后台模式，模型只规划；不是由模型自行决定执行模式，也不验证主图自动路由。"}
    if is_query and path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
    print(f"任务编号：{task_id}\n报告：{path}", flush=True)
    try:
        await context.startup()
        await container.startup()
        provider = container.get("redis")
        if not provider.enabled:
            raise RuntimeError("请设置REDIS_ENABLED=true，并启动Redis。")
        store = RedisLongTaskStore(provider.client)
        if not is_query:
            container.register("llm", LLMProvider())
            planner = PlannerAgent(llm_provider=container.get("llm"), maximum_steps=2, maximum_plan_attempts=2,
                available_agents={"general_agent": "整理用户给定信息和文本方案，不执行工具，不补造事实。"
                    "后续步骤依赖前置整理时声明depends_on，执行时从前置真实摘要读取，不预填结果。"})
            plan = await planner.create_plan(objective=question, context={"tool_catalog": []})
            report.update(question=question, plan=plan.model_dump(mode="json"), status="plan_review")
            save_report(path, report)
            print("模型生成的计划：\n" + json.dumps(report["plan"], ensure_ascii=False, indent=2), flush=True)
            if plan.status != "planned" or plan.requires_user_input:
                report["status"] = plan.status
                print("计划仍需澄清，没有提交后台。", flush=True)
                return
            validate_background_plan(plan)
            answer = await asyncio.to_thread(input, "核对整理条件→生成待办的依赖；输入 提交 后台执行：")
            if answer.strip() != "提交":
                report["status"] = "not_submitted"
                return
            task = build_initial_background_task(plan, task_id, context.user_id)
            ready_ids = [step.step_id for step in task.steps if step.status == "ready"]
            request = build_commit_request(previous_task=None, next_task=task, commit_id="planned-submit:" + task_id,
                fingerprint_payload={"action": "submit_planned_background", "task_id": task_id,
                                     "objective": task.goal.objective},
                actor_type="system", actor_id="planned_background_case", correlation_id=task.thread_id,
                queue_message=LongTaskQueueMessage(task_id=task_id, task_version=task.version,
                    reason="submitted", ready_step_ids=ready_ids, correlation_id=task.thread_id))
            report.update(status="submitting", submitted_snapshot=task.model_dump(mode="json"),
                          commit_id=request.commit_id, business_steps_executed_by_submitter=0)
            save_report(path, report)  # 先记录动作身份；响应异常后不通过重跑新任务来试探。
            receipt = await store.commit(request)
            if receipt.queue_message_id is None:
                raise RuntimeError("回执缺少队列消息编号，请按原任务编号查询，不重新提交。")
            report.update(status="submitted", submission_receipt=receipt.model_dump(mode="json"))
            save_report(path, report)
            print("后台已可靠接收；提交端没有执行任何业务步骤。", flush=True)
        else:
            print("只查询原任务，不生成计划、不重新提交。", flush=True)

        deadline = asyncio.get_running_loop().time() + wait_seconds
        last_state = None
        while True:
            current = await store.load(task_id)
            if current is None:
                raise ValueError("找不到该任务；没有自动创建替代任务。")
            if current.user_id != context.user_id:
                raise ValueError("任务不属于当前教学身份，拒绝查询。")
            report["latest_task"] = current.model_dump(mode="json")
            report["status"] = current.status
            state = (current.status, current.version)
            if state != last_state:
                print(f"当前状态：{current.status}，版本：{current.version}；" + "，".join(
                    f"{step.metadata.get('source_plan_step_id', step.step_id)}={step.status}" for step in current.steps), flush=True)
                last_state = state
            if current.status in {"completed", "failed", "cancelled", "awaiting_input"} or asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(1)
        report["events"] = [event.model_dump(mode="json") for event in await store.load_events(task_id)]
        if current.status == "completed":
            for step in current.steps:
                print(f"\n{step.title}：\n{step.output_summary}", flush=True)
            report["checks"] = {
                "submitted_with_zero_executions": bool(report.get("submitted_snapshot")) and all(
                    item["attempt_count"] == 0 for item in report["submitted_snapshot"]["steps"]),
                "all_steps_completed": all(step.status == "completed" for step in current.steps),
                "each_step_executed_once": all(step.attempt_count == 1 for step in current.steps),
            }
        print(f"\n查询原任务：REDIS_ENABLED=true PYTHONIOENCODING=utf-8 python -X utf8 -m scripts.run_planned_background_case --task-id {task_id} --wait-seconds 180", flush=True)
        if current.status == "running" and wait_seconds:
            print("本次等待结束不表示任务取消，可稍后按同一编号查询。", flush=True)
    except Exception as exc:
        report["last_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_report(path, report)
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="模型规划后从首步直接后台执行，支持按原任务编号只读查询。")
    cli.add_argument("--question", default=DEFAULT_QUESTION, help="新任务需求")
    cli.add_argument("--task-id", help="原任务编号；提供时仅查询，不提交")
    cli.add_argument("--wait-seconds", type=int, default=0, help="最多查询等待秒数，默认查询一次就退出")
    args = cli.parse_args()
    if args.wait_seconds < 0:
        cli.error("等待秒数不能为负数。")
    if args.task_id and args.question != DEFAULT_QUESTION:
        cli.error("查询已有任务时不能同时提供新问题。")
    asyncio.run(run(args.question, args.task_id, args.wait_seconds))
