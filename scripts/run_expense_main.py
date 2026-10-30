"""运行项目真实主图的记账入口，使用固定教学身份；确认后实际写账，不创建其他Agent。

运行：MCP_EXPENSE_ENABLED=true python -m scripts.run_expense_main --question "昨天吃饭20元，帮我记账"
恢复确认：沿用控制台打印的--thread-id，不重新提交--question。
补参恢复：--thread-id 原会话编号 --reply "80"，不使用interrupt恢复参数澄清。
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4


async def run(question: str | None, thread_id: str | None, reply: str | None = None) -> None:
    """启动真实主图并人工确认；question为新问题、thread_id为原检查点会话、reply为补充值；无返回值，保存报告。

    本入口固定使用教学账号，不用于生产认证；已有会话只恢复或展示，不重新创建记账动作。
    """
    from langgraph.types import Command
    from src.settings import settings
    from src.runtime.container.init import container
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.graph.graph_run import (
        create_initial_state, restore_pending_tool_clarification_state,
        restore_pending_task_collection_state,
    )

    if not settings.mcp.expense.enabled:
        raise RuntimeError("请先设置环境变量MCP_EXPENSE_ENABLED=true，再启动本命令。")
    existing_thread = thread_id is not None
    thread_id = thread_id or "expense-main-" + uuid4().hex
    context = RuntimeContext(trace_id=uuid4().hex, session_id=thread_id,
                             user_id="expense_learning_user", component="expense_main_graph")
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    config = {"configurable": {"thread_id": thread_id}}
    report_path = Path(__file__).resolve().parents[1] / "output/learning/expenses/main_runs" / thread_id / "report.json"
    print(f"主图会话编号：{thread_id}", flush=True)
    print(f"执行报告：{report_path}", flush=True)
    print(f"中断后恢复：python -m scripts.run_expense_main --thread-id {thread_id}", flush=True)
    graph = None
    try:
        await context.startup()
        await container.startup()
        graph = container.get("graph_runtime").graph
        if existing_thread:
            snapshot = await graph.aget_state(config)
            if not snapshot.values or snapshot.values.get("user_id") != context.user_id:
                raise ValueError("找不到此教学用户的会话检查点，拒绝恢复。")
            pending_clarification = (
                snapshot.values.get("tool_agent_clarification_request")
                and snapshot.values.get("tool_agent_pending_tool_call")
            )
            if pending_clarification and not snapshot.next:
                if reply is None:
                    print("当前等待参数补充：" + str(snapshot.values.get("final_answer") or ""), flush=True)
                    print(f'补参命令：python -m scripts.run_expense_main --thread-id {thread_id} --reply "补充值"', flush=True)
                    return
                # --reply明确表达继续此会话；走既有任务门卫和补参适配器，不使用Command恢复。
                user_input = reply.strip()
                if user_input.casefold() not in {"取消", "算了", "不查了", "不用了", "cancel"}:
                    user_input = "补充信息：" + user_input
                state = create_initial_state(user_input, context.trace_id, session_id=thread_id)
                state["user_id"] = context.user_id
                state = await restore_pending_task_collection_state(graph, config, state)
                state = await restore_pending_tool_clarification_state(graph, config, state)
                print("已读取待补全调用，按普通新一轮输入恢复，不使用interrupt。", flush=True)
                result = await graph.ainvoke(state, config=config)
            elif reply is not None:
                raise ValueError("此会话当前没有可补参的工具调用；未重新执行或创建动作。")
            elif not snapshot.next:
                print("该会话已经结束，仅展示已有结果，不重新执行。", flush=True)
                print(snapshot.values.get("final_answer", ""), flush=True)
                return
            else:
                result = {"__interrupt__": ["恢复原会话"]}
        else:
            state = create_initial_state(question, context.trace_id, session_id=thread_id)
            # 教学运行器绑定固定账号；不从用户自然语言提取或接受身份。
            state["user_id"] = context.user_id
            result = await graph.ainvoke(state, config=config)

        if result.get("__interrupt__"):
            snapshot = await graph.aget_state(config, subgraphs=True)
            print("主图已暂停。请核对下面的原始确认内容：", flush=True)
            if result["__interrupt__"] != ["恢复原会话"]:
                for interruption in result["__interrupt__"]:
                    print(interruption.value, flush=True)
            else:
                for task in snapshot.tasks:
                    for interruption in task.interrupts:
                        print(interruption.value, flush=True)
            decision = await asyncio.to_thread(input, "输入 y 确认；其他输入拒绝：")
            result = await graph.ainvoke(Command(resume=decision.strip()), config=config)
        # 恢复调用的返回值不作为最终展示依据；与执行报告一致，读取持久化的最终图状态。
        final_snapshot = await graph.aget_state(config)
        if (final_snapshot.values.get("tool_agent_clarification_request")
                and final_snapshot.values.get("tool_agent_pending_tool_call") and not final_snapshot.next):
            print("\n本轮已结束，业务仍等待补参：\n" + str(final_snapshot.values.get("final_answer") or ""), flush=True)
            print(f'下一轮：python -m scripts.run_expense_main --thread-id {thread_id} --reply "补充值"', flush=True)
        elif final_snapshot.next:
            print("主图尚未结束，请沿用上述会话编号恢复，不重新发起问题。", flush=True)
        else:
            final_answer = str(final_snapshot.values.get("final_answer") or "")
            if final_answer:
                print("\n主图最终回答：\n" + final_answer, flush=True)
            else:
                print("会话已结束，但最终回答为空。请核对执行报告中的工具结果，不要据此重新记账。", flush=True)
    finally:
        try:
            if graph is not None:
                snapshot = await graph.aget_state(config)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                previous_skill = None
                if report_path.exists():
                    previous_skill = json.loads(report_path.read_text(encoding="utf-8")).get("tool_planning_skill")
                report_path.write_text(json.dumps({
                    "thread_id": thread_id,
                    "state": snapshot.values,
                    "pending_nodes": list(snapshot.next),
                    "tool_planning_skill": context.metadata.get("tool_planning_skill", previous_skill),
                }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        finally:
            try:
                await container.shutdown()
            finally:
                try:
                    await context.shutdown()
                finally:
                    runtime_ctx.set(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实主图记账验证；会实际调用模型，确认后写教学账本。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="新的完整记账问题")
    group.add_argument("--thread-id", help="恢复已有会话，复用原检查点与动作键")
    parser.add_argument("--reply", help="给原会话补参；普通新一轮输入，不使用interrupt恢复")
    args = parser.parse_args()
    if args.reply is not None and (not args.thread_id or not args.reply.strip()):
        parser.error("--reply必须与--thread-id一起使用，且补充内容不能为空。")
    asyncio.run(run(args.question, args.thread_id, args.reply))
