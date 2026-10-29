"""真实模型规划的工具协作案例；复用项目调度器，不创建新主图。

运行：MCP_EXPENSE_ENABLED=true LOG_LEVEL=WARNING python -m scripts.run_expense_collaboration_case
本脚本会调用模型，人工确认后写入教学账本；阶段报告仅用于复盘，不提供退出后恢复入口。
"""

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4


ALLOWED_TOOLS = {"calculate_supply_budget", "record_expense", "record_expenses", "query_expenses"}
DEFAULT_QUESTION = (
    "这是教学账本的演示数据：今天已经购买猫砂，每袋40元买了2袋，食水碗每个19.9元买了1个。"
    "先用本地预算工具核算这批已发生消费，预算150元，单价均由我提供。"
    "再根据工具实际返回的合计金额，将整批消费合并记为一笔人民币支出，分类宠物/用品，"
    "用途为‘教学演示：猫砂与食水碗’，消费日期为今天，保存前让我确认。"
    "保存成功后查询宠物分类最近1条账单核对。最后报告实际合计、预算余额、保存结果和回查结果。"
    "预算不是消费金额；不要预填计算结果，不要把计划中的消费当成已保存账单。"
)


def restrict_catalog(catalog: list[dict], allowed: set[str]) -> list[dict]:
    """筛选工具定义；catalog为完整目录、allowed为允许名称；返回独立副本，不修改共享目录。"""
    return [deepcopy(item) for item in catalog if item["name"] in allowed]


def check_call_scope(state: dict, allowed: set[str]) -> dict:
    """检查最终调用范围；state为执行前状态、allowed为允许名称；通过返回空更新，越界抛异常。"""
    calls = state.get("tool_calls") or []
    if not calls:
        return {}  # 未提出工具调用不是越界；步骤适配器仍须核对实际执行证据。
    if any(call.get("name") not in allowed for call in calls):
        raise ValueError("调用超出本案例允许的工具范围，拒绝执行。")
    if len(calls) != 1:
        raise ValueError("本案例每一步只允许一次工具调用，依赖操作应拆为独立步骤。")
    return {}


def save_report(path: Path, report: dict) -> None:
    """保存阶段报告；path为本轮独立路径、report为报告字典；无返回值，文件不作为执行授权。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def run(question: str) -> None:
    """运行模型规划、调度、确认及聚合；question为完整需求；无返回值，打印结果并保存阶段报告。"""
    from src.settings import settings
    from src.runtime.container.init import container
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.agents.collaboration.planner import PlannerAgent
    from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
    from src.agents.collaboration.aggregator import ResultAggregator
    from src.agents.collaboration.workers.tool_agent_worker_adapter import ToolAgentWorkerAdapter, build_tool_execution_guard
    from src.agents.tool_agent.adapters.registry_adapter import build_tool_agent_tool_catalog_state_update_with_mcp
    from src.agents.tool_agent.nodes.tool_parse_node import build_llm_tool_parser
    from src.agents.tool_agent.graph import build_tool_agent_graph

    if not settings.mcp.expense.enabled:
        raise RuntimeError("请设置MCP_EXPENSE_ENABLED=true，并保持教学记账服务运行。")
    run_id = uuid4().hex
    path = Path(__file__).resolve().parents[1] / "output/learning/expenses/collaboration_runs" / run_id / "report.json"
    context = RuntimeContext(trace_id=run_id, session_id="expense-collaboration-" + run_id,
                             user_id="expense_learning_user", component="expense_collaboration_case")
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    report = {"run_id": run_id, "question": question, "status": "starting", "stages": [], "worker_runs": [],
              "runtime_environment": context.time_context.as_dict(),
              "notice": "直接复用项目调度器的教学案例；不是主图入口验证，不支持退出后恢复或自动重试写账。"}
    print(f"本轮编号：{run_id}\n报告：{path}", flush=True)
    try:
        await context.startup()
        await container.startup()
        llm = container.get("llm")
        mcp = container.get("mcp_tools")
        catalog = build_tool_agent_tool_catalog_state_update_with_mcp(
            mcp_tool_definitions=mcp.tool_definitions,
        )["tool_agent_tool_catalog"]
        catalog = restrict_catalog(catalog, ALLOWED_TOOLS)
        if {item["name"] for item in catalog} != ALLOWED_TOOLS:
            raise ValueError("案例所需工具未全部就绪，请检查服务目录。")
        report["tool_catalog"] = catalog
        base_parser = build_llm_tool_parser(llm_provider=llm)

        async def scoped_parser(parser_input: dict):
            """限制模型可见目录；parser_input为既有解析输入；返回统一工具提议，不另建解析分支。"""
            state = dict(parser_input.get("state") or {})
            state["tool_agent_tool_catalog"] = restrict_catalog(state.get("tool_agent_tool_catalog") or [], ALLOWED_TOOLS)
            return await base_parser({**parser_input, "state": state})

        async def runner(state: dict) -> dict:
            """复用真实工具图；state为步骤独立状态；返回实际结果，确认快照仍由现有步骤适配器维护。"""
            expected = state.get("tool_calls") if state.get("tool_confirmed") == "confirmed" else None
            confirmation_guard = build_tool_execution_guard(expected, state.get("tool_agent_execution_contract"))

            def guard(current: dict) -> dict:
                """执行前核对范围及确认指纹；current为当前工具状态；通过返回空更新，失败抛异常。"""
                check_call_scope(current, ALLOWED_TOOLS)
                return confirmation_guard(current)

            graph = build_tool_agent_graph(parser=scoped_parser, llm_provider=llm, mcp_provider=mcp,
                                           interrupt_func=None, before_execute=guard)
            final = dict(await graph.ainvoke(state))
            report["worker_runs"].append({
                "step_id": state.get("multi_agent_step_id"),
                "dependencies": deepcopy(state.get("multi_agent_dependency_results") or {}),
                "skill": deepcopy(context.metadata.get("tool_planning_skill")),
                "state": {key: deepcopy(final.get(key)) for key in (
                    "tool_calls", "tool_agent_response", "tool_agent_permission",
                    "tool_agent_runtime_execution_records", "tool_results", "final_answer",
                )},
            })
            save_report(path, report)
            return final

        planner = PlannerAgent(llm_provider=llm, maximum_steps=4, maximum_plan_attempts=2,
            available_agents={"tool_agent": "执行目录中的本地与MCP工具；每步只安排一种操作，需要前置实际结果时声明depends_on。"})
        plan = await planner.create_plan(objective=question, context={"tool_catalog": catalog})
        report.update(status="plan_review", plan=plan.model_dump(mode="json"))
        save_report(path, report)
        print("\n模型生成的计划（仍需人工检查业务依赖）：", flush=True)
        print(json.dumps(report["plan"], ensure_ascii=False, indent=2), flush=True)
        if plan.status != "planned" or plan.requires_user_input:
            print("计划尚不能执行，请查看报告中的澄清信息。", flush=True)
            return
        approval = await asyncio.to_thread(input, "核对先核算、再记账、最后回查的依赖；输入 执行 继续，其他输入退出：")
        if approval.strip() != "执行":
            report["status"] = "not_started"
            return
        scheduler = MultiAgentTaskScheduler(workers={"tool_agent": ToolAgentWorkerAdapter(runner)},
                                            maximum_parallel_steps=1, maximum_step_attempts=1)
        result = await scheduler.execute(plan)
        for round_number in range(4):
            report["stages"].append(result.model_dump(mode="json"))
            report["status"] = result.status
            save_report(path, report)
            print("\n步骤状态：" + "，".join(f"{item.step_id}={item.status}" for item in result.task_results), flush=True)
            if result.status != "awaiting_input":
                break
            if round_number == 3:
                print("已达到本入口的恢复次数限制，不再执行。", flush=True)
                return
            replies = {}
            for item in result.task_results:
                if item.status == "awaiting_input":
                    print(f"\n{item.step_id}：{item.clarification_prompt}", flush=True)
                    reply = await asyncio.to_thread(input, "输入 确认 或 拒绝（exit退出；本入口不支持退出后恢复）：")
                    if reply.strip().casefold() == "exit":
                        return
                    replies[item.step_id] = reply
            result = await scheduler.resume(result, user_inputs=replies)
        if result.status in {"running", "partial"}:
            result = await ResultAggregator(llm_provider=llm).aggregate(result)
            report["stages"].append(result.model_dump(mode="json"))
            report["status"] = result.status
            print("\n最终回答：\n" + str(result.final_answer or ""), flush=True)
        else:
            print(f"任务状态为{result.status}，不生成成功答案，请检查报告。", flush=True)
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
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
    parser = argparse.ArgumentParser(description="真实模型制定用品核算、确认记账及回查计划。")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="完整需求；仍仅允许本案例工具范围")
    args = parser.parse_args()
    asyncio.run(run(args.question))
