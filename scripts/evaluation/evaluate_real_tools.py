"""独立账本上的真实ToolAgent与MCP开发评估，不调用主图记忆或知识库。"""

import argparse
import asyncio
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import site
import sqlite3
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def save(report: dict) -> None:
    """原子保存阶段结果；参数为含report_path的报告；返回None，原始失败不被重跑覆盖。"""
    path = Path(report["report_path"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    """只读回查独立账本；参数为本轮数据库路径；返回按新增倒序排列的持久化记录。"""
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        return [json.loads(row[0]) for row in connection.execute("SELECT record_json FROM expenses ORDER BY rowid DESC")]


def expand_dates(value, today: date):
    """展开用例日期占位符；参数为期望结构及可信参考日；返回独立结构，不改模型输入。"""
    if isinstance(value, dict):
        return {key: expand_dates(item, today) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_dates(item, today) for item in value]
    if value == "today":
        return today.isoformat()
    if value == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    return value


def check_arguments(name: str, arguments: dict, case: dict) -> None:
    """检查执行提议不超出预审用例；参数为工具、业务参数及期望；返回None，错误提议阻止执行并判失败。

    金额、日期、分类、数量及价格来源严格核对；用途只检查非空，语义另留人工审核。
    此评估守卫不是项目本身的业务策略，不把它拦截的模型错误算成成功。
    """
    if name != case.get("tool"):
        raise ValueError("评估守卫：模型选错工具，未授权执行。")
    expected = case["expected_args"]
    for key, wanted in expected.items():
        actual = arguments.get(key)
        if key == "description":
            if not isinstance(actual, str) or not actual.strip():
                raise ValueError("用途为空。")
        elif key in {"amount", "budget", "unit_price"}:
            if Decimal(str(actual)) != Decimal(wanted):
                raise ValueError(f"评估守卫：{key}错误。")
        elif key == "items":
            if not isinstance(actual, list) or len(actual) != len(wanted):
                raise ValueError("评估守卫：明细数量错误。")
            for row, exp in zip(actual, wanted):
                check_arguments(name, row, {"tool": name, "expected_args": exp})
        elif actual != wanted:
            raise ValueError(f"评估守卫：{key}错误。")


def effective_arguments(name: str, arguments: dict) -> dict:
    """按实际工具输入契约应用默认值；参数为工具与原始提议；返回规范参数，不猜补必填字段。"""
    from src.graph.tools.schemas.supply_budget_args import SupplyBudgetArgs
    from src.mcp.expenses.schemas import ExpenseInput, ExpenseBatchInput, ExpenseQueryInput
    contracts = {"calculate_supply_budget": SupplyBudgetArgs, "record_expense": ExpenseInput,
                 "record_expenses": ExpenseBatchInput, "query_expenses": ExpenseQueryInput}
    if name not in contracts:
        raise ValueError("工具不在本轮评估范围。")
    return contracts[name].model_validate(arguments).model_dump(mode="json")


def summarize(cases: list[dict]) -> dict:
    """汇总任务与恢复；参数为全部结果；返回分层计数，拒绝不计恢复完成。"""
    normal = [c for c in cases if c["kind"] == "normal"]
    abnormal = [c for c in cases if c["kind"] == "exception"]
    recoverable = [c for c in abnormal if c.get("recoverable")]
    executions = [record for c in cases if c.get("layer") == "tool_agent"
                  for record in c.get("state", {}).get("tool_agent_runtime_execution_records", [])]
    return {"normal_tasks": {"passed": sum(c["passed"] for c in normal), "total": len(normal)},
        "exception_handling": {"passed": sum(c["passed"] for c in abnormal), "total": len(abnormal)},
        "recoverable_scenarios": {"passed": sum(c["passed"] for c in recoverable), "total": len(recoverable)},
        "actual_tool_execution_records": {"successful": sum(bool(r["tool_result"]["success"]) for r in executions),
                                          "total": len(executions)},
        "notice": "执行记录只统计工具子图，不把直接MCP幂等场景混入自然语言调用成功率。用途及最终措辞仍需人工审核。"}


async def run(dataset: Path, *, correct_source_report: Path | None = None) -> dict:
    """运行真实工具评估；参数为冻结数据集路径；返回报告并关闭本轮独立服务，不改日常教学账本。"""
    # 使用系统同版本Python时也执行依赖目录.pth初始化，MCP的Windows依赖需要它。
    dependencies = ROOT / ".venv/Lib/site-packages"
    if dependencies.is_dir():
        site.addsitedir(str(dependencies))
    folder = ROOT / "output/tool_real_evaluations" / uuid4().hex
    folder.mkdir(parents=True)
    from src.logger import base_logger
    # 本轮进程使用独立日志，避免与用户仍在运行的Worker争用Windows轮转文件。
    base_logger.remove()
    base_logger.add(str(folder / "evaluation.log"), level="WARNING", enqueue=True)
    from contextlib import AsyncExitStack
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import interrupt, Command
    from src.agents.tool_agent.graph import build_tool_agent_graph
    from src.agents.tool_agent.adapters.clarification_resume_adapter import resolve_tool_clarification_input
    from src.graph.graph_run import create_initial_state
    from src.mcp.expenses.store import ExpenseStore
    from src.mcp.expenses.schemas import ExpenseInput
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.expense_mcp_provider import ExpenseMcpProvider
    from src.runtime.container.providers.llm_provider import LLMProvider
    from src.runtime.context import RuntimeContext, runtime_ctx

    definitions = json.loads(dataset.read_text(encoding="utf-8"))
    source = json.loads(correct_source_report.read_text(encoding="utf-8")) if correct_source_report else None
    if source and source["dataset"] != definitions:
        raise ValueError("修正判定必须使用相同冻结用例。")
    old_cases = {c["case_id"]: c for c in source["cases"]} if source else {}
    ledger = Path(source["ledger"]).resolve() if source else folder / "expenses.sqlite3"
    if source and not ledger.is_relative_to(ROOT / "output/tool_real_evaluations"):
        raise ValueError("只能恢复本项目评估目录内的独立账本。")
    checkpoint_folder = correct_source_report.parent if source else folder
    store = ExpenseStore(ledger, user_id="expense_learning_user")
    # 固定夹具只供查询校验，单独标注，不计模型记账成绩。
    for index, category, subcategory in (() if source else ((0, "宠物", "用品"), (1, "宠物", "洗护"), (2, "宠物", "食品"), (3, "出行", "打车"))):
        store.record(ExpenseInput(amount="1.00", category=category, subcategory=subcategory,
            description="评估查询夹具", expense_date="2026-10-01", idempotency_key=f"fixture:{index}"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    report = {"status": "starting", "dataset": definitions, "ledger": str(ledger),
        "corrected_source_report": str(correct_source_report) if source else None,
        "report_path": str(folder / "report.json"), "cases": [], "fixture_record_count": 4,
        "notice": "真实ToolAgent子图与独立MCP账本；不含粗路由、记忆、前端。仅对预审用例参数自动确认，错误提议被评估守卫拦截并判失败。用途及最终回答语义未自动验收。"}
    save(report)
    print("报告：", report["report_path"], flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(ROOT / ".venv/Lib/site-packages") + os.pathsep + str(ROOT)
    log = (folder / "mcp_server.log").open("w", encoding="utf-8")
    bootstrap = ("import site,sys,runpy; site.addsitedir(sys.argv[1]); "
                 "from src.logger import base_logger; base_logger.remove(); "
                 "sys.argv=sys.argv[2:]; runpy.run_module(sys.argv[0],run_name='__main__')")
    child = await asyncio.create_subprocess_exec(sys.executable, "-X", "utf8", "-c", bootstrap,
        str(dependencies), "tests.learning.plus_expense_mcp_server", "--database", str(ledger), "--port", str(port),
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    container = RuntimeContainer()
    provider = ExpenseMcpProvider(enabled=True, url=f"http://127.0.0.1:{port}/mcp")
    container.register("expense", provider)
    try:
        for _ in range(100):
            if child.returncode is not None:
                raise RuntimeError("独立MCP服务启动失败，请查看服务日志。")
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(.1)
        else:
            raise TimeoutError("独立MCP启动超时。")
        await container.startup()
        container.register("llm", LLMProvider())
        report["status"] = "running"
        for raw_case in definitions["cases"]:
            case = deepcopy(raw_case)
            original = old_cases.get(case["case_id"])
            if original and original["passed"]:
                report["cases"].append({**deepcopy(original), "reused_completed_case": True})
                continue
            if original and case["case_id"] in {"E03", "E04", "E06"}:
                corrected = deepcopy(original)
                state = corrected.get("state") or {}
                if case["case_id"] == "E03":
                    assert state["tool_agent_permission"]["status"] == "rejected"
                    assert not state.get("tool_agent_runtime_execution_records")
                    key = corrected["confirmation_call"]["args"]["idempotency_key"]
                    assert not any(r["idempotency_key"] == key for r in read_rows(ledger))
                elif case["case_id"] == "E04":
                    assert not state.get("tool_calls") and not state.get("need_tool")
                    assert any(w in state.get("final_answer", "") for w in ("未来", "计划", "实际"))
                    assert not state.get("tool_agent_runtime_execution_records")
                else:
                    error = corrected["conflict_error"]
                    assert error["type"] == "RuntimeError" and error["message"].startswith("记账MCP执行错误：")
                    first = corrected["first_receipt"]["record"]
                    matches = [r for r in read_rows(ledger) if r["idempotency_key"] == first["idempotency_key"]]
                    assert len(matches) == 1 and matches[0] == first
                corrected.update(passed=True, status="completed", correction="依据已有真实状态与回执修正评估断言，未重跑模型或工具")
                report["cases"].append(corrected)
                continue
            runtime = RuntimeContext(trace_id=uuid4().hex, user_id="expense_learning_user",
                session_id=original["state"]["session_id"] if original else "tool-eval-" + uuid4().hex,
                component="tool_real_evaluation")
            old = runtime_ctx.get()
            runtime_ctx.set(runtime)
            stack = AsyncExitStack()
            item = {"case_id": case["case_id"], "kind": case["kind"], "scenario": case["scenario"],
                "recoverable": case.get("recoverable", False), "layer": case.get("layer", "tool_agent"),
                "passed": False, "status": "running", "stages": []}
            report["cases"].append(item)
            graph = None
            config = {"configurable": {"thread_id": runtime.session_id}}
            print("开始：", case["case_id"], case["scenario"], flush=True)
            try:
                await runtime.startup()
                reference = runtime.time_context.as_dict()
                case = expand_dates(case, date.fromisoformat(reference["reference_date"]))
                if original:
                    case, reference = original["expected"], original["reference_time"]
                    item["original_failure"] = {key: original.get(key) for key in ("error_type", "error_message")}
                item.update(reference_time=reference, expected=case)
                if item["layer"] == "mcp_execution":
                    args = {"amount": "8.00", "currency": "CNY", "category": "宠物", "subcategory": "用品",
                        "description": "独立评估用品", "expense_date": reference["reference_date"],
                        "idempotency_key": "tool-eval:" + uuid4().hex}
                    before = len(read_rows(ledger))
                    first = await provider.call_tool("record_expense", args)
                    item["first_receipt"] = first
                    if case["scenario"] == "conflict":
                        try:
                            await provider.call_tool("record_expense", {**args, "amount": "9.00"})
                        except Exception as error:
                            item["conflict_error"] = {"type": type(error).__name__, "message": str(error)}
                            # SDK隐藏服务端异常细节；确认这是远端工具错误而非连接失败，再核对原记录未变。
                            assert isinstance(error, RuntimeError) and str(error).startswith("记账MCP执行错误：")
                        else:
                            raise ValueError("同键不同内容未拒绝。")
                        rows = [r for r in read_rows(ledger) if r["idempotency_key"] == args["idempotency_key"]]
                        assert len(rows) == 1 and rows[0]["amount"] == "8.00"
                    else:
                        # E08故意不采用首次响应作为提交确认，复用原键重提；未制造真实断网。
                        second = await provider.call_tool("record_expense", args)
                        item["second_receipt"] = second
                        assert second["status"] == "already_recorded"
                        assert first["record"]["expense_id"] == second["record"]["expense_id"]
                    assert len(read_rows(ledger)) == before + 1
                    item.update(passed=True, status="completed")
                    continue

                def guard(state):
                    """核对冻结用例后才执行；参数为工具图状态；返回空更新，错误参数抛出并判失败。"""
                    calls = state.get("tool_calls") or []
                    if not calls and not state.get("need_tool"):
                        return {}
                    if len(calls) != 1:
                        raise ValueError("评估守卫：预期单次工具调用。")
                    call = calls[0]
                    call = call.model_dump() if hasattr(call, "model_dump") else call
                    check_arguments(call["name"], effective_arguments(call["name"], call["args"]), case)
                    item.setdefault("approved_calls", []).append(deepcopy(call))
                    return {}

                async def open_graph():
                    """打开本题SQLite检查点；无参数；返回原ToolAgent图，复用实际解析和执行节点。"""
                    saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(checkpoint_folder / f"{case['case_id']}_checkpoints.sqlite3")))
                    return build_tool_agent_graph(llm_provider=container.get("llm"), mcp_provider=provider,
                        interrupt_func=interrupt, before_execute=guard, checkpointer=saver)

                graph = await open_graph()
                state = create_initial_state(case["question"], runtime.trace_id, session_id=runtime.session_id)
                state["user_id"] = runtime.user_id
                before_rows = read_rows(ledger)
                if original:
                    saved = await graph.aget_state(config)
                    assert saved.next, "原请求已结束，不能静默生成新计划重跑。"
                    result = await graph.ainvoke(None, config)
                    item["resumed_original_checkpoint"] = True
                else:
                    result = await graph.ainvoke(state, config)
                snap = await graph.aget_state(config)
                item["stages"].append({"stage": "initial", "state": snap.values})
                if case["scenario"] == "clarification" and original:
                    first = original["stages"][0]["state"]
                    assert case["missing_field"] in first["tool_agent_clarification_request"]["missing_fields"]
                    item["original_clarification_stages"] = original["stages"]
                elif case["scenario"] == "clarification":
                    pending = snap.values.get("tool_agent_clarification_request") or {}
                    assert case["missing_field"] in pending.get("missing_fields", [])
                    assert len(read_rows(ledger)) == len(before_rows)
                    # 复用项目已有补参适配器，不再要求模型把短回复重新解析成新账单。
                    restored = {**snap.values, "question": case["reply"]}
                    resolution = resolve_tool_clarification_input(restored)
                    assert resolution["action"] == "resumed"
                    state = create_initial_state(case["reply"], runtime.trace_id, session_id=runtime.session_id)
                    state.update(resolution["state_update"], user_id=runtime.user_id)
                    result = await graph.ainvoke(state, config)
                    snap = await graph.aget_state(config)
                    item["stages"].append({"stage": "clarified", "state": snap.values})
                if case["scenario"] == "unsupported":
                    assert not result.get("__interrupt__") and not snap.values.get("tool_results")
                    assert not snap.values.get("tool_calls") and not snap.values.get("tool_agent_pending_tool_call")
                    assert len(read_rows(ledger)) == len(before_rows)
                    assert any(word in str(snap.values.get("final_answer") or "")
                               for word in ("计划", "尚未", "未发生", "未来", "实际"))
                    item.update(passed=True, status="completed")
                    continue
                if result.get("__interrupt__"):
                    calls = snap.values.get("tool_calls") or []
                    assert len(calls) == 1
                    call = calls[0].model_dump() if hasattr(calls[0], "model_dump") else calls[0]
                    check_arguments(call["name"], effective_arguments(call["name"], call["args"]), case)
                    item["confirmation_call"] = deepcopy(call)
                    if case["scenario"] == "checkpoint":
                        await stack.aclose()
                        stack = AsyncExitStack()
                        graph = await open_graph()
                        restored = await graph.aget_state(config)
                        assert restored.values["tool_calls"] == calls and restored.next
                        item["checkpoint_reopened"] = True
                    result = await graph.ainvoke(Command(resume="n" if case["scenario"] == "reject" else "y"), config)
                elif case["scenario"] in {"record", "batch", "clarification", "checkpoint", "reject"}:
                    raise ValueError("写入任务没有进入真实确认中断。")
                snap = await graph.aget_state(config)
                item["state"] = snap.values
                assert not snap.next
                if case["scenario"] == "reject":
                    assert len(read_rows(ledger)) == len(before_rows)
                    assert not snap.values.get("tool_agent_runtime_execution_records")
                    assert (snap.values.get("tool_agent_permission") or {}).get("status") == "rejected"
                else:
                    results = snap.values.get("tool_results") or []
                    assert len(results) == 1 and results[0]["success"]
                    assert item.get("approved_calls")
                    content = results[0]["content"]
                    if case["scenario"] == "budget":
                        for key, value in case["expected_output"].items():
                            assert Decimal(str(content[key])) == Decimal(value)
                        assert len(read_rows(ledger)) == len(before_rows)
                    elif case["scenario"] == "query":
                        expected = [r["expense_id"] for r in before_rows if r["category"] == case["expected_args"]["category"]][:case["expected_args"]["limit"]]
                        assert [r["expense_id"] for r in content["records"]] == expected
                        assert len(read_rows(ledger)) == len(before_rows)
                    else:
                        expected_count = len(case["expected_args"]["items"]) if case["scenario"] == "batch" else 1
                        assert len(read_rows(ledger)) == len(before_rows) + expected_count
                        original_key = item["confirmation_call"]["args"]["idempotency_key"]
                        assert item["approved_calls"][-1]["args"]["idempotency_key"] == original_key
                        assert content["status"] in {"recorded", "already_recorded"}
                item.update(passed=True, status="completed")
            except Exception as error:
                import traceback
                item.update(status="failed", error_type=type(error).__name__, error_message=str(error),
                            traceback=traceback.format_exc())
            finally:
                if graph is not None:
                    try:
                        item["state"] = (await graph.aget_state(config)).values
                    except Exception:
                        pass
                print("结果：", case["case_id"], item["status"], item.get("error_message", ""), flush=True)
                save(report)
                await stack.aclose()
                try:
                    await runtime.shutdown()
                finally:
                    runtime_ctx.set(old)
        report["summary"] = summarize(report["cases"])
        report["status"] = "completed" if all(c["passed"] for c in report["cases"]) else "completed_with_failures"
    except BaseException as error:
        report.update(status="aborted", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        save(report)
        try:
            await container.shutdown()
        finally:
            if child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 10)
                except TimeoutError:
                    child.kill()
                    await child.wait()
            log.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立账本真实工具评估，会调用模型及写入本轮测试数据库。")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/tool_real_development_v1.json")
    parser.add_argument("--correct-source-report", type=Path, help="只修正原评估器误判，保留已通过项并从原检查点继续，不重新生成计划。")
    args = parser.parse_args()
    result = asyncio.run(run(args.dataset, correct_source_report=args.correct_source_report))
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    print("报告：", result["report_path"])
