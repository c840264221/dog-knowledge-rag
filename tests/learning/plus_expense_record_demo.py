"""自然语言记账闭环：提取或复用报告、确认，经MCP提交单笔或整批并保存回执。

已有提取报告：python -m tests.learning.plus_expense_record_demo --source-report "报告完整路径"
新问题：python -m tests.learning.plus_expense_record_demo --question "昨天狗子洗澡80元" --reference-date 2026-10-24
批量：在新问题命令后增加--batch；复用报告时自动沿用报告记录的模式。
本机教学，无生产认证；执行报告属于可信本地操作材料，不接受不可信用户上传的确认报告。
同一份提取报告只代表一个动作，重复运行复用原键；跨进程工作流与多轮澄清不在本步实现。
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID

from src.mcp.expenses.extraction import (
    ExpenseBatchProposal, ExpenseProposal, build_expense_batch_request, build_expense_request,
)
from src.mcp.expenses.schemas import ExpenseBatchInput, ExpenseBatchWriteResult, ExpenseInput, ExpenseWriteResult


def request_fingerprint(arguments: dict) -> str:
    """计算参数指纹；arguments为规范化工具参数；返回SHA-256文本，只检测变化，不是签名或权限。"""
    return hashlib.sha256(json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def save_execution_report(path: Path, report: dict) -> None:
    """保存本地执行快照；path为目标路径，report为报告；无返回值，不修改来源报告或账本。

    先写同目录临时文件再替换，降低单进程写文件中断留下半份JSON的风险。
    不是跨进程锁或数据库事务；仅用于本步单进程教学操作记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_receipt(receipt: ExpenseWriteResult | ExpenseBatchWriteResult, arguments: dict) -> None:
    """核对回执；receipt为结构化回执，arguments为确认的参数；一致无返回值，不一致抛出异常。"""
    if isinstance(receipt, ExpenseBatchWriteResult):
        expected_items = arguments["items"]
        if (receipt.idempotency_key != arguments["idempotency_key"]
                or receipt.record_count != len(expected_items) or len(receipt.records) != len(expected_items)):
            raise ValueError("批次回执的原键或笔数与确认内容不一致。")
        if len({row.expense_id for row in receipt.records}) != len(expected_items):
            raise ValueError("批次回执存在重复记录编号。")
        actual_items = [row.model_dump(mode="json", include=set(expected))
                        for row, expected in zip(receipt.records, expected_items)]
        if actual_items != expected_items:
            raise ValueError("批次回执中的逐笔内容与确认清单不一致。")
    elif receipt.record.model_dump(mode="json", include=set(arguments)) != arguments:
        raise ValueError("服务回执与确认的记账参数不一致，不能宣称成功。")


async def submit_expense_once(
    arguments: dict, *, tool_name: str = "record_expense", report: dict | None = None,
) -> ExpenseWriteResult | ExpenseBatchWriteResult:
    """通过项目工具链提交已确认动作；arguments为原参数，tool_name由程序确定，report保存轨迹。

    返回值：核对后的单笔或批量回执；复用现有ToolAgent子图，不直接调用SDK，不重新规划。
    本机入口使用固定教学身份；服务Provider仅本次显式启用，不改全局容器和主图路由。
    """
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.expense_mcp_provider import ExpenseMcpProvider
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.runtime.services.expense_tool_service import ExpenseToolService

    container = RuntimeContainer()
    container.register("expense_mcp", ExpenseMcpProvider(enabled=True))
    container.register("expense_tools", ExpenseToolService(container.get("expense_mcp")))
    previous_context = runtime_ctx.get()
    context = RuntimeContext(
        trace_id=UUID(arguments["idempotency_key"].split(":")[-1]).hex,
        user_id="expense_learning_user", session_id=arguments["idempotency_key"],
        component="project_expense_tools",
    )
    runtime_ctx.set(context)
    try:
        await context.startup()
        await container.startup()
        receipt, trace = await container.get("expense_tools").execute_confirmed(
            tool_name, arguments, confirmed=True,
        )
        check_receipt(receipt, arguments)
        if report is not None:
            report["project_execution"] = trace
        return receipt
    finally:
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous_context)


def record_from_extraction_report(source_path: Path, *, output_root: Path | None = None) -> dict:
    """读取提取报告，展示确认并通过MCP记账，复用同一动作的原键。

    参数：source_path为可信本地提取报告；output_root为可选输出根目录，默认项目教学目录。
    返回值：含已完成、取消或不可执行状态的报告；通信失败保存失败或未知状态再抛出。
        只有终端本次明确确认后才提交，不以模型ready或文件里的确认标记代替确认。
    """
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source.get("status") not in {"extracted_pending_review", "needs_clarification", "unsupported"}:
        raise ValueError("来源不是成功提取报告，不允许准备记账。")
    batch = source.get("extraction_mode", "single") == "batch"
    proposal = (ExpenseBatchProposal if batch else ExpenseProposal).model_validate(source["proposal"])
    if proposal.status != "ready":
        return {
            "status": proposal.status,
            "message": proposal.message,
            "source_report": str(source_path),
            "notice": "没有进入确认或调用MCP；多轮澄清恢复后续再接入。",
        }
    # 编号来自程序生成的提取运行编号，而不是让模型生成动作编号。
    run_id = UUID(source["run_id"]).hex
    tool_name = "record_expenses" if batch else "record_expense"
    arguments = (build_expense_batch_request if batch else build_expense_request)(
        proposal, idempotency_key=("expense-natural-batch:" if batch else "expense-natural:") + run_id,
    ).model_dump(mode="json")
    fingerprint = request_fingerprint(arguments)
    root = output_root or Path(__file__).resolve().parents[2] / "output/learning/expenses/record_runs"
    report_path = root / run_id / "execution_report.json"
    report = {
        "source_report": str(source_path),
        "question": source["question"],
        "reference_date": source["reference_date"],
        "status": "awaiting_confirmation",
        "arguments": arguments,
        "tool_name": tool_name,
        "request_fingerprint": fingerprint,
        "report_path": str(report_path),
        "notice": "本机教学执行报告，不是认证授权凭据；同一提取报告只对应一个记账动作。",
    }
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (existing.get("arguments") != arguments or existing.get("request_fingerprint") != fingerprint
                or existing.get("tool_name", "record_expense") != tool_name):
            raise ValueError("来源内容或原执行参数已改变，拒绝用原动作编号提交。")
        if existing.get("status") == "completed":
            receipt = (ExpenseBatchWriteResult if batch else ExpenseWriteResult).model_validate(existing["receipt"])
            check_receipt(receipt, arguments)
            print("本地已有该动作的成功回执，直接展示；本次未再调用MCP。", flush=True)
            return existing
        if existing.get("status") == "cancelled":
            print("这个动作此前已取消，本次不重新提交。", flush=True)
            return existing
        if existing.get("status") not in {"awaiting_confirmation", "submitting", "submission_failed_or_unknown"}:
            raise ValueError("执行状态不允许继续提交。")
        print("恢复原动作编号；之前可能已提交，本次若执行只会复用原键。", flush=True)

    # 先保存参数和原键，再询问和提交；响应丢失后仍可按同一报告恢复。
    save_execution_report(report_path, report)
    print(f"用户原话：{source['question']}", flush=True)
    print(f"待记账内容（{'整批' + str(len(arguments['items'])) + '笔' if batch else '单笔'}，含实际日期）：", flush=True)
    print(json.dumps(arguments, ensure_ascii=False, indent=2), flush=True)
    print(f"执行报告：{report_path}", flush=True)
    confirmed = input("确认通过MCP保存以上完整清单？输入 y 继续，其他输入取消：").strip().casefold()
    if confirmed != "y":
        report["status"] = "cancelled"
        save_execution_report(report_path, report)
        return report

    # 再次校验本次确认的规范参数；不会重新问模型，也不会接收模型提供的确认状态。
    checked = (ExpenseBatchInput if batch else ExpenseInput).model_validate(arguments).model_dump(mode="json")
    if request_fingerprint(checked) != fingerprint:
        raise ValueError("确认后的参数发生变化，拒绝提交。")
    report["status"] = "submitting"
    save_execution_report(report_path, report)
    try:
        receipt = asyncio.run(submit_expense_once(checked, tool_name=tool_name, report=report))
        report["receipt"] = receipt.model_dump(mode="json")
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "submission_failed_or_unknown"
        report["error_type"] = type(error).__name__
        report["error_message"] = str(error)
        print("提交失败或结果未知：不要重新提取或换键；重跑同一--source-report以复用原动作。", flush=True)
        raise
    finally:
        save_execution_report(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自然语言支出提取、确认与MCP记账；确认后实际写入教学账本。")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-report", type=Path, help="复用已有提取报告，不重复调用模型")
    source_group.add_argument("--question", help="新问题，将实际调用模型提取后再进入确认")
    parser.add_argument("--reference-date", help="仅用于新问题的日期基准，YYYY-MM-DD")
    parser.add_argument("--batch", action="store_true", help="新问题按1至20笔批量提取，整批确认和保存")
    args = parser.parse_args()
    if args.source_report is not None:
        if args.reference_date is not None or args.batch:
            parser.error("复用报告时沿用其日期基准与批量模式，不另传--reference-date或--batch。")
        source_path = args.source_report
    else:
        from datetime import datetime, timedelta, timezone
        from tests.learning.plus_expense_extraction_demo import run_extraction_demo

        reference = args.reference_date or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        extracted = asyncio.run(run_extraction_demo(args.question, reference_date=reference, batch=args.batch))
        source_path = Path(extracted["report_path"])
        print("提取报告已保存；若提交异常，后续使用此路径的--source-report，不重新运行--question。", flush=True)
        print(f"来源报告：{source_path}", flush=True)
    result = record_from_extraction_report(source_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
