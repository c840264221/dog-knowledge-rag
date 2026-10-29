"""自然语言记账第一步：只提取并预览，不连接MCP、不写SQLite、不生成实际幂等键。

示例：python -m tests.learning.plus_expense_extraction_demo "昨天狗子洗澡80元"
指定对照日期：追加 --reference-date 2026-10-24；真实模型调用可能收费。
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from src.mcp.expenses.extraction import extract_expense_proposal
from src.mcp.expenses.schemas import ExpenseDate
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.context import RuntimeContext, runtime_ctx


class ExpenseJsonLLMProvider(LLMProvider):
    """教学JSON模型服务，沿用项目配置、重试与观测，只初始化本步需要的模型。"""

    @property
    def main_llm(self):
        """为主模型绑定JSON输出；无参数；返回绑定对象，不修改项目全局配置。"""
        return super().main_llm.bind(response_format={"type": "json_object"})

    @property
    def backup_llm(self):
        """为备用模型绑定同样JSON输出；无参数；返回绑定对象，避免备用路径丢失格式要求。"""
        return super().backup_llm.bind(response_format={"type": "json_object"})

    async def startup(self) -> None:
        """只初始化主模型；无参数、无返回值；备用按需创建，不启动无关的本地中文模型。"""
        _ = self.main_llm


async def run_extraction_demo(question: str, *, reference_date: str, batch: bool = False) -> dict:
    """运行独立提取并保存报告。

    参数：question为用户原话；reference_date为日期基准；batch为是否按1至20笔批量提取。
    返回值：含提议、原始响应、状态和路径的报告；失败保存诊断后继续抛出。
        本步没有确认或执行动作，不写账本；报告含用户文本，应注意本地隐私。
    """
    reference_date = TypeAdapter(ExpenseDate).validate_python(reference_date)
    root = Path(__file__).resolve().parents[2]
    run_id = uuid4().hex
    output_dir = root / "output/learning/expenses/extraction_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "extraction_report.json"
    report = {
        "run_id": run_id,
        "question": question,
        "reference_date": reference_date,
        "timezone": "Asia/Hong_Kong",
        "status": "extracting",
        "report_path": str(report_path),
        "notice": "仅提取预览，没有调用MCP或写账；契约通过不等于语义完全正确。",
    }
    print(f"本次日期基准：{reference_date}（UTC+08:00）；只提取、不记账。", flush=True)
    print(f"报告位置：{report_path}", flush=True)
    container = RuntimeContainer()
    container.register("llm", ExpenseJsonLLMProvider())
    previous_context = runtime_ctx.get()
    context = RuntimeContext(
        trace_id=run_id, user_id="expense_learning_user",
        session_id=run_id, component="expense_extraction_demo",
    )
    runtime_ctx.set(context)
    try:
        await context.startup()
        await container.startup()
        proposal = await extract_expense_proposal(
            container.get("llm"), question, reference_date=reference_date, report=report, batch=batch,
        )
        report["proposal"] = proposal.model_dump(mode="json")
        report["status"] = "extracted_pending_review" if proposal.status == "ready" else proposal.status
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error_message"] = str(error)
        raise
    finally:
        try:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            try:
                await container.shutdown()
            finally:
                try:
                    await context.shutdown()
                finally:
                    runtime_ctx.set(previous_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从自然语言提取支出，只预览，不执行记账。")
    parser.add_argument("question", help="单笔消费原话，例如：昨天狗子洗澡80元")
    parser.add_argument("--reference-date", help="今天/昨天等表达的日期基准，YYYY-MM-DD；不填取当前UTC+8日期")
    parser.add_argument("--batch", action="store_true", help="提取一批账单，任何一笔缺参则整批澄清")
    args = parser.parse_args()
    reference_date = args.reference_date or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    print(json.dumps(asyncio.run(run_extraction_demo(args.question, reference_date=reference_date, batch=args.batch)), ensure_ascii=False, indent=2))
