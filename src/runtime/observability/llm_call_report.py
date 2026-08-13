"""LLM 调用报告构建、重复候选识别和 Markdown 持久化。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.runtime.observability.llm_call_budget import (
    LLMBudgetEvaluation,
    LLMCallBudgetLimits,
    evaluate_llm_call_budgets,
)
from src.runtime.observability.llm_call_records import LLMCallRecord


DuplicateScope = Literal["agent", "multi_agent_step", "component"]


class LLMDuplicateCallCandidate(BaseModel):
    """
    描述一组疑似重复的 LLM 逻辑调用。

    功能：
        保存触发重复判定的严格业务身份和对应调用编号。它只表示值得排查，
        不直接断言业务代码存在错误。

    参数含义：
        scope：重复判定使用的身份层级。
        agent_name：调用所属 Agent；基础组件调用可以为空。
        component：实际发起调用的代码组件。
        step_id：多智能体步骤编号；普通调用为空。
        call_purpose：受控的 LLM 调用目的。
        logical_call_count：同组逻辑调用数量。
        call_ids：同组调用的唯一编号列表。

    返回值含义：
        LLMDuplicateCallCandidate：可以写入报告的疑似重复调用分组。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: DuplicateScope
    agent_name: str = ""
    component: str = ""
    step_id: str = ""
    call_purpose: str
    logical_call_count: int = Field(ge=2)
    call_ids: list[str] = Field(default_factory=list)


class LLMCallReport(BaseModel):
    """
    保存一次请求的 LLM 调用审计报告。

    功能：
        聚合逻辑调用数、物理尝试数、耗时、Token、状态、备用模型使用情况，
        并附带严格规则识别出的疑似重复调用分组和原始调用明细。

    参数含义：
        trace_id：当前用户请求的追踪编号。
        created_at：报告生成时间，使用 UTC ISO 8601 格式。
        logical_call_count：业务代码调用 safe_ainvoke 的次数。
        physical_attempt_count：主模型重试和备用模型尝试次数总和。
        status_counts：按 completed、fallback、failed 聚合的数量。
        total_latency_ms：全部逻辑调用耗时之和，不代表墙钟耗时。
        token_usage：输入、输出和总 Token 用量。
        backup_used_count：使用过备用模型的逻辑调用数量。
        invalid_record_count：无法通过调用契约校验而被忽略的记录数。
        duplicate_candidates：疑似重复调用分组。
        budget_evaluation：按调用目的执行的软预算检查结论。
        calls：经过契约校验的 LLM 调用明细。

    返回值含义：
        LLMCallReport：可用于日志摘要和 Markdown 报告的统一数据对象。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = ""
    created_at: str
    logical_call_count: int = Field(ge=0)
    physical_attempt_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    total_latency_ms: float = Field(ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    backup_used_count: int = Field(ge=0)
    invalid_record_count: int = Field(ge=0)
    duplicate_candidates: list[LLMDuplicateCallCandidate] = Field(
        default_factory=list
    )
    budget_evaluation: LLMBudgetEvaluation = Field(
        default_factory=lambda: LLMBudgetEvaluation(
            status="not_configured"
        )
    )
    calls: list[LLMCallRecord] = Field(default_factory=list)


def build_llm_call_report(
    *,
    trace_id: str,
    raw_calls: Sequence[Mapping[str, Any]] | None,
    budgets_by_purpose: Mapping[
        str,
        LLMCallBudgetLimits | Mapping[str, int | float | None],
    ] | None = None,
) -> LLMCallReport:
    """
    根据请求级 LLM 调用明细构建结构化报告。

    功能：
        先用 LLMCallRecord 校验每条原始记录，再计算调用次数、尝试次数、
        耗时、Token 和疑似重复调用。坏记录会被计数并跳过，不阻断主请求。

    参数含义：
        trace_id：当前请求的链路追踪编号。
        raw_calls：MetricsScope 中保存的 llm_calls 原始字典列表。
        budgets_by_purpose：按调用目的配置的可选软预算；空映射只统计不判定。

    返回值含义：
        LLMCallReport：本次请求的完整 LLM 调用报告。
    """

    calls: list[LLMCallRecord] = []
    invalid_record_count = 0
    for raw_call in raw_calls or []:
        try:
            calls.append(LLMCallRecord.model_validate(raw_call))
        except Exception:
            invalid_record_count += 1

    status_counts = Counter(call.status for call in calls)
    return LLMCallReport(
        trace_id=str(trace_id or "").strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
        logical_call_count=len(calls),
        physical_attempt_count=sum(call.attempt_count for call in calls),
        status_counts=dict(status_counts),
        total_latency_ms=round(sum(call.latency_ms for call in calls), 2),
        token_usage={
            "input_tokens": sum(call.input_tokens for call in calls),
            "output_tokens": sum(call.output_tokens for call in calls),
            "total_tokens": sum(call.total_tokens for call in calls),
        },
        backup_used_count=sum(call.backup_used for call in calls),
        invalid_record_count=invalid_record_count,
        duplicate_candidates=find_llm_duplicate_candidates(calls),
        budget_evaluation=evaluate_llm_call_budgets(
            calls=calls,
            budgets_by_purpose=budgets_by_purpose,
        ),
        calls=calls,
    )


def find_llm_duplicate_candidates(
    calls: Sequence[LLMCallRecord],
) -> list[LLMDuplicateCallCandidate]:
    """
    使用严格业务身份识别疑似重复的 LLM 逻辑调用。

    功能：
        多智能体调用按 Agent、step_id 和调用目的分组；普通 Agent 调用按
        Agent 和调用目的分组；没有 Agent 身份的基础调用按组件和目的分组。
        未声明调用目的的记录不参与判定，同组超过一次才形成候选项。

    参数含义：
        calls：已经通过 LLMCallRecord 校验的调用明细。

    返回值含义：
        list[LLMDuplicateCallCandidate]：稳定排序的疑似重复调用分组列表。
    """

    grouped_calls: dict[
        tuple[str, str, str, str, str],
        list[LLMCallRecord],
    ] = defaultdict(list)
    for call in calls:
        metadata = call.metadata
        purpose = str(metadata.call_purpose or "").strip()
        if not purpose or purpose == "unspecified":
            continue

        agent_name = str(metadata.agent_name or "").strip()
        component = str(metadata.component or "").strip()
        step_id = str(metadata.step_id or "").strip()
        if step_id and agent_name:
            key = (
                "multi_agent_step",
                agent_name,
                "",
                step_id,
                purpose,
            )
        elif agent_name:
            key = ("agent", agent_name, "", "", purpose)
        elif component:
            key = ("component", "", component, "", purpose)
        else:
            continue
        grouped_calls[key].append(call)

    candidates: list[LLMDuplicateCallCandidate] = []
    for key, grouped in sorted(grouped_calls.items()):
        if len(grouped) < 2:
            continue
        scope, agent_name, component, step_id, purpose = key
        observed_components = {
            str(call.metadata.component or "").strip()
            for call in grouped
        }
        if scope != "component" and len(observed_components) == 1:
            component = next(iter(observed_components))
        candidates.append(
            LLMDuplicateCallCandidate(
                scope=scope,
                agent_name=agent_name,
                component=component,
                step_id=step_id,
                call_purpose=purpose,
                logical_call_count=len(grouped),
                call_ids=[call.call_id for call in grouped],
            )
        )
    return candidates


def render_llm_call_log_summary(report: LLMCallReport) -> str:
    """
    把 LLM 调用报告压缩成一行日志摘要。

    功能：
        只展示排障最常用的调用数、尝试数、耗时、Token、失败和重复候选数，
        不输出 Prompt、回答内容或其他可能包含敏感数据的字段。

    参数含义：
        report：已经构建完成的结构化 LLM 调用报告。

    返回值含义：
        str：可以直接交给 logger.info 的单行摘要。
    """

    return (
        "LLM 调用摘要: "
        f"trace_id={report.trace_id or '-'}, "
        f"logical_calls={report.logical_call_count}, "
        f"attempts={report.physical_attempt_count}, "
        f"latency_ms={report.total_latency_ms:.2f}, "
        f"tokens={report.token_usage.get('total_tokens', 0)}, "
        f"failed={report.status_counts.get('failed', 0)}, "
        f"duplicate_candidates={len(report.duplicate_candidates)}, "
        f"budget_status={report.budget_evaluation.status}, "
        f"budget_violations={len(report.budget_evaluation.violations)}"
    )


def render_llm_call_report_markdown(report: LLMCallReport) -> str:
    """
    把结构化 LLM 调用报告渲染成 Markdown 文本。

    功能：
        生成总体摘要、疑似重复候选和逐次调用明细三个区域，便于人工审计。

    参数含义：
        report：已经构建完成的结构化 LLM 调用报告。

    返回值含义：
        str：UTF-8 Markdown 报告文本。
    """

    lines = [
        "# LLM 调用报告",
        "",
        f"- Trace ID（链路编号）：`{_escape_markdown(report.trace_id)}`",
        f"- 生成时间：`{report.created_at}`",
        f"- 逻辑调用数：{report.logical_call_count}",
        f"- 物理尝试数：{report.physical_attempt_count}",
        f"- 累计调用耗时：{report.total_latency_ms:.2f} ms",
        f"- 总 Token：{report.token_usage.get('total_tokens', 0)}",
        f"- 使用备用模型的调用数：{report.backup_used_count}",
        f"- 调用状态统计：`{report.status_counts}`",
        f"- 无效记录数：{report.invalid_record_count}",
        f"- 疑似重复分组数：{len(report.duplicate_candidates)}",
        f"- 软预算状态：`{report.budget_evaluation.status}`",
        f"- 预算超限项数：{len(report.budget_evaluation.violations)}",
        "",
        "## 软预算检查",
        "",
    ]
    if report.budget_evaluation.status == "not_configured":
        lines.append("当前没有配置任何调用目的阈值，本报告只统计实际消耗。")
    elif report.budget_evaluation.violations:
        lines.extend(
            [
                "| 调用目的 | 范围 | 指标 | 实际值 | 阈值 | Agent | Step ID | 调用编号 |",
                "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for violation in report.budget_evaluation.violations:
            lines.append(
                "| "
                f"{_escape_markdown(violation.call_purpose)} | "
                f"{violation.scope} | "
                f"{_escape_markdown(violation.metric)} | "
                f"{violation.actual} | "
                f"{violation.limit} | "
                f"{_escape_markdown(violation.agent_name)} | "
                f"{_escape_markdown(violation.step_id)} | "
                f"{_escape_markdown(violation.call_id)} |"
            )
    else:
        lines.append("已配置调用目的预算，本次请求没有发现超限。")

    lines.extend(
        [
            "",
        "## 疑似重复调用",
        "",
        ]
    )
    if report.duplicate_candidates:
        lines.extend(
            [
                "| 范围 | Agent | Step ID | 组件 | 调用目的 | 次数 | 调用编号 |",
                "| --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for candidate in report.duplicate_candidates:
            lines.append(
                "| "
                f"{candidate.scope} | "
                f"{_escape_markdown(candidate.agent_name)} | "
                f"{_escape_markdown(candidate.step_id)} | "
                f"{_escape_markdown(candidate.component)} | "
                f"{_escape_markdown(candidate.call_purpose)} | "
                f"{candidate.logical_call_count} | "
                f"{_escape_markdown(', '.join(candidate.call_ids))} |"
            )
    else:
        lines.append("没有发现符合严格判定条件的疑似重复调用。")

    lines.extend(
        [
            "",
            "## 调用明细",
            "",
            "| # | 调用编号 | Agent | Step ID | 组件 | 目的 | 模型 | 状态 | 尝试 | 耗时(ms) | Token |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for index, call in enumerate(report.calls, start=1):
        metadata = call.metadata
        lines.append(
            "| "
            f"{index} | "
            f"{_escape_markdown(call.call_id)} | "
            f"{_escape_markdown(metadata.agent_name)} | "
            f"{_escape_markdown(metadata.step_id)} | "
            f"{_escape_markdown(metadata.component)} | "
            f"{_escape_markdown(str(metadata.call_purpose))} | "
            f"{_escape_markdown(call.final_model or call.requested_model)} | "
            f"{call.status} | "
            f"{call.attempt_count} | "
            f"{call.latency_ms:.2f} | "
            f"{call.total_tokens} |"
        )
    if not report.calls:
        lines.append("| - | - | - | - | - | - | - | - | 0 | 0.00 | 0 |")
    return "\n".join(lines) + "\n"


def save_llm_call_report(
    *,
    report: LLMCallReport,
    report_dir: str | Path,
    use_date_dir: bool = True,
) -> Path:
    """
    将 LLM 调用报告保存为 Markdown 文件。

    功能：
        可按 UTC 日期创建子目录，并使用 trace_id 加微秒时间戳命名，避免
        同一 trace_id 的恢复请求覆盖前一轮报告。

    参数含义：
        report：需要持久化的结构化报告。
        report_dir：LLM 报告根目录。
        use_date_dir：是否在根目录下按 YYYY-MM-DD 创建日期目录。

    返回值含义：
        Path：已经成功写入的 Markdown 文件绝对或相对路径对象。
    """

    now = datetime.now(timezone.utc)
    output_dir = Path(report_dir)
    if use_date_dir:
        output_dir = output_dir / now.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_trace_id = _sanitize_filename(report.trace_id or "unknown_trace")
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"{safe_trace_id}_{timestamp}.md"
    output_path.write_text(
        render_llm_call_report_markdown(report),
        encoding="utf-8",
    )
    return output_path


def save_llm_call_report_json(
    *,
    report: LLMCallReport,
    markdown_report_path: str | Path,
) -> Path:
    """
    在 Markdown 报告旁保存同名 JSON 结构化报告。

    功能：
        复用 Markdown 文件路径的目录和文件名，只把扩展名改成 json，方便
        人工阅读 Markdown，也方便脚本、看板或质量门禁读取结构化数据。

    参数含义：
        report：需要持久化的结构化 LLM 调用报告。
        markdown_report_path：已经保存成功的 Markdown 报告路径。

    返回值含义：
        Path：已经成功写入的 JSON 报告路径。
    """

    output_path = Path(markdown_report_path).with_suffix(".json")
    output_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output_path


def _sanitize_filename(value: str) -> str:
    """
    将追踪编号转换成可安全用于文件名的文本。

    参数含义：
        value：原始追踪编号。

    返回值含义：
        str：只保留字母、数字、点、下划线和短横线的文件名片段。
    """

    return "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in str(value)
    )


def _escape_markdown(value: Any) -> str:
    """
    转义 Markdown 表格中会破坏列结构的字符。

    参数含义：
        value：需要展示在 Markdown 表格单元格中的值。

    返回值含义：
        str：已处理换行符和竖线的单行文本。
    """

    return str(value or "").replace("\n", " ").replace("|", "\\|")
