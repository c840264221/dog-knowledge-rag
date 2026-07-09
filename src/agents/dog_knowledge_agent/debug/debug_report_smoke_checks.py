"""
DogKnowledgeAgent Debug Report Smoke Check 工具。

功能：
    为 V1.7.2 DogKnowledgeAgent Debug Report 真实链路验证提供检查能力。

    Smoke Check 中文叫“冒烟检查”，表示：
    用较少断言确认主链路关键能力是否已经接入成功。

当前模块主要检查：
    1. state 中是否存在 dog_knowledge_debug_report。
    2. dog_knowledge_debug_report 是否是 dict。
    3. report section 是否为 dog_knowledge_agent。
    4. report 是否包含 pipeline / rag / memory / strategy / answer 五个核心 section。
    5. pipeline section 是否能看到标准 pipeline layers。
    6. report status 是否属于允许状态。
    7. 可以渲染 Markdown smoke report。

当前不负责：
    1. 不执行真实主图。
    2. 不调用 LLM。
    3. 不执行 Retriever。
    4. 不判断最终答案是否正确。
    5. 不修改 state。

专业名词：
    Debug Report：调试报告，用于复盘 Agent 执行过程。
    Smoke Check：冒烟检查，用最小成本验证核心链路是否接入。
    Section：报告片段，表示调试报告中的一个结构化区域。
    Pipeline：管线，表示 Agent 内部标准执行流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.dog_knowledge_agent.debug.debug_report import (
    render_dog_knowledge_debug_report_markdown,
)


EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS = (
    "pipeline",
    "rag",
    "memory",
    "strategy",
    "answer",
)


EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS = (
    "entry",
    "query_builder",
    "retrieval",
    "rerank",
    "quality",
    "context_builder",
    "memory_context",
    "strategy",
    "generation",
    "debug_report",
)


ALLOWED_DOG_KNOWLEDGE_DEBUG_STATUSES = (
    "ready",
    "pipeline_only",
    "missing_pipeline",
    "incomplete",
)


@dataclass(frozen=True)
class DogKnowledgeDebugReportSmokeResult:
    """
    DogKnowledgeAgent Debug Report Smoke 检查结果。

    功能：
        表示一次 dog_knowledge_debug_report 冒烟检查的结果。

    参数：
        passed:
            是否通过检查。

        errors:
            错误列表。
            有错误表示 smoke check 不通过。

        warnings:
            警告列表。
            警告表示需要注意，但不一定阻断流程。

        report_status:
            dog_knowledge_debug_report 中的 status 字段。

        observed_sections:
            实际观察到的 report section 列表。

        observed_layers:
            实际观察到的 pipeline layer 顺序。

        summary:
            中文摘要。

    返回值：
        DogKnowledgeDebugReportSmokeResult:
            不可变的 smoke 检查结果对象。
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    report_status: str
    observed_sections: tuple[str, ...]
    observed_layers: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """
        将 smoke result 转换为 dict。

        功能：
            方便写入日志、Debug Report 或测试输出。

        参数：
            无。

        返回值：
            dict[str, Any]:
                当前 smoke result 的字典表示。
        """

        return {
            "passed": self.passed,
            "errors": list(
                self.errors,
            ),
            "warnings": list(
                self.warnings,
            ),
            "report_status": self.report_status,
            "observed_sections": list(
                self.observed_sections,
            ),
            "observed_layers": list(
                self.observed_layers,
            ),
            "summary": self.summary,
        }


def extract_dog_knowledge_debug_report(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    从 state 中提取 dog_knowledge_debug_report。

    功能：
        安全读取 state["dog_knowledge_debug_report"]。
        如果不存在或不是 dict，则返回空 dict。

    参数：
        state:
            当前 LangGraph 最终状态或节点返回状态。

    返回值：
        dict[str, Any]:
            dog_knowledge_debug_report。
            如果不存在则返回空 dict。
    """

    report = state.get(
        "dog_knowledge_debug_report",
    )

    if isinstance(
            report,
            dict,
    ):
        return report

    return {}


def extract_debug_report_sections(
        debug_report: dict[str, Any],
) -> tuple[str, ...]:
    """
    提取 Debug Report 中存在的核心 section。

    功能：
        检查 debug_report 中是否包含 pipeline、rag、memory、strategy、answer。

    参数：
        debug_report:
            DogKnowledgeAgent 调试报告。

    返回值：
        tuple[str, ...]:
            实际存在的核心 section 名称。
    """

    sections: list[str] = []

    for section in EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS:
        if isinstance(
                debug_report.get(
                    section,
                ),
                dict,
        ):
            sections.append(
                section,
            )

    return tuple(
        sections,
    )


def extract_debug_report_pipeline_layers(
        debug_report: dict[str, Any],
) -> tuple[str, ...]:
    """
    从 Debug Report 中提取 pipeline layers。

    功能：
        读取 debug_report["pipeline"]["layers"]，
        并转换成 tuple[str, ...]。

    参数：
        debug_report:
            DogKnowledgeAgent 调试报告。

    返回值：
        tuple[str, ...]:
            pipeline layer 顺序。
            如果不存在则返回空 tuple。
    """

    pipeline = debug_report.get(
        "pipeline",
        {},
    )

    if not isinstance(
            pipeline,
            dict,
    ):
        return ()

    layers = pipeline.get(
        "layers",
        [],
    )

    if not isinstance(
            layers,
            list,
    ):
        return ()

    return tuple(
        layer
        for layer in layers
        if isinstance(
            layer,
            str,
        )
    )


def validate_dog_knowledge_debug_report_smoke_state(
        state: dict[str, Any],
) -> DogKnowledgeDebugReportSmokeResult:
    """
    验证 DogKnowledgeAgent Debug Report smoke state。

    功能：
        综合检查真实主图最终 state 中的 dog_knowledge_debug_report 是否结构完整。

    参数：
        state:
            当前 LangGraph 最终状态或 dog_knowledge_agent 节点返回状态。

    返回值：
        DogKnowledgeDebugReportSmokeResult:
            Debug Report smoke 检查结果。
    """

    errors: list[str] = []
    warnings: list[str] = []

    raw_report = state.get(
        "dog_knowledge_debug_report",
    )

    if raw_report is None:
        errors.append(
            "state 中缺少 dog_knowledge_debug_report。"
        )

    if raw_report is not None and not isinstance(
            raw_report,
            dict,
    ):
        errors.append(
            "dog_knowledge_debug_report 必须是 dict。"
        )

    debug_report = extract_dog_knowledge_debug_report(
        state=state,
    )

    section = debug_report.get(
        "section",
        "",
    )

    if section != "dog_knowledge_agent":
        errors.append(
            "dog_knowledge_debug_report.section 应该为 dog_knowledge_agent。"
        )

    report_status = str(
        debug_report.get(
            "status",
            "",
        )
    )

    if report_status not in ALLOWED_DOG_KNOWLEDGE_DEBUG_STATUSES:
        errors.append(
            "dog_knowledge_debug_report.status 不属于允许状态。"
        )

    observed_sections = extract_debug_report_sections(
        debug_report=debug_report,
    )

    missing_sections = tuple(
        section_name
        for section_name in EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS
        if section_name not in observed_sections
    )

    if missing_sections:
        errors.append(
            f"dog_knowledge_debug_report 缺少核心 section: {list(missing_sections)}。"
        )

    observed_layers = extract_debug_report_pipeline_layers(
        debug_report=debug_report,
    )

    if observed_layers != EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS:
        errors.append(
            "dog_knowledge_debug_report.pipeline.layers 不符合 V1.7.2 标准顺序。"
        )

    warnings.extend(
        build_debug_report_warnings(
            debug_report=debug_report,
        )
    )

    passed = not errors

    summary = build_debug_report_smoke_summary(
        passed=passed,
        report_status=report_status,
        observed_sections=observed_sections,
        observed_layers=observed_layers,
    )

    return DogKnowledgeDebugReportSmokeResult(
        passed=passed,
        errors=tuple(
            errors,
        ),
        warnings=tuple(
            warnings,
        ),
        report_status=report_status,
        observed_sections=observed_sections,
        observed_layers=observed_layers,
        summary=summary,
    )


def build_debug_report_warnings(
        debug_report: dict[str, Any],
) -> list[str]:
    """
    构建 Debug Report smoke warning 列表。

    功能：
        对非阻断问题生成 warning。
        这些问题不一定导致 smoke check 失败，但需要开发者注意。

    参数：
        debug_report:
            DogKnowledgeAgent 调试报告。

    返回值：
        list[str]:
            warning 列表。
    """

    warnings: list[str] = []

    rag = debug_report.get(
        "rag",
        {},
    )

    memory = debug_report.get(
        "memory",
        {},
    )

    strategy = debug_report.get(
        "strategy",
        {},
    )

    answer = debug_report.get(
        "answer",
        {},
    )

    if isinstance(
            rag,
            dict,
    ) and not rag.get(
        "has_rag_query",
        False,
    ):
        warnings.append(
            "Debug Report 中暂未发现 rag_query，可能当前真实业务还没有写入该字段。"
        )

    if isinstance(
            memory,
            dict,
    ) and not memory.get(
        "has_memory_context",
        False,
    ):
        warnings.append(
            "Debug Report 中暂未发现 memory_context，可能当前问题没有触发记忆召回。"
        )

    if isinstance(
            strategy,
            dict,
    ) and not strategy.get(
        "has_answer_strategy",
        False,
    ):
        warnings.append(
            "Debug Report 中暂未发现 answer_strategy，可能当前阶段还没有接入策略选择器。"
        )

    if isinstance(
            answer,
            dict,
    ) and not answer.get(
        "has_final_answer",
        False,
    ):
        warnings.append(
            "Debug Report 中暂未发现 final_answer，请确认真实业务 delegate 是否返回答案字段。"
        )

    return warnings


def build_debug_report_smoke_summary(
        passed: bool,
        report_status: str,
        observed_sections: tuple[str, ...],
        observed_layers: tuple[str, ...],
) -> str:
    """
    构建 Debug Report Smoke 中文摘要。

    功能：
        根据 smoke check 结果生成中文摘要。

    参数：
        passed:
            是否通过。

        report_status:
            report 中的 status。

        observed_sections:
            实际观察到的核心 section。

        observed_layers:
            实际观察到的 pipeline layers。

    返回值：
        str:
            中文摘要。
    """

    if passed:
        return (
            "DogKnowledgeAgent Debug Report Smoke Check 通过："
            f"report_status={report_status}，"
            f"sections={list(observed_sections)}，"
            f"pipeline_layers={list(observed_layers)}。"
        )

    return (
        "DogKnowledgeAgent Debug Report Smoke Check 未通过："
        f"report_status={report_status}，"
        f"sections={list(observed_sections)}，"
        f"pipeline_layers={list(observed_layers)}。"
    )


def render_dog_knowledge_debug_report_smoke_markdown(
        result: DogKnowledgeDebugReportSmokeResult,
        state: dict[str, Any] | None = None,
) -> str:
    """
    渲染 DogKnowledgeAgent Debug Report Smoke Markdown。

    功能：
        将 smoke check result 渲染为 Markdown。
        如果传入 state，并且其中存在 dog_knowledge_debug_report，
        会附加渲染后的 DogKnowledgeAgent Debug Report。

    参数：
        result:
            Debug Report smoke 检查结果。

        state:
            可选的 LangGraph 状态。
            用于附加完整 Debug Report Markdown。

    返回值：
        str:
            Markdown 格式 smoke report。
    """

    lines = [
        "# DogKnowledgeAgent Debug Report Smoke Check",
        "",
        f"- passed: `{result.passed}`",
        f"- report_status: `{result.report_status}`",
        f"- summary: {result.summary}",
        "",
        "## Observed Sections",
        "",
        " -> ".join(
            result.observed_sections,
        ) or "未观察到核心 sections",
        "",
        "## Observed Pipeline Layers",
        "",
        " -> ".join(
            result.observed_layers,
        ) or "未观察到 pipeline layers",
        "",
        "## Errors",
        "",
    ]

    if result.errors:
        for error in result.errors:
            lines.append(
                f"- {error}"
            )
    else:
        lines.append(
            "- 无"
        )

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )

    if result.warnings:
        for warning in result.warnings:
            lines.append(
                f"- {warning}"
            )
    else:
        lines.append(
            "- 无"
        )

    if state is not None:
        debug_report = extract_dog_knowledge_debug_report(
            state=state,
        )

        if debug_report:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## Full DogKnowledgeAgent Debug Report",
                    "",
                    render_dog_knowledge_debug_report_markdown(
                        debug_report=debug_report,
                    ),
                ]
            )

    return "\n".join(
        lines,
    )
