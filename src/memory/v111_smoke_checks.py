from dataclasses import dataclass
from typing import Any


REQUIRED_MEMORY_METADATA_FIELDS = (
    "memory_id",
    "user_id",
    "memory_type",
    "status",
    "source",
    "importance",
)


@dataclass(frozen=True)
class MemoryPipelineSmokeCheckResult:
    """
    V1.11 Memory Pipeline Smoke Check（记忆管线冒烟检查）结果。

    功能：
        保存整条记忆管线的检查状态、错误、警告和关键观测数据。

    参数：
        passed：所有阻断性检查是否通过。
        errors：导致冒烟测试失败的错误列表。
        warnings：不阻断通过、但需要关注的警告列表。
        observations：从保存、向量同步和召回阶段采集的关键数据。

    返回值：
        MemoryPipelineSmokeCheckResult：不可变的冒烟检查结果。
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    observations: dict[str, Any]


def validate_memory_pipeline_smoke(
        save_result: dict[str, Any],
        vector_documents: list[Any],
        related_state: dict[str, Any],
        unrelated_state: dict[str, Any],
) -> MemoryPipelineSmokeCheckResult:
    """
    验证 V1.11 记忆保存、向量同步和召回管线。

    功能：
        检查 SQLite 保存结果、Chroma metadata（向量元数据）、
        相关问题的记忆应用结果，以及不相关问题的语义门槛拦截结果。

    参数：
        save_result：MemoryManager.save_memory 返回的保存结果。
        vector_documents：同步到向量库的 LangChain Document（文档）列表。
        related_state：相关问题经记忆召回节点处理后的 state update。
        unrelated_state：不相关问题经记忆召回节点处理后的 state update。

    返回值：
        MemoryPipelineSmokeCheckResult：包含 PASS/FAIL 状态和详细原因。
    """

    errors: list[str] = []
    warnings: list[str] = []

    if save_result.get("action") not in {
        "created",
        "reactivated",
        "reinforced",
    }:
        errors.append("记忆没有成功保存。")

    memory_id = save_result.get("memory_id")
    if memory_id is None:
        errors.append("记忆保存结果缺少 memory_id。")

    if not vector_documents:
        errors.append("记忆没有同步到向量库。")
        metadata: dict[str, Any] = {}
    else:
        metadata = dict(
            getattr(vector_documents[-1], "metadata", {})
            or {}
        )

    missing_metadata_fields = [
        field_name
        for field_name in REQUIRED_MEMORY_METADATA_FIELDS
        if field_name not in metadata
    ]
    if missing_metadata_fields:
        errors.append(
            "向量 metadata 缺少字段："
            + "、".join(missing_metadata_fields)
            + "。"
        )

    related_report = related_state.get("memory_recall_result")
    if not isinstance(related_report, dict):
        errors.append("相关问题缺少字典格式的 memory_recall_result。")
        related_report = {}

    if related_report.get("status") != "applied":
        errors.append("相关问题没有应用记忆。")
    if int(related_report.get("selected_count", 0) or 0) < 1:
        errors.append("相关问题的 selected_count 小于 1。")
    saved_content = str(
        save_result.get("content", "")
        or ""
    ).strip()
    related_memory_context = str(
        related_state.get("memory_context", "")
        or ""
    )

    if not saved_content:
        errors.append("记忆保存结果缺少标准化后的 content。")
    elif saved_content not in related_memory_context:
        errors.append(
            "相关问题的 memory_context 中没有实际保存的标准记忆内容："
            f"{saved_content}。"
        )

    unrelated_report = unrelated_state.get("memory_recall_result")
    if not isinstance(unrelated_report, dict):
        errors.append("不相关问题缺少字典格式的 memory_recall_result。")
        unrelated_report = {}

    if unrelated_report.get("status") != "empty":
        errors.append("不相关问题没有被语义门槛拦截。")
    if int(unrelated_report.get("selected_count", 0) or 0) != 0:
        errors.append("不相关问题不应采用任何记忆。")
    if unrelated_state.get("memory_context") != "暂无用户记忆":
        errors.append("不相关问题没有返回空记忆文本。")

    observations = {
        "save_action": save_result.get("action"),
        "memory_id": memory_id,
        "vector_document_count": len(vector_documents),
        "metadata_fields": sorted(metadata.keys()),
        "related_status": related_report.get("status"),
        "related_selected_count": related_report.get("selected_count", 0),
        "unrelated_status": unrelated_report.get("status"),
        "unrelated_selected_count": unrelated_report.get("selected_count", 0),
    }

    return MemoryPipelineSmokeCheckResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        observations=observations,
    )


def render_memory_pipeline_smoke_markdown(
        result: MemoryPipelineSmokeCheckResult,
) -> str:
    """
    将 V1.11 记忆管线冒烟检查结果渲染为 Markdown。

    参数：
        result：validate_memory_pipeline_smoke 返回的检查结果。

    返回值：
        str：适合在控制台阅读的 Markdown 报告。
    """

    lines = [
        "# V1.11 Memory Pipeline Smoke Report",
        "",
        f"- status: {'PASS' if result.passed else 'FAIL'}",
        "",
        "## Observations",
    ]
    lines.extend(
        f"- {key}: {value}"
        for key, value in result.observations.items()
    )
    lines.extend(["", "## Errors"])
    lines.extend(
        (f"- {error}" for error in result.errors)
        if result.errors
        else ["- 无"]
    )
    lines.extend(["", "## Warnings"])
    lines.extend(
        (f"- {warning}" for warning in result.warnings)
        if result.warnings
        else ["- 无"]
    )
    return "\n".join(lines)
