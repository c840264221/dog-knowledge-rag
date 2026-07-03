"""
DogKnowledgeAgent Smoke Test 检查工具。

功能：
    为 V1.7.2 DogKnowledgeAgent 真实主图 Smoke Test 提供结构化检查能力。

    Smoke Test 中文叫“冒烟测试”，表示：
    不做特别细的业务断言，只验证核心链路有没有跑通。

当前模块主要检查：
    1. 是否进入 dog_knowledge_agent。
    2. 是否包含 dog_knowledge_pipeline_* metadata。
    3. pipeline steps 顺序是否正确。
    4. quality 是否位于 rerank 之后、context_builder 之前。
    5. memory_context 是否和 RagContextBuilder 分离。
    6. 是否保留真实业务节点返回字段。

当前不负责：
    1. 不调用真实主图。
    2. 不调用 LLM。
    3. 不执行真实 Retriever。
    4. 不执行真实 Reranker。
    5. 不判断业务答案是否完全正确。

专业名词：
    Smoke Test：冒烟测试，用最小成本确认主链路是否能跑通。
    Metadata：元数据，用于调试和观测的辅助信息。
    Pipeline：管线，表示按顺序执行的一组步骤。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class DogKnowledgeSmokeCheckResult:
    """
    DogKnowledgeAgent Smoke Check 结果。

    功能：
        表示一次 DogKnowledgeAgent smoke check 的检查结果。

    参数：
        passed:
            是否通过检查。

        errors:
            错误列表。
            如果为空，表示没有发现阻断问题。

        warnings:
            警告列表。
            警告不一定导致失败，但需要开发者注意。

        observed_layers:
            从 state 中观察到的 pipeline layer 顺序。

        summary:
            中文摘要说明。

    返回值：
        DogKnowledgeSmokeCheckResult:
            不可变的 smoke check 结果对象。
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    observed_layers: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """
        将 smoke check 结果转换为 dict。

        功能：
            方便写入 Debug Report、日志或测试输出。

        参数：
            无。

        返回值：
            dict[str, Any]:
                当前检查结果的字典表示。
        """

        return {
            "passed": self.passed,
            "errors": list(
                self.errors,
            ),
            "warnings": list(
                self.warnings,
            ),
            "observed_layers": list(
                self.observed_layers,
            ),
            "summary": self.summary,
        }


def extract_pipeline_layers_from_state(
        state: dict[str, Any],
) -> tuple[str, ...]:
    """
    从 state 中提取 DogKnowledgeAgent pipeline layer 顺序。

    功能：
        读取 state["dog_knowledge_pipeline_steps"]，
        并提取其中的 layer 字段。

    参数：
        state:
            当前 LangGraph 最终状态或节点返回状态。

    返回值：
        tuple[str, ...]:
            提取到的 pipeline layer 顺序。
            如果没有相关字段，则返回空 tuple。
    """

    steps = state.get(
        "dog_knowledge_pipeline_steps",
        [],
    )

    if not isinstance(
            steps,
            list,
    ):
        return ()

    layers: list[str] = []

    for step in steps:
        if not isinstance(
                step,
                dict,
        ):
            continue

        layer = step.get(
            "layer",
        )

        if isinstance(
                layer,
                str,
        ):
            layers.append(
                layer,
            )

    return tuple(
        layers,
    )


def validate_dog_knowledge_pipeline_metadata(
        state: dict[str, Any],
) -> DogKnowledgeSmokeCheckResult:
    """
    验证 DogKnowledgeAgent pipeline metadata。

    功能：
        检查真实主图执行结果中是否包含完整的 dog_knowledge_pipeline_* 字段。

    参数：
        state:
            当前 LangGraph 最终状态或 dog_knowledge_agent 节点返回状态。

    返回值：
        DogKnowledgeSmokeCheckResult:
            smoke check 检查结果。
    """

    errors: list[str] = []
    warnings: list[str] = []

    pipeline_status = state.get(
        "dog_knowledge_pipeline_status",
    )

    pipeline_version = state.get(
        "dog_knowledge_pipeline_version",
    )

    pipeline_question = state.get(
        "dog_knowledge_pipeline_question",
    )

    pipeline_steps = state.get(
        "dog_knowledge_pipeline_steps",
    )

    pipeline_trace = state.get(
        "dog_knowledge_pipeline_trace",
    )

    if pipeline_status != "skeleton_ready":
        errors.append(
            "缺少或错误的 dog_knowledge_pipeline_status，期望为 skeleton_ready。"
        )

    if not pipeline_version:
        errors.append(
            "缺少 dog_knowledge_pipeline_version。"
        )

    if not pipeline_question:
        warnings.append(
            "dog_knowledge_pipeline_question 为空，可能 state 中没有 question。"
        )

    if not isinstance(
            pipeline_steps,
            list,
    ) or not pipeline_steps:
        errors.append(
            "缺少 dog_knowledge_pipeline_steps，或该字段不是非空 list。"
        )

    if not isinstance(
            pipeline_trace,
            list,
    ) or not pipeline_trace:
        errors.append(
            "缺少 dog_knowledge_pipeline_trace，或该字段不是非空 list。"
        )

    observed_layers = extract_pipeline_layers_from_state(
        state=state,
    )

    if observed_layers != EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS:
        errors.append(
            "dog_knowledge_pipeline_steps 顺序不符合 V1.7.2 标准。"
        )

    validate_pipeline_order_errors = validate_dog_knowledge_pipeline_order(
        layers=observed_layers,
    )

    errors.extend(
        validate_pipeline_order_errors,
    )

    passed = not errors

    summary = build_smoke_check_summary(
        passed=passed,
        observed_layers=observed_layers,
    )

    return DogKnowledgeSmokeCheckResult(
        passed=passed,
        errors=tuple(
            errors,
        ),
        warnings=tuple(
            warnings,
        ),
        observed_layers=observed_layers,
        summary=summary,
    )


def validate_dog_knowledge_pipeline_order(
        layers: tuple[str, ...],
) -> list[str]:
    """
    验证 DogKnowledgeAgent pipeline 层级顺序。

    功能：
        检查关键架构顺序是否正确：
        1. rerank 在 quality 之前。
        2. quality 在 context_builder 之前。
        3. context_builder 在 memory_context 之前。

    参数：
        layers:
            pipeline layer 顺序。

    返回值：
        list[str]:
            错误信息列表。
            如果为空，表示顺序检查通过。
    """

    errors: list[str] = []

    required_layers = {
        "rerank",
        "quality",
        "context_builder",
        "memory_context",
    }

    missing_layers = [
        layer
        for layer in required_layers
        if layer not in layers
    ]

    if missing_layers:
        errors.append(
            f"pipeline 缺少关键层: {missing_layers}"
        )
        return errors

    rerank_index = layers.index(
        "rerank",
    )

    quality_index = layers.index(
        "quality",
    )

    context_builder_index = layers.index(
        "context_builder",
    )

    memory_context_index = layers.index(
        "memory_context",
    )

    if not rerank_index < quality_index < context_builder_index:
        errors.append(
            "pipeline 顺序错误：quality 必须位于 rerank 之后、context_builder 之前。"
        )

    if not context_builder_index < memory_context_index:
        errors.append(
            "pipeline 顺序错误：memory_context 必须位于 context_builder 之后，"
            "避免用户长期记忆混入 RagContext。"
        )

    return errors


def validate_dog_knowledge_business_fields(
        state: dict[str, Any],
) -> DogKnowledgeSmokeCheckResult:
    """
    验证 DogKnowledgeAgent 真实业务字段是否仍然存在。

    功能：
        检查 adapter 接入后，真实 DogKnowledgeAgent 业务结果是否被保留。

        注意：
            由于不同阶段业务字段可能不完全一致，
            这里不强制要求所有字段都存在，只给出 warnings。

    参数：
        state:
            当前 LangGraph 最终状态或 dog_knowledge_agent 节点返回状态。

    返回值：
        DogKnowledgeSmokeCheckResult:
            smoke check 检查结果。
    """

    warnings: list[str] = []

    possible_business_fields = (
        "final_answer",
        "answer",
        "rag_query",
        "rag_context",
        "retrieval_quality",
        "answer_strategy",
    )

    existing_fields = tuple(
        field
        for field in possible_business_fields
        if field in state
    )

    if not existing_fields:
        warnings.append(
            "没有发现常见业务字段。请确认真实 DogKnowledgeAgent delegate 是否被调用。"
        )

    summary = (
        "DogKnowledgeAgent 业务字段检查完成，发现字段: "
        f"{list(existing_fields)}"
    )

    return DogKnowledgeSmokeCheckResult(
        passed=True,
        errors=(),
        warnings=tuple(
            warnings,
        ),
        observed_layers=extract_pipeline_layers_from_state(
            state=state,
        ),
        summary=summary,
    )


def validate_dog_knowledge_smoke_state(
        state: dict[str, Any],
) -> DogKnowledgeSmokeCheckResult:
    """
    验证 DogKnowledgeAgent smoke state。

    功能：
        综合检查：
        1. pipeline metadata 是否存在。
        2. pipeline 顺序是否正确。
        3. 是否保留业务字段。

    参数：
        state:
            当前 LangGraph 最终状态或 dog_knowledge_agent 节点返回状态。

    返回值：
        DogKnowledgeSmokeCheckResult:
            综合 smoke check 结果。
    """

    metadata_result = validate_dog_knowledge_pipeline_metadata(
        state=state,
    )

    business_result = validate_dog_knowledge_business_fields(
        state=state,
    )

    errors = [
        *metadata_result.errors,
        *business_result.errors,
    ]

    warnings = [
        *metadata_result.warnings,
        *business_result.warnings,
    ]

    passed = not errors

    summary = build_smoke_check_summary(
        passed=passed,
        observed_layers=metadata_result.observed_layers,
    )

    return DogKnowledgeSmokeCheckResult(
        passed=passed,
        errors=tuple(
            errors,
        ),
        warnings=tuple(
            warnings,
        ),
        observed_layers=metadata_result.observed_layers,
        summary=summary,
    )


def build_smoke_check_summary(
        passed: bool,
        observed_layers: tuple[str, ...],
) -> str:
    """
    构建 Smoke Check 中文摘要。

    功能：
        根据检查结果生成简短中文说明。

    参数：
        passed:
            是否通过检查。

        observed_layers:
            实际观察到的 pipeline layer 顺序。

    返回值：
        str:
            中文摘要。
    """

    if passed:
        return (
            "DogKnowledgeAgent Smoke Check 通过："
            "pipeline metadata 存在，且职责顺序符合 V1.7.2 标准。"
        )

    return (
        "DogKnowledgeAgent Smoke Check 未通过："
        f"当前观察到的 pipeline layers 为 {list(observed_layers)}。"
    )


def render_dog_knowledge_smoke_report_markdown(
        result: DogKnowledgeSmokeCheckResult,
) -> str:
    """
    渲染 DogKnowledgeAgent Smoke Check Markdown 报告。

    功能：
        将 smoke check 结果渲染为 Markdown 文本，
        方便在控制台、文档或 Debug Report 中查看。

    参数：
        result:
            smoke check 检查结果。

    返回值：
        str:
            Markdown 格式报告。
    """

    lines = [
        "# DogKnowledgeAgent Smoke Check Report",
        "",
        f"- passed: `{result.passed}`",
        f"- summary: {result.summary}",
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

    return "\n".join(
        lines,
    )