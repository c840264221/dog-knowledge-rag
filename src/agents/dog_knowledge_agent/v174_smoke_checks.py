from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_LAYER_CONTRACT_FIELDS = (
    "dog_query_result",
    "dog_retrieval_result",
    "dog_generation_result",
    "dog_knowledge_pipeline_result",
    "dog_knowledge_answer",
    "dog_knowledge_answer_public",
    "final_answer",
)

OPTIONAL_LAYER_CONTRACT_FIELDS = (
    "dog_recommendation_result",
    "dog_fallback_result",
)


@dataclass(frozen=True)
class DogKnowledgeLayerContractSmokeCheckResult:
    """
    V1.7.4 分层契约 smoke check 结果。

    功能：
        保存一次 DogKnowledgeAgent 分层契约冒烟检查的结果，包括是否通过、
        错误列表、警告列表、观测到的字段和中文摘要。

    参数含义：
        passed:
            是否通过检查。
        errors:
            阻断性错误列表。非空表示检查失败。
        warnings:
            非阻断性警告列表。非空不一定失败，但需要关注。
        observed_fields:
            从 state 中观测到的 V1.7.4 相关字段。
        summary:
            中文摘要，方便脚本输出和人工阅读。

    返回值含义：
        DogKnowledgeLayerContractSmokeCheckResult:
            不可变的检查结果对象。
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    observed_fields: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """
        将检查结果转换为字典。

        功能：
            方便 smoke 脚本打印 JSON、写入日志或后续接入 Debug Report。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                当前检查结果的普通字典表示。
        """

        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "observed_fields": list(self.observed_fields),
            "summary": self.summary,
        }


def extract_v174_layer_contract_fields(
    state: dict[str, Any],
) -> tuple[str, ...]:
    """
    从 state 中提取 V1.7.4 分层契约相关字段。

    功能：
        扫描当前 state，返回已经存在的 V1.7.4 标准字段名称。

    参数含义：
        state:
            LangGraph 最终 state 或测试构造的 state。

    返回值含义：
        tuple[str, ...]:
            已经出现在 state 中的 V1.7.4 相关字段。
    """

    observed_fields: list[str] = []

    for field_name in (
        *REQUIRED_LAYER_CONTRACT_FIELDS,
        *OPTIONAL_LAYER_CONTRACT_FIELDS,
    ):
        if field_name in state:
            observed_fields.append(field_name)

    return tuple(observed_fields)


def validate_dog_knowledge_layer_contract_state(
    state: dict[str, Any],
) -> DogKnowledgeLayerContractSmokeCheckResult:
    """
    验证 DogKnowledgeAgent V1.7.4 分层契约 state。

    功能：
        检查真实主图或 fake state 中是否包含 V1.7.4 必需字段，
        并验证关键字段的数据形态是否符合最小契约要求。

    参数含义：
        state:
            LangGraph 最终 state 或测试构造的 state。

    返回值含义：
        DogKnowledgeLayerContractSmokeCheckResult:
            分层契约 smoke check 检查结果。
    """

    errors: list[str] = []
    warnings: list[str] = []
    observed_fields = extract_v174_layer_contract_fields(state)

    for field_name in REQUIRED_LAYER_CONTRACT_FIELDS:
        if field_name not in state:
            errors.append(f"缺少必需字段：{field_name}。")

    _validate_mapping_field(
        state=state,
        field_name="dog_query_result",
        errors=errors,
    )
    _validate_mapping_field(
        state=state,
        field_name="dog_retrieval_result",
        errors=errors,
    )
    _validate_mapping_field(
        state=state,
        field_name="dog_generation_result",
        errors=errors,
    )
    _validate_mapping_field(
        state=state,
        field_name="dog_knowledge_pipeline_result",
        errors=errors,
    )
    _validate_mapping_field(
        state=state,
        field_name="dog_knowledge_answer_public",
        errors=errors,
    )
    _validate_optional_mapping_field(
        state=state,
        field_name="dog_recommendation_result",
        errors=errors,
    )
    _validate_optional_mapping_field(
        state=state,
        field_name="dog_fallback_result",
        errors=errors,
    )
    _validate_final_answer(
        state=state,
        errors=errors,
    )
    _validate_dog_knowledge_answer(
        state=state,
        errors=errors,
    )
    _validate_pipeline_result_shape(
        state=state,
        errors=errors,
        warnings=warnings,
    )

    if "dog_fallback_result" not in state:
        warnings.append(
            "未观测到 dog_fallback_result。正常成功回答链路可以没有兜底层输出。"
        )

    passed = not errors

    return DogKnowledgeLayerContractSmokeCheckResult(
        passed=passed,
        errors=tuple(errors),
        warnings=tuple(warnings),
        observed_fields=observed_fields,
        summary=(
            "V1.7.4 分层契约字段检查通过。"
            if passed
            else "V1.7.4 分层契约字段检查失败。"
        ),
    )


def render_dog_knowledge_layer_contract_smoke_markdown(
    result: DogKnowledgeLayerContractSmokeCheckResult,
) -> str:
    """
    渲染 V1.7.4 分层契约 smoke check Markdown 报告。

    功能：
        将检查结果转换成适合终端阅读的 Markdown 文本。

    参数含义：
        result:
            validate_dog_knowledge_layer_contract_state 返回的检查结果。

    返回值含义：
        str:
            Markdown 格式的检查报告。
    """

    lines = [
        "# V1.7.4 DogKnowledgeAgent Layer Contract Smoke Report",
        "",
        f"- passed: {result.passed}",
        f"- summary: {result.summary}",
        "",
        "## Observed Fields",
    ]

    if result.observed_fields:
        lines.extend(f"- {field_name}" for field_name in result.observed_fields)
    else:
        lines.append("- 无")

    lines.extend(["", "## Errors"])
    if result.errors:
        lines.extend(f"- {error}" for error in result.errors)
    else:
        lines.append("- 无")

    lines.extend(["", "## Warnings"])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- 无")

    return "\n".join(lines)


def _validate_mapping_field(
    state: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    """
    验证必需字段是否为 dict。

    功能：
        对标准分层输出字段做最小结构检查。

    参数含义：
        state:
            当前待检查 state。
        field_name:
            要检查的字段名。
        errors:
            错误列表，发现问题时追加中文错误。

    返回值含义：
        None。
    """

    if field_name in state and not isinstance(state.get(field_name), dict):
        errors.append(f"{field_name} 必须是 dict。")


def _validate_optional_mapping_field(
    state: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    """
    验证可选字段如果存在则必须为 dict。

    功能：
        对推荐层和兜底层这类非必然出现的字段做最小结构检查。

    参数含义：
        state:
            当前待检查 state。
        field_name:
            要检查的字段名。
        errors:
            错误列表，发现问题时追加中文错误。

    返回值含义：
        None。
    """

    if field_name in state and not isinstance(state.get(field_name), dict):
        errors.append(f"{field_name} 如果存在，必须是 dict。")


def _validate_final_answer(
    state: dict[str, Any],
    errors: list[str],
) -> None:
    """
    验证 final_answer 是否为非空字符串。

    功能：
        保证旧调用方仍然能通过 final_answer 读取最终答案。

    参数含义：
        state:
            当前待检查 state。
        errors:
            错误列表，发现问题时追加中文错误。

    返回值含义：
        None。
    """

    final_answer = state.get("final_answer")

    if not isinstance(final_answer, str) or not final_answer.strip():
        errors.append("final_answer 必须是非空字符串。")


def _validate_dog_knowledge_answer(
    state: dict[str, Any],
    errors: list[str],
) -> None:
    """
    验证 dog_knowledge_answer 是否存在可用对象。

    功能：
        dog_knowledge_answer 在真实运行中可能是 Pydantic 对象，
        也可能在测试中用 dict 表示；这里做最小可用性检查。

    参数含义：
        state:
            当前待检查 state。
        errors:
            错误列表，发现问题时追加中文错误。

    返回值含义：
        None。
    """

    answer = state.get("dog_knowledge_answer")

    if answer is None:
        errors.append("dog_knowledge_answer 不能为空。")
        return

    if isinstance(answer, dict):
        return

    if hasattr(answer, "model_dump") or hasattr(answer, "to_public_dict"):
        return

    errors.append(
        "dog_knowledge_answer 必须是 dict，或提供 model_dump / to_public_dict 方法。"
    )


def _validate_pipeline_result_shape(
    state: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    验证 dog_knowledge_pipeline_result 的最小字段形态。

    功能：
        检查汇总后的 pipeline result 是否包含最终响应需要的关键字段。

    参数含义：
        state:
            当前待检查 state。
        errors:
            错误列表，发现问题时追加中文错误。
        warnings:
            警告列表，发现非阻断问题时追加中文警告。

    返回值含义：
        None。
    """

    pipeline_result = state.get("dog_knowledge_pipeline_result")

    if not isinstance(pipeline_result, dict):
        return

    for field_name in (
        "question",
        "query_type",
        "status",
        "answer",
        "confidence",
    ):
        if field_name not in pipeline_result:
            errors.append(
                f"dog_knowledge_pipeline_result 缺少字段：{field_name}。"
            )

    if not pipeline_result.get("answer"):
        errors.append("dog_knowledge_pipeline_result.answer 不能为空。")

    if "metadata" not in pipeline_result:
        warnings.append("dog_knowledge_pipeline_result 缺少 metadata 字段。")
