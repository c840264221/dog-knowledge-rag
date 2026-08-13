"""多智能体澄清回答的字段提取与步骤分配。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agents.collaboration.contracts import (
    MultiAgentClarificationExtractionResult,
)
from src.skills.extractors import extract_dog_training_plan_inputs
from src.runtime.observability.llm_call_records import (
    LLMCallPurpose,
    build_llm_call_metadata,
)


ClarificationExtractionRule = Callable[[str], Mapping[str, Any]]
logger = logging.getLogger(__name__)


class _ClarificationLLMResponse(BaseModel):
    """约束 LLM 澄清字段提取层只能返回约定的 JSON 结构。"""

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, Any] = Field(default_factory=dict)
    confidences: dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("confidences")
    @classmethod
    def validate_confidences(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """确保 LLM 返回的每个可信度都处于 0 到 1。"""

        invalid_field_ids = [
            field_id
            for field_id, confidence in value.items()
            if not 0.0 <= confidence <= 1.0
        ]
        if invalid_field_ids:
            raise ValueError(
                "字段可信度必须处于 0 到 1: "
                f"{sorted(invalid_field_ids)}"
            )
        return value


class MultiAgentClarificationFieldResolver:
    """
    从用户自然语言中提取当前澄清允许接收的字段。

    功能：
        依次运行已注册的高精度确定性规则，只保留当前等待任务申请的字段。
        多条规则对同一字段给出不同值时标记为歧义，不负责决定字段属于
        哪个步骤。

    参数含义：
        rules:
            规则名称到字段提取函数的映射。规则返回字段编号到字段值。

    返回值含义：
        MultiAgentClarificationFieldResolver:
            可以执行受字段白名单约束的澄清信息提取器。
    """

    def __init__(
        self,
        rules: Mapping[str, ClarificationExtractionRule],
        *,
        llm_provider: Any | None = None,
        minimum_llm_confidence: float = 0.75,
    ) -> None:
        if not 0.0 <= minimum_llm_confidence <= 1.0:
            raise ValueError("minimum_llm_confidence 必须处于 0 到 1")
        self.rules = dict(rules)
        self.llm_provider = llm_provider
        self.minimum_llm_confidence = minimum_llm_confidence

    def extract(
        self,
        *,
        user_text: str,
        requested_field_ids: list[str],
        existing_fields: Mapping[str, Any] | None = None,
    ) -> MultiAgentClarificationExtractionResult:
        """
        提取一轮自然语言中的澄清字段并合并前几轮结果。

        参数含义：
            user_text:
                用户本轮提供的自然语言回答。
            requested_field_ids:
                当前澄清包声明允许接收的字段编号。
            existing_fields:
                前几轮已经明确识别并保存的字段。

        返回值含义：
            MultiAgentClarificationExtractionResult:
                包含本轮字段、跨轮合并字段、缺失字段和歧义字段。
        """

        requested_ids = list(dict.fromkeys(
            str(field_id).strip()
            for field_id in requested_field_ids
            if str(field_id).strip()
        ))
        requested_set = set(requested_ids)
        resolved_fields = {
            str(field_id): value
            for field_id, value in dict(existing_fields or {}).items()
            if str(field_id) in requested_set
            and not _is_empty_value(value)
        }

        candidates_by_field: dict[str, list[tuple[str, Any]]] = {}
        normalized_text = str(user_text or "").strip()
        for rule_name, rule in self.rules.items():
            raw_fields = rule(normalized_text)
            if not isinstance(raw_fields, Mapping):
                continue
            for raw_field_id, value in raw_fields.items():
                field_id = str(raw_field_id).strip()
                if field_id not in requested_set or _is_empty_value(value):
                    continue
                candidates_by_field.setdefault(field_id, []).append(
                    (str(rule_name), value)
                )

        extracted_fields: dict[str, Any] = {}
        field_sources: dict[str, list[str]] = {}
        ambiguous_field_ids: list[str] = []
        for field_id in requested_ids:
            candidates = candidates_by_field.get(field_id, [])
            unique_values = {
                _normalize_candidate_value(value): value
                for _, value in candidates
            }
            if len(unique_values) > 1:
                ambiguous_field_ids.append(field_id)
                continue
            if not unique_values:
                continue
            extracted_fields[field_id] = next(iter(unique_values.values()))
            field_sources[field_id] = list(dict.fromkeys(
                source for source, _ in candidates
            ))

        resolved_fields.update(extracted_fields)
        missing_field_ids = [
            field_id
            for field_id in requested_ids
            if field_id not in resolved_fields
            and field_id not in ambiguous_field_ids
        ]
        coverage_ratio = (
            len(resolved_fields) / len(requested_ids)
            if requested_ids
            else 0.0
        )
        return MultiAgentClarificationExtractionResult(
            requested_field_ids=requested_ids,
            extracted_fields=extracted_fields,
            resolved_fields=resolved_fields,
            missing_field_ids=missing_field_ids,
            ambiguous_field_ids=ambiguous_field_ids,
            field_sources=field_sources,
            field_confidences={
                field_id: 1.0
                for field_id in extracted_fields
            },
            coverage_ratio=coverage_ratio,
        )

    async def extract_layered(
        self,
        *,
        user_text: str,
        requested_field_ids: list[str],
        existing_fields: Mapping[str, Any] | None = None,
        field_descriptions: Mapping[str, str] | None = None,
    ) -> MultiAgentClarificationExtractionResult:
        """
        先运行确定性规则，再使用 LLM 补充仍然缺失的字段。

        功能：
            规则结果拥有最高优先级。LLM 只接收规则执行后仍缺失的字段，
            返回值还要经过 JSON Schema、字段白名单、空值和可信度校验。
            LLM 不可用或输出非法时保留规则结果并继续请求用户澄清。

        参数含义：
            user_text:
                用户本轮提供的自然语言回答。
            requested_field_ids:
                当前澄清允许接收的全部字段编号。
            existing_fields:
                前几轮已经确认并保存的字段和值。
            field_descriptions:
                字段编号到通俗名称或说明的映射，帮助 LLM 理解字段含义。

        返回值含义：
            MultiAgentClarificationExtractionResult:
                合并规则与合法 LLM 补充结果后的标准字段提取记录。
        """

        rule_result = self.extract(
            user_text=user_text,
            requested_field_ids=requested_field_ids,
            existing_fields=existing_fields,
        )
        missing_ids = list(rule_result.missing_field_ids)
        if not missing_ids or self.llm_provider is None:
            return rule_result

        try:
            llm_result = await self._extract_missing_fields_with_llm(
                user_text=user_text,
                missing_field_ids=missing_ids,
                field_descriptions=field_descriptions or {},
            )
        except Exception as error:
            logger.warning(
                "多智能体澄清 LLM 字段提取失败，保留规则结果: %s",
                error,
            )
            return rule_result.model_copy(
                update={
                    "llm_fallback_used": True,
                    "llm_error_message": str(error),
                }
            )

        missing_set = set(missing_ids)
        accepted_fields: dict[str, Any] = {}
        accepted_confidences: dict[str, float] = {}
        rejected_field_ids: list[str] = []
        low_confidence_field_ids: list[str] = []
        for raw_field_id, value in llm_result.fields.items():
            field_id = str(raw_field_id).strip()
            if field_id not in missing_set:
                rejected_field_ids.append(field_id)
                continue
            if (
                _is_empty_value(value)
                or not _is_supported_llm_field_value(value)
            ):
                rejected_field_ids.append(field_id)
                continue
            confidence = float(
                llm_result.confidences.get(field_id, 0.0)
            )
            if confidence < self.minimum_llm_confidence:
                low_confidence_field_ids.append(field_id)
                continue
            accepted_fields[field_id] = value
            accepted_confidences[field_id] = confidence

        extracted_fields = {
            **rule_result.extracted_fields,
            **accepted_fields,
        }
        resolved_fields = {
            **rule_result.resolved_fields,
            **accepted_fields,
        }
        ambiguous_field_ids = list(dict.fromkeys([
            *rule_result.ambiguous_field_ids,
            *low_confidence_field_ids,
        ]))
        remaining_missing_ids = [
            field_id
            for field_id in rule_result.requested_field_ids
            if field_id not in resolved_fields
            and field_id not in ambiguous_field_ids
        ]
        field_sources = {
            field_id: list(sources)
            for field_id, sources in rule_result.field_sources.items()
        }
        for field_id in accepted_fields:
            field_sources[field_id] = ["llm_fallback"]
        field_confidences = {
            **rule_result.field_confidences,
            **accepted_confidences,
        }
        coverage_ratio = (
            len(resolved_fields) / len(rule_result.requested_field_ids)
            if rule_result.requested_field_ids
            else 0.0
        )
        return MultiAgentClarificationExtractionResult(
            requested_field_ids=rule_result.requested_field_ids,
            extracted_fields=extracted_fields,
            resolved_fields=resolved_fields,
            missing_field_ids=remaining_missing_ids,
            ambiguous_field_ids=ambiguous_field_ids,
            field_sources=field_sources,
            field_confidences=field_confidences,
            coverage_ratio=coverage_ratio,
            llm_fallback_used=True,
            llm_reason=llm_result.reason,
            rejected_field_ids=list(dict.fromkeys(rejected_field_ids)),
        )

    async def _extract_missing_fields_with_llm(
        self,
        *,
        user_text: str,
        missing_field_ids: list[str],
        field_descriptions: Mapping[str, str],
    ) -> _ClarificationLLMResponse:
        """
        调用项目统一 LLM Provider 提取规则尚未识别的字段。

        参数含义：
            user_text:
                用户本轮自然语言回答。
            missing_field_ids:
                规则执行后仍未识别的字段编号。
            field_descriptions:
                字段编号到字段名称或说明的映射。

        返回值含义：
            _ClarificationLLMResponse:
                经过 Pydantic 校验的 LLM 字段候选结果。
        """

        safe_ainvoke = getattr(
            self.llm_provider,
            "safe_ainvoke",
            None,
        )
        llm = getattr(self.llm_provider, "main_llm", None)
        if not callable(safe_ainvoke) or llm is None:
            raise ValueError(
                "LLM Provider 缺少 main_llm 或 safe_ainvoke"
            )

        allowed_fields = {
            field_id: str(
                field_descriptions.get(field_id) or field_id
            )
            for field_id in missing_field_ids
        }
        prompt = (
            "你是用户补充信息字段提取器。只提取用户明确表达的信息，"
            "不得猜测，只输出 JSON，不要输出 Markdown。\n"
            "输出格式：{\"fields\":{},\"confidences\":{},"
            "\"reason\":\"\"}\n"
            "fields 的键只能来自允许字段；confidences 使用 0 到 1。\n"
            f"允许字段：{json.dumps(allowed_fields, ensure_ascii=False)}\n"
            f"用户回答：{json.dumps(str(user_text or ''), ensure_ascii=False)}"
        )
        fallback_response = json.dumps(
            {
                "fields": {},
                "confidences": {},
                "reason": "LLM 字段提取不可用，保留确定性规则结果。",
            },
            ensure_ascii=False,
        )
        raw_output = await safe_ainvoke(
            llm=llm,
            prompt=prompt,
            fallback_response=fallback_response,
            call_metadata=build_llm_call_metadata(
                purpose=LLMCallPurpose.CLARIFICATION_FIELD_EXTRACTION,
                component="multi_agent_clarification_field_resolver",
                agent_name="multi_agent",
            ),
        )
        output_text = _extract_output_text(raw_output)
        return _ClarificationLLMResponse.model_validate_json(
            _strip_json_fence(output_text)
        )


def allocate_fields_to_steps(
    *,
    resolved_fields: Mapping[str, Any],
    field_consumers: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    根据字段使用步骤表把已识别字段分配给对应步骤。

    功能：
        本函数不解析自然语言，也不猜测业务归属，只执行
        field_id -> step_ids 的确定性映射。共享字段会复制给全部使用步骤。

    参数含义：
        resolved_fields:
            字段提取层已经确认的字段和值。
        field_consumers:
            字段编号到所有使用步骤编号的映射。

    返回值含义：
        dict[str, dict[str, Any]]:
            步骤编号到该步骤结构化恢复字段的映射。
    """

    step_inputs: dict[str, dict[str, Any]] = {}
    for raw_field_id, value in resolved_fields.items():
        field_id = str(raw_field_id).strip()
        raw_step_ids = field_consumers.get(field_id)
        if not isinstance(raw_step_ids, list):
            continue
        for raw_step_id in raw_step_ids:
            step_id = str(raw_step_id).strip()
            if not step_id:
                continue
            step_inputs.setdefault(step_id, {})[field_id] = value
    return step_inputs


def build_default_multi_agent_clarification_field_resolver(
    *,
    llm_provider: Any | None = None,
) -> MultiAgentClarificationFieldResolver:
    """
    构建项目当前默认的多智能体澄清字段解析器。

    功能：
        首版复用训练计划 Skill 已有的高精度领域规则。后续接入其他 Skill
        或 LLM 提取器时，可以继续向规则注册表追加来源。

    参数含义：
        无。

    返回值含义：
        MultiAgentClarificationFieldResolver:
            已注册当前狗狗训练领域确定性规则的解析器。
    """

    return MultiAgentClarificationFieldResolver(
        {
            "dog_training_plan_rule": (
                extract_dog_training_plan_inputs
            ),
        },
        llm_provider=llm_provider,
    )


def _normalize_candidate_value(value: Any) -> str:
    """把候选值转换成用于冲突比较的稳定文本。"""

    return str(value).strip().casefold()


def _is_empty_value(value: Any) -> bool:
    """判断候选字段值是否没有可用内容。"""

    return value is None or (
        isinstance(value, str) and not value.strip()
    )


def _is_supported_llm_field_value(value: Any) -> bool:
    """只允许可直接写入结构化输入的字符串、数字或布尔值。"""

    return isinstance(value, (str, int, float, bool))


def _extract_output_text(raw_output: Any) -> str:
    """从字符串或消息对象中读取 LLM 返回文本。"""

    if isinstance(raw_output, str):
        return raw_output
    return str(getattr(raw_output, "content", raw_output) or "")


def _strip_json_fence(text: str) -> str:
    """移除 LLM 可能附加在 JSON 外层的 Markdown 代码围栏。"""

    normalized = str(text or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[len("```json"):]
    elif normalized.startswith("```"):
        normalized = normalized[len("```"):]
    if normalized.endswith("```"):
        normalized = normalized[:-len("```")]
    return normalized.strip()
