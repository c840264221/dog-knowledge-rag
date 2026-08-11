"""DogKnowledgeAgent 规则与 LLM 双层查询理解节点。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, get_args

from src.agents.dog_knowledge_agent.contracts.query_understanding import (
    DogQueryLLMAnalysis,
    DogQueryUnderstandingResult,
)
from src.memory.memory_schema import PetProfileAttribute
from src.logger import logger
from src.rag.query_builders.rag_query_builder import (
    SUPPORTED_METADATA_FIELDS,
    build_rag_query_from_state,
    merge_metadata_filters,
    normalize_metadata_filter,
)


DogQueryUnderstandingNode = Callable[
    [Mapping[str, Any]],
    Awaitable[dict[str, Any]],
]


PROFILE_FIELDS_BY_KEYWORD: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("训练", "等待", "召回", "随行", "服从"),
        (
            "breed",
            "age_years",
            "health_condition",
            "activity_level",
            "training_goal",
        ),
    ),
    (
        ("饮食", "喂养", "狗粮", "减肥", "营养"),
        (
            "breed",
            "age_years",
            "weight_kg",
            "health_condition",
            "allergy",
            "diet_pattern",
        ),
    ),
    (
        ("健康", "疾病", "过敏", "生病", "症状"),
        (
            "breed",
            "age_years",
            "weight_kg",
            "sex",
            "neutered",
            "health_condition",
            "allergy",
        ),
    ),
)


def build_dog_query_understanding_node(
    *,
    llm_provider: Any,
    query_filter_parser: Any,
) -> DogQueryUnderstandingNode:
    """
    构建规则优先、LLM 补充的狗狗查询理解节点。

    参数含义：
        llm_provider：提供 main_llm 和 safe_ainvoke 的 LLM Provider（模型提供者）。
        query_filter_parser：现有 DogQueryFilterParser（狗狗查询过滤解析器）。

    返回值含义：
        DogQueryUnderstandingNode：可注册到 LangGraph 的异步查询理解节点。
    """

    async def dog_query_understanding_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行规则解析、LLM 补充解析和优先级合并。

        参数含义：
            state：当前 DogState，包含用户问题、用户标识和已有过滤条件。

        返回值含义：
            dict[str, Any]：最终 RagQuery、过滤条件、建议档案字段和审计结果。
        """

        rule_rag_query = build_rag_query_from_state(
            state=state,
            parser=query_filter_parser,
        )
        question = rule_rag_query.question
        rule_suggested_attributes = _suggest_profile_attributes_by_rule(
            question
        )
        llm_analysis = await _analyze_query_with_llm(
            llm_provider=llm_provider,
            question=question,
        )

        # LLM 条件先放入，确定性规则后放入；同字段冲突时规则结果覆盖 LLM。
        llm_filters = normalize_metadata_filter(llm_analysis.rag_filters) or {}
        merged_filters = merge_metadata_filters(
            llm_filters,
            rule_rag_query.filters,
        )

        valid_profile_attributes = frozenset(get_args(PetProfileAttribute))
        valid_llm_attributes = [
            attribute
            for attribute in _unique_strings(
                llm_analysis.pet_profile_suggested_attributes
            )
            if attribute in valid_profile_attributes
        ]
        invalid_llm_attributes = [
            attribute
            for attribute in _unique_strings(
                llm_analysis.pet_profile_suggested_attributes
            )
            if attribute not in valid_profile_attributes
        ]
        final_suggested_attributes = _unique_strings(
            [*rule_suggested_attributes, *valid_llm_attributes]
        )

        merged_rag_query = rule_rag_query.model_copy(
            update={"filters": merged_filters}
        )
        result = DogQueryUnderstandingResult(
            question=question,
            rule_rag_query=rule_rag_query.model_dump(mode="python"),
            llm_analysis=llm_analysis,
            merged_rag_query=merged_rag_query.model_dump(mode="python"),
            rule_suggested_attributes=rule_suggested_attributes,
            final_suggested_attributes=final_suggested_attributes,
            invalid_llm_profile_attributes=invalid_llm_attributes,
        )
        return {
            "rag_query": merged_rag_query.model_dump(mode="python"),
            "filters": merged_filters,
            "intent": merged_rag_query.intent,
            "top_k": merged_rag_query.top_k,
            "pet_profile_suggested_attributes": final_suggested_attributes,
            "dog_query_understanding_result": result.model_dump(
                mode="python"
            ),
        }

    return dog_query_understanding_node


async def _analyze_query_with_llm(
    *,
    llm_provider: Any,
    question: str,
) -> DogQueryLLMAnalysis:
    """
    调用 LLM 补充解析 RAG 过滤条件和回答所需档案字段。

    参数含义：
        llm_provider：项目统一 LLM Provider（模型提供者）。
        question：不包含 Skill 说明的用户业务问题。

    返回值含义：
        DogQueryLLMAnalysis：合法时返回模型结果，异常时返回空建议作为降级。
    """

    allowed_rag_fields = sorted(SUPPORTED_METADATA_FIELDS)
    allowed_profile_fields = sorted(get_args(PetProfileAttribute))
    prompt = (
        "你是狗狗知识查询理解器。只输出 JSON，不要输出 Markdown。\n"
        "输出格式：{\"rag_filters\":{},"
        "\"pet_profile_suggested_attributes\":[],\"reason\":\"\"}\n"
        f"RAG 可用字段：{allowed_rag_fields}\n"
        f"宠物档案可用字段：{allowed_profile_fields}\n"
        "只建议回答当前问题真正需要的字段，不要为了完整而全选。\n"
        f"用户问题：{question}"
    )
    fallback = json.dumps(
        {
            "rag_filters": {},
            "pet_profile_suggested_attributes": [],
            "reason": "LLM 查询理解不可用，已使用确定性规则降级。",
        },
        ensure_ascii=False,
    )
    try:
        raw_output = await llm_provider.safe_ainvoke(
            llm=llm_provider.main_llm,
            prompt=prompt,
            fallback_response=fallback,
        )
        output_text = _extract_output_text(raw_output)
        return DogQueryLLMAnalysis.model_validate_json(
            _strip_json_fence(output_text)
        )
    except Exception as error:
        logger.warning(
            "DogKnowledgeAgent LLM 查询理解失败，已回退到确定性规则: %s",
            error,
        )
        return DogQueryLLMAnalysis.model_validate_json(fallback)


def _suggest_profile_attributes_by_rule(question: str) -> list[str]:
    """
    使用稳定关键词规则建议回答所需的宠物档案字段。

    参数含义：
        question：用户业务问题。

    返回值含义：
        list[str]：按原顺序去重后的确定性字段建议。
    """

    suggestions: list[str] = []
    normalized_question = str(question or "").strip().lower()
    for keywords, attributes in PROFILE_FIELDS_BY_KEYWORD:
        if any(keyword in normalized_question for keyword in keywords):
            suggestions.extend(attributes)
    return _unique_strings(suggestions)


def _extract_output_text(raw_output: Any) -> str:
    """
    从 LLM 返回对象中提取文本。

    参数含义：
        raw_output：字符串或带 content 属性的消息对象。

    返回值含义：
        str：可交给 JSON 校验器处理的文本。
    """

    if isinstance(raw_output, str):
        return raw_output
    return str(getattr(raw_output, "content", raw_output) or "")


def _strip_json_fence(value: str) -> str:
    """
    移除模型偶尔附加的 Markdown JSON 代码围栏。

    参数含义：
        value：模型输出文本。

    返回值含义：
        str：去除代码围栏后的 JSON 文本。
    """

    normalized = str(value or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    return normalized.strip()


def _unique_strings(values: list[str]) -> list[str]:
    """
    按原顺序清理并去重字符串。

    参数含义：
        values：可能包含空白和重复项的字符串列表。

    返回值含义：
        list[str]：清理后的唯一字符串列表。
    """

    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result
