"""DogKnowledgeAgent 查询理解阶段的数据契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DogQueryLLMAnalysis(BaseModel):
    """
    保存 LLM 对狗狗知识问题的补充解析结果。

    参数含义：
        rag_filters：LLM 建议的 RAG 元数据过滤条件。
        pet_profile_suggested_attributes：回答可能需要的宠物档案字段。
        reason：LLM 给出这些建议的简短原因。

    返回值含义：
        DogQueryLLMAnalysis：经过 Pydantic 校验的模型解析结果。
    """

    model_config = ConfigDict(extra="forbid")

    rag_filters: dict[str, Any] = Field(default_factory=dict)
    pet_profile_suggested_attributes: list[str] = Field(
        default_factory=list
    )
    reason: str = ""


class DogQueryUnderstandingResult(BaseModel):
    """
    保存规则解析与 LLM 解析合并后的查询理解结果。

    参数含义：
        question：本次用于理解和检索的干净业务问题。
        rule_rag_query：确定性规则生成的 RagQuery（RAG 查询契约）。
        llm_analysis：LLM 补充解析结果。
        merged_rag_query：规则优先合并后的最终 RagQuery。
        rule_suggested_attributes：规则建议的宠物档案字段。
        final_suggested_attributes：规则与合法 LLM 建议合并后的字段。
        invalid_llm_profile_attributes：被契约拒绝的 LLM 字段。

    返回值含义：
        DogQueryUnderstandingResult：可写入 DogState 的查询理解审计结果。
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    rule_rag_query: dict[str, Any] = Field(default_factory=dict)
    llm_analysis: DogQueryLLMAnalysis
    merged_rag_query: dict[str, Any] = Field(default_factory=dict)
    rule_suggested_attributes: list[str] = Field(default_factory=list)
    final_suggested_attributes: list[str] = Field(default_factory=list)
    invalid_llm_profile_attributes: list[str] = Field(default_factory=list)
