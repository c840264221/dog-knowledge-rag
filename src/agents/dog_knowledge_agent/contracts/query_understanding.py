"""DogKnowledgeAgent 查询理解阶段的数据契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSubQuery(BaseModel):
    """模型提出的子查询；question为检索文本，collection_ids为目录标识，过滤条件仍需规则合并。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1)
    english_question: str = ""
    collection_ids: list[str] = Field(min_length=1, max_length=4)
    rag_filters: dict[str, Any] = Field(default_factory=dict)


class DogQueryLLMAnalysis(BaseModel):
    """
    保存 LLM 对狗狗知识问题的补充解析结果。

    参数含义：
        rag_filters：LLM 建议的 RAG 元数据过滤条件。
        pet_profile_suggested_attributes：回答可能需要的宠物档案字段。
        reason：LLM 给出这些建议的简短原因。
        retrieval_question：可选的单条检索文本；为空时保留改写前文本。

    返回值含义：
        DogQueryLLMAnalysis：经过 Pydantic 校验的模型解析结果。
    """

    model_config = ConfigDict(extra="forbid")

    rag_filters: dict[str, Any] = Field(default_factory=dict)
    pet_profile_suggested_attributes: list[str] = Field(
        default_factory=list
    )
    reason: str = ""
    retrieval_question: str | None = None
    # 改写后中文检索文本的忠实英文版本；缺失时兼容旧输出，使用中文单路。
    english_retrieval_question: str | None = None
    sub_queries: list[KnowledgeSubQuery] = Field(default_factory=list, max_length=8)


class DogQueryUnderstandingResult(BaseModel):
    """
    保存规则解析与 LLM 解析合并后的查询理解结果。

    参数含义：
        question：改写前的干净业务问题；实际检索文本见 merged_rag_query。
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
