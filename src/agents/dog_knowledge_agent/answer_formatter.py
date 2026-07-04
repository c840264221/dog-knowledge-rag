from collections.abc import Mapping, Sequence
from typing import Any

from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswer,
    DogKnowledgeAnswerStatus,
    DogKnowledgeEvidence,
    DogKnowledgeQueryType,
    DogKnowledgeRecommendationItem,
    DogKnowledgeSourceKind,
)


class DogKnowledgeAnswerFormatter:
    """
    DogKnowledgeAgent 答案格式化器。

    功能：
        将 DogKnowledgeAgent 内部 pipeline 的结果统一转换成 DogKnowledgeAnswer。
        内部结果可以是 dict、Pydantic Model、LangChain Document、RAG 检索结果、
        推荐结果或 fallback 结果。

    设计目标：
        1. 让 DogKnowledgeAgent 最终输出结构稳定。
        2. 降低旧 pipeline 和新 Response Contract 的耦合。
        3. 支持逐步迁移，不要求所有节点一次性改成新 Schema。
        4. 让后续 Root Agent、WebUI、API、debug report 都能消费统一结构。

    专业名词：
        Formatter：
            格式化器，负责把内部数据整理成外部稳定格式。

        Response Contract：
            响应协议，规定 Agent 最终必须返回什么字段和结构。

        Pipeline Result：
            管线结果，表示 Agent 内部多个节点执行后的最终中间结果。
    """

    QUERY_TYPE_ALIASES: dict[str, DogKnowledgeQueryType] = {
        "exact": "exact_lookup",
        "exact_info": "exact_lookup",
        "exact_search": "exact_lookup",
        "exact_lookup": "exact_lookup",
        "ask_info": "exact_lookup",
        "dog_info": "exact_lookup",
        "lookup": "exact_lookup",
        "search": "exact_lookup",
        "rag": "exact_lookup",
        "recommend": "recommendation",
        "recommendation": "recommendation",
        "recommendations": "recommendation",
        "compare": "comparison",
        "comparison": "comparison",
        "general": "general_qa",
        "general_dog_qa": "general_qa",
        "general_qa": "general_qa",
        "qa": "general_qa",
        "care_advice": "general_qa",
        "fallback": "fallback",
    }

    STATUS_ALIASES: dict[str, DogKnowledgeAnswerStatus] = {
        "success": "success",
        "ok": "success",
        "partial": "partial",
        "fallback": "fallback",
        "empty": "empty",
        "no_result": "empty",
        "error": "error",
        "failed": "error",
    }

    SOURCE_KIND_ALIASES: dict[str, DogKnowledgeSourceKind] = {
        "rag": "rag_chunk",
        "chunk": "rag_chunk",
        "rag_chunk": "rag_chunk",
        "retrieved_chunk": "rag_chunk",
        "document": "rag_chunk",
        "metadata": "metadata",
        "llm": "llm",
        "rule": "rule",
        "fallback": "fallback",
        "user_input": "user_input",
    }

    def format(
        self,
        pipeline_result: Mapping[str, Any] | Any,
        question: str | None = None,
    ) -> DogKnowledgeAnswer:
        """
        将 DogKnowledgeAgent 内部结果格式化为 DogKnowledgeAnswer。

        参数：
            pipeline_result:
                DogKnowledgeAgent 内部 pipeline 的最终结果。
                可以是 dict、Pydantic Model、对象实例等。

            question:
                用户原始问题。
                如果外部显式传入，则优先使用该参数；
                如果不传，则尝试从 pipeline_result 中读取 question、query、input 等字段。

        返回值：
            DogKnowledgeAnswer:
                标准化后的 DogKnowledgeAgent 统一答案对象。
        """

        data = self._to_dict(pipeline_result)

        final_question = (
            self._clean_text(question)
            or self._get_first_str(
                data,
                [
                    "question",
                    "user_question",
                    "query",
                    "input",
                    "user_input",
                ],
            )
            or ""
        )

        recommendations = self._build_recommendations(data)
        evidences = self._build_evidences(data)

        query_type = self._resolve_query_type(
            data=data,
            recommendations=recommendations,
        )

        is_fallback = self._resolve_is_fallback(
            data=data,
            query_type=query_type,
        )

        if is_fallback:
            query_type = "fallback"

        if query_type == "recommendation" and not recommendations and not is_fallback:
            recommendations = self._build_recommendations_from_evidences(
                evidences=evidences,
            )

        answer_text = self._get_first_str(
            data,
            [
                "answer",
                "final_answer",
                "response",
                "result",
                "text",
                "message",
                "content",
            ],
        )

        if not answer_text:
            answer_text = self._build_default_answer_text(
                query_type=query_type,
                recommendations=recommendations,
                evidences=evidences,
                is_fallback=is_fallback,
            )

        fallback_reason = None
        if is_fallback:
            fallback_reason = (
                self._get_first_str(
                    data,
                    [
                        "fallback_reason",
                        "fallback_message",
                        "reason",
                        "error",
                        "error_message",
                    ],
                )
                or "DogKnowledgeAgent 当前无法基于已有犬种知识库可靠回答该问题。"
            )

        status = self._resolve_status(
            data=data,
            answer_text=answer_text,
            recommendations=recommendations,
            evidences=evidences,
            is_fallback=is_fallback,
        )

        confidence = self._resolve_confidence(
            data=data,
            status=status,
            recommendations=recommendations,
            evidences=evidences,
            is_fallback=is_fallback,
        )

        reason = self._get_first_str(
            data,
            [
                "reason",
                "decision_reason",
                "routing_reason",
                "summary_reason",
            ],
        )

        if not reason and fallback_reason:
            reason = fallback_reason

        if not reason:
            reason = self._build_default_reason(
                query_type=query_type,
                status=status,
                recommendations=recommendations,
                evidences=evidences,
            )

        return DogKnowledgeAnswer(
            question=final_question,
            query_type=query_type,
            status=status,
            answer=answer_text,
            recommended_breeds=recommendations,
            evidences=evidences,
            confidence=confidence,
            reason=reason,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            debug=self._build_debug_info(data),
            metadata=self._as_dict(data.get("metadata")),
        )

    def _resolve_query_type(
        self,
        data: dict[str, Any],
        recommendations: list[DogKnowledgeRecommendationItem],
    ) -> DogKnowledgeQueryType:
        """
        解析问题类型。

        参数：
            data:
                pipeline_result 转换后的字典。

            recommendations:
                已解析出的推荐犬种列表。

        返回值：
            DogKnowledgeQueryType:
                标准问题类型。
        """

        raw_query_type = self._get_first_str(
            data,
            [
                "query_type",
                "intent_type",
                "intent",
                "route",
                "mode",
                "task_type",
            ],
        )

        for candidate in [
            raw_query_type,
            self._get_nested_first_str(
                data=data,
                key="answer_strategy",
                nested_keys=[
                    "task_type",
                    "answer_style",
                ],
            ),
            self._get_nested_first_str(
                data=data,
                key="rag_query",
                nested_keys=[
                    "query_type",
                    "intent",
                    "task_type",
                    "mode",
                ],
            ),
        ]:
            normalized = self._normalize_query_type(candidate)

            if normalized:
                return normalized

        if recommendations:
            return "recommendation"

        return "general_qa"

    def _resolve_status(
        self,
        data: dict[str, Any],
        answer_text: str,
        recommendations: list[DogKnowledgeRecommendationItem],
        evidences: list[DogKnowledgeEvidence],
        is_fallback: bool,
    ) -> DogKnowledgeAnswerStatus:
        """
        解析答案状态。

        参数：
            data:
                pipeline_result 转换后的字典。

            answer_text:
                最终答案文本。

            recommendations:
                推荐犬种列表。

            evidences:
                证据列表。

            is_fallback:
                是否走 fallback 兜底。

        返回值：
            DogKnowledgeAnswerStatus:
                标准答案状态。
        """

        raw_status = self._get_first_str(
            data,
            [
                "status",
                "answer_status",
                "state",
            ],
        )

        normalized = self._normalize_status(raw_status)

        if normalized:
            return normalized

        if is_fallback:
            return "fallback"

        if not answer_text and not recommendations and not evidences:
            return "empty"

        if answer_text:
            return "success"

        return "partial"

    def _resolve_confidence(
        self,
        data: dict[str, Any],
        status: DogKnowledgeAnswerStatus,
        recommendations: list[DogKnowledgeRecommendationItem],
        evidences: list[DogKnowledgeEvidence],
        is_fallback: bool,
    ) -> float:
        """
        解析或估算答案置信度。

        参数：
            data:
                pipeline_result 转换后的字典。

            status:
                答案状态。

            recommendations:
                推荐犬种列表。

            evidences:
                证据列表。

            is_fallback:
                是否走 fallback 兜底。

        返回值：
            float:
                0 到 1 之间的置信度。
        """

        raw_score = self._get_first_value(
            data,
            [
                "confidence",
                "answer_confidence",
                "score",
                "final_score",
            ],
        )

        normalized_score = self._normalize_score(raw_score)

        if normalized_score is not None:
            return normalized_score

        return self._estimate_confidence(
            status=status,
            recommendations=recommendations,
            evidences=evidences,
            is_fallback=is_fallback,
        )

    def _resolve_is_fallback(
        self,
        data: dict[str, Any],
        query_type: DogKnowledgeQueryType,
    ) -> bool:
        """
        判断是否走 fallback 兜底。

        参数：
            data:
                pipeline_result 转换后的字典。

            query_type:
                已解析出的问题类型。

        返回值：
            bool:
                True 表示 fallback；
                False 表示不是 fallback。
        """

        explicit_fallback = data.get("is_fallback")

        if isinstance(explicit_fallback, bool):
            return explicit_fallback

        if query_type == "fallback":
            return True

        if self._get_first_str(data, ["fallback_reason", "fallback_message"]):
            return True

        return False

    def _build_recommendations(
        self,
        data: dict[str, Any],
    ) -> list[DogKnowledgeRecommendationItem]:
        """
        从 pipeline_result 中构建推荐犬种列表。

        参数：
            data:
                pipeline_result 转换后的字典。

        返回值：
            list[DogKnowledgeRecommendationItem]:
                标准推荐犬种列表。
        """

        raw_items = self._get_first_list(
            data,
            [
                "recommended_breeds",
                "recommendations",
                "breed_recommendations",
                "recommendation_items",
            ],
        )

        recommendations: list[DogKnowledgeRecommendationItem] = []

        for index, raw_item in enumerate(raw_items):
            item_data = self._to_dict(raw_item)

            breed_name = self._get_first_str(
                item_data,
                [
                    "breed_name",
                    "dog_name",
                    "name",
                    "breed",
                    "id",
                ],
            )

            if not breed_name:
                continue

            display_name = self._get_first_str(
                item_data,
                [
                    "display_name",
                    "title",
                    "label",
                    "name_cn",
                ],
            )

            reason = (
                self._get_first_str(
                    item_data,
                    [
                        "reason",
                        "recommend_reason",
                        "description",
                        "summary",
                    ],
                )
                or "该犬种与用户需求存在一定匹配。"
            )

            matched_traits = self._as_str_list(
                self._get_first_value(
                    item_data,
                    [
                        "matched_traits",
                        "matched_tags",
                        "traits",
                        "tags",
                    ],
                )
            )

            warnings = self._as_str_list(
                self._get_first_value(
                    item_data,
                    [
                        "warnings",
                        "notes",
                        "risks",
                    ],
                )
            )

            evidence_ids = self._as_str_list(
                self._get_first_value(
                    item_data,
                    [
                        "evidence_ids",
                        "source_ids",
                        "chunk_ids",
                    ],
                )
            )

            score = self._normalize_score(
                self._get_first_value(
                    item_data,
                    [
                        "score",
                        "match_score",
                        "recommendation_score",
                        "confidence",
                    ],
                )
            )

            recommendations.append(
                DogKnowledgeRecommendationItem(
                    breed_name=breed_name,
                    display_name=display_name,
                    reason=reason,
                    matched_traits=matched_traits,
                    warnings=warnings,
                    evidence_ids=evidence_ids,
                    score=score,
                    metadata=self._as_dict(item_data.get("metadata")),
                )
            )

        return recommendations

    def _build_evidences(
        self,
        data: dict[str, Any],
    ) -> list[DogKnowledgeEvidence]:
        """
        从 pipeline_result 中构建答案证据列表。

        参数：
            data:
                pipeline_result 转换后的字典。

        返回值：
            list[DogKnowledgeEvidence]:
                标准证据列表。
        """

        raw_items = self._get_first_list(
            data,
            [
                "evidences",
                "evidence",
                "retrieved_chunks",
                "retrieved_items",
                "rag_chunks",
                "chunks",
                "source_documents",
                "documents",
            ],
        )

        if not raw_items:
            raw_items = self._get_nested_first_list(
                data=data,
                key="rag_context",
                nested_keys=[
                    "chunks",
                ],
            )

        evidences: list[DogKnowledgeEvidence] = []

        for index, raw_item in enumerate(raw_items):
            item_data = self._to_dict(raw_item)
            chunk_data = self._as_dict(
                item_data.get(
                    "chunk",
                )
            )
            chunk_metadata = self._as_dict(
                chunk_data.get(
                    "metadata",
                )
            )

            if isinstance(raw_item, str):
                content = raw_item
            else:
                content = self._get_first_str(
                    item_data,
                    [
                        "content",
                        "page_content",
                        "text",
                        "body",
                        "summary",
                    ],
                )

                if not content and chunk_data:
                    content = self._get_first_str(
                        chunk_data,
                        [
                            "content",
                            "page_content",
                            "text",
                            "body",
                            "summary",
                        ],
                    )

            if not content:
                continue

            evidence_id = (
                self._get_first_str(
                    item_data,
                    [
                        "evidence_id",
                        "chunk_id",
                        "id",
                        "document_id",
                        "source_id",
                    ],
                )
                or self._get_first_str(
                    chunk_data,
                    [
                        "chunk_id",
                        "id",
                        "document_id",
                        "source_id",
                    ],
                )
                or f"evidence-{index + 1}"
            )

            source_kind = self._normalize_source_kind(
                self._get_first_str(
                    item_data,
                    [
                        "source_kind",
                        "source_type",
                        "type",
                    ],
                )
            )

            title = self._get_first_str(
                item_data,
                [
                    "title",
                    "dog_name",
                    "document_title",
                    "section_title",
                    "file_name",
                ],
            )

            if not title and chunk_data:
                title = (
                    self._get_first_str(
                        chunk_data,
                        [
                            "title",
                            "dog_name",
                            "document_title",
                            "section_title",
                            "file_name",
                        ],
                    )
                    or self._get_first_str(
                        chunk_metadata,
                        [
                            "dog_name",
                            "name",
                            "breed",
                            "title",
                        ],
                    )
                )

            score = self._normalize_score(
                self._get_first_value(
                    item_data,
                    [
                        "score",
                        "relevance_score",
                        "similarity_score",
                        "rerank_score",
                        "final_score",
                        "retrieval_score",
                        "normalized_rerank_score",
                    ],
                )
            )

            metadata = {
                **chunk_metadata,
                **self._as_dict(item_data.get("metadata")),
            }

            evidences.append(
                DogKnowledgeEvidence(
                    evidence_id=evidence_id,
                    source_kind=source_kind,
                    title=title,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )

        return evidences

    def _build_recommendations_from_evidences(
        self,
        evidences: list[DogKnowledgeEvidence],
    ) -> list[DogKnowledgeRecommendationItem]:
        """
        从证据列表中合成基础推荐项。

        功能：
            在真实 pipeline 还没有稳定产出 recommended_breeds 时，
            根据 RAG evidences 中的 dog_name/title/metadata 合成推荐结果，
            用于 V1.7.3 输出契约收敛期的兼容。

        参数：
            evidences:
                已经格式化好的证据列表。

        返回值：
            list[DogKnowledgeRecommendationItem]:
                从证据中推导出的推荐犬种列表。
        """

        recommendations: list[DogKnowledgeRecommendationItem] = []
        seen_breed_names: set[str] = set()

        for evidence in evidences:
            breed_name = (
                self._get_first_str(
                    evidence.metadata,
                    [
                        "dog_name",
                        "breed_name",
                        "name",
                        "breed",
                    ],
                )
                or evidence.title
            )

            if not breed_name:
                continue

            normalized_breed_name = breed_name.strip()

            if not normalized_breed_name:
                continue

            if normalized_breed_name in seen_breed_names:
                continue

            seen_breed_names.add(
                normalized_breed_name,
            )

            recommendations.append(
                DogKnowledgeRecommendationItem(
                    breed_name=normalized_breed_name,
                    display_name=evidence.title or normalized_breed_name,
                    reason=(
                        "该犬种来自当前推荐问题的 RAG 召回证据，"
                        "可作为过渡期结构化推荐候选。"
                    ),
                    matched_traits=[],
                    warnings=[],
                    evidence_ids=[
                        evidence.evidence_id,
                    ],
                    score=evidence.score,
                    metadata=evidence.metadata,
                )
            )

        return recommendations

    def _build_default_answer_text(
        self,
        query_type: DogKnowledgeQueryType,
        recommendations: list[DogKnowledgeRecommendationItem],
        evidences: list[DogKnowledgeEvidence],
        is_fallback: bool,
    ) -> str:
        """
        构建默认答案文本。

        参数：
            query_type:
                问题类型。

            recommendations:
                推荐犬种列表。

            evidences:
                证据列表。

            is_fallback:
                是否 fallback。

        返回值：
            str:
                默认自然语言答案。
        """

        if is_fallback:
            return "我暂时无法基于当前犬种知识库可靠回答这个问题。"

        if query_type == "recommendation" and recommendations:
            return self._build_recommendation_summary(recommendations)

        if evidences:
            return "我已经从犬种知识库中找到了相关资料，但当前还没有生成完整自然语言答案。"

        return "当前没有找到足够可靠的犬种知识来回答这个问题。"

    def _build_recommendation_summary(
        self,
        recommendations: list[DogKnowledgeRecommendationItem],
    ) -> str:
        """
        根据推荐项构建默认推荐答案。

        参数：
            recommendations:
                推荐犬种列表。

        返回值：
            str:
                推荐类自然语言答案。
        """

        lines = ["我为你整理了以下候选犬种："]

        for index, item in enumerate(recommendations, start=1):
            display_name = item.display_name or item.breed_name
            lines.append(f"{index}. {display_name}：{item.reason}")

        return "\n".join(lines)

    def _build_default_reason(
        self,
        query_type: DogKnowledgeQueryType,
        status: DogKnowledgeAnswerStatus,
        recommendations: list[DogKnowledgeRecommendationItem],
        evidences: list[DogKnowledgeEvidence],
    ) -> str:
        """
        构建默认决策原因。

        参数：
            query_type:
                问题类型。

            status:
                答案状态。

            recommendations:
                推荐犬种列表。

            evidences:
                证据列表。

        返回值：
            str:
                默认原因说明。
        """

        if status == "fallback":
            return "系统无法从当前犬种知识库中获得足够可靠的信息，因此走 fallback 兜底。"

        if query_type == "recommendation" and recommendations:
            return "系统根据用户需求生成了犬种推荐结果。"

        if evidences:
            return "系统基于检索到的犬种知识证据生成答案。"

        return "系统完成了 DogKnowledgeAgent 格式化流程。"

    def _build_debug_info(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        构建 formatter 调试信息。

        参数：
            data:
                pipeline_result 转换后的字典。

        返回值：
            dict[str, Any]:
                调试信息字典。
        """

        raw_debug = self._as_dict(data.get("debug"))

        raw_debug["formatter"] = {
            "name": "DogKnowledgeAnswerFormatter",
            "version": "v1.7.3",
        }

        return raw_debug

    def _normalize_query_type(
        self,
        value: str | None,
    ) -> DogKnowledgeQueryType | None:
        """
        标准化问题类型。

        参数：
            value:
                原始问题类型字符串。

        返回值：
            DogKnowledgeQueryType | None:
                标准问题类型；无法识别时返回 None。
        """

        if not value:
            return None

        normalized = value.strip().lower()

        return self.QUERY_TYPE_ALIASES.get(normalized)

    def _normalize_status(
        self,
        value: str | None,
    ) -> DogKnowledgeAnswerStatus | None:
        """
        标准化答案状态。

        参数：
            value:
                原始状态字符串。

        返回值：
            DogKnowledgeAnswerStatus | None:
                标准答案状态；无法识别时返回 None。
        """

        if not value:
            return None

        normalized = value.strip().lower()

        return self.STATUS_ALIASES.get(normalized)

    def _normalize_source_kind(
        self,
        value: str | None,
    ) -> DogKnowledgeSourceKind:
        """
        标准化证据来源类型。

        参数：
            value:
                原始证据来源类型。

        返回值：
            DogKnowledgeSourceKind:
                标准证据来源类型。
        """

        if not value:
            return "rag_chunk"

        normalized = value.strip().lower()

        return self.SOURCE_KIND_ALIASES.get(normalized, "rag_chunk")

    def _normalize_score(
        self,
        value: Any,
    ) -> float | None:
        """
        标准化分数。

        功能：
            将不同来源的 score、confidence、match_score 统一转换到 0 到 1。
            如果输入是 88，会转换成 0.88。
            如果输入超过 1 且不超过 100，按百分制处理。
            如果输入无法转换，则返回 None。

        参数：
            value:
                原始分数。

        返回值：
            float | None:
                0 到 1 之间的分数；无法转换时返回 None。
        """

        if value is None:
            return None

        try:
            score = float(value)
        except (TypeError, ValueError):
            return None

        if score > 1.0 and score <= 100.0:
            score = score / 100.0

        if score < 0.0:
            return 0.0

        if score > 1.0:
            return 1.0

        return score

    def _estimate_confidence(
        self,
        status: DogKnowledgeAnswerStatus,
        recommendations: list[DogKnowledgeRecommendationItem],
        evidences: list[DogKnowledgeEvidence],
        is_fallback: bool,
    ) -> float:
        """
        在没有显式 confidence 时估算置信度。

        参数：
            status:
                答案状态。

            recommendations:
                推荐犬种列表。

            evidences:
                证据列表。

            is_fallback:
                是否 fallback。

        返回值：
            float:
                估算出的置信度。
        """

        if is_fallback or status == "fallback":
            return 0.1

        if status == "empty":
            return 0.0

        if recommendations and evidences:
            return 0.8

        if recommendations:
            return 0.72

        if evidences:
            return 0.7

        if status == "success":
            return 0.6

        return 0.3

    def _get_first_str(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> str | None:
        """
        从字典中按顺序读取第一个非空字符串。

        参数：
            data:
                数据字典。

            keys:
                候选字段名列表。

        返回值：
            str | None:
                第一个非空字符串；没有找到则返回 None。
        """

        value = self._get_first_value(data, keys)

        return self._clean_text(value)

    def _get_nested_first_str(
        self,
        data: dict[str, Any],
        key: str,
        nested_keys: list[str],
    ) -> str | None:
        """
        从嵌套字典中按顺序读取第一个非空字符串。

        功能：
            兼容真实 LangGraph state 中的嵌套结构，
            例如 answer_strategy.task_type 或 rag_query.intent。

        参数：
            data:
                外层数据字典。

            key:
                外层字段名。

            nested_keys:
                内层候选字段名列表。

        返回值：
            str | None:
                第一个非空字符串；如果没有找到则返回 None。
        """

        nested_data = self._as_dict(
            data.get(
                key,
            )
        )

        if not nested_data:
            return None

        return self._get_first_str(
            data=nested_data,
            keys=nested_keys,
        )

    def _get_nested_first_list(
        self,
        data: dict[str, Any],
        key: str,
        nested_keys: list[str],
    ) -> list[Any]:
        """
        从嵌套字典中按顺序读取第一个列表字段。

        功能：
            兼容真实 LangGraph state 中的嵌套结构，
            例如 rag_context.chunks。

        参数：
            data:
                外层数据字典。

            key:
                外层字段名。

            nested_keys:
                内层候选字段名列表。

        返回值：
            list[Any]:
                读取到的列表；如果没有找到则返回空列表。
        """

        nested_data = self._as_dict(
            data.get(
                key,
            )
        )

        if not nested_data:
            return []

        return self._get_first_list(
            data=nested_data,
            keys=nested_keys,
        )

    def _get_first_value(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> Any:
        """
        从字典中按顺序读取第一个存在的字段值。

        参数：
            data:
                数据字典。

            keys:
                候选字段名列表。

        返回值：
            Any:
                第一个存在的字段值；没有找到则返回 None。
        """

        for key in keys:
            if key in data and data[key] is not None:
                return data[key]

        return None

    def _get_first_list(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> list[Any]:
        """
        从字典中按顺序读取第一个列表字段。

        参数：
            data:
                数据字典。

            keys:
                候选字段名列表。

        返回值：
            list[Any]:
                读取到的列表；没有找到则返回空列表。
        """

        value = self._get_first_value(data, keys)

        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, Sequence):
            return list(value)

        return [value]

    def _clean_text(
        self,
        value: Any,
    ) -> str | None:
        """
        清洗文本字段。

        参数：
            value:
                原始值。

        返回值：
            str | None:
                清洗后的字符串；无法得到有效文本时返回 None。
        """

        if value is None:
            return None

        if not isinstance(value, str):
            return None

        cleaned = value.strip()

        if not cleaned:
            return None

        return cleaned

    def _as_str_list(
        self,
        value: Any,
    ) -> list[str]:
        """
        将任意值转换成字符串列表。

        参数：
            value:
                原始值，可以是字符串、列表、元组、集合等。

        返回值：
            list[str]:
                字符串列表。
        """

        if value is None:
            return []

        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []

        if isinstance(value, Sequence):
            result: list[str] = []

            for item in value:
                if item is None:
                    continue

                text = str(item).strip()

                if text:
                    result.append(text)

            return result

        text = str(value).strip()

        return [text] if text else []

    def _as_dict(
        self,
        value: Any,
    ) -> dict[str, Any]:
        """
        将任意值转换成字典。

        参数：
            value:
                原始值。

        返回值：
            dict[str, Any]:
                转换后的字典；无法转换时返回空字典。
        """

        if value is None:
            return {}

        if isinstance(value, Mapping):
            return dict(value)

        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)

        if hasattr(value, "dict"):
            dumped = value.dict()
            if isinstance(dumped, Mapping):
                return dict(dumped)

        if hasattr(value, "__dict__"):
            return dict(value.__dict__)

        return {}

    def _to_dict(
        self,
        value: Any,
    ) -> dict[str, Any]:
        """
        将 pipeline_result 或中间对象转换成字典。

        参数：
            value:
                原始对象，可以是 dict、Pydantic Model、普通对象等。

        返回值：
            dict[str, Any]:
                转换后的字典。
        """

        return self._as_dict(value)


def format_dog_knowledge_answer(
    pipeline_result: Mapping[str, Any] | Any,
    question: str | None = None,
) -> DogKnowledgeAnswer:
    """
    DogKnowledgeAnswerFormatter 的便捷函数。

    功能：
        外部调用时不需要手动实例化 DogKnowledgeAnswerFormatter，
        可以直接传入 pipeline_result 得到 DogKnowledgeAnswer。

    参数：
        pipeline_result:
            DogKnowledgeAgent 内部 pipeline 的最终结果。

        question:
            用户原始问题。
            如果传入，会优先使用该值。

    返回值：
        DogKnowledgeAnswer:
            标准化后的 DogKnowledgeAgent 答案。
    """

    formatter = DogKnowledgeAnswerFormatter()

    return formatter.format(
        pipeline_result=pipeline_result,
        question=question,
    )
