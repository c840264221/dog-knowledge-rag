"""
Metadata Filter Retriever。

Metadata Filter Retriever（元数据过滤召回器）：
负责基于结构化 metadata filter 和 vector search（向量检索）
从 Chroma 中召回 RAG chunks。

当前模块职责：
1. 接收 query_text 或 RagQuery
2. 构建或接收 metadata filter
3. 调用 Chroma similarity_search / similarity_search_with_score
4. 将 LangChain Document 转换成 RagChunk
5. 将 RagChunk 包装成 RagRetrievedChunk
6. 将召回结果组装成 RagContext

当前模块不负责：
1. 加载 Markdown
2. 提取 dog metadata
3. 切块
4. 写入 Chroma
5. 调用 LLM 生成答案
6. 解析用户自然语言中的过滤条件

注意：
    Query Filter Parser（查询过滤解析器）会在后续阶段实现。
    本阶段 Retriever 先支持外部传入 metadata_filter，
    或直接使用 RagQuery.filters。
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from src.rag.schemas import (
    RagChunk,
    RagContext,
    RagQuery,
    RagRetrievedChunk,
)


MetadataFilter = dict[str, Any]


class MetadataFilterRetriever:
    """
    Metadata Filter Retriever（元数据过滤召回器）。

    Retriever（召回器 / 检索器）：
    负责从向量数据库中取回和用户问题相关的 chunk。

    当前 Retriever 的核心能力：
    1. metadata filter（元数据过滤）
    2. vector search（向量检索）
    3. result normalization（结果标准化）
    4. context building（上下文构建）
    """

    RETRIEVER_NAME = "metadata_filter_retriever_v1"

    def __init__(
            self,
            vector_store: Any,
            default_top_k: int = 5,
    ):
        """
        初始化 MetadataFilterRetriever。

        参数：
            vector_store: Any
                向量数据库对象。
                当前预期是 langchain_chroma.Chroma。
                需要支持 similarity_search 或 similarity_search_with_score 方法。

            default_top_k: int
                默认召回数量。
                当外部没有传 top_k 时使用。

        返回值：
            None：
                构造函数无返回值。
        """

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k 必须大于 0"
            )

        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
            self,
            query: RagQuery | str,
            metadata_filter: MetadataFilter | None = None,
            top_k: int | None = None,
    ) -> RagContext:
        """
        执行 RAG 召回，并返回 RagContext。

        功能：
            1. 从 query 中提取 question
            2. 解析 top_k
            3. 解析 metadata_filter
            4. 优先执行严格 metadata filter 检索
            5. 如果严格过滤没有结果，则执行 fallback 放宽检索
            6. 将 LangChain Document 转换成 RagChunk
            7. 将 RagChunk 包装成 RagRetrievedChunk
            8. 构建 RagContext

        参数：
            query: RagQuery | str
                查询对象或查询字符串。
                如果传 str，则直接作为问题文本。
                如果传 RagQuery，则读取 query.question。

            metadata_filter: MetadataFilter | None
                Chroma metadata filter。
                如果外部传入，则优先使用。
                如果没有传入，则使用 RagQuery.filters。

            top_k: int | None
                本次召回数量。
                如果不传，则使用 RagQuery.top_k。
                如果 query 是 str，则使用 default_top_k。

        返回值：
            RagContext：
                RAG 上下文对象，包含 question、chunks、context_text、source_count、status。
        """

        query_text = self._get_query_text(
            query=query,
        )

        resolved_top_k = self._resolve_top_k(
            query=query,
            top_k=top_k,
        )

        resolved_filter = self._resolve_metadata_filter(
            query=query,
            metadata_filter=metadata_filter,
        )

        documents_with_scores, effective_filter, fallback_reason = (
            self._search_documents_with_fallback(
                query_text=query_text,
                metadata_filter=resolved_filter,
                top_k=resolved_top_k,
            )
        )

        retrieved_chunks = self._build_retrieved_chunks(
            documents_with_scores=documents_with_scores,
            query_text=query_text,
            metadata_filter=effective_filter,
            fallback_reason=fallback_reason,
        )

        return self._build_rag_context(
            question=query_text,
            retrieved_chunks=retrieved_chunks,
        )

    async def async_retrieve(
            self,
            query: RagQuery | str,
            metadata_filter: MetadataFilter | None = None,
            top_k: int | None = None,
    ) -> RagContext:
        """
        异步执行 RAG 召回，并返回 RagContext。

        功能：
            1. 从 query 中提取 question
            2. 解析 top_k
            3. 解析 metadata_filter
            4. 异步执行 metadata filter + vector search 检索
            5. 如果严格过滤没有结果，则异步执行 fallback 放宽检索
            6. 将 LangChain Document 转换成 RagChunk
            7. 将 RagChunk 包装成 RagRetrievedChunk
            8. 构建 RagContext

        技术名词：
            async：
                异步，表示该函数可以被 await 调用，不阻塞事件循环。

            event loop：
                事件循环，Python asyncio 用来调度异步任务的核心机制。

            to_thread：
                将同步阻塞函数放到线程池中执行，避免阻塞 event loop。

        参数：
            query:
                RagQuery 或查询字符串。
                如果是 RagQuery，则读取 query.question、query.filters、query.top_k。

            metadata_filter:
                外部传入的 Chroma metadata filter。
                如果不传，则使用 RagQuery.filters。

            top_k:
                本次召回数量。
                如果不传，则使用 RagQuery.top_k 或 default_top_k。

        返回值：
            RagContext：
                RAG 上下文对象，包含 question、chunks、context_text、source_count、status。
        """

        query_text = self._get_query_text(
            query=query,
        )

        resolved_top_k = self._resolve_top_k(
            query=query,
            top_k=top_k,
        )

        resolved_filter = self._resolve_metadata_filter(
            query=query,
            metadata_filter=metadata_filter,
        )

        documents_with_scores, effective_filter, fallback_reason = (
            await self._async_search_documents_with_fallback(
                query_text=query_text,
                metadata_filter=resolved_filter,
                top_k=resolved_top_k,
            )
        )

        retrieved_chunks = self._build_retrieved_chunks(
            documents_with_scores=documents_with_scores,
            query_text=query_text,
            metadata_filter=effective_filter,
            fallback_reason=fallback_reason,
        )

        return self._build_rag_context(
            question=query_text,
            retrieved_chunks=retrieved_chunks,
        )

    def retrieve_chunks(
            self,
            query: RagQuery | str,
            metadata_filter: MetadataFilter | None = None,
            top_k: int | None = None,
    ) -> list[RagRetrievedChunk]:
        """
        只召回 chunks，不构建额外流程。

        功能：
            内部仍然会调用 retrieve 构建 RagContext，
            然后返回 context.chunks。
            适合调试或测试时只关心召回列表的场景。

        参数：
            query: RagQuery | str
                查询对象或查询字符串。

            metadata_filter: MetadataFilter | None
                Chroma metadata filter。

            top_k: int | None
                召回数量。

        返回值：
            list[RagRetrievedChunk]：
                召回到的 chunk 列表。
        """

        context = self.retrieve(
            query=query,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

        return context.chunks

    async def async_retrieve_chunks(
            self,
            query: RagQuery | str,
            metadata_filter: MetadataFilter | None = None,
            top_k: int | None = None,
    ) -> list[RagRetrievedChunk]:
        """
        异步只召回 chunks，不构建额外业务流程。

        功能：
            内部调用 aretrieve 构建 RagContext，
            然后返回 context.chunks。

        参数：
            query:
                RagQuery 或查询字符串。

            metadata_filter:
                Chroma metadata filter。

            top_k:
                召回数量。

        返回值：
            list[RagRetrievedChunk]：
                异步召回到的 chunk 列表。
        """

        context = await self.async_retrieve(
            query=query,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

        return context.chunks

    def build_dog_metadata_filter(
            self,
            dog_name: str | None = None,
            size: str | None = None,
            max_energy: int | None = None,
            max_barking: int | None = None,
            min_trainability: int | None = None,
            good_for_apartment: bool | None = None,
            good_for_beginner: bool | None = None,
    ) -> MetadataFilter | None:
        """
        构建狗狗领域 metadata filter。

        功能：
            将犬种相关结构化条件转换成 Chroma where filter。

        参数：
            dog_name: str | None
                犬种英文名，例如 Golden Retriever。

            size: str | None
                体型，例如 small、medium、large、giant。

            max_energy: int | None
                最大精力等级，例如 3 表示 energy_level <= 3。

            max_barking: int | None
                最大吠叫等级，例如 3 表示 barking_level <= 3。

            min_trainability: int | None
                最小可训练等级，例如 3 表示 trainability_level >= 3。

            good_for_apartment: bool | None
                是否适合公寓。

            good_for_beginner: bool | None
                是否适合新手。

        返回值：
            MetadataFilter | None：
                Chroma metadata filter。
                如果没有任何条件，则返回 None。
        """

        conditions: list[MetadataFilter] = []

        if dog_name:
            conditions.append(
                {
                    "dog_name": {
                        "$eq": dog_name,
                    }
                }
            )

        if size:
            conditions.append(
                {
                    "size": {
                        "$eq": size,
                    }
                }
            )

        if max_energy is not None:
            conditions.append(
                {
                    "energy_level": {
                        "$lte": max_energy,
                    }
                }
            )

        if max_barking is not None:
            conditions.append(
                {
                    "barking_level": {
                        "$lte": max_barking,
                    }
                }
            )

        if min_trainability is not None:
            conditions.append(
                {
                    "trainability_level": {
                        "$gte": min_trainability,
                    }
                }
            )

        if good_for_apartment is not None:
            conditions.append(
                {
                    "good_for_apartment": {
                        "$eq": good_for_apartment,
                    }
                }
            )

        if good_for_beginner is not None:
            conditions.append(
                {
                    "good_for_beginner": {
                        "$eq": good_for_beginner,
                    }
                }
            )

        if not conditions:
            return None

        if len(
                conditions,
        ) == 1:
            return conditions[0]

        return {
            "$and": conditions,
        }

    def _search_documents(
            self,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            top_k: int,
    ) -> list[tuple[Any, float | None]]:
        """
        调用 Chroma 执行向量检索。

        功能：
            优先使用 similarity_search_with_score。
            如果 vector_store 不支持，则退回 similarity_search。

        参数：
            query_text: str
                查询文本。

            metadata_filter: MetadataFilter | None
                Chroma metadata filter。

            top_k: int
                召回数量。

        返回值：
            list[tuple[Any, float | None]]：
                每个元素是：
                1. document：LangChain Document
                2. score：检索分数，可能为 None

        注意：
            Chroma 返回的 score 在很多场景下是 distance（距离），
            数值越小可能越相似。
            本阶段只保存原始 retrieval_score，
            后续 Reranker 会负责重新计算 rerank_score / final_score。
        """

        kwargs: dict[str, Any] = {
            "query": query_text,
            "k": top_k,
        }

        if metadata_filter:
            kwargs[
                "filter"
            ] = metadata_filter

        if hasattr(
                self.vector_store,
                "similarity_search_with_score",
        ):
            raw_results = self.vector_store.similarity_search_with_score(
                **kwargs,
            )

            return [
                (
                    document,
                    score,
                )
                for document, score in raw_results
            ]

        if hasattr(
                self.vector_store,
                "similarity_search",
        ):
            documents = self.vector_store.similarity_search(
                **kwargs,
            )

            return [
                (
                    document,
                    None,
                )
                for document in documents
            ]

        raise AttributeError(
            "vector_store 必须提供 similarity_search 或 "
            "similarity_search_with_score 方法"
        )

    async def _async_search_documents(
            self,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            top_k: int,
    ) -> list[tuple[Any, float | None]]:
        """
        异步调用 Chroma 执行向量检索。

        功能：
            1. 如果 vector_store 提供异步 similarity_search_with_score，则直接 await 调用。
            2. 如果 vector_store 提供异步 similarity_search，则直接 await 调用。
            3. 如果 vector_store 只有同步方法，则使用 asyncio.to_thread 放入线程池执行。
            4. 避免同步向量检索阻塞 LangGraph 的 async event loop。

        参数：
            query_text:
                查询文本。

            metadata_filter:
                Chroma metadata filter。

            top_k:
                召回数量。

        返回值：
            list[tuple[Any, float | None]]：
                每个元素包含：
                1. document：LangChain Document
                2. score：检索分数，可能为 None
        """

        kwargs: dict[str, Any] = {
            "query": query_text,
            "k": top_k,
        }

        if metadata_filter:
            kwargs[
                "filter"
            ] = metadata_filter

        if hasattr(
                self.vector_store,
                "asimilarity_search_with_score",
        ):
            raw_results = await self.vector_store.asimilarity_search_with_score(
                **kwargs,
            )

            return [
                (
                    document,
                    score,
                )
                for document, score in raw_results
            ]

        if hasattr(
                self.vector_store,
                "asimilarity_search",
        ):
            documents = await self.vector_store.asimilarity_search(
                **kwargs,
            )

            return [
                (
                    document,
                    None,
                )
                for document in documents
            ]

        return await asyncio.to_thread(
            self._search_documents,
            query_text,
            metadata_filter,
            top_k,
        )

    def _search_documents_with_fallback(
            self,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            top_k: int,
    ) -> tuple[list[tuple[Any, float | None]], MetadataFilter | None, str]:
        """
        执行带 fallback 的文档检索。

        功能：
            先使用严格 metadata_filter 进行检索。
            如果严格过滤能召回结果，则直接返回。

            如果严格过滤没有召回结果，则按 fallback 策略逐步放宽：
            1. 保留 dog_name / size 等核心条件
            2. 保留 dog_name
            3. 保留 size
            4. 退回纯 vector search（纯向量检索）

        参数：
            query_text: str
                用户问题文本。

            metadata_filter: MetadataFilter | None
                原始 Chroma metadata filter。

            top_k: int
                召回数量。

        返回值：
            tuple[list[tuple[Any, float | None]], MetadataFilter | None, str]：
                第一个元素：
                    检索结果列表。
                第二个元素：
                    实际生效的 metadata filter。
                第三个元素：
                    fallback 说明。
                    如果没有发生 fallback，则为空字符串。
        """

        strict_results = self._search_documents(
            query_text=query_text,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

        if strict_results:
            return (
                strict_results,
                metadata_filter,
                "",
            )

        if not metadata_filter:
            return (
                strict_results,
                metadata_filter,
                "",
            )

        fallback_candidates = self._build_fallback_filter_candidates(
            metadata_filter=metadata_filter,
        )

        for fallback_filter, fallback_reason in fallback_candidates:
            fallback_results = self._search_documents(
                query_text=query_text,
                metadata_filter=fallback_filter,
                top_k=top_k,
            )

            if fallback_results:
                return (
                    fallback_results,
                    fallback_filter,
                    fallback_reason,
                )

        return (
            [],
            metadata_filter,
            "严格 metadata filter 和 fallback 策略都没有召回结果",
        )

    async def _async_search_documents_with_fallback(
            self,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            top_k: int,
    ) -> tuple[list[tuple[Any, float | None]], MetadataFilter | None, str]:
        """
        异步执行带 fallback 的文档检索。

        功能：
            先使用严格 metadata_filter 进行异步检索。
            如果严格过滤能召回结果，则直接返回。

            如果严格过滤没有召回结果，则按 fallback 策略逐步放宽：
            1. 保留 dog_name / size 等核心条件
            2. 保留 dog_name
            3. 保留 size
            4. 退回纯 vector search

        参数：
            query_text:
                用户问题文本。

            metadata_filter:
                原始 Chroma metadata filter。

            top_k:
                召回数量。

        返回值：
            tuple[list[tuple[Any, float | None]], MetadataFilter | None, str]：
                第一个元素是检索结果列表。
                第二个元素是实际生效的 metadata filter。
                第三个元素是 fallback 说明。
        """

        strict_results = await self._async_search_documents(
            query_text=query_text,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

        if strict_results:
            return (
                strict_results,
                metadata_filter,
                "",
            )

        if not metadata_filter:
            return (
                strict_results,
                metadata_filter,
                "",
            )

        fallback_candidates = self._build_fallback_filter_candidates(
            metadata_filter=metadata_filter,
        )

        for fallback_filter, fallback_reason in fallback_candidates:
            fallback_results = await self._async_search_documents(
                query_text=query_text,
                metadata_filter=fallback_filter,
                top_k=top_k,
            )

            if fallback_results:
                return (
                    fallback_results,
                    fallback_filter,
                    fallback_reason,
                )

        return (
            [],
            metadata_filter,
            "严格 metadata filter 和 fallback 策略都没有召回结果",
        )

    def _build_fallback_filter_candidates(
            self,
            metadata_filter: MetadataFilter,
    ) -> list[tuple[MetadataFilter | None, str]]:
        """
        构建 fallback metadata filter 候选列表。

        功能：
            当严格 metadata filter 无法召回结果时，
            按从严格到宽松的顺序生成 fallback filter。

            当前策略：
            1. 保留 dog_name 和 size
            2. 只保留 dog_name
            3. 只保留 size
            4. 去掉全部 metadata filter，退回纯向量检索

        参数：
            metadata_filter: MetadataFilter
                原始 Chroma metadata filter。

        返回值：
            list[tuple[MetadataFilter | None, str]]：
                fallback filter 和对应说明。
        """

        conditions = self._split_filter_conditions(
            metadata_filter=metadata_filter,
        )

        if not conditions:
            return [
                (
                    None,
                    "严格 metadata filter 无结果，fallback 为纯向量检索",
                )
            ]

        dog_name_conditions = [
            condition
            for condition in conditions
            if "dog_name" in condition
        ]

        size_conditions = [
            condition
            for condition in conditions
            if "size" in condition
        ]

        core_conditions = [
            condition
            for condition in conditions
            if any(
                field_name in condition
                for field_name in [
                    "dog_name",
                    "size",
                ]
            )
        ]

        candidates: list[tuple[MetadataFilter | None, str]] = []
        seen_keys: set[str] = set()

        def add_candidate(
                fallback_filter: MetadataFilter | None,
                reason: str,
        ) -> None:
            """
            添加 fallback 候选，并去重。

            参数：
                fallback_filter: MetadataFilter | None
                    fallback 使用的过滤条件。

                reason: str
                    fallback 原因说明。

            返回值：
                None。
            """

            key = str(
                fallback_filter,
            )

            if key in seen_keys:
                return

            seen_keys.add(
                key,
            )

            candidates.append(
                (
                    fallback_filter,
                    reason,
                )
            )

        if core_conditions and len(
                core_conditions,
        ) < len(
            conditions,
        ):
            add_candidate(
                fallback_filter=self._conditions_to_filter(
                    conditions=core_conditions,
                ),
                reason=(
                    "严格 metadata filter 无结果，"
                    "fallback 后仅保留 dog_name / size 等核心条件"
                ),
            )

        if dog_name_conditions and len(
                dog_name_conditions,
        ) < len(
            conditions,
        ):
            add_candidate(
                fallback_filter=self._conditions_to_filter(
                    conditions=dog_name_conditions,
                ),
                reason=(
                    "严格 metadata filter 无结果，"
                    "fallback 后仅保留 dog_name 犬种条件"
                ),
            )

        if size_conditions and len(
                size_conditions,
        ) < len(
            conditions,
        ):
            add_candidate(
                fallback_filter=self._conditions_to_filter(
                    conditions=size_conditions,
                ),
                reason=(
                    "严格 metadata filter 无结果，"
                    "fallback 后仅保留 size 体型条件"
                ),
            )

        add_candidate(
            fallback_filter=None,
            reason=(
                "严格 metadata filter 无结果，"
                "fallback 后去掉全部 metadata filter，退回纯向量检索"
            ),
        )

        return candidates

    def _split_filter_conditions(
            self,
            metadata_filter: MetadataFilter,
    ) -> list[MetadataFilter]:
        """
        拆分 metadata filter 为单字段条件列表。

        功能：
            将不同形态的 Chroma metadata filter 拆成统一格式。

            示例一：
                输入：
                    {
                        "$and": [
                            {"size": {"$eq": "small"}},
                            {"barking_level": {"$lte": 3}}
                        ]
                    }

                输出：
                    [
                        {"size": {"$eq": "small"}},
                        {"barking_level": {"$lte": 3}}
                    ]

            示例二：
                输入：
                    {
                        "size": "small",
                        "good_for_beginner": True
                    }

                输出：
                    [
                        {"size": "small"},
                        {"good_for_beginner": True}
                    ]

        参数：
            metadata_filter: MetadataFilter
                原始 metadata filter。

        返回值：
            list[MetadataFilter]：
                单字段条件列表。
        """

        if not metadata_filter:
            return []

        if "$and" in metadata_filter:
            raw_conditions = metadata_filter.get(
                "$and",
                [],
            )

            conditions: list[MetadataFilter] = []

            for condition in raw_conditions:
                if not isinstance(
                        condition,
                        dict,
                ):
                    continue

                conditions.extend(
                    self._split_filter_conditions(
                        metadata_filter=condition,
                    )
                )

            return conditions

        conditions = []

        for field_name, operator_payload in metadata_filter.items():
            if field_name.startswith(
                    "$",
            ):
                continue

            conditions.append(
                {
                    field_name: operator_payload,
                }
            )

        return conditions

    def _conditions_to_filter(
            self,
            conditions: list[MetadataFilter],
    ) -> MetadataFilter | None:
        """
        将单字段条件列表重新组装成 Chroma metadata filter。

        功能：
            如果没有条件，返回 None。
            如果只有一个条件，直接返回该条件。
            如果有多个条件，使用 $and 组合。

        参数：
            conditions: list[MetadataFilter]
                单字段条件列表。

        返回值：
            MetadataFilter | None：
                Chroma 可用的 metadata filter。
        """

        cleaned_conditions = [
            condition
            for condition in conditions
            if condition
        ]

        if not cleaned_conditions:
            return None

        if len(
                cleaned_conditions,
        ) == 1:
            return cleaned_conditions[0]

        return {
            "$and": cleaned_conditions,
        }

    def _build_retrieved_chunks(
            self,
            documents_with_scores: list[tuple[Any, float | None]],
            query_text: str,
            metadata_filter: MetadataFilter | None,
            fallback_reason: str = "",
    ) -> list[RagRetrievedChunk]:
        """
        将 Chroma 检索结果转换成 RagRetrievedChunk 列表。

        功能：
            遍历 LangChain Document，并转换成：
            1. RagChunk
            2. RagRetrievedChunk

            同时会根据 query_text、metadata_filter、fallback_reason
            和 chunk.metadata 生成更有语义价值的 reason。

        参数：
            documents_with_scores: list[tuple[Any, float | None]]
                Chroma 检索结果。

            query_text: str
                用户问题文本。

            metadata_filter: MetadataFilter | None
                本次实际使用的 metadata filter。
                如果发生 fallback，则这里是 fallback 后的 filter。

            fallback_reason: str
                fallback 说明。
                如果没有发生 fallback，则为空字符串。

        返回值：
            list[RagRetrievedChunk]：
                项目内部统一召回结果对象列表。
        """

        retrieved_chunks: list[RagRetrievedChunk] = []

        for index, item in enumerate(
                documents_with_scores,
        ):
            document, score = item

            retrieved_chunk = self._build_retrieved_chunk(
                document=document,
                score=score,
                rank=index + 1,
                query_text=query_text,
                metadata_filter=metadata_filter,
                fallback_reason=fallback_reason,
            )

            retrieved_chunks.append(
                retrieved_chunk,
            )

        return retrieved_chunks

    def _build_retrieved_chunk(
            self,
            document: Any,
            score: float | None,
            rank: int,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            fallback_reason: str = "",
    ) -> RagRetrievedChunk:
        """
        构建单个 RagRetrievedChunk。

        功能：
            将 LangChain Document 转成 RagChunk，
            再用 RagChunk 构建 RagRetrievedChunk。

            reason 字段会根据 metadata_filter、fallback_reason
            和 chunk.metadata 生成“为什么该 chunk 被选中”的语义解释。

        参数：
            document: Any
                LangChain Document 或类似对象。

            score: float | None
                检索分数。
                如果使用 similarity_search，则可能没有 score。

            rank: int
                当前结果排名，从 1 开始。
                当前 schema 中没有 rank 字段，所以写入 reason 中。

            query_text: str
                用户问题文本。

            metadata_filter: MetadataFilter | None
                本次实际使用的 metadata filter。

            fallback_reason: str
                fallback 说明。
                如果没有发生 fallback，则为空字符串。

        返回值：
            RagRetrievedChunk：
                项目内部统一召回 chunk 对象。
        """

        rag_chunk = self._document_to_rag_chunk(
            document=document,
            rank=rank,
        )

        retrieval_score = self._safe_float(
            value=score,
            default=0.0,
        )

        reason = self._build_retrieval_reason(
            query_text=query_text,
            metadata_filter=metadata_filter,
            rag_chunk=rag_chunk,
            rank=rank,
            fallback_reason=fallback_reason,
        )

        return RagRetrievedChunk(
            chunk=rag_chunk,
            retrieval_score=retrieval_score,
            rerank_score=None,
            final_score=retrieval_score,
            reason=reason,
        )

    def _document_to_rag_chunk(
            self,
            document: Any,
            rank: int,
    ) -> RagChunk:
        """
        将 LangChain Document 转换成 RagChunk。

        功能：
            Chroma similarity_search 返回的是 LangChain Document。
            Document 通常只有：
            1. page_content
            2. metadata

            本方法会从 metadata 中还原 RagChunk 需要的字段：
            1. chunk_id
            2. doc_id
            3. chunk_index
            4. source
            5. title

        参数：
            document: Any
                LangChain Document 或类似对象。

            rank: int
                当前召回排名，用于缺失字段时兜底。

        返回值：
            RagChunk：
                项目内部标准 chunk 对象。
        """

        content = self._get_document_content(
            document=document,
        )

        metadata = self._get_document_metadata(
            document=document,
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
            )
            or metadata.get(
                "id",
            )
            or self._build_fallback_chunk_id(
                content=content,
                rank=rank,
            )
        )

        doc_id = str(
            metadata.get(
                "doc_id",
            )
            or metadata.get(
                "document_id",
            )
            or "unknown_doc"
        )

        chunk_index = self._safe_int(
            value=metadata.get(
                "chunk_index",
                rank - 1,
            ),
            default=rank - 1,
        )

        source = str(
            metadata.get(
                "source",
            )
            or metadata.get(
                "relative_path",
            )
            or metadata.get(
                "file_name",
            )
            or ""
        )

        title = str(
            metadata.get(
                "title",
            )
            or metadata.get(
                "dog_name",
            )
            or metadata.get(
                "file_stem",
            )
            or ""
        )

        return RagChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            chunk_index=chunk_index,
            source=source,
            title=title,
            metadata=metadata,
        )

    def _build_rag_context(
            self,
            question: str,
            retrieved_chunks: list[RagRetrievedChunk],
    ) -> RagContext:
        """
        构建 RagContext。

        功能：
            将 question、retrieved_chunks、context_text、source_count、status
            组装成项目内部统一 RAG 上下文对象。

        参数：
            question: str
                用户问题。

            retrieved_chunks: list[RagRetrievedChunk]
                召回结果列表。

        返回值：
            RagContext：
                RAG 上下文对象。
        """

        context_text = self._build_context_text(
            retrieved_chunks=retrieved_chunks,
        )

        source_count = self._count_sources(
            retrieved_chunks=retrieved_chunks,
        )

        status = (
            "success"
            if retrieved_chunks
            else "empty"
        )

        return RagContext(
            question=question,
            context_text=context_text,
            chunks=retrieved_chunks,
            source_count=source_count,
            status=status,
        )

    def _build_context_text(
            self,
            retrieved_chunks: list[RagRetrievedChunk],
    ) -> str:
        """
        构建用于 LLM Prompt 的 context_text。

        功能：
            将多个 RagRetrievedChunk 拼接成可读上下文文本。

        参数：
            retrieved_chunks: list[RagRetrievedChunk]
                召回结果列表。

        返回值：
            str：
                拼接后的上下文文本。
        """

        blocks: list[str] = []

        for index, retrieved_chunk in enumerate(
                retrieved_chunks,
        ):
            chunk = retrieved_chunk.chunk

            dog_name = chunk.metadata.get(
                "dog_name",
                "unknown",
            )

            source = (
                    chunk.source
                    or chunk.metadata.get(
                        "relative_path",
                        "unknown",
                    )
            )

            blocks.append(
                "\n".join(
                    [
                        f"[Chunk {index + 1}]",
                        f"dog_name: {dog_name}",
                        f"source: {source}",
                        f"retrieval_score: {retrieved_chunk.retrieval_score}",
                        f"final_score: {retrieved_chunk.final_score}",
                        chunk.content,
                    ]
                )
            )

        return "\n\n".join(
            blocks,
        )

    def _get_query_text(
            self,
            query: RagQuery | str,
    ) -> str:
        """
        从查询对象中读取问题文本。

        功能：
            支持两种输入：
            1. str
            2. RagQuery

            当前项目 RagQuery 的字段是 question。

        参数：
            query: RagQuery | str
                查询对象或查询字符串。

        返回值：
            str：
                查询文本。
        """

        if isinstance(
                query,
                str,
        ):
            return query

        if not query.question:
            raise ValueError(
                "RagQuery.question 不能为空"
            )

        return query.question

    def _resolve_top_k(
            self,
            query: RagQuery | str,
            top_k: int | None,
    ) -> int:
        """
        解析本次召回数量 top_k。

        功能：
            优先级：
            1. 外部显式传入 top_k
            2. RagQuery.top_k
            3. self.default_top_k

        参数：
            query: RagQuery | str
                查询对象或查询字符串。

            top_k: int | None
                外部传入的 top_k。

        返回值：
            int：
                最终使用的 top_k。
        """

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k 必须大于 0"
                )

            return top_k

        if isinstance(
                query,
                RagQuery,
        ):
            return query.top_k

        return self.default_top_k

    def _resolve_metadata_filter(
            self,
            query: RagQuery | str,
            metadata_filter: MetadataFilter | None,
    ) -> MetadataFilter | None:
        """
        解析 metadata filter。

        功能：
            优先级：
            1. 外部显式传入 metadata_filter
            2. RagQuery.filters
            3. None

        参数：
            query: RagQuery | str
                查询对象或查询字符串。

            metadata_filter: MetadataFilter | None
                外部传入的 metadata filter。

        返回值：
            MetadataFilter | None：
                最终使用的 metadata filter。
        """

        if metadata_filter:
            return metadata_filter

        if isinstance(
                query,
                RagQuery,
        ) and query.filters:
            return dict(
                query.filters,
            )

        return None

    def _get_document_content(
            self,
            document: Any,
    ) -> str:
        """
        从 LangChain Document 中读取正文内容。

        功能：
            LangChain Document 的正文字段通常是 page_content。

        参数：
            document: Any
                LangChain Document 或类似对象。

        返回值：
            str：
                文档正文。
        """

        return str(
            getattr(
                document,
                "page_content",
                "",
            )
            or ""
        )

    def _get_document_metadata(
            self,
            document: Any,
    ) -> dict[str, Any]:
        """
        从 LangChain Document 中读取 metadata。

        功能：
            LangChain Document 的 metadata 通常是 dict。
            如果没有 metadata，则返回空 dict。

        参数：
            document: Any
                LangChain Document 或类似对象。

        返回值：
            dict[str, Any]：
                文档 metadata。
        """

        metadata = getattr(
            document,
            "metadata",
            None,
        )

        if metadata is None:
            return {}

        return dict(
            metadata,
        )

    def _count_sources(
            self,
            retrieved_chunks: list[RagRetrievedChunk],
    ) -> int:
        """
        统计来源数量。

        功能：
            根据 doc_id 或 source 统计本次召回结果来自多少个不同来源。

        参数：
            retrieved_chunks: list[RagRetrievedChunk]
                召回结果列表。

        返回值：
            int：
                不同来源数量。
        """

        sources: set[str] = set()

        for retrieved_chunk in retrieved_chunks:

            chunk = retrieved_chunk.chunk

            source_key = (
                    chunk.doc_id
                    or chunk.source
                    or chunk.metadata.get(
                        "relative_path",
                        "",
                    )
            )

            if source_key:
                sources.add(
                    str(
                        source_key,
                    )
                )

        return len(
            sources,
        )

    def _build_fallback_chunk_id(
            self,
            content: str,
            rank: int,
    ) -> str:
        """
        构建兜底 chunk_id。

        功能：
            当 Chroma metadata 中缺少 chunk_id 时，
            根据内容 hash 和 rank 构建一个临时 ID。

        参数：
            content: str
                chunk 正文内容。

            rank: int
                当前召回排名。

        返回值：
            str：
                兜底 chunk_id。
        """

        content_hash = hashlib.sha256(
            content.encode(
                "utf-8",
            )
        ).hexdigest()[
            :16
        ]

        return f"retrieved_chunk::{content_hash}::{rank}"

    def _safe_float(
            self,
            value: Any,
            default: float = 0.0,
    ) -> float:
        """
        安全转换 float。

        功能：
            将 value 转换成 float。
            如果 value 为 None 或无法转换，则返回 default。

        参数：
            value: Any
                待转换的值。

            default: float
                转换失败时使用的默认值。

        返回值：
            float：
                转换后的浮点数。
        """

        if value is None:
            return default

        try:
            return float(
                value,
            )
        except (
                TypeError,
                ValueError,
        ):
            return default

    def _safe_int(
            self,
            value: Any,
            default: int = 0,
    ) -> int:
        """
        安全转换 int。

        功能：
            将 value 转换成 int。
            如果 value 为 None 或无法转换，则返回 default。

        参数：
            value: Any
                待转换的值。

            default: int
                转换失败时使用的默认值。

        返回值：
            int：
                转换后的整数。
        """

        if value is None:
            return default

        try:
            return int(
                value,
            )
        except (
                TypeError,
                ValueError,
        ):
            return default

    def _build_retrieval_reason(
            self,
            query_text: str,
            metadata_filter: MetadataFilter | None,
            rag_chunk: RagChunk,
            rank: int,
            fallback_reason: str = "",
    ) -> str:
        """
        构建召回原因 reason。

        功能：
            根据用户问题、metadata_filter、fallback_reason
            和当前 chunk.metadata 生成更容易理解的召回原因。

            reason 不只记录技术信息，
            还会说明：
            1. 当前 chunk 匹配了哪些用户查询条件
            2. 是否发生 fallback
            3. 是否使用 metadata filter
            4. 当前结果来自哪个 retriever
            5. 当前结果排名 rank

        参数：
            query_text: str
                用户问题文本。

            metadata_filter: MetadataFilter | None
                本次实际使用的 metadata filter。
                如果发生 fallback，则这里是 fallback 后的 filter。

            rag_chunk: RagChunk
                当前召回到的 chunk。

            rank: int
                当前召回排名。

            fallback_reason: str
                fallback 说明。
                如果没有发生 fallback，则为空字符串。

        返回值：
            str：
                可读的召回原因。
        """

        semantic_reasons = self._build_semantic_reasons(
            metadata_filter=metadata_filter,
            rag_chunk=rag_chunk,
        )

        reason_parts: list[str] = []

        if fallback_reason:
            reason_parts.append(
                f"Fallback 信息：{fallback_reason}"
            )

        if semantic_reasons:
            semantic_text = "；".join(
                semantic_reasons,
            )

            reason_parts.append(
                f"匹配用户查询条件：{semantic_text}"
            )
        else:
            reason_parts.append(
                "未识别到明确 metadata 条件，"
                "该 chunk 主要通过向量语义相似度召回"
            )

        reason_parts.append(
            f"用户问题：{query_text}"
        )

        reason_parts.append(
            f"技术信息：retriever={self.RETRIEVER_NAME}，rank={rank}"
        )

        return "。".join(
            reason_parts,
        ) + "。"

    def _build_semantic_reasons(
            self,
            metadata_filter: MetadataFilter | None,
            rag_chunk: RagChunk,
    ) -> list[str]:
        """
        构建语义召回原因列表。

        功能：
            将 metadata_filter 中的结构化条件，
            转换成用户更容易理解的中文说明。

        参数：
            metadata_filter: MetadataFilter | None
                本次检索使用的 metadata filter。

            rag_chunk: RagChunk
                当前召回到的 chunk。

        返回值：
            list[str]：
                语义召回原因列表。
        """

        if not metadata_filter:
            return []

        conditions = self._flatten_metadata_filter(
            metadata_filter=metadata_filter,
        )

        reasons: list[str] = []

        for condition in conditions:

            reason = self._build_condition_reason(
                condition=condition,
                metadata=rag_chunk.metadata,
            )

            if reason:
                reasons.append(
                    reason,
                )

        return reasons

    def _flatten_metadata_filter(
            self,
            metadata_filter: MetadataFilter,
    ) -> list[MetadataFilter]:
        """
        展平 metadata filter。

        功能：
            将 Chroma where filter 转换成单条件列表。

            例如：
                {"$and": [A, B, C]}

            会转换成：
                [A, B, C]

            如果本身就是单条件，则返回：
                [metadata_filter]

        参数：
            metadata_filter: MetadataFilter
                Chroma metadata filter。

        返回值：
            list[MetadataFilter]：
                单条件列表。
        """

        if "$and" in metadata_filter:
            return list(
                metadata_filter[
                    "$and"
                ]
            )

        return [
            metadata_filter,
        ]

    def _build_condition_reason(
            self,
            condition: MetadataFilter,
            metadata: dict[str, Any],
    ) -> str:
        """
        将单个 metadata 条件转换成中文原因。

        功能：
            根据 Chroma where filter 中的字段名、操作符和值，
            生成可读的中文解释。

            支持的 operator（操作符）：
            1. $eq：等于
            2. $lte：小于等于
            3. $gte：大于等于

            如果 operator_payload 不是 dict，
            则默认按 $eq 等值匹配处理。

        参数：
            condition: MetadataFilter
                单个 metadata 条件。
                例如：
                    {"size": {"$eq": "small"}}
                    {"barking_level": {"$lte": 3}}

            metadata: dict[str, Any]
                当前 chunk 的 metadata。

        返回值：
            str：
                中文原因。
                如果无法识别，则返回空字符串。
        """

        if not condition:
            return ""

        reasons: list[str] = []

        for field_name, operator_payload in condition.items():

            actual_value = metadata.get(
                field_name,
            )

            if not isinstance(
                    operator_payload,
                    dict,
            ):
                reasons.append(
                    self._build_eq_reason(
                        field_name=field_name,
                        expected_value=operator_payload,
                        actual_value=actual_value,
                    )
                )
                continue

            if "$eq" in operator_payload:
                reasons.append(
                    self._build_eq_reason(
                        field_name=field_name,
                        expected_value=operator_payload[
                            "$eq"
                        ],
                        actual_value=actual_value,
                    )
                )
                continue

            if "$lte" in operator_payload:
                reasons.append(
                    self._build_lte_reason(
                        field_name=field_name,
                        expected_value=operator_payload[
                            "$lte"
                        ],
                        actual_value=actual_value,
                    )
                )
                continue

            if "$gte" in operator_payload:
                reasons.append(
                    self._build_gte_reason(
                        field_name=field_name,
                        expected_value=operator_payload[
                            "$gte"
                        ],
                        actual_value=actual_value,
                    )
                )
                continue

            label = self._get_field_label(
                field_name=field_name,
            )

            reasons.append(
                f"{label}使用了暂未支持的过滤操作符：{operator_payload}"
            )

        return "；".join(
            reason
            for reason in reasons
            if reason
        )

    def _build_eq_reason(
            self,
            field_name: str,
            expected_value: Any,
            actual_value: Any,
    ) -> str:
        """
        构建 $eq 条件的中文原因。

        功能：
            判断当前 metadata 实际值是否等于查询期望值，
            并生成“符合 / 不符合 / 无法判断”的中文解释。

        参数：
            field_name: str
                metadata 字段名。

            expected_value: Any
                查询要求的值。

            actual_value: Any
                当前 chunk metadata 中的实际值。

        返回值：
            str：
                中文原因。
        """

        label = self._get_field_label(
            field_name=field_name,
        )

        expected_text = self._format_reason_value(
            value=expected_value,
        )

        actual_text = self._format_reason_value(
            value=actual_value,
        )

        if actual_value is None:
            return (
                f"无法判断{label}是否符合要求："
                f"期望值为 {expected_text}，但当前 metadata 缺少该字段"
            )

        if actual_value == expected_value:
            return (
                f"{label}符合要求："
                f"期望值为 {expected_text}，实际值为 {actual_text}"
            )

        return (
            f"{label}不符合要求："
            f"期望值为 {expected_text}，实际值为 {actual_text}"
        )

    def _build_lte_reason(
            self,
            field_name: str,
            expected_value: Any,
            actual_value: Any,
    ) -> str:
        """
        构建 $lte 条件的中文原因。

        功能：
            判断当前 metadata 实际值是否小于等于查询期望值。

            $lte 是 less than or equal（小于等于）的意思。
            例如：
                {"barking_level": {"$lte": 3}}
            表示：
                吠叫等级不高于 3。

        参数：
            field_name: str
                metadata 字段名。

            expected_value: Any
                查询要求的最大值。

            actual_value: Any
                当前 chunk metadata 中的实际值。

        返回值：
            str：
                中文原因。
        """

        label = self._get_field_label(
            field_name=field_name,
        )

        expected_number = self._safe_compare_number(
            value=expected_value,
        )

        actual_number = self._safe_compare_number(
            value=actual_value,
        )

        if actual_number is None:
            return (
                f"无法判断{label}是否满足不高于 {expected_value} 的要求："
                f"当前实际值为 {actual_value}"
            )

        if expected_number is None:
            return (
                f"无法判断{label}是否满足要求："
                f"查询期望值 {expected_value} 不是有效数字"
            )

        if actual_number <= expected_number:
            return (
                f"满足{label}不高于 {expected_value} 的要求"
                f"（实际值为 {actual_value}）"
            )

        return (
            f"不满足{label}不高于 {expected_value} 的要求"
            f"（实际值为 {actual_value}）"
        )

    def _build_gte_reason(
            self,
            field_name: str,
            expected_value: Any,
            actual_value: Any,
    ) -> str:
        """
        构建 $gte 条件的中文原因。

        功能：
            判断当前 metadata 实际值是否大于等于查询期望值。

            $gte 是 greater than or equal（大于等于）的意思。
            例如：
                {"trainability_level": {"$gte": 4}}
            表示：
                可训练等级不低于 4。

        参数：
            field_name: str
                metadata 字段名。

            expected_value: Any
                查询要求的最小值。

            actual_value: Any
                当前 chunk metadata 中的实际值。

        返回值：
            str：
                中文原因。
        """

        label = self._get_field_label(
            field_name=field_name,
        )

        expected_number = self._safe_compare_number(
            value=expected_value,
        )

        actual_number = self._safe_compare_number(
            value=actual_value,
        )

        if actual_number is None:
            return (
                f"无法判断{label}是否满足不低于 {expected_value} 的要求："
                f"当前实际值为 {actual_value}"
            )

        if expected_number is None:
            return (
                f"无法判断{label}是否满足要求："
                f"查询期望值 {expected_value} 不是有效数字"
            )

        if actual_number >= expected_number:
            return (
                f"满足{label}不低于 {expected_value} 的要求"
                f"（实际值为 {actual_value}）"
            )

        return (
            f"不满足{label}不低于 {expected_value} 的要求"
            f"（实际值为 {actual_value}）"
        )

    def _get_field_label(
            self,
            field_name: str,
    ) -> str:
        """
        将 metadata 字段名转换成中文名称。

        功能：
            用于生成更容易理解的 reason。
            如果字段没有配置中文名，则返回原始字段名。

        参数：
            field_name: str
                metadata 字段名。

        返回值：
            str：
                中文字段名称。
        """

        field_labels: dict[str, str] = {
            "dog_name": "犬种",
            "size": "体型",

            "energy": "精力水平",
            "energy_level": "精力等级",

            "barking": "吠叫程度",
            "barking_level": "吠叫等级",

            "trainability": "可训练性",
            "trainability_level": "可训练等级",

            "shedding": "掉毛程度",
            "shedding_level": "掉毛等级",

            "good_for_apartment": "适合公寓",
            "good_for_beginner": "适合新手",

            "good_with_young_children_level": "适合小孩等级",
            "good_with_other_dogs_level": "适合其他狗等级",

            "drooling_level": "流口水等级",
            "coat_grooming_frequency_level": "打理频率等级",

            "height": "身高",
            "weight": "体重",
            "lifespan": "寿命",
            "tags": "标签",

            "source": "来源",
            "relative_path": "相对路径",
            "file_name": "文件名",
            "file_stem": "文件名主体",
            "section": "章节",
            "section_title": "章节标题",
            "chunk_index": "文本块序号",
        }

        return field_labels.get(
            field_name,
            field_name,
        )

    def _format_reason_value(
            self,
            value: Any,
    ) -> str:
        """
        格式化 reason 中展示的值。

        功能：
            将 Python 值转换成更适合中文 reason 展示的文本。

            例如：
                True -> 是
                False -> 否
                None -> 缺失

        参数：
            value: Any
                原始值。

        返回值：
            str：
                格式化后的文本。
        """

        if value is True:
            return "是"

        if value is False:
            return "否"

        if value is None:
            return "缺失"

        return str(
            value,
        )

    def _safe_compare_number(
            self,
            value: Any,
    ) -> float | None:
        """
        安全转换用于比较的数字。

        功能：
            将 value 转换成 float。
            如果无法转换，则返回 None。

        参数：
            value: Any
                待转换的值。

        返回值：
            float | None：
                转换后的数字。
                如果转换失败，则返回 None。
        """

        if value is None:
            return None

        try:
            return float(
                value,
            )
        except (
                TypeError,
                ValueError,
        ):
            return None