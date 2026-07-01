from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.rag.evaluation.schemas import (
    RagEvalCase,
    RagEvalResult,
    RagEvalRetrievedItem,
)

from src.rag.evaluators.retrieval_quality_evaluator import (
    evaluate_rag_context_quality,
    extract_chunks_from_rag_context,
    normalize_mapping,
)

from src.rag.evaluation.filter_utils import (
    flatten_filter_mapping,
    is_semantic_filter_subset_matched,
    normalize_semantic_filter_mapping,
)


ParseQueryFunc = Callable[[RagEvalCase], Any]
RetrieveContextFunc = Callable[[Any], Any]


class OfflineRetrievalEvaluator:
    """
    离线 RAG 检索评估器。

    功能：
        用于执行一批 RAG Evaluation 测试用例，
        将 RagEvalCase 转换成 RagEvalResult。

        它只负责批量执行评估并生成结构化结果。

    参数含义：
        parse_query_func:
            查询解析函数。
            输入 RagEvalCase，输出 RagQuery 或 dict。
            中文释义：负责把评估用例中的 question 转成系统可检索的 query 对象。

        retrieve_context_func:
            检索函数。
            输入 parse_query_func 的输出，返回 RagContext 或 dict。
            中文释义：负责执行真实 RAG 检索，返回召回上下文。

        require_quality_usable:
            是否要求 evaluate_rag_context_quality 的 is_usable=True 才算通过。
            第一版建议保持 True，这样评估更接近运行时质量门控。

    返回值含义：
        OfflineRetrievalEvaluator 实例。

    专业名词：
        Offline Evaluation：
            离线评估。指不在用户请求链路中执行，而是对评估集批量运行和统计。

        Runtime Evaluation：
            运行时评估。指在真实用户请求过程中判断当前结果是否可用。

        Quality Gate：
            质量门控。指通过规则判断当前结果是否足够好，能否进入下一步。
    """

    def __init__(
        self,
        parse_query_func: ParseQueryFunc,
        retrieve_context_func: RetrieveContextFunc,
        require_quality_usable: bool = True,
    ) -> None:
        """
        初始化离线 RAG 检索评估器。

        参数含义：
            parse_query_func:
                查询解析函数。输入 RagEvalCase，输出 RagQuery 或 dict。

            retrieve_context_func:
                检索函数。输入解析后的 query，输出 RagContext 或 dict。

            require_quality_usable:
                是否要求运行时质量评估结果可用。

        返回值含义：
            None。
        """

        self.parse_query_func = parse_query_func
        self.retrieve_context_func = retrieve_context_func
        self.require_quality_usable = require_quality_usable

    def evaluate_case(
        self,
        eval_case: RagEvalCase,
    ) -> RagEvalResult:
        """
        执行单条 RAG 评估用例。

        执行流程：
            1. 记录开始时间。
            2. 调用 parse_query_func 解析问题。
            3. 从解析结果中提取 parsed_filters。
            4. 调用 retrieve_context_func 执行检索。
            5. 归一化 RagContext。
            6. 复用 evaluate_rag_context_quality 判断召回质量。
            7. 提取召回犬种列表。
            8. 判断 hit@k、top1_hit、filter_matched。
            9. 组装 RagEvalResult。

        参数含义：
            eval_case:
                一条 RAG 评估用例。

        返回值含义：
            RagEvalResult:
                单条评估用例执行后的结果。

        异常处理：
            如果单条 case 执行失败，不会继续向外抛异常。
            而是返回 RagEvalResult.from_failed_case，
            这样批量评估不会因为一条失败而整体中断。
        """

        started_at = time.perf_counter()

        try:
            parsed_query = self.parse_query_func(eval_case)

            parsed_filters = self._extract_parsed_filters(
                parsed_query=parsed_query,
            )

            raw_rag_context = self.retrieve_context_func(parsed_query)

            rag_context = self._normalize_rag_context(
                raw_rag_context=raw_rag_context,
            )

            expected_quality_dog_name = self._extract_quality_expected_dog_name(
                eval_case=eval_case,
            )

            quality_result = evaluate_rag_context_quality(
                rag_context=rag_context,
                question=eval_case.question,
                expected_dog_name=expected_quality_dog_name,
            )

            retrieved_items = self._build_retrieved_items(
                rag_context=rag_context,
            )

            retrieved_dog_names = [
                item.dog_name
                for item in retrieved_items
                if item.dog_name
            ]

            hit_rank = self._find_first_hit_rank(
                retrieved_items=retrieved_items,
                expected_dog_names=eval_case.expected_dog_names,
            )

            hit = hit_rank is not None

            top1_hit = hit_rank == 1

            # 使用归一化工具将filters中的普通key-value字段和chroma用的$and $eq
            # 等字段归一化统一格式用于比较和阅读 同时将$lte：3 $gt：2等格式数据变为
            # 自然语言的阶级等级 例如high normal等
            expected_filters_flattened = flatten_filter_mapping(
                filters=eval_case.expected_filters,
            )

            parsed_filters_flattened = flatten_filter_mapping(
                filters=parsed_filters,
            )

            expected_filters_semantic = normalize_semantic_filter_mapping(
                filters=eval_case.expected_filters,
            )

            parsed_filters_semantic = normalize_semantic_filter_mapping(
                filters=parsed_filters,
            )

            filter_matched = self._is_filter_matched(
                expected_filters=eval_case.expected_filters,
                parsed_filters=parsed_filters,
            )

            empty_retrieval = len(retrieved_items) == 0

            passed = self._resolve_passed(
                eval_case=eval_case,
                hit=hit,
                filter_matched=filter_matched,
                empty_retrieval=empty_retrieval,
                quality_is_usable=quality_result.is_usable,
            )

            latency_ms = self._elapsed_ms(
                started_at=started_at,
            )

            return RagEvalResult(
                case_id=eval_case.case_id,
                question=eval_case.question,
                expected_dog_names=eval_case.expected_dog_names,
                expected_filters=eval_case.expected_filters,
                parsed_filters=parsed_filters,
                retrieved_items=retrieved_items,
                retrieved_dog_names=retrieved_dog_names,
                hit=hit,
                hit_rank=hit_rank,
                top1_hit=top1_hit,
                filter_matched=filter_matched,
                empty_retrieval=empty_retrieval,
                passed=passed,
                latency_ms=latency_ms,
                extra={
                    "quality_status": str(
                        quality_result.status
                        or ""
                    ),
                    "quality_score": quality_result.quality_score,
                    "quality_is_usable": quality_result.is_usable,
                    "quality_failure_type": str(
                        quality_result.failure_type
                        or ""
                    ),
                    "quality_reasons": quality_result.reasons,
                    "quality_metrics": quality_result.metrics,
                    "expected_filters_flattened": expected_filters_flattened,
                    "parsed_filters_flattened": parsed_filters_flattened,
                    "expected_filters_semantic": expected_filters_semantic,
                    "parsed_filters_semantic": parsed_filters_semantic,
                    "filter_compare_mode": "semantic_subset_match_after_flatten",
                },
            )

        except Exception as exc:
            failed_result = RagEvalResult.from_failed_case(
                eval_case=eval_case,
                error_message=str(exc),
            )

            failed_result.latency_ms = self._elapsed_ms(
                started_at=started_at,
            )

            return failed_result

    def evaluate_many(
        self,
        eval_cases: list[RagEvalCase],
    ) -> list[RagEvalResult]:
        """
        批量执行 RAG 评估用例。

        参数含义：
            eval_cases:
                多条 RAG 评估用例。

        返回值含义：
            list[RagEvalResult]:
                每条评估用例对应的评估结果列表。

        说明：
            每条 case 内部都会单独捕获异常。
            因此某一条失败不会影响后续 case 执行。
        """

        results: list[RagEvalResult] = []

        for eval_case in eval_cases:
            result = self.evaluate_case(
                eval_case=eval_case,
            )

            results.append(result)

        return results

    def _normalize_rag_context(
        self,
        raw_rag_context: Any,
    ) -> dict[str, Any]:
        """
        归一化 RagContext。

        功能：
            兼容以下几种返回形式：
            1. RagContext Pydantic 对象。
            2. dict 类型 RagContext。
            3. {"rag_context": RagContext} 这种包裹结构。

        参数含义：
            raw_rag_context:
                retrieve_context_func 返回的原始对象。

        返回值含义：
            dict[str, Any]:
                归一化后的 RagContext 字典。
        """

        context_mapping = normalize_mapping(
            raw_rag_context,
        )

        nested_context = context_mapping.get(
            "rag_context",
        )

        if nested_context is not None:
            return normalize_mapping(
                nested_context,
            )

        return context_mapping

    def _extract_parsed_filters(
        self,
        parsed_query: Any,
    ) -> dict[str, Any]:
        """
        从解析后的 query 中提取 filters。

        功能：
            兼容 Pydantic RagQuery、dict、普通对象。

        参数含义：
            parsed_query:
                parse_query_func 返回的 query 对象。

        返回值含义：
            dict[str, Any]:
                实际解析出的 filters。
        """

        query_mapping = normalize_mapping(
            parsed_query,
        )

        if query_mapping:
            raw_filters = (
                query_mapping.get("filters")
                or query_mapping.get("metadata_filters")
                or query_mapping.get("metadata_filter")
                or {}
            )

            return normalize_mapping(
                raw_filters,
            )

        raw_filters = getattr(
            parsed_query,
            "filters",
            None,
        )

        if raw_filters is None:
            raw_filters = getattr(
                parsed_query,
                "metadata_filters",
                None,
            )

        if raw_filters is None:
            raw_filters = getattr(
                parsed_query,
                "metadata_filter",
                None,
            )

        return normalize_mapping(
            raw_filters or {},
        )

    def _build_retrieved_items(
        self,
        rag_context: dict[str, Any],
    ) -> list[RagEvalRetrievedItem]:
        """
        从 RagContext 中构建评估用召回结果项。

        参数含义：
            rag_context:
                归一化后的 RagContext 字典。

        返回值含义：
            list[RagEvalRetrievedItem]:
                用于 RagEvalResult 保存和报告展示的召回结果列表。
        """

        chunks = extract_chunks_from_rag_context(
            rag_context=rag_context,
        )

        retrieved_items: list[RagEvalRetrievedItem] = []

        for index, retrieved_chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk = normalize_mapping(
                retrieved_chunk.get(
                    "chunk",
                    {},
                )
            )

            metadata = normalize_mapping(
                chunk.get(
                    "metadata",
                    {},
                )
            )

            content = self._extract_chunk_content(
                chunk=chunk,
            )

            item = RagEvalRetrievedItem(
                rank=index,
                chunk_id=self._extract_chunk_id(
                    chunk=chunk,
                    metadata=metadata,
                ),
                dog_name=self._extract_dog_name(
                    metadata=metadata,
                ),
                score=self._extract_retrieval_score(
                    retrieved_chunk=retrieved_chunk,
                ),
                source=self._extract_source(
                    metadata=metadata,
                ),
                section_title=self._extract_section_title(
                    metadata=metadata,
                    chunk=chunk,
                ),
                content_preview=self._build_content_preview(
                    content=content,
                ),
                metadata=metadata,
            )

            retrieved_items.append(item)

        return retrieved_items

    def _extract_chunk_content(
        self,
        chunk: dict[str, Any],
    ) -> str:
        """
        从 chunk 中提取正文内容。

        参数含义：
            chunk:
                召回 chunk 字典。

        返回值含义：
            str:
                chunk 正文内容。
        """

        raw_content = (
            chunk.get("content")
            or chunk.get("page_content")
            or chunk.get("text")
            or ""
        )

        return str(raw_content)

    def _extract_chunk_id(
        self,
        chunk: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str | None:
        """
        从 chunk 或 metadata 中提取 chunk_id。

        参数含义：
            chunk:
                召回 chunk 字典。

            metadata:
                chunk metadata 字典。

        返回值含义：
            str | None:
                chunk 唯一编号。
        """

        raw_chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
        )

        if raw_chunk_id is None:
            return None

        return str(raw_chunk_id)

    def _extract_dog_name(
        self,
        metadata: dict[str, Any],
    ) -> str | None:
        """
        从 metadata 中提取 dog_name。

        参数含义：
            metadata:
                chunk metadata 字典。

        返回值含义：
            str | None:
                犬种名称。
        """

        raw_dog_name = (
            metadata.get("dog_name")
            or metadata.get("name")
        )

        if not raw_dog_name:
            return None

        return str(raw_dog_name).strip()

    def _extract_retrieval_score(
        self,
        retrieved_chunk: dict[str, Any],
    ) -> float | None:
        """
        从 retrieved_chunk 中提取检索分数。

        参数含义：
            retrieved_chunk:
                RagRetrievedChunk 字典。

        返回值含义：
            float | None:
                检索分数。
                注意：你当前项目里 retrieval_score 更接近 distance，
                即数值越小通常越好。
        """

        raw_score = retrieved_chunk.get("retrieval_score")

        if raw_score is None:
            raw_score = retrieved_chunk.get("score")

        if raw_score is None:
            raw_score = retrieved_chunk.get("distance")

        if raw_score is None:
            return None

        try:
            return float(raw_score)
        except (
            TypeError,
            ValueError,
        ):
            return None

    def _extract_source(
        self,
        metadata: dict[str, Any],
    ) -> str | None:
        """
        从 metadata 中提取来源路径。

        参数含义：
            metadata:
                chunk metadata 字典。

        返回值含义：
            str | None:
                来源文件路径或来源标识。
        """

        raw_source = (
            metadata.get("source")
            or metadata.get("source_path")
            or metadata.get("file_path")
            or metadata.get("relative_path")
        )

        if not raw_source:
            return None

        return str(raw_source)

    def _extract_section_title(
        self,
        metadata: dict[str, Any],
        chunk: dict[str, Any],
    ) -> str | None:
        """
        从 metadata 或 chunk 中提取 section 标题。

        参数含义：
            metadata:
                chunk metadata 字典。

            chunk:
                chunk 字典。

        返回值含义：
            str | None:
                Markdown section 标题。
        """

        raw_section_title = (
            metadata.get("section_title")
            or metadata.get("heading")
            or chunk.get("title")
        )

        if not raw_section_title:
            return None

        return str(raw_section_title)

    def _build_content_preview(
        self,
        content: str,
        max_chars: int = 160,
    ) -> str | None:
        """
        构建 chunk 内容预览。

        参数含义：
            content:
                chunk 正文内容。

            max_chars:
                最大预览字符数。

        返回值含义：
            str | None:
                截断后的内容预览。
        """

        normalized_content = content.strip()

        if not normalized_content:
            return None

        if len(normalized_content) <= max_chars:
            return normalized_content

        return normalized_content[:max_chars] + "..."

    def _find_first_hit_rank(
        self,
        retrieved_items: list[RagEvalRetrievedItem],
        expected_dog_names: list[str],
    ) -> int | None:
        """
        查找第一次命中期望犬种的 rank。

        参数含义：
            retrieved_items:
                召回结果项列表。

            expected_dog_names:
                期望命中的犬种名称列表。

        返回值含义：
            int | None:
                第一次命中的排序位置。
                如果没有命中，返回 None。
        """

        if not expected_dog_names:
            return None

        expected_names = {
            dog_name.strip().lower()
            for dog_name in expected_dog_names
            if dog_name.strip()
        }

        for item in retrieved_items:
            if not item.dog_name:
                continue

            if item.dog_name.strip().lower() in expected_names:
                return item.rank

        return None

    def _is_filter_matched(
            self,
            expected_filters: dict[str, Any],
            parsed_filters: dict[str, Any],
    ) -> bool:
        """
        判断解析出的 filters 是否符合预期。

        参数含义：
            expected_filters:
                评估用例中定义的期望 filters。
                推荐使用业务语义格式，例如：
                {"trainability": "high"}

            parsed_filters:
                Query Parser 实际解析出的 filters。
                可以是 Chroma where 格式，例如：
                {"trainability_level": {"$gte": 4}}

        返回值含义：
            bool:
                True 表示 parsed_filters 在语义上覆盖 expected_filters。
                False 表示至少有一个期望字段没有被覆盖。

        说明：
            当前使用语义子集匹配：
            1. 先将 expected_filters 和 parsed_filters 扁平化。
            2. 再将字段和值归一化成业务语义格式。
            3. 最后判断 expected 是否是 parsed 的子集。

        专业名词：
            Semantic Match：
                语义匹配。不是要求原始结构完全一样，
                而是判断它们表达的业务含义是否一致。

            Subset Match：
                子集匹配。实际解析结果可以比期望结果多字段，
                但必须覆盖期望字段。
        """

        return is_semantic_filter_subset_matched(
            expected_filters=expected_filters,
            parsed_filters=parsed_filters,
        )

    def _extract_quality_expected_dog_name(
        self,
        eval_case: RagEvalCase,
    ) -> str | None:
        """
        提取用于运行时质量判断的 expected_dog_name。

        参数含义：
            eval_case:
                RAG 评估用例。

        返回值含义：
            str | None:
                如果能确定唯一目标犬种，则返回犬种名。
                否则返回 None。

        说明：
            evaluate_rag_context_quality 当前只接受一个 expected_dog_name。
            如果 expected_dog_names 有多个，说明这是推荐/列表型问题，
            不强行传入单个 dog_name，避免误判 metadata_mismatch。
        """

        if len(eval_case.expected_dog_names) == 1:
            return eval_case.expected_dog_names[0]

        expected_dog_name = eval_case.expected_filters.get(
            "dog_name",
        )

        if expected_dog_name:
            return str(expected_dog_name)

        return None

    def _resolve_passed(
        self,
        eval_case: RagEvalCase,
        hit: bool,
        filter_matched: bool,
        empty_retrieval: bool,
        quality_is_usable: bool,
    ) -> bool:
        """
        判断当前评估用例是否通过。

        参数含义：
            eval_case:
                RAG 评估用例。

            hit:
                是否命中期望犬种。

            filter_matched:
                filters 是否匹配。

            empty_retrieval:
                是否为空召回。

            quality_is_usable:
                复用运行时质量评估后的可用性判断。

        返回值含义：
            bool:
                True 表示当前 case 通过。
                False 表示当前 case 未通过。

        规则：
            1. 如果 expected_dog_names 非空，则必须 hit=True。
            2. expected_filters 必须匹配。
            3. 不能空召回。
            4. 如果 require_quality_usable=True，则质量评估必须可用。
        """

        if eval_case.expected_dog_names and not hit:
            return False

        if not filter_matched:
            return False

        if empty_retrieval:
            return False

        if self.require_quality_usable and not quality_is_usable:
            return False

        return True

    def _elapsed_ms(
        self,
        started_at: float,
    ) -> float:
        """
        计算耗时毫秒数。

        参数含义：
            started_at:
                time.perf_counter 记录的开始时间。

        返回值含义：
            float:
                从 started_at 到当前时间的耗时，单位毫秒。
        """

        return round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )