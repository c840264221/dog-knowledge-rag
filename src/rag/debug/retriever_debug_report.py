"""
Retriever / RAG Debug Report。

功能：
    本模块用于把 RAG 链路中的结构化对象格式化成可读调试报告。

    当前支持两类报告：

    1. Retriever Debug Report（检索调试报告）
       用于展示单次 Retriever 检索结果：
       - 用户问题
       - metadata filter
       - RagContext 状态
       - context_text 预览
       - chunks 详情
       - retrieval_score / rerank_score / final_score
       - reason

    2. RAG Debug Report（RAG 链路调试报告）
       用于展示一次完整 Graph 执行后的核心 RAG 状态：
       - route_decision
       - rag_query
       - rag_context
       - retrieval_quality
       - answer_strategy
       - final_answer

设计原则：
    1. 不执行检索。
    2. 不调用 LLM。
    3. 不修改 RagContext。
    4. 不修改 DogState。
    5. 只负责把已有状态格式化成报告。
    6. 优先支持新版 RagContext。
    7. docs 只作为旧链路兼容 fallback。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pathlib import Path

from src.rag.schemas import (
    RagChunk,
    RagContext,
    RagQuery,
    RagRetrievedChunk,
)

import shutil
from datetime import datetime


MetadataFilter = dict[str, Any]


def build_retriever_debug_report(
        query: RagQuery | str | Mapping[str, Any],
        context: RagContext | Mapping[str, Any],
        metadata_filter: MetadataFilter | None = None,
        max_context_chars: int = 1200,
) -> str:
    """
    构建 Retriever Debug Report（检索调试报告）。

    功能：
        将一次 Retriever 检索结果格式化成可读文本。
        这个函数主要用于调试 RAG 检索阶段。

        当前兼容：
        1. query 是 RagQuery。
        2. query 是 dict。
        3. query 是 str。
        4. context 是 RagContext。
        5. context 是 dict。

    参数：
        query:
            用户查询对象、查询字典或查询字符串。

        context:
            Retriever 返回的 RAG 上下文对象。
            可以是 RagContext，也可以是 model_dump 后的 dict。

        metadata_filter:
            本次检索使用的 metadata filter。
            如果不传，会优先尝试从 query.filters 中读取。

        max_context_chars:
            context_text 最大展示长度。

    返回值：
        str:
            格式化后的调试报告文本。

    专业名词：
        Retriever：
            检索器。负责从向量库或文档索引中召回候选文本块。

        RagContext：
            RAG 上下文。保存检索结果、上下文文本、来源数量和状态。
    """

    query_text = _get_query_text(
        query=query,
    )

    resolved_filter = (
        metadata_filter
        if metadata_filter is not None
        else _get_query_filters(
            query=query,
        )
    )

    fallback_detected = _detect_fallback(
        context=context,
    )

    chunks = _get_context_chunks(
        context=context,
    )

    context_text = _get_context_text(
        context=context,
    )

    blocks: list[str] = [
        "========== Retriever Debug Report ==========",
        "",
        "[Question]",
        query_text,
        "",
        "[Metadata Filter]",
        _format_metadata_filter(
            metadata_filter=resolved_filter,
        ),
        "",
        "[Context Summary]",
        f"status: {_get_context_status(context=context)}",
        f"chunk_count: {len(chunks)}",
        f"source_count: {_get_context_source_count(context=context)}",
        f"has_context_text: {bool(context_text.strip())}",
        f"fallback_detected: {fallback_detected}",
        "",
        "[Context Text Preview]",
        _truncate_text(
            text=context_text,
            max_length=max_context_chars,
        ),
        "",
    ]

    if not chunks:
        blocks.extend(
            [
                "[Chunks]",
                "没有召回任何 chunk。",
            ]
        )

        return "\n".join(
            blocks
        )

    blocks.append(
        "[Chunks]"
    )

    for index, retrieved_chunk in enumerate(
            chunks,
            start=1,
    ):
        blocks.append(
            _format_retrieved_chunk(
                retrieved_chunk=retrieved_chunk,
                index=index,
            )
        )

    return "\n".join(
        blocks
    )


def build_rag_debug_report(
        state: Mapping[str, Any],
        max_context_chars: int = 1200,
        max_answer_chars: int = 1200,
) -> str:
    """
    构建 RAG Debug Report（RAG 链路调试报告）。

    功能：
        根据一次 LangGraph 执行后的 DogState，
        生成完整的 RAG 调试报告。

        这个函数用于比 build_retriever_debug_report 更高一层的场景。
        它不仅展示 Retriever 检索结果，还展示：
        1. 路由结果。
        2. RagQuery。
        3. RagContext。
        4. retrieval_quality。
        5. answer_strategy。
        6. final_answer。

    参数：
        state:
            LangGraph 执行后的状态。
            只要求它是 Mapping，不强依赖 DogState 类型，避免循环依赖。

        max_context_chars:
            RagContext.context_text 最大展示长度。

        max_answer_chars:
            final_answer 最大展示长度。

    返回值：
        str:
            Markdown / 纯文本混合格式的 RAG 调试报告。

    专业名词：
        RAG Debug Report：
            RAG 调试报告。用于观察一次 RAG 问答背后的完整链路。

        DogState：
            Dog Agent Framework 的主状态对象。
    """

    rag_query = _model_to_dict(
        state.get(
            "rag_query",
            {},
        )
    )

    rag_context = state.get(
        "rag_context",
        {},
    )

    rag_context_dict = _model_to_dict(
        rag_context
    )

    retrieval_quality = _model_to_dict(
        state.get(
            "retrieval_quality",
            {},
        )
    )

    answer_strategy = _model_to_dict(
        state.get(
            "answer_strategy",
            {},
        )
    )

    route_decision = _model_to_dict(
        state.get(
            "route_decision",
            {},
        )
    )

    final_answer = state.get(
        "final_answer",
        state.get(
            "answer",
            "",
        ),
    )

    compact_summary = build_compact_rag_debug_summary(
        state=state,
    )

    blocks: list[str] = [
        "# RAG Debug Report",
        "",
        "## 1. Compact Summary",
        _format_json_block(
            compact_summary
        ),
        "",
        "## 2. User Question",
        "```text",
        _safe_text(
            state.get(
                "question",
                "",
            )
        ),
        "```",
        "",
        "## 3. Route Decision",
        _format_json_block(
            route_decision
        ),
        "",
        "## 4. RagQuery",
        _format_json_block(
            rag_query
        ),
        "",
        "## 5. RagContext Overview",
        _format_json_block(
            {
                "status": rag_context_dict.get(
                    "status",
                    "",
                ),
                "source_count": rag_context_dict.get(
                    "source_count",
                    0,
                ),
                "chunk_count": len(
                    _get_context_chunks(
                        context=rag_context,
                    )
                ),
                "has_context_text": bool(
                    _get_context_text(
                        context=rag_context,
                    ).strip()
                ),
            }
        ),
        "",
        "## 6. RagContext Context Text Preview",
        "```text",
        _truncate_text(
            text=_get_context_text(
                context=rag_context,
            ),
            max_length=max_context_chars,
        ),
        "```",
        "",
        "## 7. Retrieval Quality",
        _format_json_block(
            retrieval_quality
        ),
        "",
        "## 8. Top Items From Retrieval Quality",
        _format_retrieval_quality_top_items(
            retrieval_quality=retrieval_quality,
        ),
        "",
        "## 9. Retriever Detail Report",
        build_retriever_debug_report(
            query=rag_query,
            context=rag_context,
            metadata_filter=rag_query.get(
                "filters",
                state.get(
                    "filters",
                    {},
                ),
            ),
            max_context_chars=max_context_chars,
        ),
        "",
        "## 10. Answer Strategy",
        _format_json_block(
            answer_strategy
        ),
        "",
        "## 11. Final Answer Preview",
        "```text",
        _truncate_text(
            text=final_answer,
            max_length=max_answer_chars,
        ),
        "```",
    ]

    return "\n".join(
        blocks
    )


def build_compact_rag_debug_summary(
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构建精简版 RAG 调试摘要。

    功能：
        从 DogState 中提取最关键的调试字段，
        返回一个适合日志打印或快速查看的字典。

    参数：
        state:
            当前 LangGraph 状态。

    返回值：
        dict[str, Any]:
            精简调试摘要。
    """

    rag_query = _model_to_dict(
        state.get(
            "rag_query",
            {},
        )
    )

    rag_context = state.get(
        "rag_context",
        {},
    )

    rag_context_dict = _model_to_dict(
        rag_context
    )

    retrieval_quality = _model_to_dict(
        state.get(
            "retrieval_quality",
            {},
        )
    )

    answer_strategy = _model_to_dict(
        state.get(
            "answer_strategy",
            {},
        )
    )

    route_decision = _model_to_dict(
        state.get(
            "route_decision",
            {},
        )
    )

    return {
        "question": state.get(
            "question",
            "",
        ),
        "route": route_decision.get(
            "route",
            state.get(
                "next_agent",
                "",
            ),
        ),
        "current_agent": state.get(
            "current_agent",
            "",
        ),
        "next_agent": state.get(
            "next_agent",
            "",
        ),
        "intent": state.get(
            "intent",
            rag_query.get(
                "intent",
                "",
            ),
        ),
        "filters": rag_query.get(
            "filters",
            state.get(
                "filters",
                {},
            ),
        ),
        "rag_context_status": rag_context_dict.get(
            "status",
            "",
        ),
        "rag_context_source_count": rag_context_dict.get(
            "source_count",
            0,
        ),
        "rag_context_chunk_count": len(
            _get_context_chunks(
                context=rag_context,
            )
        ),
        "retrieved_count": retrieval_quality.get(
            "retrieved_count",
            0,
        ),
        "reranked_count": retrieval_quality.get(
            "reranked_count",
            0,
        ),
        "quality_status": retrieval_quality.get(
            "quality_status",
            "",
        ),
        "quality_score": retrieval_quality.get(
            "quality_score",
            None,
        ),
        "is_usable": retrieval_quality.get(
            "is_usable",
            None,
        ),
        "failure_type": retrieval_quality.get(
            "failure_type",
            "",
        ),
        "decision": retrieval_quality.get(
            "decision",
            "",
        ),
        "answer_task_type": answer_strategy.get(
            "task_type",
            "",
        ),
        "answer_style": answer_strategy.get(
            "answer_style",
            "",
        ),
        "has_final_answer": bool(
            state.get(
                "final_answer",
                state.get(
                    "answer",
                    "",
                ),
            )
        ),
    }


def _model_to_dict(
        value: Any,
) -> dict[str, Any]:
    """
    将对象安全转换成字典。

    功能：
        兼容 dict、Pydantic 对象和 None。

        当前项目里，RagQuery / RagContext 在类型声明中是 Pydantic 对象，
        但因为 SQLite checkpoint，运行时也可能是 model_dump 后的 dict。
        所以这里必须同时兼容对象和 dict。

    参数：
        value:
            任意对象。

    返回值：
        dict[str, Any]:
            字典。
    """

    if isinstance(
            value,
            Mapping,
    ):
        return dict(
            value
        )

    if hasattr(
            value,
            "model_dump",
    ):
        dumped = value.model_dump()

        if isinstance(
                dumped,
                dict,
        ):
            return dumped

    return {}


def _safe_text(
        value: Any,
        default: str = "",
) -> str:
    """
    安全转换字符串。

    功能：
        将任意值转换成字符串。
        如果值为 None，则返回 default。

    参数：
        value:
            原始值。

        default:
            默认字符串。

    返回值：
        str:
            转换后的字符串。
    """

    if value is None:
        return default

    return str(
        value
    )


def _truncate_text(
        text: Any,
        max_length: int = 800,
) -> str:
    """
    截断文本。

    功能：
        将长文本截断成适合日志和报告展示的长度。

    参数：
        text:
            原始文本。

        max_length:
            最大长度。

    返回值：
        str:
            截断后的文本。
    """

    normalized = _safe_text(
        text
    ).strip()

    if len(
            normalized
    ) <= max_length:
        return normalized

    return (
        normalized[:max_length]
        + "\n\n...【内容已截断】"
    )


def _format_json_block(
        value: Any,
) -> str:
    """
    格式化 JSON 代码块。

    功能：
        将 dict、list 或其他对象格式化成 Markdown JSON 代码块。

    参数：
        value:
            要格式化的值。

    返回值：
        str:
            Markdown JSON 代码块。
    """

    try:
        json_text = json.dumps(
            value,
            ensure_ascii=False,
            indent=4,
            default=str,
        )

    except TypeError:
        json_text = _safe_text(
            value
        )

    return (
        "```json\n"
        f"{json_text}\n"
        "```"
    )


def _get_query_text(
        query: RagQuery | str | Mapping[str, Any],
) -> str:
    """
    获取查询文本。

    功能：
        兼容 RagQuery、dict 和 str 三种输入格式。

    参数：
        query:
            查询对象、查询字典或查询字符串。

    返回值：
        str:
            用户问题文本。
    """

    if isinstance(
            query,
            str,
    ):
        return query

    query_dict = _model_to_dict(
        query
    )

    if query_dict:
        return _safe_text(
            query_dict.get(
                "question",
                "",
            )
        )

    return _safe_text(
        getattr(
            query,
            "question",
            "",
        )
    )


def _get_query_filters(
        query: RagQuery | str | Mapping[str, Any],
) -> MetadataFilter:
    """
    获取查询 filters。

    功能：
        从 RagQuery 或 dict 中读取 filters。
        如果 query 是 str，则返回空 dict。

    参数：
        query:
            查询对象、查询字典或查询字符串。

    返回值：
        dict[str, Any]:
            metadata filter。
    """

    if isinstance(
            query,
            str,
    ):
        return {}

    query_dict = _model_to_dict(
        query
    )

    filters = query_dict.get(
        "filters",
        {},
    )

    if isinstance(
            filters,
            Mapping,
    ):
        return dict(
            filters
        )

    return {}


def _get_context_status(
        context: RagContext | Mapping[str, Any],
) -> str:
    """
    获取 RagContext.status。

    参数：
        context:
            RagContext 对象或 dict。

    返回值：
        str:
            上下文状态。
    """

    context_dict = _model_to_dict(
        context
    )

    return _safe_text(
        context_dict.get(
            "status",
            "",
        )
    )


def _get_context_text(
        context: RagContext | Mapping[str, Any],
) -> str:
    """
    获取 RagContext.context_text。

    参数：
        context:
            RagContext 对象或 dict。

    返回值：
        str:
            上下文文本。
    """

    context_dict = _model_to_dict(
        context
    )

    return _safe_text(
        context_dict.get(
            "context_text",
            "",
        )
    )


def _get_context_source_count(
        context: RagContext | Mapping[str, Any],
) -> int:
    """
    获取 RagContext.source_count。

    参数：
        context:
            RagContext 对象或 dict。

    返回值：
        int:
            来源数量。
    """

    context_dict = _model_to_dict(
        context
    )

    raw_value = context_dict.get(
        "source_count",
        0,
    )

    try:
        return int(
            raw_value
        )

    except (
            TypeError,
            ValueError,
    ):
        return 0


def _get_context_chunks(
        context: RagContext | Mapping[str, Any],
) -> list[Any]:
    """
    获取 RagContext.chunks。

    功能：
        兼容 RagContext 对象和 dict。

    参数：
        context:
            RagContext 对象或 dict。

    返回值：
        list[Any]:
            chunks 列表。
    """

    context_dict = _model_to_dict(
        context
    )

    chunks = context_dict.get(
        "chunks",
        [],
    )

    if isinstance(
            chunks,
            list,
    ):
        return chunks

    return []


def _detect_fallback(
        context: RagContext | Mapping[str, Any],
) -> bool:
    """
    检测本次检索是否触发 fallback。

    功能：
        遍历 RagContext.chunks 中每条 RagRetrievedChunk 的 reason。
        如果 reason 中包含 fallback 字样，则认为本次检索触发了 fallback 降级召回。

    参数：
        context:
            RagContext 对象或 dict。

    返回值：
        bool:
            True 表示检测到 fallback。
            False 表示没有检测到 fallback。
    """

    chunks = _get_context_chunks(
        context=context,
    )

    for retrieved_chunk in chunks:
        chunk_dict = _normalize_retrieved_chunk(
            retrieved_chunk=retrieved_chunk,
        )

        reason = _safe_text(
            chunk_dict.get(
                "reason",
                "",
            )
        )

        if "fallback" in reason.lower():
            return True

    return False


def _normalize_retrieved_chunk(
        retrieved_chunk: Any,
) -> dict[str, Any]:
    """
    归一化 RagRetrievedChunk。

    功能：
        将 RagRetrievedChunk 对象或 dict 转成统一字典。

    参数：
        retrieved_chunk:
            RagRetrievedChunk 对象或 dict。

    返回值：
        dict[str, Any]:
            归一化后的 retrieved chunk 字典。
    """

    if isinstance(
            retrieved_chunk,
            RagRetrievedChunk,
    ):
        return retrieved_chunk.model_dump()

    return _model_to_dict(
        retrieved_chunk
    )


def _normalize_chunk(
        chunk: Any,
) -> dict[str, Any]:
    """
    归一化 RagChunk。

    功能：
        将 RagChunk 对象或 dict 转成统一字典。

    参数：
        chunk:
            RagChunk 对象或 dict。

    返回值：
        dict[str, Any]:
            归一化后的 chunk 字典。
    """

    if isinstance(
            chunk,
            RagChunk,
    ):
        return chunk.model_dump()

    return _model_to_dict(
        chunk
    )


def _format_retrieved_chunk(
        retrieved_chunk: RagRetrievedChunk | Mapping[str, Any],
        index: int,
) -> str:
    """
    格式化单条召回结果。

    功能：
        将 RagRetrievedChunk 中的关键信息转换成可读文本。

        当前展示：
        1. chunk_id。
        2. doc_id。
        3. dog_name。
        4. title。
        5. source。
        6. chunk_index。
        7. retrieval_score。
        8. rerank_score。
        9. final_score。
        10. reason。
        11. metadata 摘要。
        12. content_preview。

    参数：
        retrieved_chunk:
            单条召回结果对象或字典。

        index:
            当前召回结果序号，从 1 开始。

    返回值：
        str:
            单条 chunk 的调试文本。
    """

    retrieved_chunk_dict = _normalize_retrieved_chunk(
        retrieved_chunk=retrieved_chunk,
    )

    chunk_dict = _normalize_chunk(
        chunk=retrieved_chunk_dict.get(
            "chunk",
            {},
        )
    )

    metadata = chunk_dict.get(
        "metadata",
        {},
    )

    if not isinstance(
            metadata,
            Mapping,
    ):
        metadata = {}

    dog_name = metadata.get(
        "dog_name",
        "unknown",
    )

    metadata_text = _format_metadata_preview(
        metadata=dict(
            metadata
        ),
    )

    return "\n".join(
        [
            "",
            f"----- Chunk {index} -----",
            f"chunk_id: {_safe_text(chunk_dict.get('chunk_id', ''))}",
            f"doc_id: {_safe_text(chunk_dict.get('doc_id', ''))}",
            f"dog_name: {_safe_text(dog_name)}",
            f"title: {_safe_text(chunk_dict.get('title', ''))}",
            f"source: {_safe_text(chunk_dict.get('source', ''))}",
            f"chunk_index: {_safe_text(chunk_dict.get('chunk_index', ''))}",
            f"retrieval_score: {_safe_text(retrieved_chunk_dict.get('retrieval_score', ''))}",
            f"rerank_score: {_safe_text(retrieved_chunk_dict.get('rerank_score', ''))}",
            f"final_score: {_safe_text(retrieved_chunk_dict.get('final_score', ''))}",
            f"reason: {_safe_text(retrieved_chunk_dict.get('reason', ''))}",
            "metadata:",
            metadata_text,
            "content_preview:",
            _truncate_text(
                text=chunk_dict.get(
                    "content",
                    "",
                ),
                max_length=300,
            ),
        ]
    )


def _format_retrieval_quality_top_items(
        retrieval_quality: Mapping[str, Any],
) -> str:
    """
    格式化 retrieval_quality.top_items。

    功能：
        将 diagnostics 中的 top_items 转成可读文本。
        top_items 通常来自 retrieve / rerank 阶段的检索诊断信息。

    参数：
        retrieval_quality:
            检索质量 / 检索诊断字典。

    返回值：
        str:
            top_items 报告文本。

    专业名词：
        Top Items：
            排名前几条结果。通常表示最重要的上下文片段。
    """

    top_items = retrieval_quality.get(
        "top_items",
        [],
    )

    if not top_items:
        return "暂无 top_items。"

    blocks: list[str] = []

    for item in top_items:
        item_dict = _model_to_dict(
            item
        )

        if not item_dict and isinstance(
                item,
                Mapping,
        ):
            item_dict = dict(
                item
            )

        rank = item_dict.get(
            "rank",
            "",
        )

        dog_name = item_dict.get(
            "dog_name",
            "",
        )

        source = item_dict.get(
            "source",
            "",
        )

        section_title = item_dict.get(
            "section_title",
            "",
        )

        score = item_dict.get(
            "score",
            "",
        )

        preview = item_dict.get(
            "preview",
            "",
        )

        blocks.append(
            "\n".join(
                [
                    f"### Top {rank}",
                    f"- dog_name: {dog_name}",
                    f"- source: {source}",
                    f"- section_title: {section_title}",
                    f"- score: {score}",
                    "",
                    "```text",
                    _truncate_text(
                        text=preview,
                        max_length=300,
                    ),
                    "```",
                ]
            )
        )

    return "\n\n".join(
        blocks
    )


def _format_metadata_filter(
        metadata_filter: MetadataFilter | None,
) -> str:
    """
    格式化 metadata filter。

    功能：
        将 metadata filter 转换成缩进后的 JSON 字符串，
        方便在终端或日志中观察 Parser 生成的结构化过滤条件。

    参数：
        metadata_filter:
            metadata filter。

    返回值：
        str:
            格式化后的 filter 文本。
    """

    if not metadata_filter:
        return "无 metadata filter。"

    return json.dumps(
        metadata_filter,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _format_metadata_preview(
        metadata: dict[str, Any],
) -> str:
    """
    格式化 metadata 摘要。

    功能：
        从完整 metadata 中挑选调试时最重要的字段，
        并转换成多行文本，避免终端输出过长。

    参数：
        metadata:
            chunk.metadata 元数据字典。

    返回值：
        str:
            metadata 摘要文本。
    """

    if not metadata:
        return "  无 metadata。"

    important_fields = [
        "dog_name",
        "size",
        "height",
        "weight",
        "lifespan",
        "energy",
        "energy_level",
        "barking",
        "barking_level",
        "trainability",
        "trainability_level",
        "shedding",
        "shedding_level",
        "good_for_apartment",
        "good_for_beginner",
        "source",
        "relative_path",
        "file_name",
        "section_title",
        "chunk_index",
    ]

    lines: list[str] = []

    for field_name in important_fields:
        if field_name not in metadata:
            continue

        value = metadata.get(
            field_name,
        )

        lines.append(
            f"  {field_name}: {_format_value(value=value)}"
        )

    if not lines:
        return "  metadata 中没有命中预设的重要字段。"

    return "\n".join(
        lines
    )


def _format_value(
        value: Any,
) -> str:
    """
    格式化 metadata 字段值。

    功能：
        将 Python 值转换成适合调试报告展示的字符串。
        主要处理 bool、None、list、dict 等常见类型。

    参数：
        value:
            metadata 中的原始字段值。

    返回值：
        str:
            格式化后的文本。
    """

    if value is True:
        return "True"

    if value is False:
        return "False"

    if value is None:
        return "None"

    if isinstance(
            value,
            list,
    ):
        return ", ".join(
            str(
                item
            )
            for item in value
        )

    if isinstance(
            value,
            dict,
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(
        value
    )

def resolve_trace_id_for_report(
        state: Mapping[str, Any],
        trace_id: str | None = None,
) -> str:
    """
    解析 RAG Debug Report 使用的 trace_id。

    功能：
        优先使用外部传入的 trace_id。
        如果没有传入，则从 state["trace_id"] 中读取。
        如果仍然没有，则使用 unknown_trace。

    参数：
        state:
            当前 DogState 或状态字典。

        trace_id:
            外部传入的 trace_id。

    返回值：
        str:
            可用于报告文件名的 trace_id。
    """

    if trace_id:
        return str(
            trace_id
        )

    state_trace_id = state.get(
        "trace_id",
        "",
    )

    if state_trace_id:
        return str(
            state_trace_id
        )

    return "unknown_trace"


def sanitize_report_file_name(
        value: str,
) -> str:
    """
    清理报告文件名。

    功能：
        将 trace_id 转换成安全文件名。
        避免出现路径分隔符或特殊字符导致文件写入异常。

    参数：
        value:
            原始文件名文本。

    返回值：
        str:
            安全文件名。
    """

    safe_chars: list[str] = []

    for char in str(
            value
            or ""
    ):
        if char.isalnum() or char in {
            "-",
            "_",
        }:
            safe_chars.append(
                char
            )
        else:
            safe_chars.append(
                "_"
            )

    file_name = "".join(
        safe_chars
    ).strip(
        "_"
    )

    if not file_name:
        return "unknown_trace"

    return file_name


def save_rag_debug_report(
        state: Mapping[str, Any],
        report_dir: str | Path,
        trace_id: str | None = None,
        max_context_chars: int = 1200,
        max_answer_chars: int = 1200,
        use_date_dir: bool = True,
) -> Path:
    """
    保存 RAG Debug Report 到 Markdown 文件。

    功能：
        根据当前 DogState 构建 RAG Debug Report，
        并保存到指定目录下。

        支持两种目录模式：

        1. 按日期分目录：
            logs/report/rag_debug/2026-06-27/<trace_id>.md

        2. 不按日期分目录：
            logs/report/rag_debug/<trace_id>.md

        如果目录不存在，会自动创建。

    参数：
        state:
            当前 LangGraph 最终状态。

        report_dir:
            报告保存根目录。

        trace_id:
            当前请求 trace_id。
            如果不传，则从 state["trace_id"] 中读取。

        max_context_chars:
            RagContext.context_text 最大展示长度。

        max_answer_chars:
            final_answer 最大展示长度。

        use_date_dir:
            是否按日期分目录保存。

    返回值：
        Path:
            实际写入的报告文件路径。
    """

    resolved_trace_id = resolve_trace_id_for_report(
        state=state,
        trace_id=trace_id,
    )

    safe_trace_id = sanitize_report_file_name(
        resolved_trace_id
    )

    output_dir = resolve_rag_debug_report_output_dir(
        report_dir=report_dir,
        use_date_dir=use_date_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = output_dir / f"{safe_trace_id}.md"

    report_text = build_rag_debug_report(
        state=state,
        max_context_chars=max_context_chars,
        max_answer_chars=max_answer_chars,
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    if not report_path.exists():
        raise RuntimeError(
            f"RAG Debug Report 写入后未找到文件: {report_path}"
        )

    return report_path


def resolve_rag_debug_report_output_dir(
        report_dir: str | Path,
        use_date_dir: bool = True,
        now: datetime | None = None,
) -> Path:
    """
    解析 RAG Debug Report 输出目录。

    功能：
        根据配置决定 report 文件是否按日期分目录保存。

        示例：
            use_date_dir=True:
                logs/report/rag_debug/2026-06-27/

            use_date_dir=False:
                logs/report/rag_debug/

    参数：
        report_dir:
            RAG Debug Report 根目录。

        use_date_dir:
            是否按日期分目录。

        now:
            当前时间。
            主要用于测试，正常业务不需要传。

    返回值：
        Path:
            最终输出目录。

    专业名词：
        Date Partition:
            日期分区。按日期拆分目录，方便检索和清理历史文件。
    """

    base_dir = Path(
        report_dir
    ).expanduser()

    if not use_date_dir:
        return base_dir

    current_time = now or datetime.now()

    date_dir_name = current_time.strftime(
        "%Y-%m-%d"
    )

    return base_dir / date_dir_name


def cleanup_old_rag_debug_reports(
        report_dir: str | Path,
        retention_days: int = 7,
        now: datetime | None = None,
) -> int:
    """
    清理过期的 RAG Debug Report 日期目录。

    功能：
        只清理 report_dir 下符合 YYYY-MM-DD 命名格式的日期目录。
        不会删除其他非日期目录，避免误删。

    参数：
        report_dir:
            RAG Debug Report 根目录。

        retention_days:
            保留天数。
            小于等于 0 时不执行清理。

        now:
            当前时间。
            主要用于测试，正常业务不需要传。

    返回值：
        int:
            实际删除的日期目录数量。

    专业名词：
        Retention:
            保留策略。表示日志或报告保留多久。

        Cleanup:
            清理。删除过期文件或目录，避免磁盘无限增长。
    """

    if retention_days <= 0:
        return 0

    base_dir = Path(
        report_dir
    ).expanduser()

    if not base_dir.exists():
        return 0

    current_time = now or datetime.now()

    removed_count = 0

    for child in base_dir.iterdir():

        if not child.is_dir():
            continue

        try:
            folder_date = datetime.strptime(
                child.name,
                "%Y-%m-%d",
            )

        except ValueError:
            continue

        age_days = (
            current_time.date()
            - folder_date.date()
        ).days

        if age_days <= retention_days:
            continue

        shutil.rmtree(
            child
        )

        removed_count += 1

    return removed_count