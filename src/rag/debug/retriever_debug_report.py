"""
Retriever Debug Report。

Retriever Debug Report（检索调试报告）：
用于把 RagQuery、metadata filter、RagContext、RagRetrievedChunk
格式化成可读的调试文本。

当前模块职责：
1. 展示用户问题
2. 展示 Parser 生成的 metadata filter
3. 展示 RagContext 的整体状态
4. 展示是否触发 fallback
5. 展示每条召回 chunk 的关键信息
6. 展示每条召回结果的 reason

当前模块不负责：
1. 执行向量检索
2. 执行 metadata filter
3. 执行 fallback 策略
4. 调用 LLM 生成答案
5. 修改 RagContext 数据
"""

from __future__ import annotations

import json
from typing import Any

from src.rag.schemas import (
    RagContext,
    RagQuery,
    RagRetrievedChunk,
)


MetadataFilter = dict[str, Any]


def build_retriever_debug_report(
        query: RagQuery | str,
        context: RagContext,
        metadata_filter: MetadataFilter | None = None,
) -> str:
    """
    构建 Retriever Debug Report（检索调试报告）。

    功能：
        将一次 RAG 检索结果格式化成可读文本，方便在命令行中调试。
        报告中会包含用户问题、metadata filter、context 状态、
        fallback 检测结果、召回 chunk 数量、source_count、
        以及每条召回结果的详细信息。

    参数：
        query: RagQuery | str
            用户查询对象或查询字符串。
            如果是 RagQuery，则读取 query.question。
            如果是 str，则直接作为问题文本。

        context: RagContext
            Retriever 返回的 RAG 上下文对象。
            里面包含 context_text、chunks、source_count、status 等字段。

        metadata_filter: MetadataFilter | None
            本次检索使用的 metadata filter。
            一般可以传入 parser 生成的 query.filters，
            或 Retriever 实际使用的 filter。

    返回值：
        str：
            格式化后的调试报告文本。
    """

    query_text = _get_query_text(
        query=query,
    )

    fallback_detected = _detect_fallback(
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
            metadata_filter=metadata_filter,
        ),
        "",
        "[Context Summary]",
        f"status: {context.status}",
        f"chunk_count: {len(context.chunks)}",
        f"source_count: {context.source_count}",
        f"fallback_detected: {fallback_detected}",
        "",
    ]

    if not context.chunks:
        blocks.extend(
            [
                "[Chunks]",
                "没有召回任何 chunk。",
            ]
        )

        return "\n".join(
            blocks,
        )

    blocks.append(
        "[Chunks]"
    )

    for index, retrieved_chunk in enumerate(
            context.chunks,
            start=1,
    ):
        blocks.append(
            _format_retrieved_chunk(
                retrieved_chunk=retrieved_chunk,
                index=index,
            )
        )

    return "\n".join(
        blocks,
    )

def _get_query_text(
        query: RagQuery | str,
) -> str:
    """
    获取查询文本。

    功能：
        兼容 RagQuery 和 str 两种输入格式。
        如果传入 RagQuery，则读取 question 字段。
        如果传入 str，则直接返回该字符串。

    参数：
        query: RagQuery | str
            查询对象或查询字符串。

    返回值：
        str：
            用户问题文本。
    """

    if isinstance(
            query,
            str,
    ):
        return query

    return query.question

def _format_metadata_filter(
        metadata_filter: MetadataFilter | None,
) -> str:
    """
    格式化 metadata filter。

    功能：
        将 metadata filter 转换成缩进后的 JSON 字符串，
        方便在终端中观察 Parser 生成的结构化过滤条件。

    参数：
        metadata_filter: MetadataFilter | None
            metadata filter（元数据过滤条件）。

    返回值：
        str：
            格式化后的 filter 文本。
            如果 filter 为空，则返回“无 metadata filter”。
    """

    if not metadata_filter:
        return "无 metadata filter。"

    return json.dumps(
        metadata_filter,
        ensure_ascii=False,
        indent=2,
    )

def _detect_fallback(
        context: RagContext,
) -> bool:
    """
    检测本次检索是否触发 fallback。

    功能：
        遍历 RagContext.chunks 中每条 RagRetrievedChunk 的 reason。
        如果 reason 中包含 Fallback 或 fallback 字样，
        则认为本次检索触发了 fallback 降级召回。

    参数：
        context: RagContext
            Retriever 返回的上下文对象。

    返回值：
        bool：
            True 表示检测到 fallback；
            False 表示没有检测到 fallback。
    """

    for retrieved_chunk in context.chunks:
        reason = retrieved_chunk.reason or ""

        if "fallback" in reason.lower():
            return True

    return False

def _format_retrieved_chunk(
        retrieved_chunk: RagRetrievedChunk,
        index: int,
) -> str:
    """
    格式化单条召回结果。

    功能：
        将 RagRetrievedChunk 中的关键信息转换成可读文本。
        主要展示 chunk_id、doc_id、dog_name、source、
        retrieval_score、final_score、reason 和 metadata 摘要。

    参数：
        retrieved_chunk: RagRetrievedChunk
            单条召回结果对象。

        index: int
            当前召回结果的序号，从 1 开始。

    返回值：
        str：
            单条 chunk 的调试文本。
    """

    chunk = retrieved_chunk.chunk

    dog_name = chunk.metadata.get(
        "dog_name",
        "unknown",
    )

    metadata_text = _format_metadata_preview(
        metadata=chunk.metadata,
    )

    return "\n".join(
        [
            "",
            f"----- Chunk {index} -----",
            f"chunk_id: {chunk.chunk_id}",
            f"doc_id: {chunk.doc_id}",
            f"dog_name: {dog_name}",
            f"title: {chunk.title}",
            f"source: {chunk.source}",
            f"chunk_index: {chunk.chunk_index}",
            f"retrieval_score: {retrieved_chunk.retrieval_score}",
            f"rerank_score: {retrieved_chunk.rerank_score}",
            f"final_score: {retrieved_chunk.final_score}",
            f"reason: {retrieved_chunk.reason}",
            "metadata:",
            metadata_text,
            "content_preview:",
            chunk.content[:300],
        ]
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
        metadata: dict[str, Any]
            chunk.metadata 元数据字典。

    返回值：
        str：
            metadata 摘要文本。
    """

    if not metadata:
        return "  无 metadata。"

    important_fields = [
        "dog_name",
        "size",
        "energy_level",
        "barking_level",
        "trainability_level",
        "good_for_apartment",
        "good_for_beginner",
        "shedding_level",
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
        lines,
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
        value: Any
            metadata 中的原始字段值。

    返回值：
        str：
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
            str(item)
            for item in value
        )

    if isinstance(
            value,
            dict,
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(
        value,
    )