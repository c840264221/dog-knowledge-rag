from typing import (
    Any,
    Mapping,
)

from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
)

from collections.abc import Mapping

from langchain_core.documents import Document

from src.rag.schemas import (
    RagChunk,
    RagContext,
    RagRetrievedChunk,
)



class RetrievedItemDiagnostic(BaseModel):
    """
    单条检索结果诊断信息。

    功能：
        用于描述一条被 Retriever 或 Reranker 返回的文档信息。
        该对象不会参与业务决策，只用于 Debug、日志、评估和可观测。

    参数：
        rank:
            当前文档排名，从 1 开始。

        source:
            文档来源，例如 Markdown 文件路径、chunk source 或 metadata 中的 source 字段。

        dog_name:
            犬种名称。
            如果 metadata 中没有 dog_name，则为空字符串。

        section_title:
            文档所在章节标题。
            如果 metadata 中没有 section_title，则为空字符串。

        score:
            检索或重排序分数。
            如果当前文档没有分数，则为 None。

        metadata_keys:
            当前文档 metadata 中包含的字段名列表。

        preview:
            文档内容预览，默认截断后展示。

    返回值：
        RetrievedItemDiagnostic:
            单条检索结果诊断对象。

    专业名词：
        Rank:
            排名。表示当前文档在结果列表中的顺序。

        Metadata:
            元数据。描述文档的结构化信息，例如 dog_name、source、section_title。

        Score:
            分数。表示检索或重排序模型对文档相关性的评分。
    """

    rank: int = Field(
        ...,
        description="当前文档排名，从 1 开始",
    )

    source: str = Field(
        default="",
        description="文档来源",
    )

    dog_name: str = Field(
        default="",
        description="犬种名称",
    )

    section_title: str = Field(
        default="",
        description="章节标题",
    )

    score: float | None = Field(
        default=None,
        description="检索或重排序分数",
    )

    metadata_keys: list[str] = Field(
        default_factory=list,
        description="metadata 字段名列表",
    )

    preview: str = Field(
        default="",
        description="文档内容预览",
    )


class RetrievalDiagnostics(BaseModel):
    """
    RAG 检索诊断信息。

    功能：
        用于描述一次 RAG 检索链路的关键诊断信息。

        它可以被写入：
            state["retrieval_quality"]

        后续可以用于：
        1. Debug 调试。
        2. Trace 链路追踪。
        3. RAG Evaluation 检索评估。
        4. 判断是否需要 retry / ask_user / generate。
        5. 生成 RAG Debug Report。

    参数：
        question:
            用户原始问题。

        stage:
            当前诊断发生在哪个阶段，例如 retrieve、evaluate、rerank、generate。

        route:
            当前主图路由，例如 exact_agent、recommendation_agent。

        intent:
            当前解析出的用户意图，例如 ask_info、recommend、general。

        filters:
            当前检索过滤条件。

        filter_fields:
            filters 中出现过的字段名。

        requested_top_k:
            用户或系统请求的 top_k 数量。

        retrieved_count:
            初始检索返回的文档数量。

        reranked_count:
            rerank 后保留的文档数量。

        has_context:
            是否存在可用于生成回答的上下文。

        has_metadata_filter:
            是否使用了 metadata filter。

        has_dog_name_filter:
            是否使用了 dog_name 过滤。

        top_items:
            排名前几条文档摘要。

        failure_type:
            检索失败类型。
            例如 empty_result、weak_context、ambiguous_query。

        decision:
            当前阶段建议的下一步动作。
            例如 rerank、retry、ask_user、generate。

        reason:
            当前诊断或决策原因。

    返回值：
        RetrievalDiagnostics:
            检索诊断对象。

    专业名词：
        Diagnostics:
            诊断信息。用于定位系统运行过程中的问题。

        Retrieval Quality:
            检索质量。表示召回结果是否足够支持回答。

        Metadata Filter:
            元数据过滤。根据 dog_name、size、energy 等结构化字段过滤文档。

        Failure Type:
            失败类型。用于标准化表示检索失败原因。
    """

    question: str = Field(
        default="",
        description="用户原始问题",
    )

    stage: str = Field(
        default="unknown",
        description="当前诊断阶段",
    )

    route: str = Field(
        default="",
        description="当前主图路由",
    )

    intent: str = Field(
        default="",
        description="当前用户意图",
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="当前检索过滤条件",
    )

    filter_fields: list[str] = Field(
        default_factory=list,
        description="filters 中出现过的字段名",
    )

    requested_top_k: int | None = Field(
        default=None,
        description="请求的 top_k 数量",
    )

    retrieved_count: int = Field(
        default=0,
        description="初始检索数量",
    )

    reranked_count: int = Field(
        default=0,
        description="rerank 后数量",
    )

    has_context: bool = Field(
        default=False,
        description="是否有可用上下文",
    )

    has_metadata_filter: bool = Field(
        default=False,
        description="是否使用 metadata filter",
    )

    has_dog_name_filter: bool = Field(
        default=False,
        description="是否使用 dog_name filter",
    )

    top_items: list[RetrievedItemDiagnostic] = Field(
        default_factory=list,
        description="top 检索结果摘要",
    )

    failure_type: str = Field(
        default="",
        description="检索失败类型",
    )

    decision: str = Field(
        default="",
        description="建议的下一步动作",
    )

    reason: str = Field(
        default="",
        description="诊断原因",
    )

    context_text_preview: str = Field(
        default="",
        description="RagContext.context_text 的预览文本",
    )


def normalize_metadata(
        item: Any,
) -> dict[str, Any]:
    """
    归一化 metadata。

    功能：
        从不同类型的检索结果中提取 metadata。

        当前兼容：
        1. LangChain Document。
        2. RagRetrievedChunk 对象。
        3. RagChunk 对象。
        4. dict 形式 RagRetrievedChunk。
        5. dict 形式 RagChunk。
        6. 普通对象 metadata 属性。

    参数：
        item:
            检索结果对象。

    返回值：
        dict[str, Any]:
            metadata 字典。
    """

    if isinstance(
            item,
            Document,
    ):
        return dict(
            item.metadata or {}
        )

    if isinstance(
            item,
            RagRetrievedChunk,
    ):
        return dict(
            item.chunk.metadata or {}
        )

    if isinstance(
            item,
            RagChunk,
    ):
        return dict(
            item.metadata or {}
        )

    if isinstance(
            item,
            Mapping,
    ):
        metadata = item.get(
            "metadata",
            {},
        )

        if isinstance(
                metadata,
                Mapping,
        ) and metadata:
            return dict(
                metadata
            )

        chunk = item.get(
            "chunk",
            {},
        )

        chunk_dict = model_to_dict(
            chunk
        )

        chunk_metadata = chunk_dict.get(
            "metadata",
            {},
        )

        if isinstance(
                chunk_metadata,
                Mapping,
        ):
            return dict(
                chunk_metadata
            )

        return {}

    metadata = getattr(
        item,
        "metadata",
        {},
    )

    if isinstance(
            metadata,
            Mapping,
    ):
        return dict(
            metadata
        )

    return {}


def normalize_content(
        item: Any,
) -> str:
    """
    归一化内容文本。

    功能：
        从不同类型的检索结果中提取正文内容。

        当前兼容：
        1. Document.page_content。
        2. RagRetrievedChunk.chunk.content。
        3. RagChunk.content。
        4. dict["chunk"]["content"]。
        5. dict["content"]。
        6. dict["page_content"]。

    参数：
        item:
            检索结果对象。

    返回值：
        str:
            文本内容。
    """

    if isinstance(
            item,
            Document,
    ):
        return str(
            item.page_content or ""
        )

    if isinstance(
            item,
            RagRetrievedChunk,
    ):
        return str(
            item.chunk.content or ""
        )

    if isinstance(
            item,
            RagChunk,
    ):
        return str(
            item.content or ""
        )

    if isinstance(
            item,
            Mapping,
    ):
        direct_content = (
            item.get(
                "content"
            )
            or item.get(
                "page_content"
            )
        )

        if direct_content:
            return str(
                direct_content
            )

        chunk = item.get(
            "chunk",
            {},
        )

        chunk_dict = model_to_dict(
            chunk
        )

        return str(
            chunk_dict.get(
                "content",
                "",
            )
            or ""
        )

    content = getattr(
        item,
        "content",
        None,
    )

    if content is not None:
        return str(
            content
        )

    page_content = getattr(
        item,
        "page_content",
        None,
    )

    if page_content is not None:
        return str(
            page_content
        )

    return ""

def model_to_dict(
        value: Any,
) -> dict[str, Any]:
    """
    将对象归一化为字典。

    功能：
        兼容 Pydantic 对象、dict 和普通对象。

        在当前项目中，RagContext 在 DogState 类型里是对象，
        但因为 SQLite checkpoint，实际运行时也可能是 model_dump 后的 dict。
        所以 diagnostics 不能只支持其中一种。

    参数：
        value:
            任意对象，可能是 dict、Pydantic BaseModel 或 None。

    返回值：
        dict[str, Any]:
            归一化后的字典。
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


def normalize_rag_context(
        rag_context: Any,
) -> dict[str, Any]:
    """
    归一化 RagContext。

    功能：
        将 RagContext 对象或 dict 统一转换为 dict。

    参数：
        rag_context:
            可能是 RagContext 对象，也可能是 dict。

    返回值：
        dict[str, Any]:
            归一化后的 RagContext 字典。

    专业名词：
        RagContext：
            RAG 上下文对象，包含 context_text、chunks、source_count、status 等字段。
    """

    return model_to_dict(
        rag_context
    )


def normalize_diagnostic_text(
        value: Any,
        default: str = "",
) -> str:
    """
    归一化诊断文本字段。

    功能：
        将 diagnostics 中的文本字段统一转换成字符串。
        主要用于处理 None，避免 Pydantic 校验失败。

        例如：
            None -> ""
            "empty_result" -> "empty_result"
            123 -> "123"

    参数：
        value:
            原始值。

        default:
            当 value 为 None 时使用的默认值。

    返回值：
        str:
            归一化后的字符串。

    专业名词：
        Diagnostics：
            诊断信息。用于记录 RAG 链路运行过程中的状态、质量和决策原因。

        Normalization：
            归一化。把不同类型的数据统一转换成稳定格式。
    """

    if value is None:
        return default

    return str(
        value
    )

def extract_chunks_from_rag_context(
        rag_context: Any,
) -> list[Any]:
    """
    从 RagContext 中提取 chunks。

    功能：
        chunks 是结构化检索结果列表。
        diagnostics 的 top_items 应该优先从 chunks 中提取，
        因为 chunks 中保留了 dog_name、source、score、metadata 等结构化信息。

    参数：
        rag_context:
            RagContext 对象或 dict。

    返回值：
        list[Any]:
            chunks 列表。
    """

    normalized_context = normalize_rag_context(
        rag_context
    )

    chunks = normalized_context.get(
        "chunks",
        [],
    )

    if not isinstance(
            chunks,
            list,
    ):
        return []

    return list(
        chunks
    )

def resolve_diagnostic_items(
        rag_context: Any = None,
        docs: list[Any] | None = None,
) -> list[Any]:
    """
    解析 diagnostics 展示用的检索结果列表。

    功能：
        新版 RAG 优先使用 RagContext.chunks。
        只有当 RagContext.chunks 不存在时，才 fallback 到 docs。

    参数：
        rag_context:
            RagContext 对象或 dict。

        docs:
            旧版兼容 Document 列表。

    返回值：
        list[Any]:
            用于 summarize_retrieved_items 的结果列表。
    """

    chunks = extract_chunks_from_rag_context(
        rag_context
    )

    if chunks:
        return chunks

    return list(
        docs or []
    )

def has_rag_context(
        rag_context: Any = None,
        docs: list[Any] | None = None,
) -> bool:
    """
    判断当前是否存在可用上下文。

    功能：
        优先使用 RagContext.context_text 判断是否有最终上下文。
        如果 context_text 为空，再检查 RagContext.chunks。
        最后才 fallback 到 docs。

    参数：
        rag_context:
            RagContext 对象或 dict。

        docs:
            旧版兼容 Document 列表。

    返回值：
        bool:
            如果存在可用上下文，返回 True。
    """

    context_text = extract_context_text_from_rag_context(
        rag_context
    )

    if context_text.strip():
        return True

    chunks = extract_chunks_from_rag_context(
        rag_context
    )

    if chunks:
        return True

    return bool(
        docs
    )

def extract_context_text_from_rag_context(
        rag_context: Any,
) -> str:
    """
    从 RagContext 中提取 context_text。

    功能：
        context_text 是最终注入 LLM Prompt 的上下文文本。
        它适合用于判断当前是否存在可生成答案的上下文，
        也适合做整体上下文预览。

    参数：
        rag_context:
            RagContext 对象或 dict。

    返回值：
        str:
            context_text 字符串。
    """

    normalized_context = normalize_rag_context(
        rag_context
    )

    return str(
        normalized_context.get(
            "context_text",
            "",
        )
        or ""
    )


def resolve_visible_items_for_diagnostics(
        rag_context: Any = None,
        docs: list[Any] | None = None,
) -> list[Any]:
    """
    解析诊断展示用的检索结果。

    功能：
        v1.5 新版 RAG 中，诊断信息应该优先基于 rag_context.chunks。
        只有当 rag_context.chunks 不存在时，才回退使用 docs。

    参数：
        rag_context:
            新版 RAG 上下文。

        docs:
            旧版兼容 Document 列表。

    返回值：
        list[Any]:
            用于 summarize_retrieved_items 的结果列表。
    """

    chunks = extract_chunks_from_rag_context(
        rag_context
    )

    if chunks:
        return chunks

    return list(
        docs or []
    )


def has_context_for_diagnostics(
        rag_context: Any = None,
        docs: list[Any] | None = None,
) -> bool:
    """
    判断是否存在可用上下文。

    功能：
        优先根据 rag_context.context_text 和 rag_context.chunks 判断。
        只有新版 RAG 信息不存在时，才回退检查 docs。

    参数：
        rag_context:
            新版 RAG 上下文。

        docs:
            旧版兼容 Document 列表。

    返回值：
        bool:
            如果存在可用上下文，返回 True。
    """

    context_text = extract_context_text_from_rag_context(
        rag_context
    )

    if context_text.strip():
        return True

    chunks = extract_chunks_from_rag_context(
        rag_context
    )

    if chunks:
        return True

    return bool(
        docs
    )


def normalize_score(
        item: Any,
) -> float | None:
    """
    归一化检索或重排序分数。

    功能：
        从 RagRetrievedChunk、dict、Document.metadata 中提取分数。

        优先级：
        1. final_score
        2. normalized_rerank_score
        3. rerank_score
        4. retrieval_score
        5. score

    参数：
        item:
            检索结果对象。

    返回值：
        float | None:
            分数。
    """

    candidate_scores: list[Any] = []

    if isinstance(
            item,
            RagRetrievedChunk,
    ):
        candidate_scores.extend(
            [
                item.final_score,
                item.rerank_score,
                item.retrieval_score,
            ]
        )

    elif isinstance(
            item,
            Mapping,
    ):
        candidate_scores.extend(
            [
                item.get("final_score"),
                item.get("normalized_rerank_score"),
                item.get("rerank_score"),
                item.get("retrieval_score"),
                item.get("score"),
            ]
        )

    else:
        candidate_scores.extend(
            [
                getattr(item, "final_score", None),
                getattr(item, "normalized_rerank_score", None),
                getattr(item, "rerank_score", None),
                getattr(item, "retrieval_score", None),
                getattr(item, "score", None),
            ]
        )

    metadata = normalize_metadata(
        item
    )

    candidate_scores.extend(
        [
            metadata.get("final_score"),
            metadata.get("normalized_rerank_score"),
            metadata.get("rerank_score"),
            metadata.get("retrieval_score"),
            metadata.get("score"),
        ]
    )

    for raw_score in candidate_scores:

        if raw_score is None:
            continue

        try:
            return float(
                raw_score
            )

        except (
                TypeError,
                ValueError,
        ):
            continue

    return None


def truncate_text(
        text: str,
        max_length: int = 160,
) -> str:
    """
    截断文本。

    功能：
        将长文本截断为适合日志和诊断展示的短文本。

    参数：
        text:
            原始文本。

        max_length:
            最大长度。

    返回值：
        str:
            截断后的文本。
    """

    normalized = " ".join(
        str(
            text or ""
        ).split()
    )

    if len(
            normalized
    ) <= max_length:
        return normalized

    return (
        normalized[:max_length]
        + "..."
    )


def extract_filter_fields(
        filters: Mapping[str, Any] | None,
) -> list[str]:
    """
    提取 filters 中出现过的字段名。

    功能：
        支持两种 filter 结构：

        1. 扁平结构：
            {
                "dog_name": "Golden Retriever"
            }

        2. Chroma $and 结构：
            {
                "$and": [
                    {"dog_name": "Golden Retriever"},
                    {"energy": "high"}
                ]
            }

    参数：
        filters:
            检索过滤条件。

    返回值：
        list[str]:
            filters 中出现过的字段名列表。
    """

    if not isinstance(
            filters,
            Mapping,
    ):
        return []

    fields: list[str] = []

    for key, value in filters.items():

        if key == "$and" and isinstance(
                value,
                list,
        ):
            for child in value:
                fields.extend(
                    extract_filter_fields(
                        child
                    )
                )

            continue

        if key == "$or" and isinstance(
                value,
                list,
        ):
            for child in value:
                fields.extend(
                    extract_filter_fields(
                        child
                    )
                )

            continue

        if not key.startswith(
                "$"
        ):
            fields.append(
                str(
                    key
                )
            )

    return sorted(
        set(
            fields
        )
    )


def contains_filter_field(
        filters: Mapping[str, Any] | None,
        field_name: str,
) -> bool:
    """
    判断 filters 中是否包含指定字段。

    功能：
        用于判断是否使用了 dog_name、energy、size 等 metadata filter。

    参数：
        filters:
            检索过滤条件。

        field_name:
            要检查的字段名。

    返回值：
        bool:
            如果 filters 中包含该字段，返回 True。
            否则返回 False。
    """

    return field_name in extract_filter_fields(
        filters
    )


def resolve_route_from_state(
        state: Mapping[str, Any],
) -> str:
    """
    从 state 中解析当前路由。

    功能：
        优先读取 route_decision.route。
        如果不存在，则读取 next_agent。
        如果仍不存在，则读取 current_agent。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            当前路由名称。
    """

    route_decision = state.get(
        "route_decision",
        {},
    )

    if isinstance(
            route_decision,
            Mapping,
    ):
        route = route_decision.get(
            "route"
        )

        if route:
            return str(
                route
            )

    next_agent = state.get(
        "next_agent"
    )

    if next_agent:
        return str(
            next_agent
        )

    current_agent = state.get(
        "current_agent"
    )

    if current_agent:
        return str(
            current_agent
        )

    return ""


def resolve_filters_from_state(
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从 state 中解析检索 filters。

    功能：
        优先读取 rag_query.filters。
        如果没有 rag_query，则读取旧字段 filters。

        当前兼容：
        1. rag_query 是 dict。
        2. rag_query 是 Pydantic 对象。
        3. state["filters"] 旧字段。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            检索 filters。
    """

    rag_query = state.get(
        "rag_query"
    )

    if isinstance(
            rag_query,
            Mapping,
    ):
        filters = rag_query.get(
            "filters",
            {},
        )

        if isinstance(
                filters,
                dict,
        ):
            return filters

    rag_query_filters = getattr(
        rag_query,
        "filters",
        None,
    )

    if isinstance(
            rag_query_filters,
            dict,
    ):
        return rag_query_filters

    filters = state.get(
        "filters",
        {},
    )

    if isinstance(
            filters,
            dict,
    ):
        return filters

    return {}


def resolve_top_k_from_state(
        state: Mapping[str, Any],
) -> int | None:
    """
    从 state 中解析 top_k。

    功能：
        优先读取 rag_query.top_k。
        如果没有 rag_query，则读取 state["top_k"]。

    参数：
        state:
            当前 DogState。

    返回值：
        int | None:
            top_k 数量。
            如果不存在或无法转换，则返回 None。
    """

    rag_query = state.get(
        "rag_query"
    )

    if isinstance(
            rag_query,
            Mapping,
    ):
        top_k = rag_query.get(
            "top_k"
        )

        if top_k is not None:
            try:
                return int(
                    top_k
                )

            except (
                    TypeError,
                    ValueError,
            ):
                return None

    top_k = getattr(
        rag_query,
        "top_k",
        None,
    )

    if top_k is not None:
        try:
            return int(
                top_k
            )

        except (
                TypeError,
                ValueError,
        ):
            return None

    state_top_k = state.get(
        "top_k"
    )

    if state_top_k is not None:
        try:
            return int(
                state_top_k
            )

        except (
                TypeError,
                ValueError,
        ):
            return None

    return None


def summarize_retrieved_items(
        items: list[Any] | None,
        max_items: int = 5,
) -> list[RetrievedItemDiagnostic]:
    """
    汇总检索结果。

    功能：
        将 docs 或 reranked_docs 转成统一的诊断摘要。

    参数：
        items:
            检索结果列表。

        max_items:
            最多保留多少条摘要。

    返回值：
        list[RetrievedItemDiagnostic]:
            检索结果诊断摘要列表。
    """

    if not items:
        return []

    summaries: list[RetrievedItemDiagnostic] = []

    for index, item in enumerate(
            items[:max_items],
            start=1,
    ):
        metadata = normalize_metadata(
            item
        )

        content = normalize_content(
            item
        )

        summaries.append(
            RetrievedItemDiagnostic(
                rank=index,
                source=str(
                    metadata.get(
                        "source",
                        metadata.get(
                            "file_name",
                            metadata.get(
                                "relative_path",
                                "",
                            ),
                        ),
                    )
                    or ""
                ),
                dog_name=str(
                    metadata.get(
                        "dog_name",
                        metadata.get(
                            "name",
                            "",
                        ),
                    )
                    or ""
                ),
                section_title=str(
                    metadata.get(
                        "section_title",
                        metadata.get(
                            "heading",
                            "",
                        ),
                    )
                    or ""
                ),
                score=normalize_score(
                    item
                ),
                metadata_keys=sorted(
                    metadata.keys()
                ),
                preview=truncate_text(
                    content
                ),
            )
        )

    return summaries


def build_retrieval_diagnostics(
        state: Mapping[str, Any],
        stage: str,
        docs: list[Any] | None = None,
        reranked_docs: list[Any] | None = None,
        rag_context: Any = None,
        reranked_rag_context: Any = None,
        failure_type: str = "",
        decision: str = "",
        reason: str = "",
        max_items: int = 5,
) -> dict[str, Any]:
    """
    构建 RAG 检索诊断信息。

    功能：
        根据当前 DogState、RagContext、检索文档、重排序文档生成统一诊断字典。

        v1.5 新版 RAG 数据优先级：
        1. 优先使用 reranked_rag_context.chunks。
        2. 其次使用 rag_context.chunks。
        3. 再 fallback 到 reranked_docs。
        4. 最后 fallback 到 docs。

        context_text 的职责：
        1. 判断是否有最终上下文。
        2. 提供最终 Prompt 上下文预览。
        3. 不用于反向解析 top_items。

        chunks 的职责：
        1. 提供结构化 top_items。
        2. 提供 dog_name、source、score、metadata 等诊断字段。

    参数：
        state:
            当前 DogState。

        stage:
            当前诊断阶段。

        docs:
            旧版兼容 Document 列表。

        reranked_docs:
            旧版兼容 rerank 后 Document 列表。

        rag_context:
            retrieve / evaluate 阶段的 RagContext。

        reranked_rag_context:
            rerank 后的 RagContext。

        failure_type:
            失败类型。

        decision:
            当前建议动作。

        reason:
            诊断原因。

        max_items:
            top_items 最多数量。

    返回值：
        dict[str, Any]:
            可直接写入 DogState["retrieval_quality"] 的诊断字典。
    """

    filters = resolve_filters_from_state(
        state
    )

    filter_fields = extract_filter_fields(
        filters
    )

    base_rag_context = (
        rag_context
        if rag_context is not None
        else state.get(
            "rag_context",
            {},
        )
    )

    final_rag_context = (
        reranked_rag_context
        if reranked_rag_context is not None
        else base_rag_context
    )

    retrieved_items = resolve_diagnostic_items(
        rag_context=base_rag_context,
        docs=docs,
    )

    reranked_items = resolve_diagnostic_items(
        rag_context=final_rag_context,
        docs=reranked_docs,
    )

    visible_items = (
        reranked_items
        if reranked_items
        else retrieved_items
    )

    context_text = extract_context_text_from_rag_context(
        final_rag_context
    )

    retrieved_count = len(
        retrieved_items
    )

    has_rerank_input = (
        reranked_rag_context is not None
        or reranked_docs is not None
        or stage == "rerank"
    )

    reranked_count = (
        len(
            reranked_items
        )
        if has_rerank_input
        else 0
    )

    safe_stage = normalize_diagnostic_text(
        stage,
        default="unknown",
    )

    safe_failure_type = normalize_diagnostic_text(
        failure_type,
        default="",
    )

    safe_decision = normalize_diagnostic_text(
        decision,
        default="",
    )

    safe_reason = normalize_diagnostic_text(
        reason,
        default="",
    )

    diagnostics = RetrievalDiagnostics(
        question=str(
            state.get(
                "question",
                "",
            )
            or ""
        ),
        stage=safe_stage,
        route=resolve_route_from_state(
            state
        ),
        intent=str(
            state.get(
                "intent",
                "",
            )
            or ""
        ),
        filters=filters,
        filter_fields=filter_fields,
        requested_top_k=resolve_top_k_from_state(
            state
        ),
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        has_context=has_rag_context(
            rag_context=final_rag_context,
            docs=visible_items,
        ),
        has_metadata_filter=bool(
            filter_fields
        ),
        has_dog_name_filter=contains_filter_field(
            filters=filters,
            field_name="dog_name",
        ),
        top_items=summarize_retrieved_items(
            items=visible_items,
            max_items=max_items,
        ),
        context_text_preview=truncate_text(
            text=context_text,
            max_length=300,
        ),
        failure_type=safe_failure_type,
        decision=safe_decision,
        reason=safe_reason,
    )

    return diagnostics.model_dump()

def merge_retrieval_diagnostics(
        old_diagnostics: Mapping[str, Any] | None,
        new_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """
    合并检索诊断信息。

    功能：
        保留旧诊断信息，并使用新诊断信息覆盖同名字段。

        使用场景：
            retrieve 阶段先写 retrieved_count。
            evaluate 阶段再写 decision / failure_type。
            rerank 阶段再写 reranked_count / top_items。

    参数：
        old_diagnostics:
            旧的检索诊断信息。

        new_diagnostics:
            新的检索诊断信息。

    返回值：
        dict[str, Any]:
            合并后的检索诊断信息。
    """

    merged: dict[str, Any] = {}

    if isinstance(
            old_diagnostics,
            Mapping,
    ):
        merged.update(
            dict(
                old_diagnostics
            )
        )

    merged.update(
        dict(
            new_diagnostics
        )
    )

    return merged


def build_retrieval_quality_log_summary(
        retrieval_quality: dict[str, Any] | None,
        max_top_items: int = 3,
) -> dict[str, Any]:
    """
    构建 retrieval_quality 控制台日志摘要。

    功能：
        将完整 retrieval_quality 压缩成适合控制台打印的摘要。
        避免把 top_items、metadata_keys、context_text_preview 等长字段完整输出。

    参数：
        retrieval_quality:
            检索诊断信息。
            通常来自 state["retrieval_quality"]。

        max_top_items:
            最多展示多少条 top_items 摘要。

    返回值：
        dict[str, Any]:
            控制台友好的短摘要。

    专业名词：
        retrieval_quality:
            检索质量 / 检索诊断信息。

        log summary:
            日志摘要。只保留排查问题最需要的核心字段。
    """

    if not isinstance(
            retrieval_quality,
            dict,
    ):
        return {}

    top_items = retrieval_quality.get(
        "top_items",
        [],
    )

    compact_top_items: list[dict[str, Any]] = []

    if isinstance(
            top_items,
            list,
    ):
        for item in top_items[:max_top_items]:

            if not isinstance(
                    item,
                    dict,
            ):
                continue

            source = str(
                item.get(
                    "source",
                    "",
                )
                or ""
            )

            compact_top_items.append(
                {
                    "rank": item.get(
                        "rank",
                        "",
                    ),
                    "dog_name": item.get(
                        "dog_name",
                        "",
                    ),
                    "section_title": item.get(
                        "section_title",
                        "",
                    ),
                    "score": item.get(
                        "score",
                        None,
                    ),
                    "source_file": Path(
                        source
                    ).name if source else "",
                }
            )

    return {
        "stage": retrieval_quality.get(
            "stage",
            "",
        ),
        "route": retrieval_quality.get(
            "route",
            "",
        ),
        "intent": retrieval_quality.get(
            "intent",
            "",
        ),
        "filter_fields": retrieval_quality.get(
            "filter_fields",
            [],
        ),
        "retrieved_count": retrieval_quality.get(
            "retrieved_count",
            0,
        ),
        "reranked_count": retrieval_quality.get(
            "reranked_count",
            0,
        ),
        "has_context": retrieval_quality.get(
            "has_context",
            False,
        ),
        "has_metadata_filter": retrieval_quality.get(
            "has_metadata_filter",
            False,
        ),
        "has_dog_name_filter": retrieval_quality.get(
            "has_dog_name_filter",
            False,
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
        "top_items": compact_top_items,
    }