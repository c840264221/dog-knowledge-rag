from typing import Any, Literal

from pydantic import BaseModel, Field


class RagDocument(BaseModel):
    """
    RAG 原始文档模型。

    功能：
        用来表示进入 RAG（Retrieval-Augmented Generation，检索增强生成）
        系统之前的原始文档数据。

    字段说明：
        doc_id:
            文档唯一 ID，用来区分不同文档。

        source:
            文档来源，例如本地文件路径、网页 URL、数据库记录 ID。

        title:
            文档标题，方便后续检索结果展示和引用。

        content:
            文档正文内容。

        metadata:
            元数据（Metadata），用于保存额外信息，例如标签、分类、创建时间等。

    返回值：
        这是一个 Pydantic 数据模型，本身不直接返回业务结果，
        但会作为 RAG Pipeline 的标准输入结构。
    """

    doc_id: str = Field(
        ...,
        description="文档唯一 ID"
    )

    source: str = Field(
        ...,
        description="文档来源"
    )

    title: str = Field(
        default="",
        description="文档标题"
    )

    content: str = Field(
        ...,
        description="文档正文内容"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="文档元数据"
    )


class RagChunk(BaseModel):
    """
    RAG 文本块模型。

    功能：
        用来表示原始文档经过 Chunking（文本切分）之后得到的小文本块。
        Chunk 是后续 Embedding（向量化）和 Retrieval（检索）的基本单位。

    字段说明：
        chunk_id:
            文本块唯一 ID。

        doc_id:
            当前文本块所属的原始文档 ID。

        content:
            文本块正文内容。

        chunk_index:
            当前文本块在原始文档中的顺序编号。

        source:
            文本块来源，通常继承自 RagDocument.source。

        title:
            文本块所属文档标题。

        metadata:
            文本块元数据，可以包含章节、标签、文档类型等信息。

    返回值：
        这是一个标准数据模型，主要用于索引、检索和重排。
    """

    chunk_id: str = Field(
        ...,
        description="文本块唯一 ID"
    )

    doc_id: str = Field(
        ...,
        description="所属文档 ID"
    )

    content: str = Field(
        ...,
        description="文本块正文内容"
    )

    chunk_index: int = Field(
        default=0,
        ge=0,
        description="文本块顺序编号"
    )

    source: str = Field(
        default="",
        description="文本块来源"
    )

    title: str = Field(
        default="",
        description="所属文档标题"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="文本块元数据"
    )


class RagQuery(BaseModel):
    """
    RAG 检索请求模型。

    功能：
        用来表示一次 RAG 检索请求。
        后续 Retriever（检索器）会根据该结构执行检索。

    字段说明：
        question:
            用户问题。

        user_id:
            用户 ID，用于多用户隔离和个性化检索。

        top_k:
            初次检索返回数量。

        filters:
            检索过滤条件，例如只检索某个 dog_name、category 或 tag。

        intent:
            用户意图，例如 general、dog_info、health、training 等。

    返回值：
        这是一个检索请求模型，会传入 Retriever 或 RAG Pipeline。
    """

    question: str = Field(
        ...,
        description="用户问题"
    )

    user_id: str = Field(
        default="default",
        description="用户 ID"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="检索返回数量"
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="检索过滤条件"
    )

    intent: str = Field(
        default="general",
        description="用户意图"
    )


class RagRetrievedChunk(BaseModel):
    """
    RAG 检索结果文本块模型。

    功能：
        用来表示 Retriever（检索器）或 Reranker（重排器）返回的单条检索结果。

    字段说明：
        chunk:
            被检索到的文本块。

        retrieval_score:
            向量检索阶段的分数。

        rerank_score:
            重排阶段的分数。如果还没有经过重排，可以为空。

        final_score:
            最终排序分数。可以由 retrieval_score、rerank_score、metadata 加权共同计算。

        reason:
            当前文本块被选中的原因，方便 debug 和可观测分析。

    返回值：
        这是一个检索结果模型，用于后续上下文组装和回答生成。
    """

    chunk: RagChunk = Field(
        ...,
        description="检索到的文本块"
    )

    retrieval_score: float = Field(
        default=0.0,
        description="向量检索分数"
    )

    rerank_score: float | None = Field(
        default=None,
        description="重排分数"
    )

    final_score: float = Field(
        default=0.0,
        description="最终排序分数"
    )

    reason: str = Field(
        default="",
        description="选中原因"
    )


class RagContext(BaseModel):
    """
    RAG 上下文模型。

    功能：
        用来表示最终注入 LLM Prompt 的上下文内容。
        Context Builder（上下文构建器）会把多个 RagRetrievedChunk 组合成 RagContext。

    字段说明：
        question:
            用户问题。

        context_text:
            最终拼接好的上下文文本。

        chunks:
            被选中的检索结果列表。

        source_count:
            来源数量。

        status:
            上下文构建状态。
            success 表示成功；
            empty 表示没有检索到内容；
            truncated 表示因为长度限制被截断。

    返回值：
        这是最终进入 Answer Generation（答案生成）节点前的标准上下文模型。
    """

    question: str = Field(
        ...,
        description="用户问题"
    )

    context_text: str = Field(
        default="",
        description="最终上下文文本"
    )

    chunks: list[RagRetrievedChunk] = Field(
        default_factory=list,
        description="检索结果列表"
    )

    source_count: int = Field(
        default=0,
        ge=0,
        description="来源数量"
    )

    status: Literal["success", "empty", "truncated"] = Field(
        default="empty",
        description="上下文状态"
    )