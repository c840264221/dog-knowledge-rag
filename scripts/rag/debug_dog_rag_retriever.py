"""
Dog RAG Retriever Debug Script。

Dog RAG Retriever Debug Script（狗狗 RAG 召回器调试脚本）：
用于连接真实 RuntimeContainer 和真实 Chroma，
测试 MetadataFilterRetriever 是否可以正常完成召回。

当前脚本职责：
1. 启动 RuntimeContainer
2. 从 container 中读取 VectorStoreProvider
3. 从 VectorStoreProvider 中读取真实 Chroma vector store
4. 构建 MetadataFilterRetriever
5. 构建 RagQuery
6. 执行真实召回
7. 打印 RagContext、RagRetrievedChunk、RagChunk、metadata

当前脚本不负责：
1. Markdown 入库
2. metadata 提取
3. chunk 切分
4. Rerank 重排
5. LLM 回答生成

运行示例：
    python -m scripts.rag.debug_dog_rag_retriever

    python -m scripts.rag.debug_dog_rag_retriever --dog-name "Affenpinscher"

    python -m scripts.rag.debug_dog_rag_retriever --question "small apartment dog that does not bark much"

    python -m scripts.rag.debug_dog_rag_retriever --size small --max-barking 3 --good-for-apartment
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from typing import Any

from src.rag.retrievers import (
    MetadataFilterRetriever,
)

from src.rag.schemas import (
    RagContext,
    RagQuery,
    RagRetrievedChunk,
)

from src.runtime.container.init import (
    container,
)


MetadataFilter = dict[str, Any]


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    功能：
        定义 debug_dog_rag_retriever.py 支持的命令行参数。

    参数：
        无。

    返回值：
        argparse.ArgumentParser：
            Python 标准库 argparse 的参数解析器对象。
    """

    parser = argparse.ArgumentParser(
        description="Debug real Dog RAG MetadataFilterRetriever.",
    )

    parser.add_argument(
        "--vector-store-provider",
        default="vectorstore",
        help=(
            "RuntimeContainer 中注册的 VectorStoreProvider 名称。"
        ),
    )

    parser.add_argument(
        "--vector-store-attr",
        default="db",
        help=(
            "VectorStoreProvider 上的向量库属性名。"
            "默认是 db。如果 Provider 本身就是 vector store，可以传 self。"
        ),
    )

    parser.add_argument(
        "--question",
        default="small apartment dog that does not bark much",
        help="用于真实向量检索的用户问题。",
    )

    parser.add_argument(
        "--user-id",
        default="default",
        help="RagQuery.user_id，默认是 default。",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="召回数量，默认是 5。",
    )

    parser.add_argument(
        "--dog-name",
        default=None,
        help=(
            "按 dog_name 精确过滤。"
            "例如 Affenpinscher、Golden Retriever。"
            "注意：这里要用 metadata 中真实保存的犬种名。"
        ),
    )

    parser.add_argument(
        "--size",
        default="small",
        help=(
            "按 size 过滤。"
            "默认是 small。"
            "如果不想使用 size 过滤，可以传空字符串：--size \"\"。"
        ),
    )

    parser.add_argument(
        "--max-energy",
        type=int,
        default=None,
        help="最大 energy_level。例如 3 表示 energy_level <= 3。",
    )

    parser.add_argument(
        "--max-barking",
        type=int,
        default=3,
        help="最大 barking_level。默认是 3。",
    )

    parser.add_argument(
        "--min-trainability",
        type=int,
        default=None,
        help="最小 trainability_level。例如 3 表示 trainability_level >= 3。",
    )

    parser.add_argument(
        "--good-for-apartment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否过滤 good_for_apartment=True。"
            "默认启用。"
            "如需关闭，使用 --no-good-for-apartment。"
        ),
    )

    parser.add_argument(
        "--good-for-beginner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "是否过滤 good_for_beginner。"
            "默认不使用该过滤条件。"
        ),
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="不使用任何 metadata filter，只做纯向量检索。",
    )

    parser.add_argument(
        "--content-preview-length",
        type=int,
        default=500,
        help="每条 chunk 内容预览长度，默认是 500。",
    )

    parser.add_argument(
        "--print-full-json",
        action="store_true",
        help="是否额外打印完整 RagContext JSON。",
    )

    return parser


def parse_args(
        argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    解析命令行参数。

    功能：
        将命令行参数解析成 argparse.Namespace 对象。
        测试时可以传入 argv，真实命令行运行时使用默认 None。

    参数：
        argv: list[str] | None
            命令行参数列表。

    返回值：
        argparse.Namespace：
            解析后的命令行参数对象。
    """

    parser = build_argument_parser()

    return parser.parse_args(
        args=argv,
    )


async def main(
        argv: list[str] | None = None,
) -> None:
    """
    脚本主入口。

    功能：
        解析命令行参数，并执行真实 Retriever 调试流程。

    参数：
        argv: list[str] | None
            命令行参数列表。
            None 表示使用真实命令行参数。

    返回值：
        None：
            主入口无返回值。
    """

    args = parse_args(
        argv=argv,
    )

    await run_debug_from_args(
        args=args,
    )


async def run_debug_from_args(
        args: argparse.Namespace,
        runtime_container: Any | None = None,
) -> RagContext:
    """
    根据命令行参数执行真实 Retriever 调试。

    功能：
        1. 启动 RuntimeContainer
        2. 获取真实 vector store
        3. 构建 MetadataFilterRetriever
        4. 构建 metadata filter
        5. 构建 RagQuery
        6. 执行 retrieve
        7. 打印结果
        8. 关闭 RuntimeContainer

    参数：
        args: argparse.Namespace
            命令行参数对象。

        runtime_container: Any | None
            运行时容器。
            默认使用项目全局 container。
            测试时可以传入 FakeContainer。

    返回值：
        RagContext：
            真实 Retriever 返回的 RAG 上下文对象。
    """

    active_container = runtime_container or container

    await _maybe_await(
        active_container.startup(),
    )

    try:
        vector_store_provider = active_container.get(
            args.vector_store_provider,
        )

        vector_store = get_vector_store_from_provider(
            vector_store_provider=vector_store_provider,
            vector_store_attr=args.vector_store_attr,
        )

        retriever = MetadataFilterRetriever(
            vector_store=vector_store,
            default_top_k=args.top_k,
        )

        metadata_filter = build_metadata_filter_from_args(
            retriever=retriever,
            args=args,
        )

        rag_query = build_rag_query_from_args(
            args=args,
            metadata_filter=metadata_filter,
        )

        rag_context = retriever.retrieve(
            query=rag_query,
        )

        print_debug_result(
            rag_context=rag_context,
            metadata_filter=metadata_filter,
            args=args,
        )

        return rag_context

    finally:
        await _maybe_await(
            active_container.shutdown(),
        )


def get_vector_store_from_provider(
        vector_store_provider: Any,
        vector_store_attr: str,
) -> Any:
    """
    从 VectorStoreProvider 中获取 vector store。

    功能：
        默认读取 provider.db。
        如果 vector_store_attr 是 self，则直接返回 provider 自身。

    参数：
        vector_store_provider: Any
            向量库 Provider 对象。

        vector_store_attr: str
            向量库属性名。
            例如 db、vector_store、self。

    返回值：
        Any：
            Chroma vector store 对象。
    """

    if vector_store_attr in {
        "",
        "self",
    }:
        return vector_store_provider

    if not hasattr(
            vector_store_provider,
            vector_store_attr,
    ):
        raise AttributeError(
            f"VectorStoreProvider 不存在属性：{vector_store_attr}"
        )

    return getattr(
        vector_store_provider,
        vector_store_attr,
    )


def build_metadata_filter_from_args(
        retriever: MetadataFilterRetriever,
        args: argparse.Namespace,
) -> MetadataFilter | None:
    """
    根据命令行参数构建 metadata filter。

    功能：
        将命令行参数中的 dog_name、size、max_energy、max_barking、
        min_trainability、good_for_apartment、good_for_beginner
        转换成 Chroma metadata filter。

    参数：
        retriever: MetadataFilterRetriever
            元数据过滤召回器。
            复用它内部的 build_dog_metadata_filter 方法。

        args: argparse.Namespace
            命令行参数对象。

    返回值：
        MetadataFilter | None：
            Chroma metadata filter。
            如果传入 --no-filter，则返回 None。
    """

    if args.no_filter:
        return None

    size = normalize_optional_string(
        value=args.size,
    )

    dog_name = normalize_optional_string(
        value=args.dog_name,
    )

    return retriever.build_dog_metadata_filter(
        dog_name=dog_name,
        size=size,
        max_energy=args.max_energy,
        max_barking=args.max_barking,
        min_trainability=args.min_trainability,
        good_for_apartment=args.good_for_apartment,
        good_for_beginner=args.good_for_beginner,
    )


def build_rag_query_from_args(
        args: argparse.Namespace,
        metadata_filter: MetadataFilter | None,
) -> RagQuery:
    """
    根据命令行参数构建 RagQuery。

    功能：
        将 question、user_id、top_k、filters、intent 组装成 RagQuery。

    参数：
        args: argparse.Namespace
            命令行参数对象。

        metadata_filter: MetadataFilter | None
            本次检索使用的 metadata filter。

    返回值：
        RagQuery：
            RAG 检索请求模型。
    """

    return RagQuery(
        question=args.question,
        user_id=args.user_id,
        top_k=args.top_k,
        filters=metadata_filter or {},
        intent="dog_info",
    )


def print_debug_result(
        rag_context: RagContext,
        metadata_filter: MetadataFilter | None,
        args: argparse.Namespace,
) -> None:
    """
    打印真实 Retriever 调试结果。

    功能：
        以适合终端阅读的格式打印：
        1. question
        2. metadata filter
        3. status
        4. source_count
        5. chunk 数量
        6. 每条 RagRetrievedChunk 的分数、原因、RagChunk 字段、metadata

    参数：
        rag_context: RagContext
            Retriever 返回的 RAG 上下文对象。

        metadata_filter: MetadataFilter | None
            本次使用的 metadata filter。

        args: argparse.Namespace
            命令行参数对象。

    返回值：
        None：
            打印函数无返回值。
    """

    print(
        "\n========== Dog RAG Retriever Debug Result ==========\n"
    )

    print(
        f"question: {rag_context.question}"
    )

    print(
        f"status: {rag_context.status}"
    )

    print(
        f"source_count: {rag_context.source_count}"
    )

    print(
        f"chunk_count: {len(rag_context.chunks)}"
    )

    print(
        "\nmetadata_filter:"
    )

    print(
        json.dumps(
            metadata_filter,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print(
        "\n---------------- Retrieved Chunks ----------------"
    )

    if not rag_context.chunks:
        print(
            "没有召回到任何 chunk。"
        )
        return

    for index, retrieved_chunk in enumerate(
            rag_context.chunks,
    ):
        print_retrieved_chunk(
            index=index,
            retrieved_chunk=retrieved_chunk,
            content_preview_length=args.content_preview_length,
        )

    print(
        "\n---------------- Context Text Preview ----------------"
    )

    print(
        truncate_text(
            text=rag_context.context_text,
            max_length=1500,
        )
    )

    if args.print_full_json:
        print(
            "\n---------------- Full RagContext JSON ----------------"
        )

        print(
            json.dumps(
                rag_context.model_dump(),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


def print_retrieved_chunk(
        index: int,
        retrieved_chunk: RagRetrievedChunk,
        content_preview_length: int,
) -> None:
    """
    打印单条 RagRetrievedChunk。

    功能：
        展示召回结果中的：
        1. retrieval_score
        2. rerank_score
        3. final_score
        4. reason
        5. RagChunk 核心字段
        6. metadata
        7. content preview

    参数：
        index: int
            当前 chunk 在列表中的索引，从 0 开始。

        retrieved_chunk: RagRetrievedChunk
            单条召回结果。

        content_preview_length: int
            内容预览长度。

    返回值：
        None：
            打印函数无返回值。
    """

    chunk = retrieved_chunk.chunk

    print(
        f"\n[Retrieved Chunk {index + 1}]"
    )

    print(
        f"retrieval_score: {retrieved_chunk.retrieval_score}"
    )

    print(
        f"rerank_score: {retrieved_chunk.rerank_score}"
    )

    print(
        f"final_score: {retrieved_chunk.final_score}"
    )

    print(
        f"reason: {retrieved_chunk.reason}"
    )

    print(
        "\nRagChunk:"
    )

    print(
        f"  chunk_id: {chunk.chunk_id}"
    )

    print(
        f"  doc_id: {chunk.doc_id}"
    )

    print(
        f"  chunk_index: {chunk.chunk_index}"
    )

    print(
        f"  source: {chunk.source}"
    )

    print(
        f"  title: {chunk.title}"
    )

    print(
        "\nmetadata:"
    )

    print(
        json.dumps(
            chunk.metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print(
        "\ncontent_preview:"
    )

    print(
        truncate_text(
            text=chunk.content,
            max_length=content_preview_length,
        )
    )


def truncate_text(
        text: str,
        max_length: int = 500,
) -> str:
    """
    截断文本。

    功能：
        控制终端输出长度，避免一次性打印过长内容。

    参数：
        text: str
            原始文本。

        max_length: int
            最大输出长度。

    返回值：
        str：
            截断后的文本。
    """

    clean_text = str(
        text
        or ""
    ).replace(
        "\n",
        " ",
    ).strip()

    if len(
            clean_text,
    ) <= max_length:
        return clean_text

    return clean_text[
        :max_length
    ] + "..."


def normalize_optional_string(
        value: str | None,
) -> str | None:
    """
    规范化可选字符串。

    功能：
        将 None、空字符串、纯空格字符串统一转换成 None。

    参数：
        value: str | None
            原始字符串。

    返回值：
        str | None：
            规范化后的字符串。
    """

    if value is None:
        return None

    clean_value = value.strip()

    if not clean_value:
        return None

    return clean_value


async def _maybe_await(
        value: Any,
) -> Any:
    """
    如果 value 是 awaitable，则 await 它。

    功能：
        兼容同步和异步生命周期方法。
        例如 container.startup 可能是同步方法，也可能是异步方法。

    参数：
        value: Any
            可能是普通值，也可能是 coroutine / awaitable。

    返回值：
        Any：
            如果是 awaitable，返回 await 后的结果；
            如果不是 awaitable，原样返回。
    """

    if inspect.isawaitable(
            value,
    ):
        return await value

    return value


if __name__ == "__main__":
    asyncio.run(
        main(),
    )