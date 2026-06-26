"""
Dog RAG Index Inspect Script。

Dog RAG Index Inspect Script（狗狗 RAG 索引检查脚本）：
用于检查已经写入 Chroma 的狗狗 RAG 数据是否正常。

当前脚本只读不写，主要检查：

1. Chroma collection 中有多少条数据
2. 随机读取若干条 sample metadata
3. 使用 dog_name 做 metadata filter 检查
4. 使用 size / energy_level / barking_level 做组合 metadata filter 检查
5. 使用 metadata filter + vector search 检查向量召回是否正常

本脚本不负责：
1. 加载 Markdown
2. 提取 metadata
3. 切块
4. 写入 Chroma
5. 修改或删除任何 Chroma 数据
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from typing import Any

from src.runtime.container.init import (
    container,
)


MetadataFilter = dict[str, Any]


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    功能：
        定义 inspect_dog_rag_index.py 支持的命令行参数。

    参数：
        无。

    返回值：
        argparse.ArgumentParser：
            Python 标准库 argparse 的参数解析器对象。
    """

    parser = argparse.ArgumentParser(
        description="Inspect Dog Agent Framework RAG Chroma index.",
    )

    parser.add_argument(
        "--vector-store-provider",
        default="vectorstore",
        help=(
            "Container 中注册的 VectorStoreProvider 名称，"
            "默认是 vector_store_provider。"
        ),
    )

    parser.add_argument(
        "--vector-store-attr",
        default="db",
        help=(
            "VectorStoreProvider 上的 Chroma 属性名，"
            "默认是 db。如果 Provider 本身就是 Chroma，可以传 self。"
        ),
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="随机查看多少条已入库样本，默认是 5。",
    )

    parser.add_argument(
        "--dog-name",
        default="Golden Retriever",
        help="用于 dog_name 精确过滤检查的犬种名，默认是 Golden Retriever。",
    )

    parser.add_argument(
        "--size",
        default="small",
        help="用于 metadata filter 检查的 size，默认是 small。",
    )

    parser.add_argument(
        "--max-energy",
        type=int,
        default=3,
        help="用于 metadata filter 检查的最大 energy_level，默认是 3。",
    )

    parser.add_argument(
        "--max-barking",
        type=int,
        default=3,
        help="用于 metadata filter 检查的最大 barking_level，默认是 3。",
    )

    parser.add_argument(
        "--query",
        default="small dog suitable for apartment",
        help="用于 vector search 检查的查询文本。",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="向量检索返回多少条结果，默认是 5。",
    )

    return parser


def parse_args(
        argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    解析命令行参数。

    功能：
        将命令行输入解析成 argparse.Namespace 对象。

    参数：
        argv: list[str] | None
            命令行参数列表。
            None 表示使用真实命令行参数。
            测试时可以传入自定义列表。

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
        解析命令行参数，并执行 RAG 索引检查流程。

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

    await run_inspection_from_args(
        args=args,
    )


async def run_inspection_from_args(
        args: argparse.Namespace,
        runtime_container: Any | None = None,
) -> dict[str, Any]:
    """
    根据命令行参数执行 RAG 索引检查。

    功能：
        1. 启动 container
        2. 从 container 获取 VectorStoreProvider
        3. 从 VectorStoreProvider 中获取 Chroma vector store
        4. 执行索引检查
        5. 打印检查结果
        6. 关闭 container

    参数：
        args: argparse.Namespace
            命令行参数对象。

        runtime_container: Any | None
            运行时容器对象。
            默认使用项目中的全局 container。
            测试时可以传入 FakeContainer。

    返回值：
        dict[str, Any]：
            索引检查结果。
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

        result = inspect_vector_store(
            vector_store=vector_store,
            sample_limit=args.sample_limit,
            dog_name=args.dog_name,
            size=args.size,
            max_energy=args.max_energy,
            max_barking=args.max_barking,
            query=args.query,
            top_k=args.top_k,
        )

        print_result(
            result=result,
        )

        return result

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


def inspect_vector_store(
        vector_store: Any,
        sample_limit: int,
        dog_name: str,
        size: str | None,
        max_energy: int | None,
        max_barking: int | None,
        query: str,
        top_k: int,
) -> dict[str, Any]:
    """
    检查 Chroma vector store 中的 RAG 数据。

    功能：
        1. 统计 collection 数据量
        2. 读取 sample metadata
        3. 按 dog_name 检查 metadata filter
        4. 按 size / energy_level / barking_level 检查组合 filter
        5. 执行 metadata filter + vector search

    参数：
        vector_store: Any
            Chroma 向量库对象。

        sample_limit: int
            读取多少条样本。

        dog_name: str
            用于 dog_name 精确过滤的犬种名。

        size: str | None
            用于 size 过滤的犬种体型。

        max_energy: int | None
            最大 energy_level。

        max_barking: int | None
            最大 barking_level。

        query: str
            向量检索查询文本。

        top_k: int
            向量检索返回条数。

    返回值：
        dict[str, Any]：
            检查结果。
    """

    collection_count = get_collection_count(
        vector_store=vector_store,
    )

    sample_records = get_records(
        vector_store=vector_store,
        limit=sample_limit,
    )

    dog_name_filter = build_dog_name_filter(
        dog_name=dog_name,
    )

    dog_name_records = get_records(
        vector_store=vector_store,
        where=dog_name_filter,
        limit=sample_limit,
    )

    metadata_filter = build_metadata_filter(
        size=size,
        max_energy=max_energy,
        max_barking=max_barking,
    )

    filtered_records = get_records(
        vector_store=vector_store,
        where=metadata_filter,
        limit=sample_limit,
    )

    vector_search_records = similarity_search(
        vector_store=vector_store,
        query=query,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    return {
        "script": "inspect_dog_rag_index_v1",
        "collection_count": collection_count,
        "sample_check": {
            "limit": sample_limit,
            "records": sample_records,
        },
        "dog_name_filter_check": {
            "filter": dog_name_filter,
            "match_count": len(
                dog_name_records,
            ),
            "records": dog_name_records,
        },
        "metadata_filter_check": {
            "filter": metadata_filter,
            "match_count": len(
                filtered_records,
            ),
            "records": filtered_records,
        },
        "vector_search_check": {
            "query": query,
            "top_k": top_k,
            "filter": metadata_filter,
            "match_count": len(
                vector_search_records,
            ),
            "records": vector_search_records,
        },
    }


def build_dog_name_filter(
        dog_name: str | None,
) -> MetadataFilter | None:
    """
    构建 dog_name 精确过滤条件。

    功能：
        将 dog_name 转换成 Chroma where filter。

    参数：
        dog_name: str | None
            犬种名称。

    返回值：
        MetadataFilter | None：
            Chroma where filter。
    """

    if not dog_name:
        return None

    return {
        "dog_name": {
            "$eq": dog_name,
        }
    }


def build_metadata_filter(
        size: str | None,
        max_energy: int | None,
        max_barking: int | None,
) -> MetadataFilter | None:
    """
    构建组合 metadata filter。

    功能：
        将 size、energy_level、barking_level 转换成 Chroma where filter。

    参数：
        size: str | None
            犬种体型，例如 small、medium、large、giant。

        max_energy: int | None
            最大 energy_level。

        max_barking: int | None
            最大 barking_level。

    返回值：
        MetadataFilter | None：
            Chroma where filter。
            如果没有任何过滤条件，则返回 None。
    """

    conditions: list[MetadataFilter] = []

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

    if not conditions:
        return None

    if len(
            conditions,
    ) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


def get_collection_count(
        vector_store: Any,
) -> int | None:
    """
    获取 Chroma collection 数据量。

    功能：
        优先从 vector_store._collection.count() 获取数量。
        如果 vector_store 本身支持 count()，也可以兼容。

    参数：
        vector_store: Any
            Chroma 向量库对象。

    返回值：
        int | None：
            collection 数据量。
            如果无法获取，返回 None。
    """

    collection = get_collection(
        vector_store=vector_store,
    )

    if hasattr(
            collection,
            "count",
    ):
        return collection.count()

    if hasattr(
            vector_store,
            "count",
    ):
        return vector_store.count()

    return None


def get_collection(
        vector_store: Any,
) -> Any:
    """
    获取底层 Chroma collection。

    功能：
        兼容 LangChain Chroma 的 _collection 属性。
        如果没有 _collection，则返回 vector_store 自身。

    参数：
        vector_store: Any
            Chroma 向量库对象。

    返回值：
        Any：
            底层 collection 或 vector_store 自身。
    """

    if hasattr(
            vector_store,
            "_collection",
    ):
        return vector_store._collection

    if hasattr(
            vector_store,
            "collection",
    ):
        return vector_store.collection

    return vector_store


def get_records(
        vector_store: Any,
        where: MetadataFilter | None = None,
        limit: int = 5,
) -> list[dict[str, Any]]:
    """
    从 Chroma 中读取记录。

    功能：
        使用 vector_store.get 或 collection.get 读取已入库数据。
        可以传入 where metadata filter。

    参数：
        vector_store: Any
            Chroma 向量库对象。

        where: MetadataFilter | None
            Chroma metadata filter。

        limit: int
            最多读取多少条。

    返回值：
        list[dict[str, Any]]：
            规范化后的记录列表。
    """

    getter = (
        vector_store.get
        if hasattr(
            vector_store,
            "get",
        )
        else get_collection(
            vector_store=vector_store,
        ).get
    )

    kwargs: dict[str, Any] = {
        "limit": limit,
        "include": [
            "metadatas",
            "documents",
        ],
    }

    if where:
        kwargs[
            "where"
        ] = where

    raw_result = getter(
        **kwargs,
    )

    return normalize_get_result(
        raw_result=raw_result,
    )


def similarity_search(
        vector_store: Any,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    """
    执行 metadata filter + vector search。

    功能：
        调用 vector_store.similarity_search 进行向量检索。
        如果传入 metadata_filter，则只在满足 metadata filter 的 chunk 中搜索。

    参数：
        vector_store: Any
            Chroma 向量库对象。

        query: str
            查询文本。

        top_k: int
            返回条数。

        metadata_filter: MetadataFilter | None
            Chroma metadata filter。

    返回值：
        list[dict[str, Any]]：
            规范化后的检索结果。
    """

    if not hasattr(
            vector_store,
            "similarity_search",
    ):
        return []

    kwargs: dict[str, Any] = {
        "query": query,
        "k": top_k,
    }

    if metadata_filter:
        kwargs[
            "filter"
        ] = metadata_filter

    documents = vector_store.similarity_search(
        **kwargs,
    )

    return normalize_documents(
        documents=documents,
    )


def normalize_get_result(
        raw_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    规范化 Chroma get 返回结果。

    功能：
        Chroma get 通常返回 dict，里面包含：
        1. ids
        2. documents
        3. metadatas

        本方法将其转换成更容易阅读的 list[dict]。

    参数：
        raw_result: dict[str, Any]
            Chroma get 原始结果。

    返回值：
        list[dict[str, Any]]：
            规范化后的记录列表。
    """

    if not raw_result:
        return []

    ids = raw_result.get(
        "ids",
        [],
    )

    documents = raw_result.get(
        "documents",
        [],
    )

    metadatas = raw_result.get(
        "metadatas",
        [],
    )

    records: list[dict[str, Any]] = []

    for index, item_id in enumerate(
            ids,
    ):

        content = (
            documents[index]
            if index < len(
                documents,
            )
            else ""
        )

        metadata = (
            metadatas[index]
            if index < len(
                metadatas,
            )
            else {}
        )

        records.append(
            {
                "id": item_id,
                "content_preview": truncate_text(
                    text=content,
                ),
                "metadata": metadata or {},
            }
        )

    return records


def normalize_documents(
        documents: list[Any],
) -> list[dict[str, Any]]:
    """
    规范化 LangChain Document 检索结果。

    功能：
        将 similarity_search 返回的 Document 对象转换成 dict。

    参数：
        documents: list[Any]
            LangChain Document 列表。

    返回值：
        list[dict[str, Any]]：
            规范化后的检索结果。
    """

    records: list[dict[str, Any]] = []

    for index, document in enumerate(
            documents,
    ):
        records.append(
            {
                "rank": index + 1,
                "content_preview": truncate_text(
                    text=getattr(
                        document,
                        "page_content",
                        "",
                    ),
                ),
                "metadata": getattr(
                    document,
                    "metadata",
                    {},
                ) or {},
            }
        )

    return records


def truncate_text(
        text: str,
        max_length: int = 220,
) -> str:
    """
    截断文本。

    功能：
        控制输出内容长度，避免终端打印太长。

    参数：
        text: str
            原始文本。

        max_length: int
            最大长度。

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


async def _maybe_await(
        value: Any,
) -> Any:
    """
    如果 value 是 awaitable，则 await 它。

    功能：
        兼容同步和异步生命周期方法。

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


def print_result(
        result: dict[str, Any],
) -> None:
    """
    打印检查结果。

    功能：
        将检查结果以 JSON 格式打印出来。
        ensure_ascii=False 可以保证中文不被转义。

    参数：
        result: dict[str, Any]
            检查结果。

    返回值：
        None：
            打印函数无返回值。
    """

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(
        main(),
    )