"""
Dog RAG Index Rebuild Script。

Dog RAG Index Rebuild Script（狗狗 RAG 索引重建脚本）：
用于把 data/dog_markdown 目录下的 Markdown 犬种文档重新写入 Chroma 向量库。

当前脚本执行流程：

1. 从 Container 中获取 VectorStoreProvider（向量库 Provider）
2. 从 VectorStoreProvider 中惰性获取 Chroma vector store
3. 创建 MarkdownDocumentLoader
4. 创建 DogBreedMetadataExtractor
5. 创建 MarkdownChunker
6. 创建 RagChromaIndexer
7. 创建 RagIndexPipeline
8. 执行索引 Pipeline 并打印入库统计结果

本脚本只负责 CLI 执行入口，不负责具体业务算法。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.rag.chunking.markdown_chunker import (
    MarkdownChunker,
)
from src.rag.extractors import (
    DogBreedMetadataExtractor,
)
from src.rag.indexers import (
    RagChromaIndexer,
)
from src.rag.loaders.markdown_loader import (
    MarkdownDocumentLoader,
)
from src.rag.pipelines import (
    RagIndexPipeline,
)
from src.runtime.container.init import (
    container,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    功能：
        定义 rebuild_dog_rag_index.py 支持的命令行参数。

    参数：
        无。

    返回值：
        argparse.ArgumentParser：
            Python 标准库 argparse 的参数解析器对象。
    """

    parser = argparse.ArgumentParser(
        description="Rebuild Dog Agent Framework RAG Chroma index.",
    )

    parser.add_argument(
        "--data-dir",
        default="data/dog_markdown",
        help="Markdown 犬种文档目录，默认是 data/dog_markdown。",
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
            "默认是 db。如果你的 Provider 直接就是 Chroma，可以传 self。"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每批写入 Chroma 的 chunk 数量，默认是 64。",
    )

    parser.add_argument(
        "--overwrite-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否覆盖已有 chunk。"
            "默认开启。可以使用 --no-overwrite-existing 关闭。"
        ),
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
            解析后的参数对象。
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
        解析命令行参数，并执行 RAG 索引重建流程。

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

    await run_rebuild_from_args(
        args=args,
    )


async def run_rebuild_from_args(
        args: argparse.Namespace,
        runtime_container: Any | None = None,
) -> dict[str, Any]:
    """
    根据命令行参数执行 RAG 索引重建。

    功能：
        1. 从 container 获取 vector store provider
        2. 惰性初始化当前入库所需的 vector store
        3. 构建 RAG Index Pipeline
        4. 执行目录入库
        5. 打印结果

        本脚本不会启动整个 RuntimeContainer。索引重建只依赖默认
        Embedding 和 VectorStore；启动完整容器会额外初始化 LLM、Memory、
        Tool 等无关服务，并引入不必要的密钥和外部服务依赖。

    参数：
        args: argparse.Namespace
            命令行参数对象。

        runtime_container: Any | None
            运行时容器对象。
            默认使用项目中的全局 container。
            测试时可以传入 FakeContainer。

    返回值：
        dict[str, Any]：
            Pipeline 返回的入库统计结果。
    """

    active_container = runtime_container or container

    data_dir = Path(
        args.data_dir,
    )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"RAG 数据目录不存在：{data_dir}"
        )

    vector_store_provider = active_container.get(
        args.vector_store_provider,
    )

    vector_store = _get_vector_store_from_provider(
        vector_store_provider=vector_store_provider,
        vector_store_attr=args.vector_store_attr,
    )

    pipeline = build_pipeline(
        input_path=data_dir,
        vector_store=vector_store,
        batch_size=args.batch_size,
        overwrite_existing=args.overwrite_existing,
    )

    result = run_pipeline(
        pipeline=pipeline,
    )

    print_result(
        result=result,
    )

    return result


def build_pipeline(
        input_path: str | Path,
        vector_store: Any,
        batch_size: int,
        overwrite_existing: bool,
) -> RagIndexPipeline:
    """
    构建 Dog RAG Index Pipeline。

    功能：
        创建 Loader、Extractor、Chunker、Indexer，
        并组装成 RagIndexPipeline。

        当前项目中的 MarkdownDocumentLoader 需要 input_path 参数。
        input_path 可以是单个 Markdown 文件路径，也可以是 Markdown 目录路径。

    参数：
        input_path: str | Path
            Markdown 输入路径。

        vector_store: Any
            Chroma 向量库对象。
            通常来自 VectorStoreProvider.db。

        batch_size: int
            Chroma 批量写入大小。

        overwrite_existing: bool
            是否覆盖已有 chunk。

    返回值：
        RagIndexPipeline：
            已经组装好的 RAG 入库流水线。
    """

    loader = MarkdownDocumentLoader(
        input_path=input_path,
    )

    metadata_extractor = DogBreedMetadataExtractor()

    chunker = MarkdownChunker()

    indexer = RagChromaIndexer(
        vector_store=vector_store,
        batch_size=batch_size,
        overwrite_existing=overwrite_existing,
    )

    return RagIndexPipeline(
        loader=loader,
        metadata_extractor=metadata_extractor,
        chunker=chunker,
        indexer=indexer,
    )


def run_pipeline(
        pipeline: RagIndexPipeline,
) -> dict[str, Any]:
    """
    执行 RAG Index Pipeline。

    功能：
        调用 pipeline.index()，
        由 Pipeline 内部调用 loader.load() 完成文档加载。

    参数：
        pipeline: RagIndexPipeline
            RAG 入库流水线对象。

    返回值：
        dict[str, Any]：
            入库统计结果。
    """

    return pipeline.index()


def _get_vector_store_from_provider(
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


def print_result(
        result: dict[str, Any],
) -> None:
    """
    打印入库结果。

    功能：
        将 Pipeline 返回的统计结果以 JSON 格式打印出来。
        ensure_ascii=False 可以保证中文不被转义。

    参数：
        result: dict[str, Any]
            Pipeline 入库统计结果。

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
