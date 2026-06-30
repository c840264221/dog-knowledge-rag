from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.rag.evaluation import (
    OfflineRetrievalEvaluator,
    RagEvalCase,
    RagEvalReport,
    calculate_rag_eval_metrics,
    load_rag_eval_cases,
    write_rag_eval_report,
)

from src.runtime.container.init import container

from src.settings import settings


DEFAULT_DATASET_PATH = Path(
    "data/rag_eval/dog_rag_eval_cases.json"
)

DEFAULT_REPORT_DIR = Path(settings.path.RAG_EVALUATE_REPORT_DIR)


def build_parser() -> Any:
    """
    构建 Dog RAG Query Parser。

    功能：
        创建或获取当前项目中用于解析用户问题的 Query Parser。
        Query Parser 负责把自然语言问题转换成 RagQuery 或包含 filters 的 dict。

    参数含义：
        无。

    返回值含义：
        Any:
            Query Parser 实例。

    注意：
        如果你当前项目里的 Parser 路径或类名不同，
        只需要修改这个函数里的 import 和实例化逻辑。

    专业名词：
        Query Parser：
            查询解析器。负责把用户自然语言问题解析成结构化查询条件。

        Filter：
            过滤条件。用于限制向量库检索范围，例如 dog_name、energy、size。
    """

    from src.rag.query_parsers import DogQueryFilterParser

    return DogQueryFilterParser()


def build_retriever() -> Any:
    """
    构建 Dog RAG Retriever。

    功能：
        创建或获取当前项目中用于执行 metadata filter 检索的 Retriever。
        Retriever 负责根据 RagQuery 从向量库中召回 RagContext。

    参数含义：
        无。

    返回值含义：
        Any:
            Retriever 实例。

    注意：
        如果你的 MetadataFilterRetriever 构造函数需要参数，
        例如 vectorstore、embedding、collection 等，
        可以在这里从 container 或 provider 中获取后传入。

    专业名词：
        Retriever：
            检索器。负责从向量库或知识库中召回相关文档 / chunk。

        RagContext：
            RAG 上下文。通常包含 context_text、chunks、status 等字段。
    """

    from src.rag.retrievers import MetadataFilterRetriever

    vectorstore_provider = container.get("vectorstore")
    db = vectorstore_provider.db

    return MetadataFilterRetriever(db)


def parse_eval_case(
    parser: Any,
    eval_case: RagEvalCase,
) -> Any:
    """
    将 RagEvalCase 解析成项目实际使用的 RagQuery 或 dict。

    功能：
        适配不同 Parser 方法名。
        优先尝试 parse 方法，其次尝试 parse_query、invoke、__call__。

    参数含义：
        parser:
            Query Parser 实例。

        eval_case:
            当前评估用例。

    返回值含义：
        Any:
            解析后的 query 对象。
            可以是 RagQuery、dict，或者你项目里的其他查询对象。

    异常：
        AttributeError:
            当 parser 没有可识别的调用方法时抛出。
    """

    if hasattr(parser, "parse"):
        return parser.parse(
            question=eval_case.question,
            top_k=eval_case.top_k,
        )

    if hasattr(parser, "parse_query"):
        return parser.parse_query(
            question=eval_case.question,
            top_k=eval_case.top_k,
        )

    if hasattr(parser, "invoke"):
        return parser.invoke(
            {
                "question": eval_case.question,
                "top_k": eval_case.top_k,
            }
        )

    if callable(parser):
        return parser(eval_case)

    raise AttributeError(
        "无法调用 Query Parser：未找到 parse / parse_query / invoke / __call__ 方法。"
    )


def retrieve_rag_context(
    retriever: Any,
    parsed_query: Any,
) -> Any:
    """
    根据解析后的 query 执行 RAG 检索。

    功能：
        适配不同 Retriever 方法名。
        优先尝试 retrieve 方法，其次尝试 search、invoke、__call__。

    参数含义：
        retriever:
            Retriever 实例。

        parsed_query:
            Query Parser 输出的查询对象。

    返回值含义：
        Any:
            RagContext、dict，或者 {"rag_context": RagContext} 结构。

    异常：
        AttributeError:
            当 retriever 没有可识别的调用方法时抛出。
    """

    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(
            parsed_query
        )

    if hasattr(retriever, "search"):
        return retriever.search(
            parsed_query
        )

    if hasattr(retriever, "invoke"):
        return retriever.invoke(
            parsed_query
        )

    if callable(retriever):
        return retriever(parsed_query)

    raise AttributeError(
        "无法调用 Retriever：未找到 retrieve / search / invoke / __call__ 方法。"
    )


def build_parse_query_func(
    parser: Any,
):
    """
    构建 OfflineRetrievalEvaluator 需要的 parse_query_func。

    功能：
        将 parser 实例包装成一个函数，使其符合：
        Callable[[RagEvalCase], Any]

    参数含义：
        parser:
            Query Parser 实例。

    返回值含义：
        callable:
            输入 RagEvalCase，输出 parsed_query 的函数。
    """

    def parse_query_func(
        eval_case: RagEvalCase,
    ) -> Any:
        """
        解析单条评估用例。

        参数含义：
            eval_case:
                当前评估用例。

        返回值含义：
            Any:
                解析后的 query 对象。
        """

        return parse_eval_case(
            parser=parser,
            eval_case=eval_case,
        )

    return parse_query_func


def build_retrieve_context_func(
    retriever: Any,
):
    """
    构建 OfflineRetrievalEvaluator 需要的 retrieve_context_func。

    功能：
        将 retriever 实例包装成一个函数，使其符合：
        Callable[[Any], Any]

    参数含义：
        retriever:
            Retriever 实例。

    返回值含义：
        callable:
            输入 parsed_query，输出 RagContext 的函数。
    """

    def retrieve_context_func(
        parsed_query: Any,
    ) -> Any:
        """
        执行单次 RAG 检索。

        参数含义：
            parsed_query:
                Query Parser 输出的查询对象。

        返回值含义：
            Any:
                RagContext 或 dict。
        """

        return retrieve_rag_context(
            retriever=retriever,
            parsed_query=parsed_query,
        )

    return retrieve_context_func


def build_run_id() -> str:
    """
    构建本次评估运行 ID。

    功能：
        使用当前时间生成唯一 run_id，方便报告文件命名和历史追踪。

    参数含义：
        无。

    返回值含义：
        str:
            评估运行 ID，例如 rag_eval_20260630_153012。
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return f"rag_eval_{timestamp}"


def main() -> None:
    """
    执行 Dog RAG Retrieval Evaluation。

    功能：
        1. 加载评估数据集。
        2. 构建 Parser。
        3. 构建 Retriever。
        4. 执行离线检索评估。
        5. 计算汇总指标。
        6. 生成评估报告。
        7. 写入 Markdown 文件。
        8. 在终端打印关键指标和报告路径。

    参数含义：
        无。

    返回值含义：
        None。
    """

    dataset_path = DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"评估数据集不存在: {dataset_path}"
        )

    print("=" * 80)
    print("开始执行 Dog RAG Retrieval Evaluation")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")

    eval_cases = load_rag_eval_cases(
        dataset_path=dataset_path,
    )

    print(f"成功加载评估用例数量: {len(eval_cases)}")

    parser = build_parser()
    retriever = build_retriever()

    evaluator = OfflineRetrievalEvaluator(
        parse_query_func=build_parse_query_func(
            parser=parser,
        ),
        retrieve_context_func=build_retrieve_context_func(
            retriever=retriever,
        ),
        require_quality_usable=True,
    )

    results = evaluator.evaluate_many(
        eval_cases=eval_cases,
    )

    metrics = calculate_rag_eval_metrics(
        results=results,
    )

    run_id = build_run_id()

    report = RagEvalReport(
        run_id=run_id,
        dataset_path=dataset_path.as_posix(),
        metrics=metrics,
        results=results,
        metadata={
            "version": "v1.6.0",
            "stage": "RAG Evaluation MVP",
            "evaluator": "OfflineRetrievalEvaluator",
            "parser": type(parser).__name__,
            "retriever": type(retriever).__name__,
            "dataset_path": dataset_path.as_posix(),
            "total_cases": len(eval_cases),
        },
    )

    report_path = write_rag_eval_report(
        report=report,
        output_dir=DEFAULT_REPORT_DIR,
    )

    print("=" * 80)
    print("Dog RAG Retrieval Evaluation 完成")
    print("=" * 80)
    print(f"Report: {report_path.as_posix()}")
    print("-" * 80)
    print(f"total_cases: {metrics.total_cases}")
    print(f"passed_cases: {metrics.passed_cases}")
    print(f"failed_cases: {metrics.failed_cases}")
    print(f"hit_at_k: {metrics.hit_at_k:.2%}")
    print(f"top1_accuracy: {metrics.top1_accuracy:.2%}")
    print(f"filter_match_rate: {metrics.filter_match_rate:.2%}")
    print(f"empty_retrieval_rate: {metrics.empty_retrieval_rate:.2%}")

    if metrics.average_latency_ms is None:
        print("average_latency_ms: N/A")
    else:
        print(f"average_latency_ms: {metrics.average_latency_ms:.3f} ms")

    failed_results = [
        result
        for result in results
        if not result.is_successful()
    ]

    if failed_results:
        print("-" * 80)
        print("失败用例:")
        for result in failed_results:
            failure_type = result.extra.get(
                "quality_failure_type",
                "",
            )

            print(
                f"- {result.case_id}: "
                f"hit={result.hit}, "
                f"top1={result.top1_hit}, "
                f"filter_matched={result.filter_matched}, "
                f"empty={result.empty_retrieval}, "
                f"failure_type={failure_type}, "
                f"error={result.error_message or ''}"
            )

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("=" * 80)
        print("Dog RAG Retrieval Evaluation 执行失败")
        print("=" * 80)
        print(str(exc))
        sys.exit(1)