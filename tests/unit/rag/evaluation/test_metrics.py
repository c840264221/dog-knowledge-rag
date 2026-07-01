from src.rag.evaluation import (
    RagEvalResult,
    calculate_rag_eval_metrics,
)


def test_calculate_metrics_with_empty_results() -> None:
    """
    测试空评估结果列表时，指标能返回默认值。

    参数含义：
        无。

    返回值含义：
        None。
    """

    metrics = calculate_rag_eval_metrics([])

    assert metrics.total_cases == 0
    assert metrics.passed_cases == 0
    assert metrics.failed_cases == 0
    assert metrics.hit_at_k == 0.0
    assert metrics.top1_accuracy == 0.0
    assert metrics.filter_match_rate == 0.0
    assert metrics.empty_retrieval_rate == 0.0
    assert metrics.average_latency_ms is None


def test_calculate_metrics_success() -> None:
    """
    测试正常计算 RAG 评估指标。

    参数含义：
        无。

    返回值含义：
        None。
    """

    results = [
        RagEvalResult(
            case_id="dog_eval_001",
            question="问题 1",
            expected_dog_names=["Poodle"],
            expected_filters={"size": "small"},
            parsed_filters={"size": "small"},
            retrieved_dog_names=["Poodle"],
            hit=True,
            hit_rank=1,
            top1_hit=True,
            filter_matched=True,
            empty_retrieval=False,
            passed=True,
            latency_ms=100.0,
        ),
        RagEvalResult(
            case_id="dog_eval_002",
            question="问题 2",
            expected_dog_names=["Golden Retriever"],
            expected_filters={"dog_name": "Golden Retriever"},
            parsed_filters={"dog_name": "Golden Retriever"},
            retrieved_dog_names=["Beagle", "Golden Retriever"],
            hit=True,
            hit_rank=2,
            top1_hit=False,
            filter_matched=True,
            empty_retrieval=False,
            passed=True,
            latency_ms=200.0,
        ),
        RagEvalResult(
            case_id="dog_eval_003",
            question="问题 3",
            expected_dog_names=["Chihuahua"],
            expected_filters={"dog_name": "Chihuahua"},
            parsed_filters={},
            retrieved_dog_names=[],
            hit=False,
            hit_rank=None,
            top1_hit=False,
            filter_matched=False,
            empty_retrieval=True,
            passed=False,
            latency_ms=300.0,
        ),
    ]

    metrics = calculate_rag_eval_metrics(results)

    assert metrics.total_cases == 3
    assert metrics.passed_cases == 2
    assert metrics.failed_cases == 1

    assert metrics.hit_at_k == 2 / 3
    assert metrics.top1_accuracy == 1 / 3
    assert metrics.filter_match_rate == 2 / 3
    assert metrics.empty_retrieval_rate == 1 / 3
    assert metrics.average_latency_ms == 200.0


def test_result_with_error_should_not_be_counted_as_passed() -> None:
    """
    测试存在 error_message 的结果不会被统计为通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    results = [
        RagEvalResult(
            case_id="dog_eval_001",
            question="问题 1",
            expected_dog_names=["Poodle"],
            expected_filters={},
            parsed_filters={},
            retrieved_dog_names=["Poodle"],
            hit=True,
            hit_rank=1,
            top1_hit=True,
            filter_matched=True,
            empty_retrieval=False,
            passed=True,
            error_message="模拟异常",
        )
    ]

    metrics = calculate_rag_eval_metrics(results)

    assert metrics.total_cases == 1
    assert metrics.passed_cases == 0
    assert metrics.failed_cases == 1
    assert metrics.hit_at_k == 1.0
    assert metrics.top1_accuracy == 1.0
    assert metrics.filter_match_rate == 1.0
    assert metrics.empty_retrieval_rate == 0.0


def test_average_latency_ignores_none() -> None:
    """
    测试平均耗时统计时会忽略 None。

    参数含义：
        无。

    返回值含义：
        None。
    """

    results = [
        RagEvalResult(
            case_id="dog_eval_001",
            question="问题 1",
            expected_dog_names=[],
            expected_filters={},
            parsed_filters={},
            retrieved_dog_names=[],
            hit=False,
            top1_hit=False,
            filter_matched=False,
            empty_retrieval=True,
            passed=False,
            latency_ms=None,
        ),
        RagEvalResult(
            case_id="dog_eval_002",
            question="问题 2",
            expected_dog_names=[],
            expected_filters={},
            parsed_filters={},
            retrieved_dog_names=[],
            hit=True,
            top1_hit=True,
            filter_matched=True,
            empty_retrieval=False,
            passed=True,
            latency_ms=120.0,
        ),
    ]

    metrics = calculate_rag_eval_metrics(results)

    assert metrics.average_latency_ms == 120.0