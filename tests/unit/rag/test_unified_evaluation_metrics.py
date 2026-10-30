"""评估统计关键口径：失败不能从分母删除，双主题必须分别命中。"""

from scripts.rag.evaluate_unified_knowledge import MODES, summarize


def test_failed_request_stays_in_denominator_and_partial_is_not_full():
    """检查失败和部分覆盖；无参数；返回None，一命中不能冒充两个主题全部完成。"""
    variant = {"topics": [{"topic_id": "a", "first_expected_rank": 2},
                           {"topic_id": "b", "first_expected_rank": None}]}
    report = {"dataset": {"cases": [{"expected_groups": [1, 2]}, {"expected_groups": [3]}]},
              "cases": [{"variants": {mode: variant for mode in MODES}}, {"variants": {}}]}
    summary = summarize(report)
    for result in summary.values():
        assert result["evidence_topic_count"] == 3
        assert result["completed_requests"] == 1
        assert result["hit_rate_at_5"] == 1 / 3
        assert result["mrr_at_5"] == .5 / 3
        assert result["fully_covered_requests"] == 0
