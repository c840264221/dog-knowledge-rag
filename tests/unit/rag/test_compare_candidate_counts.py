"""候选数量对照的离线测试，不访问真实模型或索引库。"""

from unittest.mock import Mock

from scripts.rag import compare_candidate_counts as demo


def test_comparison_changes_candidates_but_keeps_final_count(monkeypatch):
    """检查两组仅改变候选数量、保留完整重排排名且最终只取五条。

    参数：monkeypatch 为 pytest 的依赖替换工具。
    返回：无；以模拟的第八位运动块验证候选与最终选择的区别。
    """
    items = [
        {
            "chunk": {
                "chunk_id": f"chunk-{index}",
                "content": "测试正文",
                "metadata": {
                    "dog_name": "Golden Retriever",
                    "section_title": "Exercise" if index == 7 else "Other",
                },
            },
            "retrieval_score": float(index),
        }
        for index in range(10)
    ]
    retriever = Mock()
    retriever.retrieve.side_effect = [
        Mock(model_dump=Mock(return_value={"chunks": items[:5]})),
        Mock(model_dump=Mock(return_value={"chunks": items})),
    ]
    reorder = Mock(side_effect=[
        {"rag_context": {"chunks": items[:5]}},
        {"rag_context": {"chunks": [items[7], *items[:7], *items[8:]]}},
    ])
    monkeypatch.setattr(demo, "rerank_rag_context_chunks", reorder)

    report = demo.compare_candidate_counts(retriever, Mock())

    queries = [call.kwargs["query"] for call in retriever.retrieve.call_args_list]
    assert [query.top_k for query in queries] == [5, 10]
    assert queries[0].question == queries[1].question == demo.QUESTION
    assert queries[0].filters == queries[1].filters == demo.FILTERS
    assert [call.kwargs["top_k"] for call in reorder.call_args_list] == [5, 10]
    small, large = report["groups"]
    assert small["retrieval"]["exercise_ranks"] == []
    assert large["retrieval"]["exercise_ranks"] == [8]
    assert large["reranked"]["exercise_ranks"] == [1]
    assert not small["exercise_in_final_top_k"]
    assert large["exercise_in_final_top_k"]
    assert len(large["selected_chunk_ids"]) == 5
    assert report["five_candidates_subset_of_ten"]
