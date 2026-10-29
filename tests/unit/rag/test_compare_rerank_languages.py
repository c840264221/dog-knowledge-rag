"""语言对照的离线测试；不连接真实索引或调用模型。"""

from copy import deepcopy
from unittest.mock import Mock

from scripts.rag import compare_rerank_languages as demo


def test_languages_share_candidates_and_retrieve_only_once(monkeypatch):
    """验证只召回一次、两组正文及顺序一致，并隔离组间修改。

    参数：monkeypatch 为 pytest 的依赖替换工具。
    返回：无；通过输入快照和调用次数断言验证控制变量。
    """
    items = [{
        "chunk": {
            "chunk_id": "exercise",
            "content": "Daily exercise information.",
            "metadata": {
                "dog_name": "Golden Retriever",
                "section_title": "Exercise",
            },
        },
    }]
    retriever = Mock()
    retriever.retrieve.return_value.model_dump.return_value = {"chunks": items}
    inputs = []

    def reorder(**kwargs):
        """保存重排输入并模拟修改。参数为重排参数；返回模拟重排上下文。"""
        inputs.append(deepcopy(kwargs))
        kwargs["chunks"][0]["chunk"]["content"] = "模拟内部修改"
        return {"rag_context": {"chunks": kwargs["chunks"]}}

    monkeypatch.setattr(demo, "rerank_rag_context_chunks", reorder)
    report = demo.compare_rerank_languages(retriever, Mock())

    retriever.retrieve.assert_called_once()
    query = retriever.retrieve.call_args.kwargs["query"]
    assert query.question == demo.CHINESE_QUESTION
    assert query.top_k == 10
    assert len(inputs) == 2
    assert inputs[0]["question"] == demo.CHINESE_QUESTION
    assert inputs[1]["question"] == demo.ENGLISH_QUESTION
    assert inputs[0]["chunks"] == inputs[1]["chunks"] == items
    assert items[0]["chunk"]["content"] == "Daily exercise information."
    assert report["groups"][0]["input_chunk_ids"] == report["groups"][1]["input_chunk_ids"]
    assert all(group["exercise_in_final_top_k"] for group in report["groups"])
