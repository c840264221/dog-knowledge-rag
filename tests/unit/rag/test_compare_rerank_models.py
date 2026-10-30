"""模型对照及提供者配置隔离测试；不加载真实模型。"""

from copy import deepcopy
from unittest.mock import Mock

from scripts.rag import compare_rerank_models as demo
from src.runtime.container.providers import reranker_provider as provider_module


def test_model_comparison_reuses_candidate_snapshot(monkeypatch):
    """验证只检索一次，两个模型接收同一问题和相同候选。

    参数：monkeypatch 为 pytest 的依赖替换工具。
    返回：无；通过调用参数和结果断言验证实验变量。
    """
    items = [{
        "chunk": {
            "chunk_id": "exercise",
            "content": "Daily exercise.",
            "metadata": {"dog_name": "Golden Retriever", "section_title": "Exercise"},
        },
    }]
    retriever = Mock()
    retriever.retrieve.return_value.model_dump.return_value = {"chunks": items}
    inputs = []

    def reorder(**kwargs):
        """记录输入并返回候选。参数为重排参数；返回模拟重排结果。"""
        inputs.append((kwargs["question"], deepcopy(kwargs["chunks"])))
        return {"rag_context": {"chunks": kwargs["chunks"]}}

    monkeypatch.setattr(demo, "rerank_rag_context_chunks", reorder)
    report = demo.compare_rerank_models(
        retriever, {"baseline": Mock(), "candidate": Mock()},
    )
    retriever.retrieve.assert_called_once()
    assert inputs[0] == inputs[1]
    assert [group["configured_model"] for group in report["groups"]] == [
        "baseline", "candidate",
    ]
    assert all(group["exercise_in_final_top_k"] for group in report["groups"])


def test_provider_uses_isolated_config_without_changing_default(monkeypatch):
    """验证提供者支持独立配置，且旧构造方式保持不变。

    参数：monkeypatch 为 pytest 的依赖替换工具。
    返回：无；替换模型构造函数，不下载权重。
    """
    original_name = provider_module.settings.reranker.model_name
    custom = provider_module.settings.reranker.model_copy(
        update={"model_name": "test-model", "device": "cpu", "huggingface_token": None}
    )
    constructor = Mock()
    monkeypatch.setattr(provider_module, "CrossEncoder", constructor)
    provider = provider_module.RerankerProvider(reranker_settings=custom)
    assert provider.reranker is provider.reranker
    constructor.assert_called_once_with("test-model", model_kwargs={}, device="cpu")
    assert provider_module.settings.reranker.model_name == original_name
    assert provider_module.RerankerProvider()._settings is provider_module.settings.reranker
