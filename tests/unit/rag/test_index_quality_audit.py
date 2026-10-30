"""索引审计规则测试：纯数据，不使用真实向量库或模型。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rag.evaluation.index_quality_audit import audit_index_records, is_heading_only
from src.runtime.container.providers import vectorstore_provider as vector_module


@pytest.mark.parametrize("text,expected", [
    ("## 养护指南", True), ("\n## Parent\n### Child\n", True),
    ("## Exercise\nDaily exercise is needed.", False), ("Drink water.", False),
    ("", False), ("#hashtag", False), ("Exercise", False),
])
def test_heading_detection(text, expected):
    """验证标题规则不误伤短正文；参数为输入与预期；无返回值。"""
    assert is_heading_only(text) is expected


def test_audit_counts_and_finds_exercise():
    """验证噪声计数、文档去重及目标正文定位；无参数，无返回值。"""
    report = audit_index_records({
        "ids": ["h1", "h2", "exercise", "empty"],
        "documents": ["## Care", "## Care", "### Exercise\nGoldens need daily exercise.", ""],
        "metadatas": [
            {"doc_id": "other"}, {"doc_id": "other"},
            {"title": "Golden Retriever", "doc_id": "golden"}, {},
        ],
    })
    assert report["heading_only_count"] == 2
    assert report["affected_document_count"] == 1
    assert report["empty_content_count"] == 1
    assert report["heading_only_fraction"] == 0.5
    assert report["explicit_exercise_record_count"] == 1
    assert report["explicit_exercise_records"][0]["record_id"] == "exercise"


def test_source_matching_does_not_match_another_breed():
    """验证来源文件可确定目标且不把其他犬种运动混入；无参数，无返回值。"""
    report = audit_index_records({
        "ids": ["golden", "other"],
        "documents": ["### Exercise\nBody.", "### Exercise\nOther body."],
        "metadatas": [{"source": "D:/data/Golden_Retriever.md"}, {"title": "Corgi"}],
    })
    assert report["target_record_count"] == 1
    assert report["explicit_exercise_records"][0]["record_id"] == "golden"


def test_misaligned_snapshot_is_rejected():
    """验证三列长度不一致时拒绝审计，避免错误配对；无参数，无返回值。"""
    with pytest.raises(ValueError):
        audit_index_records({"ids": ["id"], "documents": [], "metadatas": []})


def test_audit_provider_does_not_require_embedding(monkeypatch):
    """验证审计提供者传入空向量函数且禁止建集合；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    provider = vector_module.VectorStoreProvider(
        None, enable_memory=False, collection_name="langchain", create_collection_if_not_exists=False,
    )
    _ = provider.db
    assert constructor.call_args.kwargs["embedding_function"] is None
    assert constructor.call_args.kwargs["create_collection_if_not_exists"] is False


def test_regular_provider_still_uses_embedding(monkeypatch):
    """验证普通提供者仍注入原向量模型；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    embedding = object()
    provider = vector_module.VectorStoreProvider(SimpleNamespace(embedding=embedding))
    _ = provider.db
    assert constructor.call_args.kwargs["embedding_function"] is embedding
