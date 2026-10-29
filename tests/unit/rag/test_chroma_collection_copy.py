"""原向量复制测试：内存替身验证，不初始化Chroma或下载模型。"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rag.indexers.chroma_collection_copy import copy_without_headings, records_fingerprint
from src.runtime.container.providers import vectorstore_provider as vector_module


class FakeCollection:
    """支持显式向量增添与读取的测试替身，不具备向量化能力。"""

    def __init__(self, name: str, records: dict):
        """保存名称与记录；参数name为名称，records为记录快照；无返回值。"""
        self.name = name
        self.records = deepcopy(records)
        self.add_calls = 0
        self.schema = SimpleNamespace(serialize_to_json=lambda: {
            "keys": {"#embedding": {"float_list": {"vector_index": {"config": {
                "space": "l2", "hnsw": {"ef_search": 100, "max_neighbors": 16},
            }}}}},
        })

    def get(self, **kwargs):
        """按替身规则返回完整记录；kwargs为兼容读取参数；返回深拷贝字典。"""
        return deepcopy(self.records)

    def count(self):
        """获取当前条数；无参数；返回整数。"""
        return len(self.records["ids"])

    def add(self, *, ids, documents, metadatas, embeddings):
        """只接收明确提供的向量；参数为四列等长数据；追加后无返回值。"""
        self.add_calls += 1
        for key, values in (("ids", ids), ("documents", documents), ("metadatas", metadatas), ("embeddings", embeddings)):
            self.records[key].extend(deepcopy(values))


def make_records() -> dict:
    """准备一个标题及两个正文记录，各有原向量；无参数；返回四列快照。"""
    return {
        "ids": ["heading", "exercise", "short"],
        "documents": ["## Care", "### Exercise\nDaily exercise.", "Drink water."],
        "metadatas": [{"doc_id": "d"}, {"doc_id": "d", "section_title": "Exercise"}, {"doc_id": "e"}],
        "embeddings": [[1.0, 0.0], [0.25, 0.75], [0.5, 0.5]],
    }


def empty_records() -> dict:
    """创建空四列记录；无参数；返回新字典，不共享可变列表。"""
    return {key: [] for key in ("ids", "documents", "metadatas", "embeddings")}


def test_copy_keeps_vectors_metadata_and_original_source():
    """验证排除标题且向量与元数据原样保留，源不变；无参数，无返回值。"""
    records = make_records()
    source = FakeCollection("source", records)
    target = FakeCollection("target", empty_records())
    report = copy_without_headings(source, target, records, batch_size=1)
    assert target.records["ids"] == ["exercise", "short"]
    assert target.records["embeddings"] == records["embeddings"][1:]
    assert target.records["metadatas"] == records["metadatas"][1:]
    assert source.records == records
    assert target.add_calls == 2
    assert report["embedding_reused"] is True
    assert report["source_unchanged_verified"] is True


def test_nonempty_target_is_never_overwritten():
    """验证已有目标拒绝写入而不是先清空；无参数，无返回值。"""
    records = make_records()
    target = FakeCollection("target", records)
    with pytest.raises(ValueError, match="已有数据"):
        copy_without_headings(FakeCollection("source", records), target, records)
    assert target.add_calls == 0
    assert target.records == records


def test_changed_source_is_rejected_before_add():
    """验证源集合与读取快照不同就停止；无参数，无返回值。"""
    records = make_records()
    source = FakeCollection("source", records)
    source.records["documents"][1] = "changed"
    target = FakeCollection("target", empty_records())
    with pytest.raises(ValueError, match="源集合发生变化"):
        copy_without_headings(source, target, records)
    assert target.add_calls == 0


def test_fingerprint_ignores_record_order_but_not_vector_changes():
    """验证只调整返回顺序不变指纹，向量变化可检出；无参数，无返回值。"""
    records = make_records()
    reordered = {key: list(reversed(deepcopy(values))) for key, values in records.items()}
    assert records_fingerprint(records) == records_fingerprint(reordered)
    reordered["embeddings"][0][0] = 0.9
    assert records_fingerprint(records) != records_fingerprint(reordered)


def test_provider_passes_explicit_index_configuration(monkeypatch):
    """验证提供者传递索引配置且仍不注入模型；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    configuration = {"hnsw": {"space": "l2"}}
    provider = vector_module.VectorStoreProvider(None, enable_memory=False, collection_configuration=configuration)
    _ = provider.native_collection
    assert constructor.call_args.kwargs["collection_configuration"] == configuration
    assert constructor.call_args.kwargs["embedding_function"] is None
