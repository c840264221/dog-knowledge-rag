"""验证维护更新顺序及失败保护；不访问真实向量库或向量模型。"""

from unittest.mock import Mock

import pytest

from scripts.rag.preview_document_update import prepare_document_chunks
from scripts.rag.update_document_index import update_document_index


@pytest.fixture
def update_case(tmp_path):
    """准备单篇源文档及仅存在于内存中的模拟存储。

    参数：tmp_path 为 pytest 提供的临时目录。
    返回：路径、文档编号、模拟存储、记录字典及操作序列。
    """
    path = tmp_path / "golden.md"
    path.write_text("# 金毛\n## 养护指南\n### 运动\n运动正文", encoding="utf-8")
    document, _ = prepare_document_chunks(path)
    rows = {
        "old-heading": {
            "content": "## 养护指南",
            "metadata": {"doc_id": document.doc_id},
        },
        "other-document": {
            "content": "其他文档",
            "metadata": {"doc_id": "other"},
        },
    }
    events = []
    store = Mock()

    def get(*, where, include):
        """模拟按文档读取。参数为过滤条件及返回字段；返回 Chroma 风格数据。"""
        events.append("get")
        selected = [
            (key, row) for key, row in rows.items()
            if row["metadata"]["doc_id"] == where["doc_id"]
        ]
        return {
            "ids": [key for key, _ in selected],
            "documents": [row["content"] for _, row in selected],
            "metadatas": [row["metadata"] for _, row in selected],
        }

    def add_documents(*, documents, ids):
        """模拟新增或覆盖。参数为正文对象和编号；返回本次编号列表。"""
        events.append("write")
        for key, doc in zip(ids, documents):
            rows[key] = {"content": doc.page_content, "metadata": dict(doc.metadata)}
        return ids

    def delete(*, ids):
        """模拟精确删除。参数为待删编号；返回无，不涉及其他记录。"""
        events.append("delete")
        for key in ids:
            rows.pop(key, None)

    store.get.side_effect = get
    store.add_documents.side_effect = add_documents
    store.delete.side_effect = delete
    return path, document.doc_id, store, rows, events


def test_update_verifies_before_cleanup_and_can_repeat(update_case):
    """验证先写后查再删、不影响其他文档及同源重跑。

    参数：update_case 为模拟更新环境。
    返回：无，通过断言验证顺序和最终状态。
    """
    path, doc_id, store, rows, events = update_case
    result = update_document_index(
        path, store, confirmed_doc_id=doc_id, maintenance_confirmed=True,
    )
    assert events == ["get", "write", "get", "delete", "get"]
    assert result["verified"] is True
    assert result["removed_obsolete_count"] == 1
    assert "old-heading" not in rows
    assert rows["other-document"]["content"] == "其他文档"
    store.delete.assert_called_once_with(ids=["old-heading"])

    repeated = update_document_index(
        path, store, confirmed_doc_id=doc_id, maintenance_confirmed=True,
    )
    assert repeated["removed_obsolete_count"] == 0
    assert repeated["final_chunk_count"] == result["final_chunk_count"]
    assert store.delete.call_count == 1


@pytest.mark.parametrize("failure_mode", ["write_error", "missing_new_rows"])
def test_update_does_not_cleanup_when_new_data_not_verified(update_case, failure_mode):
    """验证写入异常或写后缺块时不清理旧记录。

    参数：update_case 为模拟环境；failure_mode 为失败类型。
    返回：无，通过异常及未删除断言验证保护逻辑。
    """
    path, doc_id, store, rows, _ = update_case
    store.add_documents.side_effect = (
        RuntimeError("模拟写入失败") if failure_mode == "write_error" else None
    )
    with pytest.raises(RuntimeError):
        update_document_index(
            path, store, confirmed_doc_id=doc_id, maintenance_confirmed=True,
        )
    store.delete.assert_not_called()
    assert "old-heading" in rows


def test_update_requires_maintenance_and_matching_identity(update_case):
    """验证未确认维护状态或文档身份不一致时不访问存储。

    参数：update_case 为模拟环境。
    返回：无，通过异常和未调用断言验证入口保护。
    """
    path, doc_id, store, _, _ = update_case
    with pytest.raises(ValueError, match="暂停"):
        update_document_index(path, store, confirmed_doc_id=doc_id)
    with pytest.raises(ValueError, match="编号"):
        update_document_index(
            path, store, confirmed_doc_id="wrong", maintenance_confirmed=True,
        )
    assert store.mock_calls == []
