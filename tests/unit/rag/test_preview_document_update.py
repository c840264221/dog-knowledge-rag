"""单文档更新预览测试：只使用临时文件及模拟存储，不连接真实向量库。"""

from unittest.mock import Mock

import pytest

from scripts.rag.preview_document_update import preview_document_update


def test_preview_classifies_ids_without_writes(tmp_path):
    """验证编号差集和标题路径，且存储仅被读取。

    参数：tmp_path 为 pytest 提供的临时目录。
    返回：无；通过断言检查预览结果与唯一的存储调用。
    """
    path = tmp_path / "dog.md"
    path.write_text("# 金毛\n## 养护指南\n### 运动\n运动正文", encoding="utf-8")
    store = Mock()
    store.get.return_value = {"ids": [], "metadatas": []}
    first = preview_document_update(path, store)
    same_id = first["new_ids"][0]

    store.reset_mock()
    store.get.return_value = {
        "ids": [same_id, "old-heading-only"],
        "metadatas": [{"doc_id": first["doc_id"]}] * 2,
    }
    result = preview_document_update(path, store)

    assert result["same_ids"] == [same_id]
    assert result["new_ids"] == []
    assert result["obsolete_ids"] == ["old-heading-only"]
    assert result["new_chunks"][0]["heading_path"] == "金毛 > 养护指南 > 运动"
    assert result["mode"] == "preview_only"
    store.get.assert_called_once_with(
        where={"doc_id": first["doc_id"]}, include=["metadatas"],
    )
    assert len(store.mock_calls) == 1


def test_preview_rejects_wrong_document_and_empty_body(tmp_path):
    """验证非目标文档和无正文文件不能产生清理预览。

    参数：tmp_path 为 pytest 临时目录。
    返回：无；通过异常断言验证保护条件。
    """
    path = tmp_path / "dog.md"
    path.write_text("# 金毛\n运动正文", encoding="utf-8")
    store = Mock()
    store.get.return_value = {
        "ids": ["foreign"], "metadatas": [{"doc_id": "another-document"}],
    }
    with pytest.raises(ValueError, match="非目标文档"):
        preview_document_update(path, store)

    store.reset_mock()
    path.write_text("# 金毛\n## 养护指南", encoding="utf-8")
    with pytest.raises(ValueError, match="没有生成正文块"):
        preview_document_update(path, store)
    store.get.assert_not_called()
