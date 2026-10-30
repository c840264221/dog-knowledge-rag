"""只检查新案例入口的目录隔离和执行边界，不调用模型或记账服务。"""

import pytest

from scripts.run_expense_collaboration_case import check_call_scope, restrict_catalog


def test_catalog_is_filtered_without_mutating_shared_definitions():
    """验证目录隔离；无参数、无返回值，筛选结果独立且不向模型暴露其他工具。"""
    original = [{"name": "query_expenses", "input_schema": {"required": ["limit"]}},
                {"name": "sqlite_execute", "input_schema": {}}]
    selected = restrict_catalog(original, {"query_expenses"})
    assert [item["name"] for item in selected] == ["query_expenses"]
    selected[0]["input_schema"]["required"].append("extra")
    assert original[0]["input_schema"]["required"] == ["limit"]


def test_execution_scope_rejects_unknown_or_combined_calls():
    """验证执行前检查；无参数、无返回值，未知工具与同一步混合操作均拒绝。"""
    allowed = {"query_expenses", "record_expense"}
    assert check_call_scope({"tool_calls": []}, allowed) == {}
    assert check_call_scope({"tool_calls": [{"name": "query_expenses"}]}, allowed) == {}
    with pytest.raises(ValueError, match="范围"):
        check_call_scope({"tool_calls": [{"name": "sqlite_execute"}]}, allowed)
    with pytest.raises(ValueError, match="独立步骤"):
        check_call_scope({"tool_calls": [{"name": "record_expense"}, {"name": "query_expenses"}]}, allowed)


