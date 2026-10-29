"""真实工具评估的最小统计与执行前边界检查，不代替实际模型运行。"""

import pytest
from scripts.evaluation.evaluate_real_tools import check_arguments, effective_arguments, summarize


def test_wrong_amount_is_failure_before_write():
    """错误提议必须拦截；无参数；返回None，不把金额解析错误算工具任务成功。"""
    with pytest.raises(ValueError):
        check_arguments("record_expense", {"amount": "800"},
                        {"tool": "record_expense", "expected_args": {"amount": "80.00"}})


def test_rejection_not_counted_as_recovery():
    """正确拒绝不计恢复完成；无参数；返回None，失败保持在原始分母。"""
    summary = summarize([
        {"kind": "normal", "passed": False},
        {"kind": "exception", "passed": True, "recoverable": False},
        {"kind": "exception", "passed": False, "recoverable": True}])
    assert summary["normal_tasks"] == {"passed": 0, "total": 1}
    assert summary["exception_handling"] == {"passed": 1, "total": 2}
    assert summary["recoverable_scenarios"] == {"passed": 0, "total": 1}


def test_contract_defaults_are_not_missing_business_parameters():
    """默认币种缺省不应误拒绝；无参数；返回None，仅采用现有契约默认值，不猜必填数据。"""
    normalized = effective_arguments("calculate_supply_budget", {
        "budget": "100", "price_source": "user_provided",
        "items": [{"name": "猫砂", "unit_price": "30", "quantity": 2}]})
    check_arguments("calculate_supply_budget", normalized,
        {"tool": "calculate_supply_budget", "expected_args": {"currency": "CNY"}})
