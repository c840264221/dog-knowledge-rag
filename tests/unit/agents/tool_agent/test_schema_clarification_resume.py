"""验证通用缺参恢复的关键边界，不调用模型、不写账。"""

from copy import deepcopy

import pytest

from src.agents.tool_agent.adapters.clarification_resume_adapter import resolve_tool_clarification_input
from src.agents.tool_agent.adapters.tool_call_validation_adapter import validate_tool_calls_from_state
from src.mcp.expenses.schemas import ExpenseBatchInput, ExpenseInput


@pytest.mark.parametrize("batch", [False, True])
def test_missing_amount_resume_preserves_other_values_and_action_key(batch):
    """验证单笔和批量补参；batch选择输入契约；无返回值，断言原键、日期和其他账单保留。"""
    detail = {"currency": "CNY", "category": "宠物", "subcategory": "洗护",
              "description": "狗子洗澡", "expense_date": "2026-10-24"}
    arguments = ({"items": [{**detail, "amount": "20.00"}, detail]} if batch else dict(detail))
    arguments["idempotency_key"] = "tool-action:original-test-key"
    before = deepcopy(arguments)
    name = "record_expenses" if batch else "record_expense"
    schema = (ExpenseBatchInput if batch else ExpenseInput).model_json_schema()
    state = {"question": "昨天狗子洗澡，帮我记账", "tool_calls": [{"name": name, "args": arguments}],
             "tool_agent_tool_catalog": [{"name": name, "input_schema": schema}]}
    state.update(validate_tool_calls_from_state(state))
    field = "items.1.amount" if batch else "amount"
    assert state["tool_agent_clarification_request"]["missing_fields"] == [field]
    assert not state["tool_calls"] and state["tool_results"] == []
    assert state["tool_agent_pending_tool_call"]["args"] == before

    # 明确归属旧任务但格式非法，仍保留等待状态，不执行，也不重新生成动作。
    state.update(question="abc", task_relation_decision={"relation": "resume"})
    invalid = resolve_tool_clarification_input(state)
    assert invalid["action"] == "partial"
    state.update(invalid["state_update"])
    assert state["tool_agent_pending_tool_call"]["args"] == before

    state["question"] = "80"
    resumed = resolve_tool_clarification_input(state)
    assert resumed["action"] == "resumed"
    state.update(resumed["state_update"])
    result = validate_tool_calls_from_state(state)
    assert result["tool_call_validation_ok"] is True
    saved = result["tool_calls"][0]["args"]
    assert saved["idempotency_key"] == before["idempotency_key"]
    assert arguments == before
    if batch:
        assert saved["items"][0] == before["items"][0]
        assert saved["items"][1] == {**detail, "amount": "80"}
    else:
        assert saved == {**before, "amount": "80"}


def test_generic_integer_field_converts_and_checks_bounds():
    """验证不是金额专用解析；无参数，无返回值，整数按工具契约转换并校验范围。"""
    state = {"question": "0", "task_relation_decision": {"relation": "resume"},
             "tool_agent_clarification_request": {"missing_fields": ["quantity"], "options": {},
                 "field_schemas": {"quantity": {"type": "integer", "minimum": 1, "maximum": 9}}},
             "tool_agent_pending_tool_call": {"name": "example", "args": {"idempotency_key": "same"}}}
    assert resolve_tool_clarification_input(state)["action"] == "partial"
    state["question"] = "2"
    result = resolve_tool_clarification_input(state)
    assert result["state_update"]["tool_calls"][0]["args"] == {"idempotency_key": "same", "quantity": 2}
