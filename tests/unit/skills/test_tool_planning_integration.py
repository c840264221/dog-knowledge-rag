"""只验证技能扩展与程序参数的关键边界，不调用模型或写账。"""

from pathlib import Path
import json
from types import SimpleNamespace
import pytest

from src.graph.tools.runtime.argument_binding import bind_runtime_arguments
from src.graph.tools.schemas.tool_call_schema import ToolCall, ToolParseResult
from src.skills.file_loader import load_file_skill_registry, select_agent_skill, build_file_skill_context
from src.skills.selector import SkillSelector
from src.skills.tool_planning import load_tool_planning_registry, build_tool_planning_context, select_tool_planning_skill


SKILL_FILE = str(Path(__file__).resolve().parents[3] / "skills/expense-recording/SKILL.md")


@pytest.mark.asyncio
async def test_bound_tool_preloads_declared_skill_and_only_exposes_bound_tool():
    """验证明确工具步骤；无参数、无返回值，无触发词时按绑定预加载技能，只用一轮模型提议且不暴露其他工具。"""
    from src.agents.tool_agent.nodes.tool_parse_node import build_llm_tool_parser
    from src.runtime.context import RuntimeContext, runtime_ctx

    prompts = []

    async def fake_call(**kwargs):
        """记录解析提示；kwargs为调用参数；返回固定单笔提议，不执行账本写入。"""
        prompts.append(kwargs["prompt"].to_string())
        return json.dumps({"need_tool": True, "tool_calls": [{"name": "record_expense", "args": {"amount": "99.90"}}]})

    context = RuntimeContext()
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        parser = build_llm_tool_parser(llm_provider=SimpleNamespace(backup_llm=object(), safe_ainvoke=fake_call))
        result = await parser({"question": "保存这笔消费", "state": {
            "tool_agent_execution_contract": {"execution_mode": "tool", "tool_name": "record_expense"},
            "tool_agent_tool_catalog": [
                {"name": "record_expense", "input_schema": {"x-planning-skill": "expense-recording"}},
                {"name": "record_expenses", "input_schema": {"x-planning-skill": "expense-recording"}},
                {"name": "query_expenses", "input_schema": {}},
            ],
        }})
        assert len(prompts) == 1 and result.tool_calls[0].name == "record_expense"
        assert "自然语言支出记账" in prompts[0]
        assert "query_expenses" not in prompts[0]
        assert context.metadata["tool_planning_skill"]["loading_reason"] == "tool_binding"
    finally:
        runtime_ctx.set(previous)


@pytest.mark.asyncio
async def test_tool_dependency_loads_unmatched_skill_and_replans_once(tmp_path, monkeypatch):
    """验证通用工具依赖加载；tmp_path存新技能、monkeypatch注入目录；无返回值，不靠记账关键词也能加载。"""
    from src.agents.tool_agent.nodes import tool_parse_node as module
    from src.runtime.context import RuntimeContext, runtime_ctx

    path = tmp_path / "SKILL.md"
    path.write_text("""---
name: dispatch-policy
description: 规范物流登记提议。
metadata:
  target_agents: [tool_agent]
  activation_hints: [不会出现在本次问题中的触发短语]
  allowed_tools: [register_dispatch]
---
承运商和运单号必须来自输入，不生成确认状态。
""", encoding="utf-8")
    registry, locations = load_file_skill_registry([path])
    monkeypatch.setattr(module, "load_tool_planning_registry", lambda: (registry, locations))
    prompts = []

    async def fake_call(**kwargs):
        """记录真实规划提示；kwargs为调用参数；返回固定提议，不调用外部工具。"""
        prompts.append(kwargs["prompt"].to_string())
        return json.dumps({"need_tool": True, "tool_calls": [{"name": "register_dispatch", "args": {"number": "A001"}}]})

    ctx = RuntimeContext()
    previous = runtime_ctx.get()
    runtime_ctx.set(ctx)
    try:
        parser = module.build_llm_tool_parser(llm_provider=SimpleNamespace(backup_llm=object(), safe_ainvoke=fake_call))
        result = await parser({"question": "登记运单A001", "state": {"tool_agent_tool_catalog": [{
            "name": "register_dispatch", "input_schema": {"type": "object", "x-planning-skill": "dispatch-policy"},
        }]}})
        assert len(prompts) == 2
        assert "承运商和运单号必须来自输入" not in prompts[0]
        assert "承运商和运单号必须来自输入" in prompts[1]
        assert result.tool_calls[0].name == "register_dispatch"
        assert ctx.metadata["tool_planning_skill"]["loading_reason"] == "tool_requirement"
        assert ctx.metadata["tool_planning_skill"]["planning_attempt_count"] == 2
    finally:
        runtime_ctx.set(previous)


@pytest.mark.asyncio
async def test_dependency_text_does_not_activate_unrelated_skill(monkeypatch):
    """验证当前步骤选择隔离；monkeypatch注入模型替身；无返回值，前置结果中的记账词不限制只读查询。"""
    from src.agents.tool_agent.nodes import tool_parse_node as module

    calls = []

    async def fake_call(**kwargs):
        """提供只读提议；kwargs为调用参数；记录次数并返回合法JSON。"""
        calls.append(kwargs)
        return json.dumps({"need_tool": True, "tool_calls": [{"name": "query_expenses", "args": {"limit": 1}}]})

    parser = module.build_llm_tool_parser(llm_provider=SimpleNamespace(backup_llm=object(), safe_ainvoke=fake_call))
    result = await parser({"question": "查询最新一条\n前置结果：已记账", "state": {
        "retrieval_question": "查询最新一条", "tool_agent_tool_catalog": [
            {"name": "query_expenses", "input_schema": {}},
            {"name": "record_expense", "input_schema": {"x-planning-skill": "expense-recording"}},
            {"name": "record_expenses", "input_schema": {"x-planning-skill": "expense-recording"}},
        ],
    }})
    assert len(calls) == 1
    assert result.tool_calls[0].name == "query_expenses"


def test_skill_requires_available_tools_and_loads_reference():
    """校验加载与筛选分离；无参数；无返回值，工具不齐时仍加载定义，但不会选择依赖它的技能。"""
    question = "昨天狗子洗澡80元，帮我记账"
    registry, locations = load_tool_planning_registry([SKILL_FILE])
    assert registry.get("expense-recording") is not None
    assert select_tool_planning_skill(question, registry, locations, {"weather"}) is None
    selected = select_tool_planning_skill(
        question, registry, locations, {"record_expense", "record_expenses"},
    )
    context, metadata = build_tool_planning_context(selected, registry, locations)
    assert metadata["skill_id"] == "expense-recording"
    assert metadata["reference_paths"]
    assert '"classification_rules"' in context
    assert select_tool_planning_skill("查询成都天气", registry, locations, {"record_expense", "record_expenses"}) is None


def test_rule_only_skill_and_agent_filter_select_once(tmp_path, monkeypatch):
    """验证纯规则技能和单次选择；tmp_path为临时目录、monkeypatch记录选择次数；无返回值。"""
    path = tmp_path / "SKILL.md"
    path.write_text("""---
name: interview-format
description: 整理面试回答结构。
metadata:
  version: "1.0.0"
  target_agents: [general_agent]
  activation_hints: [整理面试回答]
  output_contract: 按问题、方案、验证结果、限制组织回答。
---
只整理已有经历，不编造数据。
""", encoding="utf-8")
    registry, locations = load_file_skill_registry([path])
    calls = []
    original = SkillSelector.select

    def track_selection(self, question):
        """记录真实选择调用；self为选择器、question为问题；返回原选择结果。"""
        calls.append(question)
        return original(self, question)

    monkeypatch.setattr(SkillSelector, "select", track_selection)
    skill = select_agent_skill("整理面试回答", "general_agent", registry, locations, set())
    assert skill is not None and skill.allowed_tools == []
    context, metadata = build_file_skill_context(skill, registry, locations)
    assert len(calls) == 1
    assert metadata["skill_id"] == "interview-format"
    assert "按问题、方案、验证结果、限制组织回答" in context
    assert "ToolParseResult" not in context
    assert select_agent_skill("整理面试回答", "tool_agent", registry, locations, set()) is None


def test_runtime_key_ignores_model_value_without_changing_business_fields():
    """校验程序动作键所有权；无参数；无返回值，新调用独立生成，业务参数与原提议不被修改。"""
    proposal = ToolParseResult(need_tool=True, tool_calls=[
        ToolCall(name="write_tool", args={"amount": "35.00", "idempotency_key": "model-invented"}),
        ToolCall(name="date", args={}),
    ])
    catalog = [{"name": "write_tool", "input_schema": {"properties": {
        "idempotency_key": {"type": "string", "x-runtime-generated": "uuid"},
    }}}]
    first = bind_runtime_arguments(proposal, catalog)
    second = bind_runtime_arguments(proposal, catalog)
    assert first.tool_calls[0].args["idempotency_key"].startswith("tool-action:")
    assert first.tool_calls[0].args["idempotency_key"] != second.tool_calls[0].args["idempotency_key"]
    assert first.tool_calls[0].args["amount"] == "35.00"
    assert proposal.tool_calls[0].args["idempotency_key"] == "model-invented"
    assert first.tool_calls[1] == proposal.tool_calls[1]
