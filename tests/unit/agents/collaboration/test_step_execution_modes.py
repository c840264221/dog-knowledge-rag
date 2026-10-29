"""验证能力级与明确工具步骤的正式契约及执行约束，不调用真实模型或外部工具。"""

import json

import pytest
from pydantic import ValidationError

from src.agents.collaboration.contracts import AgentTaskStep
from src.agents.collaboration.planner.output_parser import parse_planner_output
from src.agents.collaboration.workers.tool_agent_worker_adapter import ToolAgentWorkerAdapter, build_tool_execution_guard
from src.agents.tool_agent.graph import build_tool_agent_graph
from tests.unit.agents.tool_agent.test_tool_agent_graph import FakeAinvokeParser, FakeExecutor, build_fake_registry


def test_execution_contract_defaults_and_legacy_promotion():
    """验证兼容与冲突；无参数、无返回值，旧工具字段提升为正式绑定，能力任务默认保持可选工具。"""
    capability = AgentTaskStep(step_id="a", title="查询资料", assigned_agent="dog_knowledge_agent")
    assert capability.execution_mode == "capability" and capability.tool_name is None
    original = {"tool_name": "weather", "question": "查询成都天气"}
    tool = AgentTaskStep(step_id="b", title="查天气", assigned_agent="tool_agent", input_data=original)
    assert tool.execution_mode == "tool" and tool.tool_name == "weather"
    assert "tool_name" not in tool.input_data
    assert original["tool_name"] == "weather"
    with pytest.raises(ValidationError):
        AgentTaskStep(step_id="b", title="查天气", assigned_agent="tool_agent", execution_mode="tool")
    with pytest.raises(ValidationError):
        AgentTaskStep(step_id="b", title="查天气", assigned_agent="tool_agent", execution_mode="capability", tool_name="weather")
    with pytest.raises(ValidationError):
        AgentTaskStep(step_id="b", title="查天气", assigned_agent="tool_agent", tool_name="weather", input_data={"tool": "date"})


def test_planner_checks_bound_tool_against_supplied_directory():
    """验证规划预检；无参数、无返回值，已有目录时未知绑定直接拒绝，不留到外部调用。"""
    raw = json.dumps({"plan_id": "p", "objective": "查询天气", "steps": [{
        "step_id": "a", "title": "查天气", "assigned_agent": "tool_agent", "execution_mode": "tool", "tool_name": "weather",
    }]})
    options = dict(raw_output=raw, expected_plan_id="p", expected_objective="查询天气",
                   allowed_agent_names={"tool_agent"}, maximum_steps=2)
    assert parse_planner_output(**options, allowed_tool_names={"weather"}).steps[0].tool_name == "weather"
    with pytest.raises(ValueError, match="目录不存在"):
        parse_planner_output(**options, allowed_tool_names={"date"})


@pytest.mark.asyncio
async def test_bound_step_rejects_other_tool_but_capability_step_can_choose_it():
    """验证真实子图执行边界；无参数、无返回值，能力步骤可选日期，绑定天气的步骤不能执行日期。"""
    parser = FakeAinvokeParser({"need_tool": True, "tool_calls": [{"name": "date", "args": {}}]})
    executor = FakeExecutor()

    async def runner(state):
        """运行既有工具子图；state为宿主步骤状态；返回离线执行结果。"""
        graph = build_tool_agent_graph(parser=parser, executor=executor, tool_registry=build_fake_registry(),
            interrupt_func=None, before_execute=build_tool_execution_guard(None, state.get("tool_agent_execution_contract")))
        return await graph.ainvoke(state)

    worker = ToolAgentWorkerAdapter(runner)
    bound = AgentTaskStep(step_id="a", title="查天气", assigned_agent="tool_agent", execution_mode="tool", tool_name="weather")
    rejected = await worker(bound, {})
    assert rejected.status == "failed" and executor.calls == []
    assert bound.tool_name == "weather"
    capability = AgentTaskStep(step_id="b", title="读取日期", assigned_agent="tool_agent")
    completed = await worker(capability, {})
    assert completed.status == "completed"
    assert executor.calls == [{"tool_name": "date", "args": {}}]


@pytest.mark.asyncio
async def test_changed_binding_cannot_reuse_previous_confirmation():
    """验证确认与绑定关联；无参数、无返回值，改绑工具后旧确认失效，不调用执行器。"""
    parser = FakeAinvokeParser({"need_tool": True, "tool_calls": [{"name": "weather", "args": {"city": "成都"}}]})
    executor = FakeExecutor()

    async def runner(state):
        """运行离线工具图；state为步骤状态；返回待确认状态或工具结果。"""
        expected = state.get("tool_calls") if state.get("tool_confirmed") == "confirmed" else None
        graph = build_tool_agent_graph(parser=parser, executor=executor, tool_registry=build_fake_registry(),
            interrupt_func=None, before_execute=build_tool_execution_guard(expected, state.get("tool_agent_execution_contract")))
        return await graph.ainvoke(state)

    worker = ToolAgentWorkerAdapter(runner)
    step = AgentTaskStep(step_id="a", title="查天气", assigned_agent="tool_agent", execution_mode="tool", tool_name="weather")
    waiting = await worker(step, {})
    assert waiting.status == "awaiting_input"
    changed = step.model_copy(deep=True)
    changed.tool_name = "date"
    changed.input_data.update(multi_agent_is_resuming=True, multi_agent_previous_worker_output=waiting.output, multi_agent_resume_input="确认")
    rejected = await worker(changed, {})
    assert rejected.status == "failed" and executor.calls == []
