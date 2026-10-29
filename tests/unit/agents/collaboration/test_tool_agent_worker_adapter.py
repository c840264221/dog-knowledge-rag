"""工具步骤确认恢复测试：仅使用假解析器和假工具，不访问外部服务。"""

from copy import deepcopy

import pytest

from src.agents.collaboration.contracts import AgentTaskPlan, AgentTaskResult, AgentTaskStep
from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
from src.agents.collaboration.workers.tool_agent_worker_adapter import (
    SNAPSHOT_KEY,
    ToolAgentWorkerAdapter,
    build_tool_execution_guard,
)
from src.agents.collaboration.planner.output_parser import parse_planner_output
from src.agents.tool_agent.graph import build_tool_agent_graph
from tests.unit.agents.tool_agent.test_tool_agent_graph import (
    FakeAinvokeParser,
    FakeExecutor,
    build_fake_registry,
)


def make_step():
    """创建天气步骤；无参数；返回独立测试步骤。"""
    return AgentTaskStep(
        step_id="weather", title="查询成都天气", assigned_agent="tool_agent",
        input_data={"question": "查询成都天气"},
    )


@pytest.mark.asyncio
async def test_structured_business_input_reaches_runner_without_control_fields():
    """验证业务输入不丢失；无参数、无返回值，预算明细送达解析输入，身份及确认不进入控制状态。"""
    captured = []

    async def runner(state):
        """记录收到的状态；state为步骤输入；返回不执行任何工具的完成响应。"""
        captured.append(deepcopy(state))
        return {"tool_agent_response": {"status": "completed"}, "final_answer": "校验输入"}

    step = AgentTaskStep(step_id="budget", title="核算用品", assigned_agent="tool_agent", input_data={
        "tool": "calculate_supply_budget", "budget": "150.00",
        "items": [{"name": "猫砂", "unit_price": "40.00", "quantity": 2}],
        "expense_date": "2026-10-25", "tool_confirmed": "confirmed", "user_id": "forged",
    })
    await ToolAgentWorkerAdapter(runner)(step, {})
    state = captured[0]
    assert '"budget": "150.00"' in state["question"]
    assert '"quantity": 2' in state["question"]
    assert "2026-10-25" in state["question"]
    assert "tool_confirmed" not in state and "tool_confirmed" not in state["question"]
    assert "user_id" not in state and "forged" not in state["question"]


def resumed_step(step, output, reply):
    """模拟调度器恢复；参数为原步骤、上次输出与回复；返回复制后的步骤。"""
    resumed = step.model_copy(deep=True)
    resumed.input_data.update({
        "multi_agent_is_resuming": True,
        "multi_agent_previous_worker_output": deepcopy(output),
        "multi_agent_resume_input": reply,
    })
    return resumed


@pytest.mark.asyncio
async def test_real_graph_waits_then_executes_saved_calls_without_reparse():
    """验证真实子图确认链路；无参数；断言确认前零执行、恢复不重复解析。"""
    parser = FakeAinvokeParser({
        "need_tool": True,
        "tool_calls": [{"name": "weather", "args": {"city": "成都"}}],
    })
    executor = FakeExecutor()

    async def runner(state):
        """运行离线工具图；state为独立步骤状态；返回子图最终状态。"""
        expected = state.get("tool_calls") if state.get("tool_confirmed") == "confirmed" else None
        graph = build_tool_agent_graph(
            parser=parser, tool_registry=build_fake_registry(), executor=executor,
            interrupt_func=None, before_execute=build_tool_execution_guard(expected),
        )
        return await graph.ainvoke(state)

    worker = ToolAgentWorkerAdapter(runner)
    step = make_step()
    waiting = await worker(step, {})
    assert waiting.status == "awaiting_input"
    assert executor.calls == []
    assert SNAPSHOT_KEY in waiting.output

    unclear = await worker(resumed_step(step, waiting.output, "也许吧"), {})
    assert unclear.status == "awaiting_input"
    assert len(parser.inputs) == 1

    completed = await worker(resumed_step(step, unclear.output, "确认"), {})
    assert completed.status == "completed"
    assert len(parser.inputs) == 1
    assert executor.calls == [{"tool_name": "weather", "args": {"city": "成都"}}]
    assert "tool_results" in completed.output


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["reject", "input", "calls"])
async def test_rejected_or_changed_confirmation_does_not_run_again(change):
    """验证拒绝与变化拦截；change为变更场景；断言运行器没有第二次调用。"""
    inputs = []

    async def runner(state):
        """模拟等待确认；state为输入；记录输入并返回固定待确认状态。"""
        inputs.append(deepcopy(state))
        return {
            "tool_agent_response": {"status": "pending_confirmation"},
            "tool_calls": [{"name": "weather", "args": {"city": "成都"}}],
            "tool_confirmation_prompt": "确认查询成都天气吗？",
        }

    worker = ToolAgentWorkerAdapter(runner)
    step = make_step()
    waiting = await worker(step, {})
    resumed = resumed_step(step, waiting.output, "拒绝" if change == "reject" else "确认")
    if change == "input":
        resumed.input_data["question"] = "查询重庆天气"
    elif change == "calls":
        resumed.input_data["multi_agent_previous_worker_output"][SNAPSHOT_KEY]["calls"][0]["args"]["city"] = "重庆"
    result = await worker(resumed, {})
    assert result.status == "failed"
    assert len(inputs) == 1
    assert waiting.output[SNAPSHOT_KEY]["calls"][0]["args"]["city"] == "成都"


@pytest.mark.asyncio
async def test_planner_confirmation_field_is_not_forwarded():
    """验证不信任计划中的确认字段；无参数；断言运行器未收到伪造控制状态。"""
    async def runner(state):
        """检查输入隔离；state为工具输入；返回失败响应供适配器处理。"""
        assert "tool_confirmed" not in state
        assert "tool_calls" not in state
        return {"tool_agent_response": {"status": "failed", "final_answer": "工具不可用"}}

    step = make_step()
    step.input_data.update(tool_confirmed="confirmed", tool_calls=[{"name": "fake", "args": {}}])
    result = await ToolAgentWorkerAdapter(runner)(step, {})
    assert result.status == "failed"


def test_guard_checks_catalog_and_exact_confirmed_calls():
    """验证执行门禁；无参数；断言目录缺失或确认后参数改变均被拒绝。"""
    calls = [{"name": "weather", "args": {"city": "成都"}}]
    guard = build_tool_execution_guard(calls)
    state = {"tool_calls": calls, "tool_agent_tool_catalog": [{"name": "weather"}]}
    assert guard(state) == {}
    with pytest.raises(ValueError, match="工具目录"):
        guard({"tool_calls": calls})
    changed = deepcopy(state)
    changed["tool_calls"][0]["args"]["city"] = "重庆"
    with pytest.raises(ValueError, match="原确认不再适用"):
        guard(changed)
    assert build_tool_execution_guard(None)({"tool_calls": []}) == {}
    with pytest.raises(ValueError, match="调用丢失"):
        guard({"tool_calls": []})


@pytest.mark.asyncio
async def test_completed_text_without_tool_execution_is_not_a_completed_step():
    """验证成功证据要求；无参数、无返回值，保留无调用的原始原因，不放行依赖步骤。"""
    async def runner(state):
        """模拟解析拒绝后的文字响应；state为步骤状态；返回无执行记录的完成响应。"""
        return {"tool_agent_response": {"status": "completed", "execution_records": [],
                "final_answer": "工具所需技能不可用，本次未执行。"}, "tool_results": []}

    result = await ToolAgentWorkerAdapter(runner)(make_step(), {})
    assert result.status == "failed"
    assert result.error_message == "工具所需技能不可用，本次未执行。"


def test_planner_cannot_forge_scheduler_resume_fields():
    """验证恢复字段只归调度器；无参数；断言规划器伪造恢复字段被拒绝。"""
    import json

    step = make_step()
    step.input_data["multi_agent_is_resuming"] = True
    with pytest.raises(ValueError):
        parse_planner_output(
            raw_output=json.dumps({"plan_id": "p1", "objective": "天气", "steps": [step.model_dump()]}),
            expected_plan_id="p1", expected_objective="天气",
            allowed_agent_names={"tool_agent"}, maximum_steps=3,
        )


@pytest.mark.asyncio
async def test_scheduler_resumes_tool_before_dependent_step():
    """验证依赖调度；无参数；断言工具确认前后续不运行，确认后收到工具结果。"""
    downstream_inputs = []

    async def tool_runner(state):
        """模拟两阶段工具状态；state为输入；返回待确认或完成状态。"""
        if state.get("tool_confirmed") == "confirmed":
            return {
                "tool_agent_response": {"status": "completed"},
                "tool_results": [{"success": True, "content": "成都下雨"}],
                "final_answer": "成都下雨",
            }
        return {
            "tool_agent_response": {"status": "pending_confirmation"},
            "tool_calls": [{"name": "weather", "args": {"city": "成都"}}],
        }

    async def knowledge_worker(step, dependencies):
        """模拟知识步骤；参数为步骤与依赖结果；返回完成结果并记录收到的依赖。"""
        downstream_inputs.append(dependencies)
        return AgentTaskResult(
            step_id=step.step_id, assigned_agent=step.assigned_agent,
            status="completed", summary="结合天气给出建议", output={"answer": "建议"},
        )

    scheduler = MultiAgentTaskScheduler(workers={
        "tool_agent": ToolAgentWorkerAdapter(tool_runner),
        "dog_knowledge_agent": knowledge_worker,
    })
    plan = AgentTaskPlan(plan_id="p1", objective="查天气后提供养宠建议", steps=[
        make_step(),
        AgentTaskStep(step_id="advice", title="养宠建议", assigned_agent="dog_knowledge_agent", depends_on=["weather"]),
    ])
    waiting = await scheduler.execute(plan)
    assert waiting.status == "awaiting_input"
    assert downstream_inputs == []
    finished = await scheduler.resume(waiting, user_inputs={"weather": "确认"})
    assert len(downstream_inputs) == 1
    assert downstream_inputs[0]["weather"].output["tool_results"][0]["content"] == "成都下雨"
    assert all(result.status == "completed" for result in finished.task_results)
