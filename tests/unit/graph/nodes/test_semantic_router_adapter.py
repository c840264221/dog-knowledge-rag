"""
semantic_router Adapter 单元测试。

功能：
    测试旧主图节点 semantic_router_node 是否已经正确转调新版 RootAgent。

背景：
    V1.7.1 阶段采用 Adapter 过渡方案：
        1. 主图节点名 semantic_router 暂时不变。
        2. 真实路由逻辑迁移到 src.agents.root_agent.supervisor。
        3. router_node.py 只作为兼容入口。

测试目标：
    确保 semantic_router_node 的输出和 RootAgent 标准输出一致。
"""

import pytest

import src.graph.nodes.router_node as router_module
from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)
from src.graph.nodes.router_node import (
    semantic_router_node,
)
from src.runtime.resume import resolve_pending_task_relation


@pytest.mark.asyncio
async def test_semantic_router_should_not_repeat_processed_task_relation(
        monkeypatch,
) -> None:
    """
    验证独立门卫执行后语义路由不再重复判断新旧任务关系。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于让重复调用立即失败。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    def fail_if_called(_state):
        """
        在任务关系分类器被错误重复调用时抛出异常。

        参数含义：
            _state:
                语义路由错误传入的主图状态，本测试不使用。

        返回值含义：
            无；函数始终抛出 AssertionError。
        """

        raise AssertionError("任务关系门卫不应重复执行")

    monkeypatch.setattr(
        router_module,
        "resolve_pending_task_relation",
        fail_if_called,
    )

    result = await semantic_router_node(
        {
            "question": "帮我查询成都天气。",
            "task_relation_guard_processed": True,
            "task_relation_decision": {
                "relation": "new_task",
            },
        }
    )

    assert result["route_decision"]["route"] == "tool_agent"


@pytest.mark.asyncio
async def test_selected_tool_task_should_not_run_multi_agent_resume(
        monkeypatch,
) -> None:
    """
    验证用户选中工具等待任务后不会让多智能体适配器消费同一输入。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于检测错误的跨模块恢复调用。

    返回值含义：
        None。
    """

    async def fail_if_multi_agent_called(*_args, **_kwargs):
        """
        在未选中的多智能体恢复适配器被调用时让测试失败。

        参数含义：
            *_args、**_kwargs:
                错误调用传入的参数，本测试不使用。

        返回值含义：
            无；函数始终抛出 AssertionError。
        """

        raise AssertionError("未选中的多智能体任务不应消费输入")

    monkeypatch.setattr(
        router_module,
        "resolve_multi_agent_resume_input",
        fail_if_multi_agent_called,
    )

    result = await semantic_router_node(
        {
            "question": "memory",
            "task_relation_guard_processed": True,
            "task_relation_decision": {"relation": "resume"},
            "task_relation_pending_kind": "tool",
            "tool_agent_clarification_request": {
                "status": "pending",
                "missing_fields": ["database_name"],
                "options": {"database_name": ["memory", "rag"]},
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_list_tables",
                "args": {},
            },
            "multi_agent_task_result": {"status": "awaiting_input"},
        }
    )

    assert result["next_agent"] == "tool_agent"
    assert result["tool_calls"][0]["args"]["database_name"] == "memory"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question, expected_route",
    [
        (
            "推荐几种适合公寓养的狗",
            "dog_knowledge_agent",
        ),
        (
            "金毛寿命多久？",
            "dog_knowledge_agent",
        ),
        (
            "现在几点？",
            "tool_agent",
        ),
        (
            "你好，你是谁？",
            "general_agent",
        ),
        (
            "为狗狗制定健康和训练综合方案",
            "multi_agent",
        ),
    ],
)
async def test_semantic_router_adapter_calls_root_supervisor(
        question: str,
        expected_route: str,
) -> None:
    """
    测试 semantic_router_node 适配新版 RootAgent。

    功能：
        调用旧入口 semantic_router_node，
        验证它返回新版 route_decision 结构。

    参数：
        question:
            用户问题。

        expected_route:
            预期 route。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        "question": question,
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = await semantic_router_node(
        state,
    )

    route_decision = result.get(
        "route_decision",
        {},
    )

    assert route_decision.get(
        "route",
    ) == expected_route

    assert result.get(
        "next_agent",
    ) == expected_route

    assert result.get(
        "current_agent",
    ) == "root_agent"


@pytest.mark.asyncio
async def test_semantic_router_adapter_does_not_return_legacy_query_parse_fields() -> None:
    """
    测试 semantic_router_node 不再返回旧 query_parse 字段。

    功能：
        验证旧入口不会再输出 intent、filters、tags、features、dog_name
        这些早期 QueryParseResult 字段。

    参数：
        无。

    返回值：
        None:
            pytest 通过 assert 断言测试结果。
    """

    state = {
        "question": "推荐适合新手养的狗",
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = await semantic_router_node(
        state,
    )

    assert "intent" not in result
    assert "filters" not in result
    assert "tags" not in result
    assert "features" not in result
    assert "dog_name" not in result


@pytest.mark.asyncio
async def test_semantic_router_adapter_should_restore_pending_tool_argument() -> None:
    """
    测试语义路由入口恢复上一轮缺失的工具参数。

    功能：
        Checkpoint state 中存在待补全 database_name 时，输入 memory 应补回参数，
        清理澄清请求并路由到 ToolAgent。

    参数：
        无。

    返回值：
        None:
            pytest 根据断言判断测试结果。
    """

    result = await semantic_router_node(
        {
            "question": "memory",
            "user_id": "test_user",
            "session_id": "test_session",
            "trace_id": "test_trace",
            "tool_agent_clarification_request": {
                "status": "pending",
                "missing_fields": ["database_name"],
                "options": {
                    "database_name": ["memory", "rag"],
                },
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_list_tables",
                "args": {},
            },
        }
    )

    assert result["next_agent"] == "tool_agent"
    assert result["tool_calls"][0]["args"]["database_name"] == "memory"
    assert result["tool_agent_clarification_request"] is None
    assert result["tool_agent_clarification_resume_ready"] is True
    pending_task = result["pending_tasks"][
        "tool:sqlite_list_tables"
    ]
    assert pending_task["status"] == "running"
    assert pending_task["version"] == 2


@pytest.mark.asyncio
async def test_semantic_router_should_keep_partial_clarification_in_tool_agent() -> None:
    """测试只补完一个字段时仍留在 ToolAgent 继续询问剩余字段。"""

    result = await semantic_router_node(
        {
            "question": "memory",
            "user_id": "test_user",
            "session_id": "test_session",
            "trace_id": "test_trace",
            "tool_agent_clarification_request": {
                "status": "pending",
                "tool_name": "sqlite_describe_table",
                "missing_fields": [
                    "database_name",
                    "table_name",
                ],
                "options": {
                    "database_name": ["memory", "rag"],
                    "table_name": [],
                },
                "question": "请补充数据库别名和表名。",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_describe_table",
                "args": {},
            },
        }
    )

    assert result["next_agent"] == "tool_agent"
    assert result["tool_agent_clarification_resolution"]["action"] == "partial"
    assert result["tool_agent_pending_tool_call"]["args"] == {
        "database_name": "memory",
    }
    assert result["tool_agent_clarification_request"]["missing_fields"] == [
        "table_name",
    ]
    pending_task = result["pending_tasks"][
        "tool:sqlite_describe_table"
    ]
    assert pending_task["status"] == "awaiting_input"
    assert pending_task["version"] == 1


@pytest.mark.asyncio
async def test_semantic_router_should_stop_when_task_transition_fails() -> None:
    """
    测试任务已经处于 running 时不会重复恢复并执行同一个工具。

    参数含义：
        无。

    返回值含义：
        None。
    """

    state = {
        "question": "memory",
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
        "tool_agent_clarification_request": {
            "status": "pending",
            "missing_fields": ["database_name"],
            "options": {"database_name": ["memory", "rag"]},
            "question": "请选择数据库别名。",
        },
        "tool_agent_pending_tool_call": {
            "name": "sqlite_list_tables",
            "args": {},
        },
    }
    relation_update = resolve_pending_task_relation(state)["state_update"]
    pending_tasks = relation_update["pending_tasks"]
    pending_tasks["tool:sqlite_list_tables"]["status"] = "running"
    pending_tasks["tool:sqlite_list_tables"]["version"] = 2

    result = await semantic_router_node(
        {
            **state,
            **relation_update,
            "pending_tasks": pending_tasks,
            "task_relation_guard_processed": True,
        }
    )

    assert result["next_agent"] == "FINISH"
    assert result["route_decision"]["source"] == (
        "pending_task_state_guard"
    )
    assert result["tool_calls"] == []
    assert result["need_tool"] is False
    assert "本轮没有继续执行" in result["final_answer"]


@pytest.mark.asyncio
async def test_semantic_router_should_route_multi_agent_resume() -> None:
    """
    测试暂停任务的用户回答会先转换成恢复输入并路由到 multi_agent。

    参数：无。
    返回值：None。
    """

    step = AgentTaskStep(
        step_id="confirm_profile",
        title="确认读取资料",
        assigned_agent="dog_knowledge_agent",
        status="awaiting_input",
    )
    paused_result = MultiAgentTaskResult(
        collaboration_id="router_resume_task",
        plan=AgentTaskPlan(
            plan_id="router_resume_plan",
            objective="生成综合方案",
            steps=[step],
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="是否允许读取资料？",
        ),
        status="awaiting_input",
        task_results=[
            AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="是否允许读取资料？",
            )
        ],
    )

    result = await semantic_router_node(
        {
            "question": "允许读取",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert result["next_agent"] == "multi_agent"
    assert result["multi_agent_resume_action"] == "resume"
    assert result["multi_agent_resume_inputs"] == {
        "confirm_profile": "允许读取"
    }
    assert result["multi_agent_resume_ready"] is True


@pytest.mark.asyncio
async def test_semantic_router_should_keep_saved_skill_target_on_resume() -> None:
    """
    测试 Skill 恢复时不会用简短补充回答重新判断目标 Agent。

    功能：
        用户只补充“6岁”时，RootAgent 应读取检查点保存的目标，继续进入
        dog_knowledge_agent，而不是把简短回答误判成 general_agent。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = await semantic_router_node(
        {
            "question": "6岁",
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_target_agent": "dog_knowledge_agent",
        }
    )

    assert result["next_agent"] == "dog_knowledge_agent"
    assert result["route_decision"]["hints"]["skill_resume"] is True


@pytest.mark.asyncio
async def test_semantic_router_should_not_resume_old_task_for_new_request() -> None:
    """
    测试完整新请求不会被单个等待步骤错误吸收为恢复答案。

    参数含义：
        无。

    返回值含义：
        None。
    """

    step = AgentTaskStep(
        step_id="step_profile",
        title="补全档案",
        assigned_agent="profile_agent",
        status="awaiting_input",
    )
    paused_result = MultiAgentTaskResult(
        collaboration_id="old_task",
        plan=AgentTaskPlan(
            plan_id="old_plan",
            objective="生成旧健康方案",
            steps=[step],
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="请补充年龄。",
        ),
        status="awaiting_input",
        task_results=[
            AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="请补充年龄。",
            )
        ],
    )

    result = await semantic_router_node(
        {
            "question": "请使用多个智能体制定一份新的训练方案。",
            "multi_agent_task_result": paused_result.model_dump(
                mode="python"
            ),
        }
    )

    assert result["next_agent"] == "multi_agent"
    assert result["multi_agent_task_result"] == {}
    assert result["multi_agent_resume_action"] == "new_question"
    assert result["multi_agent_resume_ready"] is False
    assert result["task_relation_decision"]["relation"] == "new_task"


@pytest.mark.asyncio
async def test_semantic_router_should_pause_on_ambiguous_task_relation() -> None:
    """
    测试无法判断新旧任务时不会冒险恢复等待步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = await semantic_router_node(
        {
            "question": "成都天气",
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_target_agent": "dog_knowledge_agent",
            "skill_pending_prompt": "请补充狗狗年龄。",
        }
    )

    assert result["next_agent"] == "FINISH"
    assert result["task_relation_requires_confirmation"] is True
    assert result["waiting_user_input"] is True
    assert result["task_relation_decision"]["relation"] == "ambiguous"
