from typing import Any

import pytest

from src.agents.collaboration import (
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStep,
    MultiAgentTaskResult,
)
from src.graph import graph_run
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
    GraphInterruptType,
)
from src.runtime.resume.pending_tasks import (
    PendingInputContract,
    PendingTaskCollection,
    PendingTaskSnapshot,
    ToolPendingPayload,
)


class FakeRuntimeContext:
    """
    测试用 RuntimeContext。

    功能：
        模拟 graph_run.py 需要写入的 trace_id、user_id 和 session_id 字段。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContext:
            一个最小可用的运行时上下文对象。
    """

    def __init__(self) -> None:
        self.trace_id = None
        self.user_id = None
        self.session_id = None


class FakeInterrupt:
    """
    测试用 interrupt 对象。

    功能：
        模拟 LangGraph interrupt 对象的 value 字段。

    参数含义：
        value:
            interrupt 提示文本。

    返回值含义：
        FakeInterrupt:
            一个包含 value 的测试对象。
    """

    def __init__(
        self,
        value: str,
    ) -> None:
        self.value = value


class FakeTask:
    """
    测试用 task 对象。

    功能：
        模拟 LangGraph current_state.tasks 中保存 interrupts 的结构。

    参数含义：
        prompt:
            interrupt 提示文本。

    返回值含义：
        FakeTask:
            一个包含 interrupts 列表的测试对象。
    """

    def __init__(
        self,
        prompt: str,
    ) -> None:
        self.interrupts = [
            FakeInterrupt(prompt),
        ]


class FakeCurrentState:
    """
    测试用 current state。

    功能：
        模拟 app.aget_state(config) 返回的 LangGraph 当前状态对象。

    参数含义：
        values:
            当前 state values。
        next_nodes:
            current_state.next，非空表示图处于中断状态。
        prompt:
            interrupt 提示文本。

    返回值含义：
        FakeCurrentState:
            一个最小 current state 对象。
    """

    def __init__(
        self,
        values: dict[str, Any] | None = None,
        next_nodes: tuple[str, ...] = (),
        prompt: str = "是否继续？",
    ) -> None:
        self.values = values or {}
        self.next = next_nodes
        self.tasks = [
            FakeTask(prompt),
        ] if next_nodes else []


class FakeGraphApp:
    """
    测试用 graph app。

    功能：
        模拟 LangGraph compiled graph 的 aget_state 和 astream 方法。

    参数含义：
        current_state:
            aget_state 返回的状态。
        resume_events:
            astream 恢复执行时产出的事件列表。

    返回值含义：
        FakeGraphApp:
            一个支持异步调用的测试 graph app。
    """

    def __init__(
        self,
        current_state: FakeCurrentState,
        resume_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.current_state = current_state
        self.resume_events = resume_events or []
        self.received_configs: list[dict[str, Any]] = []
        self.astream_call_count = 0

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> FakeCurrentState:
        """
        模拟读取 LangGraph 当前状态。

        功能：
            记录传入 config，并返回预设 current_state。

        参数含义：
            config:
                LangGraph config。

        返回值含义：
            FakeCurrentState:
                预设状态对象。
        """

        self.received_configs.append(config)
        return self.current_state

    async def astream(
        self,
        *_args: Any,
        **_kwargs: Any,
    ):
        """
        模拟恢复执行的异步事件流。

        功能：
            按顺序 yield 预设 resume_events。

        参数含义：
            *_args:
                兼容 LangGraph astream 位置参数。
            **_kwargs:
                兼容 LangGraph astream 关键字参数。

        返回值含义：
            AsyncIterator[dict[str, Any]]:
                异步事件流。
        """

        self.astream_call_count += 1
        for event in self.resume_events:
            yield event


async def fake_stream_runner(
    **_kwargs: Any,
):
    """
    测试用主图流式执行函数。

    功能：
        模拟 safe_stream_graph 产出一个状态更新事件。

    参数含义：
        **_kwargs:
            兼容 safe_stream_graph 的 graph、state、config、stream_mode 参数。

    返回值含义：
        AsyncIterator[dict[str, Any]]:
            异步状态事件。
    """

    yield {
        "memory_extract": {},
    }


@pytest.fixture()
def runtime_context() -> FakeRuntimeContext:
    """
    创建测试 runtime context。

    功能：
        为每个测试提供独立的 FakeRuntimeContext。

    参数含义：
        无。

    返回值含义：
        FakeRuntimeContext:
            测试运行时上下文。
    """

    return FakeRuntimeContext()


@pytest.fixture(autouse=True)
def patch_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    固定测试 user_id。

    功能：
        避免 create_initial_state 依赖真实用户管理模块。

    参数含义：
        monkeypatch:
            pytest 提供的动态替换工具。

    返回值含义：
        None。
    """

    monkeypatch.setattr(
        graph_run,
        "get_user_id",
        lambda: "test_user",
    )
    monkeypatch.setattr(
        graph_run,
        "write_rag_debug_report_if_enabled",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        graph_run,
        "write_llm_call_report_if_enabled",
        lambda **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_return_final_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试新问题正常完成时返回 GraphFinalResult。

    功能：
        使用 mock graph app 模拟完整主图运行完成，并验证结构化结果字段。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "answer": "金毛通常很友好。",
            },
        )
    )

    result = await graph_run.run_main_graph_with_result(
        question="金毛性格怎么样？",
        thread_id="thread_001",
        trace_id="trace_001",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "金毛通常很友好。"
    assert result.thread_id == "thread_001"
    assert result.trace_id == "trace_001"
    assert runtime_context.user_id == "test_user"
    assert runtime_context.session_id == "thread_001"
    assert app.received_configs[0]["configurable"] == {
        "thread_id": "thread_001",
    }


@pytest.mark.asyncio
async def test_restore_active_pet_state_should_use_user_scoped_whitelist() -> None:
    """
    验证当前宠物恢复会校验用户，并且只复制宠物身份白名单字段。

    参数含义：
        无。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "user_id": "test_user",
                "active_pet_key": "pet_v1_abc123",
                "active_pet_name": "豆豆",
                "final_answer": "不应恢复的旧答案",
                "route_decision": {"route": "general_agent"},
                "rag_context": {"chunks": ["不应恢复"]},
            }
        )
    )

    restored = await graph_run.restore_active_pet_state(
        app=app,
        config={"configurable": {"thread_id": "pet-thread"}},
        state={
            "user_id": "test_user",
            "question": "它现在30公斤",
            "active_pet_key": "",
            "active_pet_name": "",
            "final_answer": "",
            "route_decision": {},
            "rag_context": None,
        },
    )

    assert restored["active_pet_key"] == "pet_v1_abc123"
    assert restored["active_pet_name"] == "豆豆"
    assert restored["final_answer"] == ""
    assert restored["route_decision"] == {}
    assert restored["rag_context"] is None


@pytest.mark.asyncio
async def test_restore_active_pet_state_should_reject_other_user() -> None:
    """
    验证检查点属于其他用户时不会恢复宠物身份。

    参数含义：
        无。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "user_id": "other_user",
                "active_pet_key": "pet_v1_other",
                "active_pet_name": "其他用户的宠物",
            }
        )
    )

    restored = await graph_run.restore_active_pet_state(
        app=app,
        config={"configurable": {"thread_id": "shared-thread"}},
        state={
            "user_id": "test_user",
            "active_pet_key": "",
            "active_pet_name": "",
        },
    )

    assert restored["active_pet_key"] == ""
    assert restored["active_pet_name"] == ""


@pytest.mark.asyncio
async def test_restore_pending_task_registry_should_validate_scope() -> None:
    """
    测试统一等待任务注册表只恢复属于当前用户和线程的合法任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    task = PendingTaskSnapshot(
        task_id="pending_tool_checkpoint_001",
        task_kind="tool",
        user_id="test_user",
        thread_id="pending-thread",
        title="查询数据库表",
        pending_prompt="请选择数据库别名。",
        input_contracts=[
            PendingInputContract(
                field_id="database_name",
                value_type="string",
                enum_values=["memory", "rag"],
            )
        ],
        payload=ToolPendingPayload(
            tool_name="sqlite_list_tables",
            missing_fields=["database_name"],
        ),
    )
    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "pending_tasks": PendingTaskCollection([task]).to_state(),
                "final_answer": "不应恢复的旧答案",
            }
        )
    )

    restored = await graph_run.restore_pending_task_collection_state(
        app=app,
        config={"configurable": {"thread_id": "pending-thread"}},
        state={
            "user_id": "test_user",
            "pending_tasks": {},
            "final_answer": "",
        },
    )

    assert list(restored["pending_tasks"]) == [task.task_id]
    assert restored["pending_tasks"][task.task_id]["task_kind"] == "tool"
    assert restored["final_answer"] == ""

    rejected = await graph_run.restore_pending_task_collection_state(
        app=app,
        config={"configurable": {"thread_id": "other-thread"}},
        state={
            "user_id": "test_user",
            "pending_tasks": {},
        },
    )

    assert rejected["pending_tasks"] == {}


@pytest.mark.asyncio
async def test_restore_pending_tool_clarification_state_should_use_whitelist() -> None:
    """测试主图入口只从检查点恢复参数澄清所需字段。"""

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "tool_agent_clarification_request": {
                    "status": "pending",
                    "missing_fields": ["database_name"],
                },
                "tool_agent_pending_tool_call": {
                    "name": "sqlite_list_tables",
                    "args": {},
                },
                "tool_agent_pending_original_question": "查询数据库中的表",
                "final_answer": "不应恢复的旧答案",
                "tool_results": ["不应恢复的旧结果"],
            }
        )
    )

    restored = await graph_run.restore_pending_tool_clarification_state(
        app=app,
        config={
            "configurable": {
                "thread_id": "conversation-1",
            }
        },
        state={
            "question": "memory",
            "final_answer": "",
            "tool_results": [],
        },
    )

    assert restored["tool_agent_pending_tool_call"]["name"] == (
        "sqlite_list_tables"
    )
    assert restored["tool_agent_pending_original_question"] == "查询数据库中的表"
    assert restored["final_answer"] == ""
    assert restored["tool_results"] == []


@pytest.mark.asyncio
async def test_restore_pending_multi_agent_state_should_use_whitelist() -> None:
    """
    测试主图入口只恢复暂停中的多 Agent 任务字段。

    参数含义：无。
    返回值含义：None。
    """

    plan = AgentTaskPlan(
        plan_id="checkpoint_multi_agent_plan",
        objective="等待用户确认",
        steps=[
            AgentTaskStep(
                step_id="confirm_profile",
                title="确认读取资料",
                assigned_agent="profile_agent",
                status="awaiting_input",
            )
        ],
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="是否允许读取资料？",
    )
    paused_result = MultiAgentTaskResult(
        collaboration_id="checkpoint_multi_agent_task",
        plan=plan,
        status="awaiting_input",
        task_results=[
            AgentTaskResult(
                step_id="confirm_profile",
                assigned_agent="profile_agent",
                status="awaiting_input",
                requires_user_input=True,
                clarification_prompt="是否允许读取资料？",
            )
        ],
    )
    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "multi_agent_task_result": paused_result.model_dump(
                    mode="python"
                ),
                "multi_agent_pending_prompt": "是否允许读取资料？",
                "final_answer": "不应恢复的旧答案",
                "route_decision": {"route": "general_agent"},
            }
        )
    )

    restored = await graph_run.restore_pending_multi_agent_state(
        app=app,
        config={
            "configurable": {
                "thread_id": "conversation-multi-agent",
            }
        },
        state={
            "question": "允许读取",
            "final_answer": "",
            "route_decision": {},
        },
    )

    assert restored["multi_agent_task_result"]["status"] == (
        "awaiting_input"
    )
    assert restored["multi_agent_pending_prompt"] == "是否允许读取资料？"
    assert restored["final_answer"] == ""
    assert restored["route_decision"] == {}


@pytest.mark.asyncio
async def test_restore_pending_skill_state_should_use_whitelist() -> None:
    """
    测试主图入口只恢复顶层 Skill 继续执行所需的白名单字段。

    功能：
        验证原始问题、技能编号、已有输入和目标 Agent 会恢复，同时旧答案
        和旧 RAG 结果不会进入本轮状态。

    参数含义：
        无。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "skill_runtime_result": {"status": "awaiting_input"},
                "skill_selected_id": "dog-training-plan",
                "skill_inputs": {
                    "breed": "Golden Retriever",
                    "age": "6岁",
                },
                "skill_status": "awaiting_input",
                "skill_pending_prompt": "请补充当前行为和训练目标。",
                "skill_original_question": "为6岁金毛制定训练计划。",
                "skill_target_agent": "dog_knowledge_agent",
                "skill_execution_mode": "degraded",
                "skill_ignored_input_ids": ["current_behavior"],
                "skill_degradation_reason": (
                    "user_selected_degraded_execution"
                ),
                "skill_degradation_user_input": "简化执行",
                "final_answer": "不应恢复的旧答案",
                "rag_context": {"chunks": ["不应恢复"]},
            }
        )
    )

    restored = await graph_run.restore_pending_skill_state(
        app=app,
        config={"configurable": {"thread_id": "skill-thread"}},
        state={
            "question": "它会坐下，希望学习等待和召回。",
            "final_answer": "",
            "rag_context": None,
        },
    )

    assert restored["skill_status"] == "awaiting_input"
    assert restored["skill_selected_id"] == "dog-training-plan"
    assert restored["skill_original_question"] == "为6岁金毛制定训练计划。"
    assert restored["skill_target_agent"] == "dog_knowledge_agent"
    assert restored["skill_execution_mode"] == "degraded"
    assert restored["skill_ignored_input_ids"] == ["current_behavior"]
    assert restored["skill_degradation_reason"] == (
        "user_selected_degraded_execution"
    )
    assert restored["skill_degradation_user_input"] == "简化执行"
    assert restored["final_answer"] == ""
    assert restored["rag_context"] is None


@pytest.mark.asyncio
async def test_restore_pending_task_selection_should_use_whitelist() -> None:
    """
    测试主图入口只恢复合法的多等待任务候选和未绑定输入。

    参数含义：
        无。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "user_id": "test_user",
                "task_relation_candidates": [
                    {
                        "task_id": "tool:sqlite_list_tables",
                        "task_kind": "tool",
                        "title": "补充工具参数",
                        "pending_prompt": "请选择数据库。",
                    },
                    {
                        "task_id": "multi_agent:task_001",
                        "task_kind": "multi_agent",
                        "title": "补充多智能体任务",
                        "pending_prompt": "请补充狗狗年龄。",
                    },
                ],
                "task_relation_unassigned_input": "500",
                "task_relation_selection_action": "resume",
                "final_answer": "不应恢复的旧答案",
            }
        )
    )

    restored = await graph_run.restore_pending_task_selection_state(
        app=app,
        config={"configurable": {"thread_id": "selection-thread"}},
        state={
            "user_id": "test_user",
            "question": "2",
            "final_answer": "",
        },
    )

    assert restored["task_relation_unassigned_input"] == "500"
    assert restored["task_relation_selection_action"] == "resume"
    assert len(restored["task_relation_candidates"]) == 2
    assert restored["task_relation_candidates"][1]["task_id"] == (
        "multi_agent:task_001"
    )
    assert restored["final_answer"] == ""


def test_skill_logical_wait_should_return_interrupt_result() -> None:
    """
    测试顶层 Skill 的逻辑等待会转换成统一主图中断结果。

    功能：
        即使主图已经走到 END，只要 Skill 状态仍为 awaiting_input，API 也应
        收到 interrupted，而不是把提示文字误当成正常完成答案。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = graph_run.build_graph_result_from_current_state(
        current_state=FakeCurrentState(
            values={
                "skill_status": "awaiting_input",
                "skill_selected_id": "dog-training-plan",
                "skill_pending_prompt": "请补充当前行为和训练目标。",
                "skill_target_agent": "dog_knowledge_agent",
                "waiting_user_input": True,
            }
        ),
        thread_id="skill-thread",
        checkpoint_ns="main_graph",
        trace_id="skill-trace",
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "请补充当前行为和训练目标。"
    assert result.interrupt_type == GraphInterruptType.USER_CLARIFICATION
    assert result.metadata["skill_selected_id"] == "dog-training-plan"
    assert result.metadata["skill_target_agent"] == "dog_knowledge_agent"


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_return_interrupt_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试新问题触发中断时返回 GraphInterruptResult。

    功能：
        使用 mock current_state.next 模拟图停在 interrupt 节点。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            next_nodes=("ask_confirm",),
            prompt="是否允许调用天气工具？",
        )
    )

    result = await graph_run.run_main_graph_with_result(
        question="今天成都天气怎么样？",
        thread_id="thread_002",
        trace_id="trace_002",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "是否允许调用天气工具？"
    assert result.thread_id == "thread_002"
    assert result.trace_id == "trace_002"


def test_multi_agent_logical_wait_should_return_interrupt_result() -> None:
    """
    测试多 Agent 写入 State 的逻辑等待会转换成统一中断结果。

    功能：
        即使 current_state.next 为空，只要 DogState 明确表示多 Agent 正在
        等待输入，也不能把本轮误报成 completed。

    参数含义：
        无。

    返回值含义：
        None。
    """

    current_state = FakeCurrentState(
        values={
            "current_agent": "multi_agent",
            "waiting_user_input": True,
            "multi_agent_pending_prompt": "请补充狗狗年龄。",
            "multi_agent_task_result": {
                "collaboration_id": "multi_agent_task_waiting",
                "status": "awaiting_input",
            },
        }
    )

    result = graph_run.build_graph_result_from_current_state(
        current_state=current_state,
        thread_id="thread_multi_waiting",
        checkpoint_ns="default",
        trace_id="trace_multi_waiting",
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "请补充狗狗年龄。"
    assert (
        result.interrupt_type
        == GraphInterruptType.USER_CLARIFICATION
    )
    assert result.metadata["logical_interrupt"] is True
    assert result.metadata["business_status"] == "awaiting_input"
    assert result.metadata["state_waiting_user_input"] is True
    assert result.metadata["waiting_state_consistent"] is True
    assert (
        result.metadata["multi_agent_task_id"]
        == "multi_agent_task_waiting"
    )


def test_multi_agent_result_should_override_stale_waiting_flag() -> None:
    """
    测试多 Agent 标准结果可以覆盖未同步的主图等待标记。

    功能：
        当任务结果已经是 awaiting_input、但 waiting_user_input 仍为 False
        时，仍返回中断，并通过 metadata 暴露状态不一致。

    参数含义：
        无。

    返回值含义：
        None。
    """

    current_state = FakeCurrentState(
        values={
            "current_agent": "multi_agent",
            "waiting_user_input": False,
            "multi_agent_pending_prompt": "请补充狗狗体重。",
            "multi_agent_task_result": {
                "collaboration_id": "multi_agent_task_stale_flag",
                "status": "awaiting_input",
            },
        }
    )

    result = graph_run.build_graph_result_from_current_state(
        current_state=current_state,
        thread_id="thread_stale_waiting_flag",
        checkpoint_ns="default",
        trace_id="trace_stale_waiting_flag",
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "请补充狗狗体重。"
    assert result.metadata["state_waiting_user_input"] is False
    assert result.metadata["waiting_state_consistent"] is False


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_return_tool_interrupt_metadata(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试工具确认中断会返回 ToolAgent 元数据。

    功能：
        当当前 state 中包含工具确认字段时，
        GraphInterruptResult 应该标记为 tool_confirmation，
        并携带 current_agent、tool_calls 等恢复所需信息。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "next_agent": "tool_agent",
                "route_decision": {
                    "route": "tool_agent",
                },
                "tool_calls": [
                    {
                        "name": "weather",
                        "args": {
                            "city": "成都",
                        },
                    }
                ],
                "tool_confirmed": "pending",
                "tool_confirmation_required": True,
                "tool_agent_permission": {
                    "status": "pending",
                },
            },
            next_nodes=("tool_confirm",),
            prompt="是否允许调用天气工具？",
        )
    )

    result = await graph_run.run_main_graph_with_result(
        question="今天成都天气怎么样？",
        thread_id="thread_tool_interrupt",
        trace_id="trace_tool_interrupt",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.interrupt_type == GraphInterruptType.TOOL_CONFIRMATION
    assert result.metadata["current_agent"] == "tool_agent"
    assert result.metadata["route"] == "tool_agent"
    assert result.metadata["tool_confirmed"] == "pending"
    assert result.metadata["tool_confirmation_required"] is True
    assert result.metadata["tool_calls"][0]["name"] == "weather"


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_resume_to_final_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试恢复执行后直接返回最终结果。

    功能：
        使用 mock astream 模拟 Command resume 后产出 answer 事件。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(),
        resume_events=[
            {
                "answer": "已完成工具调用。",
            },
        ],
    )
    result = await graph_run.run_main_graph_with_result(
        question="y",
        thread_id="thread_003",
        trace_id="trace_003",
        resume_value="y",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "已完成工具调用。"
    assert result.metadata["source"] == "resume_stream_event"


@pytest.mark.asyncio
async def test_multi_agent_logical_resume_should_start_new_graph_turn(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试多 Agent 逻辑等待不会错误调用原生 Command resume。

    功能：
        模拟 Planner 等待用户补充信息后调用 API resume；用户回答应作为
        新一轮 question 进入主图，并携带 Checkpoint 中的暂停任务结果。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    paused_result = MultiAgentTaskResult(
        collaboration_id="multi_agent_task_logical_resume",
        plan=AgentTaskPlan(
            plan_id="logical_resume_plan",
            objective="制定健康饮食训练方案",
            steps=[
                AgentTaskStep(
                    step_id="step_plan",
                    title="生成综合方案",
                    assigned_agent="general_agent",
                )
            ],
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="请补充狗狗档案。",
        ),
        status="awaiting_input",
        task_results=[],
        final_answer="请补充狗狗档案。",
    )
    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "multi_agent_task_result": paused_result.model_dump(
                    mode="python"
                ),
                "multi_agent_pending_prompt": "请补充狗狗档案。",
                "waiting_user_input": True,
                "final_answer": "请补充狗狗档案。",
            }
        )
    )
    received_state: dict[str, Any] = {}

    async def logical_resume_stream_runner(
        **kwargs: Any,
    ):
        """记录重新进入主图的 State，并模拟恢复完成。"""

        received_state.update(kwargs["state"])
        app.current_state = FakeCurrentState(
            values={
                "final_answer": "已根据6岁金毛档案生成方案。",
                "multi_agent_task_result": {
                    "status": "completed",
                },
            }
        )
        yield {
            "multi_agent": {
                "final_answer": "已根据6岁金毛档案生成方案。",
            }
        }

    result = await graph_run.run_main_graph_with_result(
        question="是一只6岁的金毛，体重30公斤。",
        thread_id="thread_logical_resume",
        trace_id="trace_logical_resume",
        resume_value="是一只6岁的金毛，体重30公斤。",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=logical_resume_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "已根据6岁金毛档案生成方案。"
    assert received_state["question"] == "是一只6岁的金毛，体重30公斤。"
    assert (
        received_state["multi_agent_task_result"]["status"]
        == "awaiting_input"
    )
    assert app.astream_call_count == 0


@pytest.mark.asyncio
async def test_multi_agent_skill_state_should_survive_checkpoint_resume(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试多 Agent Step 内部的 Skill 状态能否经过 Checkpoint 恢复。

    功能：
        模拟训练 Step 因 Skill 缺少必要输入而暂停。下一轮用户通过 API
        resume 补充信息时，主图应从相同 thread_id 的 Checkpoint 恢复完整
        多 Agent 任务，其中已经提取的 Skill 编号和输入不能丢失；用户新回答
        应作为新的 question 进入主图，而不是错误调用原生 Command resume。

    参数含义：
        runtime_context:
            保存本次测试 trace_id、user_id 和 session_id 的运行时上下文。

    返回值含义：
        None。
    """

    # 这是上一轮写入 Checkpoint、尚未完成的训练步骤结果。
    paused_step_result = AgentTaskResult(
        step_id="step_training",
        assigned_agent="dog_knowledge_agent",
        status="awaiting_input",
        requires_user_input=True,
        clarification_prompt="请补充当前行为和训练目标。",
        output={
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_inputs": {
                "breed": "Golden Retriever",
                "age": "6岁",
            },
            "skill_pending_prompt": "请补充当前行为和训练目标。",
        },
        metadata={
            "skill_runtime": {
                "status": "awaiting_input",
                "selected_skill_id": "dog-training-plan",
            }
        },
    )

    # 这是总编排器返回并由主图整体保存的暂停任务。
    paused_task_result = MultiAgentTaskResult(
        collaboration_id="multi_agent_task_skill_checkpoint",
        plan=AgentTaskPlan(
            plan_id="skill_checkpoint_plan",
            objective="为6岁的金毛制定训练计划",
            steps=[
                AgentTaskStep(
                    step_id="step_training",
                    title="制定训练计划",
                    description="为6岁的金毛制定训练计划",
                    assigned_agent="dog_knowledge_agent",
                    status="awaiting_input",
                )
            ],
            status="awaiting_input",
            requires_user_input=True,
            clarification_prompt="请补充当前行为和训练目标。",
        ),
        status="awaiting_input",
        task_results=[paused_step_result],
    )

    # FakeGraphApp.current_state 模拟相同 thread_id 对应的最新 Checkpoint。
    app = FakeGraphApp(
        current_state=FakeCurrentState(
            values={
                "multi_agent_task_result": paused_task_result.model_dump(
                    mode="python"
                ),
                "multi_agent_pending_prompt": (
                    "请补充当前行为和训练目标。"
                ),
                "waiting_user_input": True,
                "final_answer": "不应恢复成最终答案的旧提示。",
            }
        )
    )

    # 保存主图重新启动时真正收到的 State，用于检查 Checkpoint 恢复结果。
    received_state: dict[str, Any] = {}

    async def skill_resume_stream_runner(
        **kwargs: Any,
    ):
        """
        记录恢复后的主图输入并模拟多 Agent 最终执行完成。

        参数含义：
            kwargs:
                safe_stream_graph 风格参数，其中 state 是重新进入主图的状态。

        返回值含义：
            AsyncIterator[dict[str, Any]]:
                通过 yield 产生一条模拟的多 Agent 完成事件。
        """

        received_state.update(kwargs["state"])
        app.current_state = FakeCurrentState(
            values={
                "final_answer": "已根据补充信息生成训练计划。",
                "multi_agent_task_result": {
                    "status": "completed",
                },
            }
        )
        yield {
            "multi_agent": {
                "final_answer": "已根据补充信息生成训练计划。",
            }
        }

    result = await graph_run.run_main_graph_with_result(
        question="它目前会坐下，希望学习等待和召回。",
        thread_id="thread_skill_checkpoint",
        trace_id="trace_skill_checkpoint_resume",
        resume_value="它目前会坐下，希望学习等待和召回。",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=skill_resume_stream_runner,
    )

    assert isinstance(result, GraphFinalResult)
    assert result.answer == "已根据补充信息生成训练计划。"

    # 用户补充内容作为新问题重新进入路由，而不是丢失或变成旧问题。
    assert received_state["question"] == (
        "它目前会坐下，希望学习等待和召回。"
    )

    # Checkpoint 中 Step 已提取出的 Skill 数据完整保留，供 Worker 恢复使用。
    restored_step_result = received_state[
        "multi_agent_task_result"
    ]["task_results"][0]
    assert restored_step_result["status"] == "awaiting_input"
    assert restored_step_result["output"]["skill_selected_id"] == (
        "dog-training-plan"
    )
    assert restored_step_result["output"]["skill_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }

    # 多 Agent 逻辑等待通过新一轮主图恢复，不会走工具确认使用的原生恢复流。
    assert app.astream_call_count == 0


@pytest.mark.asyncio
async def test_resume_multi_agent_logical_wait_should_return_interrupt(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试恢复事件再次等待输入时仍返回统一中断结果。

    功能：
        防止恢复执行事件中的 final_answer 提示被提前当成正常完成答案。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(),
        resume_events=[
            {
                "current_agent": "multi_agent",
                "waiting_user_input": True,
                "multi_agent_pending_prompt": "还需要补充狗狗体重。",
                "final_answer": "还需要补充狗狗体重。",
                "multi_agent_task_result": {
                    "collaboration_id": "multi_agent_task_resume_wait",
                    "status": "awaiting_input",
                },
            }
        ],
    )

    result = await graph_run.run_main_graph_with_result(
        question="6岁",
        thread_id="thread_resume_waiting",
        trace_id="trace_resume_waiting",
        resume_value="6岁",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "还需要补充狗狗体重。"
    assert (
        result.interrupt_type
        == GraphInterruptType.USER_CLARIFICATION
    )
    assert result.metadata["source"] == "resume_stream_event"


def test_ambiguous_task_relation_should_return_interrupt_result() -> None:
    """
    测试无法区分新旧任务时返回统一澄清中断结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = graph_run.build_task_relation_interrupt_result_from_state(
        state={
            "task_relation_requires_confirmation": True,
            "task_relation_pending_kind": "skill",
            "task_relation_decision": {
                "relation": "ambiguous",
                "normalized_input": "成都天气",
                "confidence": 0.5,
                "reason": "无法确定",
                "source": "fallback",
            },
            "pending_prompt": "请明确是继续旧任务还是开始新问题。",
            "waiting_user_input": True,
        },
        thread_id="thread_relation",
        checkpoint_ns="main_graph",
        trace_id="trace_relation",
        source="current_state",
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "请明确是继续旧任务还是开始新问题。"
    assert result.metadata["pending_task_kind"] == "skill"
    assert result.metadata["task_relation"]["relation"] == "ambiguous"


@pytest.mark.asyncio
async def test_run_main_graph_with_result_should_resume_to_interrupt_result(
    runtime_context: FakeRuntimeContext,
) -> None:
    """
    测试恢复执行后再次中断。

    功能：
        使用 mock astream 返回空事件，再通过 aget_state 模拟图再次停在 interrupt。

    参数含义：
        runtime_context:
            测试运行时上下文。

    返回值含义：
        None。
    """

    app = FakeGraphApp(
        current_state=FakeCurrentState(
            next_nodes=("ask_confirm",),
            prompt="还需要二次确认吗？",
        ),
        resume_events=[],
    )
    result = await graph_run.run_main_graph_with_result(
        question="y",
        thread_id="thread_004",
        trace_id="trace_004",
        resume_value="y",
        graph_app=app,
        runtime_context=runtime_context,
        stream_runner=fake_stream_runner,
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "还需要二次确认吗？"
    assert result.thread_id == "thread_004"
