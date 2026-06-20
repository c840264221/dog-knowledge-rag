"""
state scope 单元测试。

StateScope（状态作用域）：
用于管理一次 Agent / Graph 执行过程中的运行时状态，例如当前 agent、当前 node、当前 tool、执行阶段、重试次数等。

RuntimeState（运行时状态）：
StateScope 内部持有的数据对象，用于保存具体状态字段。

execution history（执行历史）：
记录 Agent / Graph 执行过程中的关键步骤，方便调试和生成 debug report。

export（导出）：
把当前状态转换成 dict 字典格式，方便持久化、日志记录或 checkpoint 保存。

restore（恢复）：
从 dict 字典中恢复状态，常用于 checkpoint resume（检查点恢复）。
"""

import pytest

from src.runtime.scopes.state_scope import StateScope


def test_state_scope_can_be_created():
    """
    测试 StateScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    assert state_scope is not None
    assert state_scope.state is not None


def test_state_scope_get_state_should_return_runtime_state_instance():
    """
    测试 get_state 是否返回内部 state 对象。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    runtime_state = state_scope.get_state()

    assert runtime_state is state_scope.state


def test_state_scope_set_and_get_agent():
    """
    测试 set_agent 和 get_agent 是否可以正确设置和读取当前 agent。

    agent（智能体）：
    表示当前正在执行的智能体名称，例如 general_qa_agent。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_agent(
        "general_qa_agent",
    )

    assert state_scope.get_agent() == "general_qa_agent"
    assert state_scope.state.current_agent == "general_qa_agent"


def test_state_scope_set_and_get_node():
    """
    测试 set_node 和 get_node 是否可以正确设置和读取当前 node。

    node（节点）：
    表示 LangGraph 中当前正在执行的节点，例如 tool_parse_node、answer_gen_node。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_node(
        "tool_parse_node",
    )

    assert state_scope.get_node() == "tool_parse_node"
    assert state_scope.state.current_node == "tool_parse_node"


def test_state_scope_set_tool_should_update_current_tool():
    """
    测试 set_tool 是否可以正确设置当前 tool。

    tool（工具）：
    表示当前正在执行的外部能力或函数，例如 date、weather、retriever。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_tool(
        "weather_tool",
    )

    assert state_scope.state.current_tool == "weather_tool"


def test_state_scope_set_phase_should_update_phase():
    """
    测试 set_phase 是否可以正确设置当前执行阶段。

    phase（阶段）：
    表示当前执行流程所处阶段，例如 parsing、retrieving、answering。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_phase(
        "retrieving",
    )

    assert state_scope.state.phase == "retrieving"


def test_state_scope_add_history_should_append_item():
    """
    测试 add_history 是否可以向 execution_history 中追加记录。

    execution_history（执行历史）：
    用于保存 Graph / Agent 执行过程中的关键步骤。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    history_item = {
        "node": "tool_parse_node",
        "status": "success",
    }

    state_scope.add_history(
        history_item,
    )

    assert state_scope.state.execution_history == [
        history_item,
    ]


def test_state_scope_add_history_should_keep_order():
    """
    测试 execution_history 是否保持追加顺序。

    order（顺序）：
    执行历史需要按照真实执行先后保存，方便后续 debug report 分析。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    first_item = {
        "node": "tool_parse_node",
        "status": "start",
    }

    second_item = {
        "node": "tool_parse_node",
        "status": "end",
    }

    state_scope.add_history(
        first_item,
    )

    state_scope.add_history(
        second_item,
    )

    assert state_scope.state.execution_history == [
        first_item,
        second_item,
    ]


def test_state_scope_set_and_get_retry_count():
    """
    测试 set_retry_count 和 get_retry_count 是否可以正确设置和读取重试次数。

    retry count（重试次数）：
    表示当前执行流程已经重试的次数，例如 retrieval retry、tool retry。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_retry_count(
        3,
    )

    assert state_scope.get_retry_count() == 3
    assert state_scope.state.retry_count == 3


def test_state_scope_export_should_return_state_dict():
    """
    测试 export 是否可以导出当前状态字典。

    state dict（状态字典）：
    使用 dict 格式保存当前状态，方便日志、checkpoint、debug report 使用。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.set_agent(
        "general_qa_agent",
    )

    state_scope.set_node(
        "answer_gen_node",
    )

    state_scope.set_tool(
        "weather_tool",
    )

    state_scope.set_phase(
        "answering",
    )

    state_scope.set_retry_count(
        2,
    )

    exported_state = state_scope.export()

    assert exported_state == {
        "current_agent": "general_qa_agent",
        "current_node": "answer_gen_node",
        "current_tool": "weather_tool",
        "phase": "answering",
        "retry_count": 2,
    }


def test_state_scope_restore_should_restore_state_from_dict():
    """
    测试 restore 是否可以从字典恢复状态。

    restore（恢复）：
    从 dict 中读取字段，并写回 RuntimeState。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    restore_data = {
        "current_agent": "exact_search_agent",
        "current_node": "retrieve_node",
        "current_tool": "retriever",
        "phase": "retrieving",
        "retry_count": 1,
    }

    state_scope.restore(
        restore_data,
    )

    assert state_scope.state.current_agent == "exact_search_agent"
    assert state_scope.state.current_node == "retrieve_node"
    assert state_scope.state.current_tool == "retriever"
    assert state_scope.state.phase == "retrieving"
    assert state_scope.state.retry_count == 1


def test_state_scope_restore_should_default_retry_count_to_zero():
    """
    测试 restore 在缺少 retry_count 时是否默认恢复为 0。

    当前 StateScope.restore 中使用 data.get("retry_count", 0)，
    所以当数据中没有 retry_count 字段时，retry_count 应该是 0。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    restore_data = {
        "current_agent": "general_qa_agent",
        "current_node": "answer_gen_node",
        "current_tool": "weather_tool",
        "phase": "answering",
    }

    state_scope.restore(
        restore_data,
    )

    assert state_scope.state.retry_count == 0


def test_state_scope_restore_should_allow_missing_optional_fields():
    """
    测试 restore 在缺少部分字段时是否可以正常工作。

    optional fields（可选字段）：
    有些状态字段在某些阶段可能不存在，例如当前没有 tool 时 current_tool 可以为 None。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    restore_data = {}

    state_scope.restore(
        restore_data,
    )

    assert state_scope.state.current_agent is None
    assert state_scope.state.current_node is None
    assert state_scope.state.current_tool is None
    assert state_scope.state.phase is None
    assert state_scope.state.retry_count == 0


@pytest.mark.asyncio
async def test_state_scope_startup_should_not_change_state():
    """
    测试 startup 是否不会修改当前 state。

    当前 StateScope.startup 是 pass，
    所以 startup 前后的 export 结果应该一致。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.restore(
        {
            "current_agent": "general_qa_agent",
            "current_node": "tool_parse_node",
            "current_tool": "date_tool",
            "phase": "tool_calling",
            "retry_count": 1,
        }
    )

    before_startup = state_scope.export()

    await state_scope.startup()

    after_startup = state_scope.export()

    assert after_startup == before_startup


@pytest.mark.asyncio
async def test_state_scope_shutdown_should_clear_execution_history():
    """
    测试 shutdown 是否会清空 execution_history。

    当前 StateScope.shutdown 中调用：
        self.state.execution_history.clear()

    所以 shutdown 后 execution_history 应该为空。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.add_history(
        {
            "node": "tool_parse_node",
            "status": "success",
        }
    )

    assert len(state_scope.state.execution_history) == 1

    await state_scope.shutdown()

    assert state_scope.state.execution_history == []


@pytest.mark.asyncio
async def test_state_scope_shutdown_should_not_clear_exported_fields():
    """
    测试 shutdown 是否不会清空 current_agent、current_node、current_tool、phase、retry_count。

    当前 StateScope.shutdown 只清空 execution_history，
    并没有重置其他字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    state_scope = StateScope()

    state_scope.restore(
        {
            "current_agent": "general_qa_agent",
            "current_node": "answer_gen_node",
            "current_tool": "weather_tool",
            "phase": "answering",
            "retry_count": 2,
        }
    )

    await state_scope.shutdown()

    assert state_scope.export() == {
        "current_agent": "general_qa_agent",
        "current_node": "answer_gen_node",
        "current_tool": "weather_tool",
        "phase": "answering",
        "retry_count": 2,
    }