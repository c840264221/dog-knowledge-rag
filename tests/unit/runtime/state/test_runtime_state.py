"""
runtime state 单元测试。

RuntimeState（运行时状态）：
用于保存一次 Agent / Graph 执行过程中的基础状态信息。

dataclass（数据类）：
Python 提供的简化类定义方式，适合保存结构化数据。

execution_history（执行历史）：
用于兼容旧版本执行历史记录。
当前项目已经使用 TimelineScope 作为更完整的时间线系统，
但 execution_history 暂时保留作为兼容字段。

default_factory（默认工厂）：
dataclasses.field 提供的能力，用于为每个实例创建独立的默认对象。
常用于 list、dict 这类可变对象，避免多个实例共享同一个列表。
"""

from dataclasses import is_dataclass

from src.runtime.state.runtime_state import RuntimeState


def test_runtime_state_should_be_dataclass():
    """
    测试 RuntimeState 是否是 dataclass。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert is_dataclass(RuntimeState)


def test_runtime_state_can_be_created():
    """
    测试 RuntimeState 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState()

    assert runtime_state is not None


def test_runtime_state_default_values_should_be_correct():
    """
    测试 RuntimeState 默认字段值是否符合预期。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState()

    assert runtime_state.current_agent is None
    assert runtime_state.current_node is None
    assert runtime_state.current_tool is None
    assert runtime_state.phase is None
    assert runtime_state.retry_count == 0
    assert runtime_state.execution_history == []


def test_runtime_state_can_be_created_with_custom_values():
    """
    测试 RuntimeState 是否可以通过构造函数传入自定义字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState(
        current_agent="general_qa_agent",
        current_node="tool_parse_node",
        current_tool="weather_tool",
        phase="tool_calling",
        retry_count=2,
        execution_history=[
            {
                "node": "tool_parse_node",
                "status": "success",
            }
        ],
    )

    assert runtime_state.current_agent == "general_qa_agent"
    assert runtime_state.current_node == "tool_parse_node"
    assert runtime_state.current_tool == "weather_tool"
    assert runtime_state.phase == "tool_calling"
    assert runtime_state.retry_count == 2
    assert runtime_state.execution_history == [
        {
            "node": "tool_parse_node",
            "status": "success",
        }
    ]


def test_runtime_state_fields_can_be_updated():
    """
    测试 RuntimeState 字段是否可以正常修改。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState()

    runtime_state.current_agent = "exact_search_agent"
    runtime_state.current_node = "retrieve_node"
    runtime_state.current_tool = "retriever"
    runtime_state.phase = "retrieving"
    runtime_state.retry_count = 1

    assert runtime_state.current_agent == "exact_search_agent"
    assert runtime_state.current_node == "retrieve_node"
    assert runtime_state.current_tool == "retriever"
    assert runtime_state.phase == "retrieving"
    assert runtime_state.retry_count == 1


def test_runtime_state_execution_history_can_append_items():
    """
    测试 execution_history 是否可以追加执行历史记录。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState()

    history_item = {
        "node": "answer_gen_node",
        "status": "success",
    }

    runtime_state.execution_history.append(
        history_item,
    )

    assert runtime_state.execution_history == [
        history_item,
    ]


def test_runtime_state_execution_history_should_be_isolated_between_instances():
    """
    测试不同 RuntimeState 实例之间的 execution_history 是否相互隔离。

    为什么要测这个：
    execution_history 是 list（列表），属于可变对象。
    如果没有使用 field(default_factory=list)，多个 RuntimeState 实例可能会共享同一个列表，
    导致一个实例追加历史记录，另一个实例也受到影响。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_state = RuntimeState()
    second_state = RuntimeState()

    first_state.execution_history.append(
        {
            "node": "first_node",
            "status": "success",
        }
    )

    assert first_state.execution_history == [
        {
            "node": "first_node",
            "status": "success",
        }
    ]

    assert second_state.execution_history == []
    assert first_state.execution_history is not second_state.execution_history


def test_runtime_state_retry_count_can_be_incremented():
    """
    测试 retry_count 是否可以作为重试计数器正常递增。

    retry_count（重试次数）：
    用于记录检索重试、工具重试或节点重试次数。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    runtime_state = RuntimeState()

    runtime_state.retry_count += 1
    runtime_state.retry_count += 1

    assert runtime_state.retry_count == 2