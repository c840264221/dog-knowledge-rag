"""
execute_tool_node 单元测试。

execute_tool_node（工具执行节点）：
用于执行当前 state.tool_calls 中的第一个工具调用，并把工具结果追加到 tool_results 列表中。

build_execute_tool_node（工具执行节点构建函数）：
用于通过 Dependency Injection（依赖注入）方式，把 tool_executor、
checkpoint_manager、runtime_context_getter 注入到 node 中。

Tool Results（工具结果）：
工具执行后的结果集合。这里统一使用 list 结构，支持多个工具调用结果。

Chain Tool Execution（链式工具执行）：
每次只执行 tool_calls 中的第一个工具，然后把剩余 tool_calls 放回 state，
由 Graph 下一轮继续执行。
"""

import pytest

from src.graph.nodes.execute_tool_node import (
    build_execute_tool_node,
    _normalize_tool_results,
    _dump_tool_result,
)


class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 Graph 正在执行的 node 和 tool。
    """

    def __init__(self):
        """
        初始化假 StateScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.current_node = None
        self.current_tool = None

    def set_node(self, node_name):
        """
        设置当前 node 名称。

        参数：
            node_name：
                当前执行的节点名称。

        返回值：
            None：无业务返回值。
        """

        self.current_node = node_name

    def set_tool(self, tool_name):
        """
        设置当前 tool 名称。

        参数：
            tool_name：
                当前执行的工具名称。

        返回值：
            None：无业务返回值。
        """

        self.current_tool = tool_name


class FakeTimelineScope:
    """
    测试用假 TimelineScope。

    TimelineScope（时间线作用域）：
    用于记录当前请求执行过程中的事件。
    """

    def __init__(self):
        """
        初始化假 TimelineScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.events = []

    def add_event(self, event_type, name, metadata=None):
        """
        添加时间线事件。

        参数：
            event_type：
                事件类型，例如 node 或 tool。

            name：
                事件名称，例如 execute_tool_node 或 weather。

            metadata：
                事件附加信息，默认为 None。

        返回值：
            None：无业务返回值。
        """

        self.events.append(
            {
                "event_type": event_type,
                "name": name,
                "metadata": metadata,
            }
        )


class FakeRuntimeContext:
    """
    测试用假 RuntimeContext。

    RuntimeContext（运行时上下文）：
    表示一次请求执行过程中的运行时环境。
    """

    def __init__(self):
        """
        初始化假 RuntimeContext。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self):
        """
        获取假 StateScope。

        参数：
            无。

        返回值：
            FakeStateScope：
                测试用状态作用域。
        """

        return self.state_scope

    def timeline(self):
        """
        获取假 TimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope：
                测试用时间线作用域。
        """

        return self.timeline_scope


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于保存、恢复、清理运行时检查点。
    """

    def __init__(self):
        """
        初始化假 CheckpointManager。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.save_count = 0

    def save_checkpoint(self):
        """
        模拟保存 checkpoint。

        参数：
            无。

        返回值：
            None：无业务返回值。
        """

        self.save_count += 1


class FakeToolResultModel:
    """
    测试用假 Pydantic Model 工具结果。

    Pydantic Model（Pydantic 模型）：
    通常支持 model_dump 方法，可以转换成 dict。
    """

    def __init__(self, data):
        """
        初始化假工具结果模型。

        参数：
            data：
                model_dump 需要返回的数据。

        返回值：
            None：构造函数无返回值。
        """

        self.data = data

    def model_dump(self):
        """
        模拟 Pydantic model_dump。

        参数：
            无。

        返回值：
            dict：
                工具结果字典。
        """

        return self.data


class FakeToolExecutor:
    """
    测试用假 ToolExecutor。

    ToolExecutor（工具执行器）：
    用于根据工具名称和参数执行工具。
    """

    def __init__(self, result=None, error=None):
        """
        初始化假工具执行器。

        参数：
            result：
                execute 成功时返回的工具结果。

            error：
                execute 需要主动抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.result = result
        self.error = error
        self.calls = []

    async def execute(self, name, args):
        """
        模拟异步执行工具。

        参数：
            name：
                工具名称。

            args：
                工具参数。

        返回值：
            object：
                模拟工具执行结果。
        """

        self.calls.append(
            {
                "name": name,
                "args": args,
            }
        )

        if self.error:
            raise self.error

        return self.result


def build_test_node(
    tool_result=None,
    tool_error=None,
    with_checkpoint=True,
    with_runtime_context=True,
):
    """
    构建测试用 execute_tool_node。

    功能：
        创建 fake tool_executor、fake checkpoint_manager、fake runtime_context。
        然后通过 build_execute_tool_node 注入依赖，得到真正可执行的 node。

    参数：
        tool_result：
            假工具执行器成功返回的结果。

        tool_error：
            假工具执行器需要抛出的异常。

        with_checkpoint：
            是否传入 checkpoint_manager。
            True 表示传入 fake checkpoint_manager。
            False 表示传入 None。

        with_runtime_context：
            是否提供 fake runtime context。
            True 表示 runtime_context_getter 返回 fake_ctx。
            False 表示 runtime_context_getter 返回 None。

    返回值：
        tuple：
            node, fake_ctx, fake_tool_executor, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()

    fake_tool_executor = FakeToolExecutor(
        result=tool_result,
        error=tool_error,
    )

    fake_checkpoint_manager = (
        FakeCheckpointManager()
        if with_checkpoint
        else None
    )

    def runtime_context_getter():
        """
        获取测试用 RuntimeContext。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None：
                根据 with_runtime_context 决定是否返回 fake_ctx。
        """

        if not with_runtime_context:
            return None

        return fake_ctx

    node = build_execute_tool_node(
        tool_executor=fake_tool_executor,
        checkpoint_manager=fake_checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    )


def test_normalize_tool_results_should_return_empty_list_when_none():
    """
    测试 tool_results 为 None 时，是否归一化为空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert _normalize_tool_results(
        None,
    ) == []


def test_normalize_tool_results_should_keep_list():
    """
    测试 tool_results 已经是 list 时，是否保持 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_results = [
        "result1",
        "result2",
    ]

    result = _normalize_tool_results(
        tool_results,
    )

    assert result == tool_results
    assert result is tool_results


def test_normalize_tool_results_should_convert_string_to_list():
    """
    测试历史遗留字符串 tool_results 是否转换成 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = _normalize_tool_results(
        "用户取消了工具调用。",
    )

    assert result == [
        "用户取消了工具调用。",
    ]


def test_normalize_tool_results_should_wrap_unknown_object():
    """
    测试未知类型 tool_results 是否被包装成 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    raw_result = {
        "tool_name": "date",
        "result": "2026-06-20",
    }

    result = _normalize_tool_results(
        raw_result,
    )

    assert result == [
        raw_result,
    ]


def test_dump_tool_result_should_use_model_dump_when_available():
    """
    测试工具结果有 model_dump 方法时，是否转换成 dict。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    model_result = FakeToolResultModel(
        {
            "tool_name": "weather",
            "result": "北京天气：晴",
            "success": True,
        }
    )

    result = _dump_tool_result(
        model_result,
    )

    assert result == {
        "tool_name": "weather",
        "result": "北京天气：晴",
        "success": True,
    }


def test_dump_tool_result_should_keep_plain_dict():
    """
    测试工具结果本身是 dict 时，是否直接返回。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    raw_result = {
        "tool_name": "date",
        "result": "2026-06-20",
        "success": True,
    }

    result = _dump_tool_result(
        raw_result,
    )

    assert result == raw_result


@pytest.mark.asyncio
async def test_execute_tool_node_should_return_existing_results_when_no_tool_calls():
    """
    测试没有 tool_calls 时，是否安全返回已有 tool_results。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node()

    state = {
        "tool_results": [
            {
                "tool_name": "date",
                "result": "2026-06-20",
            }
        ]
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            {
                "tool_name": "date",
                "result": "2026-06-20",
            }
        ],
    }

    assert fake_ctx.state_scope.current_node == "execute_tool_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "execute_tool_node",
            "metadata": None,
        }
    ]

    assert fake_tool_executor.calls == []
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_execute_tool_node_should_normalize_string_results_when_no_tool_calls():
    """
    测试没有 tool_calls 且已有 tool_results 是字符串时，是否归一化成 list。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node()

    state = {
        "tool_results": "用户取消了工具调用。",
    }

    result = await node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            "用户取消了工具调用。",
        ],
    }

    assert fake_tool_executor.calls == []
    assert fake_checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_execute_tool_node_should_execute_single_tool_and_append_result():
    """
    测试单个工具调用时，是否执行工具并追加工具结果。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = FakeToolResultModel(
        {
            "tool_name": "weather",
            "result": "北京天气：晴",
            "success": True,
        }
    )

    (
        node,
        fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
    )

    state = {
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                    "date": "2025-03-15",
                },
            }
        ],
        "tool_results": [],
        "tool_round": 2,
    }

    result = await node(
        state,
    )

    assert result == {
        "tool_results": [
            {
                "tool_name": "weather",
                "result": "北京天气：晴",
                "success": True,
            }
        ],
        "tool_calls": [],
        "need_tool": False,
        "tool_round": 3,
    }

    assert fake_tool_executor.calls == [
        {
            "name": "weather",
            "args": {
                "city": "北京",
                "date": "2025-03-15",
            },
        }
    ]

    assert fake_ctx.state_scope.current_node == "execute_tool_node"
    assert fake_ctx.state_scope.current_tool == "weather"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "execute_tool_node",
            "metadata": None,
        },
        {
            "event_type": "tool",
            "name": "weather",
            "metadata": {
                "args": {
                    "city": "北京",
                    "date": "2025-03-15",
                }
            },
        },
    ]

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_execute_tool_node_should_execute_first_tool_and_keep_remaining_calls():
    """
    测试多个 tool_calls 时，是否只执行第一个，并保留剩余工具调用。

    Chain Tool Execution（链式工具执行）：
    当前节点每次只执行第一个工具，剩余工具留给下一轮 Graph 执行。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = FakeToolResultModel(
        {
            "tool_name": "date",
            "result": "2026-06-20",
            "success": True,
        }
    )

    (
        node,
        _fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
    )

    remaining_weather_call = {
        "name": "weather",
        "args": {
            "city": "上海",
            "date": "2026-06-20",
        },
    }

    state = {
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            },
            remaining_weather_call,
        ],
        "tool_results": [
            {
                "tool_name": "previous",
                "result": "之前的结果",
            }
        ],
    }

    result = await node(
        state,
    )

    assert result == {
        "tool_results": [
            {
                "tool_name": "previous",
                "result": "之前的结果",
            },
            {
                "tool_name": "date",
                "result": "2026-06-20",
                "success": True,
            },
        ],
        "tool_calls": [
            remaining_weather_call,
        ],
        "need_tool": True,
        "tool_round": 1,
    }

    assert fake_tool_executor.calls == [
        {
            "name": "date",
            "args": {},
        }
    ]

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_execute_tool_node_should_normalize_existing_string_result_and_append_new_result():
    """
    测试已有 tool_results 是字符串时，是否先归一化再追加新工具结果。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = {
        "tool_name": "date",
        "result": "2026-06-20",
        "success": True,
    }

    (
        node,
        _fake_ctx,
        _fake_tool_executor,
        _fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
    )

    state = {
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
        "tool_results": "历史字符串结果",
    }

    result = await node(
        state,
    )

    assert result["tool_results"] == [
        "历史字符串结果",
        {
            "tool_name": "date",
            "result": "2026-06-20",
            "success": True,
        },
    ]


@pytest.mark.asyncio
async def test_execute_tool_node_should_use_default_args_when_args_missing():
    """
    测试 tool_call 缺少 args 时，是否默认使用空 dict。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = {
        "tool_name": "date",
        "result": "2026-06-20",
    }

    (
        node,
        _fake_ctx,
        fake_tool_executor,
        _fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
    )

    state = {
        "tool_calls": [
            {
                "name": "date",
            }
        ],
    }

    result = await node(
        state,
    )

    assert fake_tool_executor.calls == [
        {
            "name": "date",
            "args": {},
        }
    ]

    assert result["tool_results"] == [
        {
            "tool_name": "date",
            "result": "2026-06-20",
        }
    ]


@pytest.mark.asyncio
async def test_execute_tool_node_should_use_default_tool_round_when_missing():
    """
    测试缺少 tool_round 时，是否默认从 0 递增到 1。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = {
        "tool_name": "date",
        "result": "2026-06-20",
    }

    (
        node,
        _fake_ctx,
        _fake_tool_executor,
        _fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
    )

    state = {
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = await node(
        state,
    )

    assert result["tool_round"] == 1


@pytest.mark.asyncio
async def test_execute_tool_node_should_work_without_checkpoint_manager():
    """
    测试未注入 checkpoint_manager 时，节点是否仍可正常工作。

    checkpoint_manager=None：
    表示当前节点不保存 checkpoint，但不应该影响工具执行逻辑。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = {
        "tool_name": "date",
        "result": "2026-06-20",
    }

    (
        node,
        _fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
        with_checkpoint=False,
    )

    state = {
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = await node(
        state,
    )

    assert result["need_tool"] is False
    assert len(fake_tool_executor.calls) == 1
    assert fake_checkpoint_manager is None


@pytest.mark.asyncio
async def test_execute_tool_node_should_work_without_runtime_context():
    """
    测试 runtime_context_getter 返回 None 时，节点是否仍可正常工作。

    runtime_context_getter（运行时上下文获取函数）：
    用于在 node 执行时获取当前 RuntimeContext。
    如果返回 None，node 应该跳过 state/timeline 写入，而不是报错。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    tool_result = {
        "tool_name": "date",
        "result": "2026-06-20",
    }

    (
        node,
        fake_ctx,
        fake_tool_executor,
        fake_checkpoint_manager,
    ) = build_test_node(
        tool_result=tool_result,
        with_runtime_context=False,
    )

    state = {
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = await node(
        state,
    )

    assert result["need_tool"] is False
    assert len(fake_tool_executor.calls) == 1
    assert fake_checkpoint_manager.save_count == 1

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.state_scope.current_tool is None
    assert fake_ctx.timeline_scope.events == []