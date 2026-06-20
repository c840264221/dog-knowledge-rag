"""
ask_confirm_tool_node 单元测试。

ask_confirm_tool_node（工具确认节点）：
用于在真正执行工具之前，通过 interrupt（中断）询问用户是否确认执行工具。

build_ask_confirm_tool_node（工具确认节点构建函数）：
用于通过 Dependency Injection（依赖注入）方式，把 checkpoint_manager、
interrupt_func、runtime_context_getter 注入到 node 中。

Interrupt（中断）：
LangGraph 中用于暂停当前 Graph 执行，并等待用户输入的机制。

Graph Node（图节点）：
LangGraph 中的执行节点，接收 state，返回 dict，用于更新 Graph 状态。
"""

from src.graph.nodes.ask_confirm_tool_node import (
    build_ask_confirm_tool_node,
)


class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 Graph 正在执行的 node。
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
                事件类型，例如 node。

            name：
                事件名称，例如 ask_confirm_tool_node。

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


class FakeInterrupt:
    """
    测试用假 Interrupt。

    Interrupt（中断）：
    在真实 LangGraph 中会暂停执行并等待用户输入。
    测试中用 FakeInterrupt 直接返回预设用户输入。
    """

    def __init__(self, user_input):
        """
        初始化假中断函数。

        参数：
            user_input：
                模拟用户输入，例如 y 或 n。

        返回值：
            None：构造函数无返回值。
        """

        self.user_input = user_input
        self.prompts = []

    def __call__(self, prompt):
        """
        模拟 interrupt(prompt)。

        参数：
            prompt：
                节点生成的用户确认提示。

        返回值：
            str：
                模拟用户输入。
        """

        self.prompts.append(
            prompt
        )

        return self.user_input


def build_test_node(
    user_input="y",
    with_checkpoint=True,
    with_runtime_context=True,
):
    """
    构建测试用 ask_confirm_tool_node。

    功能：
        创建 fake checkpoint_manager、fake interrupt、fake runtime_context。
        然后通过 build_ask_confirm_tool_node 注入依赖，得到真正可执行的 node。

    参数：
        user_input：
            模拟用户输入。
            y 表示确认执行工具。
            其他输入表示拒绝执行工具。

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
            node, fake_ctx, fake_checkpoint_manager, fake_interrupt。
    """

    fake_ctx = FakeRuntimeContext()

    fake_checkpoint_manager = (
        FakeCheckpointManager()
        if with_checkpoint
        else None
    )

    fake_interrupt = FakeInterrupt(
        user_input=user_input
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

    node = build_ask_confirm_tool_node(
        checkpoint_manager=fake_checkpoint_manager,
        interrupt_func=fake_interrupt,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        fake_checkpoint_manager,
        fake_interrupt,
    )


def test_ask_confirm_tool_node_should_return_no_tool_when_tool_calls_missing():
    """
    测试缺少 tool_calls 时，是否返回 need_tool=False。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_checkpoint_manager,
        fake_interrupt,
    ) = build_test_node()

    state = {}

    result = node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            "没有工具需要确认。"
        ]
    }

    assert fake_ctx.state_scope.current_node == "ask_confirm_tool_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "ask_confirm_tool_node",
            "metadata": None,
        }
    ]

    assert fake_interrupt.prompts == []
    assert fake_checkpoint_manager.save_count == 0


def test_ask_confirm_tool_node_should_return_no_tool_when_tool_calls_empty():
    """
    测试 tool_calls 为空列表时，是否返回 need_tool=False。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        fake_interrupt,
    ) = build_test_node()

    state = {
        "tool_calls": [],
    }

    result = node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            "没有工具需要确认。"
        ]
    }

    assert fake_interrupt.prompts == []
    assert fake_checkpoint_manager.save_count == 0


def test_ask_confirm_tool_node_should_return_empty_dict_when_user_confirms():
    """
    测试用户输入 y 时，是否返回空 dict。

    empty dict（空字典）：
    在 LangGraph node 中通常表示不更新 state。
    这里表示保留 need_tool=True 和 tool_calls，继续进入 execute_tool。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_checkpoint_manager,
        fake_interrupt,
    ) = build_test_node(
        user_input="y",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "成都",
                    "date": "2026-06-20",
                },
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {}

    assert fake_ctx.state_scope.current_node == "ask_confirm_tool_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "ask_confirm_tool_node",
            "metadata": None,
        }
    ]

    assert fake_interrupt.prompts == [
        "即将执行以下工具：\n"
        "1. 工具：【weather】，参数：【{'city': '成都', 'date': '2026-06-20'}】\n"
        "是否继续？(y/n)"
    ]

    assert fake_checkpoint_manager.save_count == 2


def test_ask_confirm_tool_node_should_accept_uppercase_y():
    """
    测试用户输入大写 Y 时，是否也视为确认。

    lower（转小写）：
    当前实现使用 user_input.strip().lower() == "y"，
    所以大写 Y 会被转换成小写 y。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        _fake_interrupt,
    ) = build_test_node(
        user_input="Y",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {}
    assert fake_checkpoint_manager.save_count == 2


def test_ask_confirm_tool_node_should_strip_user_input():
    """
    测试用户输入前后带空格时，是否会 strip 后再判断。

    strip（去除空白）：
    去掉字符串前后的空格、换行符等空白字符。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        _fake_interrupt,
    ) = build_test_node(
        user_input="  y  ",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {}
    assert fake_checkpoint_manager.save_count == 2


def test_ask_confirm_tool_node_should_cancel_tool_when_user_rejects():
    """
    测试用户输入 n 时，是否取消工具调用。

    cancel tool call（取消工具调用）：
    清空 tool_calls，并设置 need_tool=False，防止后续误执行工具。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        fake_interrupt,
    ) = build_test_node(
        user_input="n",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "成都",
                    "date": "2026-06-20",
                },
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            "用户取消了工具调用。"
        ],
        "tool_calls": []
    }

    assert fake_interrupt.prompts == [
        "即将执行以下工具：\n"
        "1. 工具：【weather】，参数：【{'city': '成都', 'date': '2026-06-20'}】\n"
        "是否继续？(y/n)"
    ]

    assert fake_checkpoint_manager.save_count == 2


def test_ask_confirm_tool_node_should_cancel_tool_when_user_input_unknown():
    """
    测试用户输入非 y 内容时，是否默认取消工具调用。

    fallback（兜底）：
    当用户输入不是 y 时，统一走拒绝分支，避免误调用工具。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        _fake_interrupt,
    ) = build_test_node(
        user_input="abc",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {
        "need_tool": False,
        "tool_results": [
            "用户取消了工具调用。"
        ],
        "tool_calls": []
    }

    assert fake_checkpoint_manager.save_count == 2


def test_ask_confirm_tool_node_should_only_use_first_tool_call():
    """
    测试存在多个 tool_calls 时，是否只使用第一个生成确认提示。

    当前实现：
        tool_call = tool_calls[0]

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_checkpoint_manager,
        fake_interrupt,
    ) = build_test_node(
        user_input="y",
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "成都",
                },
            },
            {
                "name": "date",
                "args": {},
            },
        ],
    }

    result = node(
        state,
    )

    assert result == {}

    assert fake_interrupt.prompts == [
        "即将执行以下工具：\n"
        "1. 工具：【weather】，参数：【{'city': '成都'}】\n"
        "2. 工具：【date】，参数：【{}】\n"
        "是否继续？(y/n)"
    ]


def test_ask_confirm_tool_node_should_work_without_checkpoint_manager():
    """
    测试未注入 checkpoint_manager 时，节点是否仍可正常工作。

    checkpoint_manager=None：
    表示当前节点不保存 checkpoint，但不应该影响确认逻辑。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_checkpoint_manager,
        _fake_interrupt,
    ) = build_test_node(
        user_input="y",
        with_checkpoint=False,
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {}
    assert fake_checkpoint_manager is None


def test_ask_confirm_tool_node_should_work_without_runtime_context():
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

    (
        node,
        fake_ctx,
        fake_checkpoint_manager,
        _fake_interrupt,
    ) = build_test_node(
        user_input="y",
        with_runtime_context=False,
    )

    state = {
        "need_tool": True,
        "tool_calls": [
            {
                "name": "date",
                "args": {},
            }
        ],
    }

    result = node(
        state,
    )

    assert result == {}

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.timeline_scope.events == []
    assert fake_checkpoint_manager.save_count == 2