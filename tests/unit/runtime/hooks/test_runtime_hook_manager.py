"""
runtime hook manager 单元测试。

RuntimeHookManager（运行时钩子管理器）：
用于管理 runtime hook（运行时钩子），允许系统在关键执行点插入扩展逻辑。

hook（钩子）：
在某个特定时机被触发的扩展对象，例如 before_node、after_node、on_error。

hook_name（钩子名称）：
用于区分不同钩子触发点的字符串，例如 "before_node"、"after_node"、"on_error"。

emit（触发）：
根据 hook_name 找到已注册的 hooks，并依次调用 hook.execute。

execute（执行）：
hook 对象中真正处理逻辑的方法。
"""

import pytest

from src.runtime.hooks.hook_manager import RuntimeHookManager


class FakeAsyncHook:
    """
    测试用异步 Hook。

    FakeAsyncHook（假异步钩子）：
    用于测试 RuntimeHookManager.emit 是否会调用 hook.execute。
    """

    def __init__(self):
        """
        初始化测试 Hook。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.executed = False
        self.calls = []

    async def execute(self, *args, **kwargs):
        """
        异步执行 Hook。

        参数：
            *args：位置参数，格式为 tuple。
            **kwargs：关键字参数，格式为 dict。

        返回值：
            None：无业务返回值，只记录调用信息。
        """

        self.executed = True

        self.calls.append(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )


class FakeSyncHook:
    """
    测试用同步 Hook。

    FakeSyncHook（假同步钩子）：
    用于测试 RuntimeHookManager.emit 是否可以兼容普通 def execute。
    """

    def __init__(self):
        """
        初始化同步测试 Hook。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.executed = False
        self.calls = []

    def execute(self, *args, **kwargs):
        """
        同步执行 Hook。

        参数：
            *args：位置参数，格式为 tuple。
            **kwargs：关键字参数，格式为 dict。

        返回值：
            None：无业务返回值，只记录调用信息。
        """

        self.executed = True

        self.calls.append(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )


def test_runtime_hook_manager_can_be_created():
    """
    测试 RuntimeHookManager 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    assert hook_manager is not None
    assert hook_manager._hooks is not None
    assert len(hook_manager._hooks) == 0


def test_runtime_hook_manager_register_should_add_hook():
    """
    测试 register 是否可以注册 hook。

    register（注册）：
    将 hook 添加到指定 hook_name 对应的钩子列表中。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()
    hook = FakeAsyncHook()

    hook_manager.register(
        "before_node",
        hook,
    )

    assert "before_node" in hook_manager._hooks
    assert hook in hook_manager._hooks["before_node"]


def test_runtime_hook_manager_register_multiple_hooks_should_keep_order():
    """
    测试同一个 hook_name 下注册多个 hook 时是否保持注册顺序。

    register order（注册顺序）：
    hooks 被 register 进入列表的先后顺序。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    first_hook = FakeAsyncHook()
    second_hook = FakeAsyncHook()

    hook_manager.register(
        "after_node",
        first_hook,
    )

    hook_manager.register(
        "after_node",
        second_hook,
    )

    assert hook_manager._hooks["after_node"] == [
        first_hook,
        second_hook,
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_execute_registered_hook():
    """
    测试 emit 是否会执行已注册的 hook。

    emit（触发）：
    根据 hook_name 找到对应 hooks，并依次调用 execute。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()
    hook = FakeAsyncHook()

    hook_manager.register(
        "before_node",
        hook,
    )

    await hook_manager.emit(
        "before_node",
    )

    assert hook.executed is True
    assert len(hook.calls) == 1


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_pass_args_and_kwargs():
    """
    测试 emit 是否会把 *args 和 **kwargs 传给 hook.execute。

    *args（位置参数）：
    按位置传递的参数。

    **kwargs（关键字参数）：
    按名称传递的参数。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()
    hook = FakeAsyncHook()

    hook_manager.register(
        "before_node",
        hook,
    )

    await hook_manager.emit(
        "before_node",
        "tool_parse_node",
        trace_id="test_trace_id",
        state={
            "question": "金毛适合新手养吗？",
        },
    )

    assert hook.calls == [
        {
            "args": (
                "tool_parse_node",
            ),
            "kwargs": {
                "trace_id": "test_trace_id",
                "state": {
                    "question": "金毛适合新手养吗？",
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_execute_multiple_hooks():
    """
    测试 emit 是否会执行同一个 hook_name 下的多个 hook。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    first_hook = FakeAsyncHook()
    second_hook = FakeAsyncHook()

    hook_manager.register(
        "after_node",
        first_hook,
    )

    hook_manager.register(
        "after_node",
        second_hook,
    )

    await hook_manager.emit(
        "after_node",
        "answer_gen_node",
        output={
            "answer": "可以，金毛通常适合新手。",
        },
    )

    assert first_hook.executed is True
    assert second_hook.executed is True

    assert first_hook.calls == [
        {
            "args": (
                "answer_gen_node",
            ),
            "kwargs": {
                "output": {
                    "answer": "可以，金毛通常适合新手。",
                },
            },
        }
    ]

    assert second_hook.calls == [
        {
            "args": (
                "answer_gen_node",
            ),
            "kwargs": {
                "output": {
                    "answer": "可以，金毛通常适合新手。",
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_only_execute_matching_hook_name():
    """
    测试 emit 是否只执行匹配 hook_name 的 hook。

    matching hook name（匹配的钩子名称）：
    只有注册在当前 hook_name 下的 hook 才应该被触发。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    before_node_hook = FakeAsyncHook()
    after_node_hook = FakeAsyncHook()

    hook_manager.register(
        "before_node",
        before_node_hook,
    )

    hook_manager.register(
        "after_node",
        after_node_hook,
    )

    await hook_manager.emit(
        "before_node",
        "tool_parse_node",
    )

    assert before_node_hook.executed is True
    assert after_node_hook.executed is False


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_without_hooks_should_not_raise_error():
    """
    测试没有 hook 时 emit 是否不会报错。

    no hooks（无钩子）：
    当前 hook_name 没有注册任何 hook。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    await hook_manager.emit(
        "missing_hook",
        "some_node",
    )

    assert hook_manager._hooks.get(
        "missing_hook",
        [],
    ) == []


def test_runtime_hook_manager_should_group_hooks_by_hook_name():
    """
    测试 RuntimeHookManager 是否按 hook_name 分组保存 hooks。

    group by hook_name（按钩子名称分组）：
    不同 hook_name 下的 hooks 应该互不混淆。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    before_hook = FakeAsyncHook()
    error_hook = FakeAsyncHook()

    hook_manager.register(
        "before_node",
        before_hook,
    )

    hook_manager.register(
        "on_error",
        error_hook,
    )

    assert hook_manager._hooks["before_node"] == [
        before_hook,
    ]

    assert hook_manager._hooks["on_error"] == [
        error_hook,
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_support_sync_hook():
    """
    测试 RuntimeHookManager.emit 是否支持同步 hook。

    sync hook（同步钩子）：
    execute 方法使用普通 def 定义，调用后会立即执行并返回 None。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()
    hook = FakeSyncHook()

    hook_manager.register(
        "before_node",
        hook,
    )

    await hook_manager.emit(
        "before_node",
        "tool_parse_node",
        trace_id="test_trace_id",
    )

    assert hook.executed is True

    assert hook.calls == [
        {
            "args": (
                "tool_parse_node",
            ),
            "kwargs": {
                "trace_id": "test_trace_id",
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_support_sync_and_async_hooks_together():
    """
    测试 RuntimeHookManager.emit 是否可以同时支持同步 hook 和异步 hook。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    sync_hook = FakeSyncHook()
    async_hook = FakeAsyncHook()

    hook_manager.register(
        "after_node",
        sync_hook,
    )

    hook_manager.register(
        "after_node",
        async_hook,
    )

    await hook_manager.emit(
        "after_node",
        "answer_gen_node",
        output={
            "answer": "金毛通常适合新手。",
        },
    )

    expected_call = {
        "args": (
            "answer_gen_node",
        ),
        "kwargs": {
            "output": {
                "answer": "金毛通常适合新手。",
            },
        },
    }

    assert sync_hook.executed is True
    assert async_hook.executed is True
    assert sync_hook.calls == [expected_call]
    assert async_hook.calls == [expected_call]

class FakeFailingHook:
    """
    测试用失败 Hook。

    FakeFailingHook（假失败钩子）：
    用于测试某个 hook 报错时，RuntimeHookManager 是否能隔离异常。
    """

    def __init__(
        self,
        error_message: str = "hook failed",
    ):
        """
        初始化失败 Hook。

        参数：
            error_message：
                抛出异常时使用的错误信息。

        返回值：
            None：构造函数无返回值。
        """

        self.error_message = error_message

    async def execute(self, *args, **kwargs):
        """
        执行 Hook 并主动抛出异常。

        参数：
            *args：位置参数，格式为 tuple。
            **kwargs：关键字参数，格式为 dict。

        返回值：
            None：本方法会抛出 RuntimeError，不会正常返回。
        """

        raise RuntimeError(
            self.error_message
        )

@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_continue_when_hook_failed():
    """
    测试某个 hook 报错时，RuntimeHookManager 默认是否继续执行后续 hook。

    fault isolation（故障隔离）：
    一个 hook 失败，不影响其他 hook 继续处理。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    failing_hook = FakeFailingHook()
    normal_hook = FakeAsyncHook()

    hook_manager.register(
        "after_node",
        failing_hook,
    )

    hook_manager.register(
        "after_node",
        normal_hook,
    )

    await hook_manager.emit(
        "after_node",
        "answer_gen_node",
        output={
            "answer": "金毛通常适合新手。",
        },
    )

    assert normal_hook.executed is True

    assert normal_hook.calls == [
        {
            "args": (
                "answer_gen_node",
            ),
            "kwargs": {
                "output": {
                    "answer": "金毛通常适合新手。",
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_raise_error_in_strict_mode():
    """
    测试 strict mode 下 hook 报错时是否会向外抛出异常。

    strict mode（严格模式）：
    hook 报错时不吞掉异常，而是直接 raise 给调用方。

    参数：
        无。

    返回值：
        None：pytest 会根据 pytest.raises 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager(
        raise_hook_errors=True,
    )

    failing_hook = FakeFailingHook(
        error_message="strict hook failed",
    )

    hook_manager.register(
        "on_error",
        failing_hook,
    )

    with pytest.raises(
        RuntimeError,
        match="strict hook failed",
    ):
        await hook_manager.emit(
            "on_error",
            "tool_parse_node",
            error={
                "message": "parse failed",
            },
        )

class FakeSyncFailingHook:
    """
    测试用同步失败 Hook。

    FakeSyncFailingHook（假同步失败钩子）：
    用于测试同步 execute 抛错时，RuntimeHookManager 是否能隔离异常。
    """

    def __init__(
        self,
        error_message: str = "sync hook failed",
    ):
        """
        初始化同步失败 Hook。

        参数：
            error_message：
                抛出异常时使用的错误信息。

        返回值：
            None：构造函数无返回值。
        """

        self.error_message = error_message

    def execute(self, *args, **kwargs):
        """
        同步执行 Hook 并主动抛出异常。

        参数：
            *args：位置参数，格式为 tuple。
            **kwargs：关键字参数，格式为 dict。

        返回值：
            None：本方法会抛出 RuntimeError，不会正常返回。
        """

        raise RuntimeError(
            self.error_message
        )

@pytest.mark.asyncio
async def test_runtime_hook_manager_emit_should_continue_when_sync_hook_failed():
    """
    测试同步 hook 报错时，RuntimeHookManager 默认是否继续执行后续 hook。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    hook_manager = RuntimeHookManager()

    failing_hook = FakeSyncFailingHook()
    normal_hook = FakeSyncHook()

    hook_manager.register(
        "before_node",
        failing_hook,
    )

    hook_manager.register(
        "before_node",
        normal_hook,
    )

    await hook_manager.emit(
        "before_node",
        "retrieve_node",
        trace_id="test_trace_id",
    )

    assert normal_hook.executed is True

    assert normal_hook.calls == [
        {
            "args": (
                "retrieve_node",
            ),
            "kwargs": {
                "trace_id": "test_trace_id",
            },
        }
    ]
