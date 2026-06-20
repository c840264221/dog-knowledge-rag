"""
memory scope 单元测试。

MemoryScope（记忆作用域）：
用于管理一次请求执行过程中的 memory snapshot（记忆快照）。

memory snapshot（记忆快照）：
表示当前请求中召回或整理出来的记忆数据，例如用户偏好、历史事实、上下文记忆等。

RequestScope（请求作用域）：
底层 key-value store（键值存储），用于保存一次请求内的临时数据。

KEY（键）：
MemoryScope 使用固定 key "memory_snapshot" 把记忆数据存入 RequestScope。
"""

import pytest

from src.runtime.context.request_scope import RequestScope

from src.runtime.scopes.memory_scope import MemoryScope


def test_memory_scope_can_be_created():
    """
    测试 MemoryScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    assert memory_scope is not None
    assert memory_scope.scope is request_scope


def test_memory_scope_key_should_be_memory_snapshot():
    """
    测试 MemoryScope.KEY 是否为 memory_snapshot。

    KEY（键）：
    用于在 RequestScope 中保存 memory 数据的固定名称。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert MemoryScope.KEY == "memory_snapshot"


def test_memory_scope_get_memories_should_return_empty_list_by_default():
    """
    测试未设置 memories 时，get_memories 是否默认返回空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    assert memory_scope.get_memories() == []


def test_memory_scope_set_and_get_memories():
    """
    测试 set_memories 和 get_memories 是否可以正确写入和读取记忆。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "memory_type": "preference",
            "content": "用户喜欢金毛",
        },
        {
            "memory_type": "dislike",
            "content": "用户不喜欢掉毛严重的犬种",
        },
    ]

    memory_scope.set_memories(
        memories,
    )

    assert memory_scope.get_memories() == memories


def test_memory_scope_set_memories_should_store_data_in_request_scope():
    """
    测试 set_memories 是否真的把数据保存到底层 RequestScope。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "memory_type": "favorite_dog",
            "content": "用户喜欢边牧",
        }
    ]

    memory_scope.set_memories(
        memories,
    )

    assert request_scope.get(
        MemoryScope.KEY,
    ) == memories


def test_memory_scope_get_memories_should_read_from_request_scope():
    """
    测试 get_memories 是否从底层 RequestScope 读取数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "memory_type": "profile",
            "content": "用户正在开发 Dog Agent Framework",
        }
    ]

    request_scope.set(
        MemoryScope.KEY,
        memories,
    )

    assert memory_scope.get_memories() == memories


def test_memory_scope_set_memories_should_override_old_memories():
    """
    测试重复 set_memories 时是否会覆盖旧记忆。

    override（覆盖）：
    新数据替换旧数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    old_memories = [
        {
            "content": "旧记忆",
        }
    ]

    new_memories = [
        {
            "content": "新记忆",
        }
    ]

    memory_scope.set_memories(
        old_memories,
    )

    memory_scope.set_memories(
        new_memories,
    )

    assert memory_scope.get_memories() == new_memories


def test_memory_scope_clear_should_remove_memories():
    """
    测试 clear 是否可以清空记忆数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "content": "用户喜欢金毛",
        }
    ]

    memory_scope.set_memories(
        memories,
    )

    assert memory_scope.get_memories() == memories

    memory_scope.clear()

    assert memory_scope.get_memories() == []
    assert request_scope.get(MemoryScope.KEY) is None


def test_memory_scope_clear_without_memories_should_not_raise_error():
    """
    测试没有 memories 时调用 clear 是否不会报错。

    当前 clear 内部调用 RequestScope.remove，
    RequestScope.remove 使用 pop(key, None)，所以 key 不存在时不会报错。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memory_scope.clear()

    assert memory_scope.get_memories() == []


def test_memory_scope_should_share_data_when_using_same_request_scope():
    """
    测试多个 MemoryScope 使用同一个 RequestScope 时是否共享记忆数据。

    shared scope（共享作用域）：
    多个 scope wrapper（作用域包装器）底层使用同一个 RequestScope，
    因此可以读写同一份请求级数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    first_memory_scope = MemoryScope(
        request_scope,
    )

    second_memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "content": "共享记忆",
        }
    ]

    first_memory_scope.set_memories(
        memories,
    )

    assert second_memory_scope.get_memories() == memories


def test_memory_scope_should_be_isolated_when_using_different_request_scopes():
    """
    测试不同 RequestScope 下的 MemoryScope 是否互不污染。

    isolated（隔离）：
    一个请求作用域中的记忆数据，不应该影响另一个请求作用域。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_request_scope = RequestScope()
    second_request_scope = RequestScope()

    first_memory_scope = MemoryScope(
        first_request_scope,
    )

    second_memory_scope = MemoryScope(
        second_request_scope,
    )

    first_memory_scope.set_memories(
        [
            {
                "content": "第一个请求的记忆",
            }
        ]
    )

    assert first_memory_scope.get_memories() == [
        {
            "content": "第一个请求的记忆",
        }
    ]

    assert second_memory_scope.get_memories() == []


@pytest.mark.asyncio
async def test_memory_scope_startup_should_not_change_memories():
    """
    测试 startup 是否不会改变已有 memories。

    当前 MemoryScope.startup 是 pass，
    所以 startup 前后 memories 应保持一致。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "content": "startup 前已有记忆",
        }
    ]

    memory_scope.set_memories(
        memories,
    )

    await memory_scope.startup()

    assert memory_scope.get_memories() == memories


@pytest.mark.asyncio
async def test_memory_scope_shutdown_should_clear_memories():
    """
    测试 shutdown 是否会清空 memories。

    当前 MemoryScope.shutdown 内部调用 self.clear()，
    所以 shutdown 后 get_memories 应返回空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    memory_scope = MemoryScope(
        request_scope,
    )

    memories = [
        {
            "content": "shutdown 前已有记忆",
        }
    ]

    memory_scope.set_memories(
        memories,
    )

    await memory_scope.shutdown()

    assert memory_scope.get_memories() == []
    assert request_scope.get(MemoryScope.KEY) is None