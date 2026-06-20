"""
request scope 单元测试。

RequestScope（请求作用域）：
用于保存一次请求执行过程中的临时数据。

request-level data（请求级数据）：
只在当前一次请求中有效的数据，例如检索结果、记忆上下文、统计信息等。

key-value store（键值存储）：
用 key 和 value 的形式保存数据，例如 {"user_id": "test_user"}。

set（设置）：
向 RequestScope 写入一个 key 对应的 value。

get（获取）：
从 RequestScope 中读取某个 key 对应的 value。

remove（移除）：
从 RequestScope 中删除某个 key。

clear（清空）：
清除 RequestScope 中保存的全部数据。
"""

from src.runtime.context.request_scope import RequestScope


def test_request_scope_can_be_created():
    """
    测试 RequestScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    assert request_scope is not None
    assert isinstance(
        request_scope._data,
        dict,
    )
    assert request_scope._data == {}


def test_request_scope_set_and_get_value():
    """
    测试 set 和 get 是否可以正确写入和读取数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    request_scope.set(
        "user_id",
        "test_user",
    )

    assert request_scope.get(
        "user_id",
    ) == "test_user"


def test_request_scope_get_missing_key_should_return_none_by_default():
    """
    测试读取不存在的 key 时是否默认返回 None。

    missing key（缺失键）：
    表示 RequestScope 中不存在该 key。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    assert request_scope.get(
        "missing_key",
    ) is None


def test_request_scope_get_missing_key_should_return_custom_default():
    """
    测试读取不存在的 key 时是否可以返回自定义默认值。

    default value（默认值）：
    当 key 不存在时返回的备用值。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    assert request_scope.get(
        "missing_key",
        default="fallback",
    ) == "fallback"


def test_request_scope_set_should_override_existing_value():
    """
    测试重复 set 同一个 key 时是否会覆盖旧值。

    override（覆盖）：
    新值替换旧值。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    request_scope.set(
        "phase",
        "retrieving",
    )

    request_scope.set(
        "phase",
        "answering",
    )

    assert request_scope.get(
        "phase",
    ) == "answering"


def test_request_scope_can_store_dict_value():
    """
    测试 RequestScope 是否可以保存 dict 类型数据。

    dict（字典）：
    Python 的键值结构，常用于保存 metadata、tool result、retrieval result 等结构化数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metadata = {
        "trace_id": "test_trace",
        "component": "general_qa_agent",
    }

    request_scope.set(
        "metadata",
        metadata,
    )

    assert request_scope.get(
        "metadata",
    ) == metadata


def test_request_scope_can_store_list_value():
    """
    测试 RequestScope 是否可以保存 list 类型数据。

    list（列表）：
    常用于保存 docs、events、messages 等多个对象。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    docs = [
        "doc_1",
        "doc_2",
    ]

    request_scope.set(
        "docs",
        docs,
    )

    assert request_scope.get(
        "docs",
    ) == docs


def test_request_scope_remove_should_delete_existing_key():
    """
    测试 remove 是否可以删除已存在的 key。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    request_scope.set(
        "memory_context",
        "用户喜欢金毛",
    )

    assert request_scope.get(
        "memory_context",
    ) == "用户喜欢金毛"

    request_scope.remove(
        "memory_context",
    )

    assert request_scope.get(
        "memory_context",
    ) is None


def test_request_scope_remove_missing_key_should_not_raise_error():
    """
    测试 remove 删除不存在的 key 时是否不会报错。

    当前实现使用 pop(key, None)，所以 key 不存在时不会抛出 KeyError。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    request_scope.remove(
        "missing_key",
    )

    assert request_scope.get(
        "missing_key",
    ) is None


def test_request_scope_clear_should_remove_all_data():
    """
    测试 clear 是否可以清空所有数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    request_scope.set(
        "user_id",
        "test_user",
    )

    request_scope.set(
        "phase",
        "retrieving",
    )

    request_scope.set(
        "docs",
        [
            "doc_1",
            "doc_2",
        ],
    )

    assert request_scope._data != {}

    request_scope.clear()

    assert request_scope._data == {}
    assert request_scope.get("user_id") is None
    assert request_scope.get("phase") is None
    assert request_scope.get("docs") is None


def test_request_scope_data_should_be_isolated_between_instances():
    """
    测试不同 RequestScope 实例之间的数据是否相互隔离。

    isolated（隔离）：
    一个实例中的数据变化，不应该影响另一个实例。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_scope = RequestScope()
    second_scope = RequestScope()

    first_scope.set(
        "user_id",
        "first_user",
    )

    assert first_scope.get(
        "user_id",
    ) == "first_user"

    assert second_scope.get(
        "user_id",
    ) is None

    assert first_scope._data is not second_scope._data