"""
metrics scope 单元测试。

MetricsScope（指标作用域）：
用于管理一次请求执行过程中的 runtime metrics（运行时指标）。

runtime metrics（运行时指标）：
表示系统运行过程中的统计数据，例如工具调用次数、LLM 调用次数、错误次数、工具耗时、LLM 耗时等。

observability（可观测性）：
通过 metrics（指标）、trace（链路追踪）、timeline（时间线）、logs（日志）观察系统内部运行状态的能力。

RequestScope（请求作用域）：
底层 key-value store（键值存储），用于保存一次请求内的临时数据。

KEY（键）：
MetricsScope 使用固定 key "runtime_metrics" 把指标数据存入 RequestScope。
"""

import pytest

from src.runtime.context.request_scope import RequestScope

from src.runtime.scopes.metrics_scope import MetricsScope


def test_metrics_scope_can_be_created():
    """
    测试 MetricsScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    assert metrics_scope is not None
    assert metrics_scope.scope is request_scope


def test_metrics_scope_key_should_be_runtime_metrics():
    """
    测试 MetricsScope.KEY 是否为 runtime_metrics。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert MetricsScope.KEY == "runtime_metrics"


def test_metrics_scope_get_metrics_should_return_empty_dict_by_default():
    """
    测试未初始化 metrics 时，get_metrics 是否默认返回空字典。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    assert metrics_scope.get_metrics() == {}


def test_metrics_scope_init_metrics_should_create_default_metrics():
    """
    测试 init_metrics 是否可以初始化默认指标。

    default metrics（默认指标）：
    MetricsScope 初始化时创建的一组基础统计字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    assert metrics_scope.get_metrics() == {
        "tool_count": 0,
        "llm_count": 0,
        "error_count": 0,
        "tool_latency": 0,
        "llm_latency": 0,
    }


def test_metrics_scope_init_metrics_should_store_data_in_request_scope():
    """
    测试 init_metrics 是否真的把指标保存到底层 RequestScope。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    assert request_scope.get(
        MetricsScope.KEY,
    ) == {
        "tool_count": 0,
        "llm_count": 0,
        "error_count": 0,
        "tool_latency": 0,
        "llm_latency": 0,
    }


def test_metrics_scope_update_should_set_metric_value():
    """
    测试 update 是否可以设置指定指标的值。

    update（更新）：
    直接把某个指标 key 设置为指定 value。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    metrics_scope.update(
        "tool_count",
        3,
    )

    assert metrics_scope.get_metrics()["tool_count"] == 3


def test_metrics_scope_update_should_add_new_metric_key():
    """
    测试 update 是否可以添加新的指标字段。

    custom metric（自定义指标）：
    默认指标之外额外添加的统计字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    metrics_scope.update(
        "retrieval_count",
        2,
    )

    assert metrics_scope.get_metrics()["retrieval_count"] == 2


def test_metrics_scope_update_without_init_should_work():
    """
    测试未 init_metrics 时，update 是否也可以正常工作。

    当前 get_metrics 默认返回 {}，
    所以 update 会基于空字典添加字段。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.update(
        "tool_count",
        1,
    )

    assert metrics_scope.get_metrics() == {
        "tool_count": 1,
    }


def test_metrics_scope_increment_should_increase_existing_metric():
    """
    测试 increment 是否可以递增已有指标。

    increment（递增）：
    在当前指标值基础上增加指定数值，默认增加 1。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    metrics_scope.increment(
        "tool_count",
    )

    assert metrics_scope.get_metrics()["tool_count"] == 1


def test_metrics_scope_increment_should_support_custom_amount():
    """
    测试 increment 是否支持自定义递增数量。

    amount（数量）：
    每次递增的数值，默认是 1。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    metrics_scope.increment(
        "tool_latency",
        amount=60,
    )

    metrics_scope.increment(
        "tool_latency",
        amount=40,
    )

    assert metrics_scope.get_metrics()["tool_latency"] == 100


def test_metrics_scope_increment_missing_metric_should_start_from_zero():
    """
    测试 increment 遇到不存在的指标时是否从 0 开始递增。

    当前实现使用 metrics.get(key, 0) + amount，
    所以不存在的 key 会先当作 0 处理。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.increment(
        "retrieval_count",
    )

    assert metrics_scope.get_metrics()["retrieval_count"] == 1


def test_metrics_scope_restore_should_replace_metrics():
    """
    测试 restore 是否可以恢复指标数据。

    restore（恢复）：
    从外部数据恢复当前 runtime metrics，常用于 checkpoint resume（检查点恢复）。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    restored_metrics = {
        "tool_count": 5,
        "llm_count": 2,
        "error_count": 1,
        "tool_latency": 300,
        "llm_latency": 900,
    }

    metrics_scope.restore(
        restored_metrics,
    )

    assert metrics_scope.get_metrics() == restored_metrics


def test_metrics_scope_restore_should_store_data_in_request_scope():
    """
    测试 restore 是否真的把数据保存到底层 RequestScope。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    restored_metrics = {
        "tool_count": 1,
    }

    metrics_scope.restore(
        restored_metrics,
    )

    assert request_scope.get(
        MetricsScope.KEY,
    ) == restored_metrics


def test_metrics_scope_should_share_data_when_using_same_request_scope():
    """
    测试多个 MetricsScope 使用同一个 RequestScope 时是否共享 metrics。

    shared scope（共享作用域）：
    多个 scope wrapper（作用域包装器）底层使用同一个 RequestScope，
    因此可以读写同一份请求级数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    first_metrics_scope = MetricsScope(
        request_scope,
    )

    second_metrics_scope = MetricsScope(
        request_scope,
    )

    first_metrics_scope.increment(
        "tool_count",
    )

    assert second_metrics_scope.get_metrics() == {
        "tool_count": 1,
    }


def test_metrics_scope_should_be_isolated_when_using_different_request_scopes():
    """
    测试不同 RequestScope 下的 MetricsScope 是否互不污染。

    isolated（隔离）：
    一个请求作用域中的 metrics，不应该影响另一个请求作用域。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_request_scope = RequestScope()
    second_request_scope = RequestScope()

    first_metrics_scope = MetricsScope(
        first_request_scope,
    )

    second_metrics_scope = MetricsScope(
        second_request_scope,
    )

    first_metrics_scope.increment(
        "tool_count",
    )

    assert first_metrics_scope.get_metrics() == {
        "tool_count": 1,
    }

    assert second_metrics_scope.get_metrics() == {}


@pytest.mark.asyncio
async def test_metrics_scope_startup_should_init_metrics_when_empty():
    """
    测试 startup 在 metrics 为空时是否会初始化默认指标。

    当前 MetricsScope.startup 逻辑：
    - 如果 get_metrics() 返回空字典，则调用 init_metrics()
    - 如果已有 metrics，则直接 return

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    await metrics_scope.startup()

    assert metrics_scope.get_metrics() == {
        "tool_count": 0,
        "llm_count": 0,
        "error_count": 0,
        "tool_latency": 0,
        "llm_latency": 0,
    }


@pytest.mark.asyncio
async def test_metrics_scope_startup_should_not_override_existing_metrics():
    """
    测试 startup 在已有 metrics 时是否不会覆盖原有数据。

    这个测试很重要：
    如果 checkpoint resume（检查点恢复）后已有 metrics，
    startup 不应该把它重新初始化为 0。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    existing_metrics = {
        "tool_count": 3,
        "llm_count": 1,
        "error_count": 0,
        "tool_latency": 250,
        "llm_latency": 800,
    }

    metrics_scope.restore(
        existing_metrics,
    )

    await metrics_scope.startup()

    assert metrics_scope.get_metrics() == existing_metrics


@pytest.mark.asyncio
async def test_metrics_scope_shutdown_should_remove_metrics():
    """
    测试 shutdown 是否会删除 metrics。

    当前 MetricsScope.shutdown 内部调用：
        self.scope.remove(self.KEY)

    所以 shutdown 后 get_metrics 应返回空字典。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    metrics_scope = MetricsScope(
        request_scope,
    )

    metrics_scope.init_metrics()

    await metrics_scope.shutdown()

    assert metrics_scope.get_metrics() == {}
    assert request_scope.get(MetricsScope.KEY) is None