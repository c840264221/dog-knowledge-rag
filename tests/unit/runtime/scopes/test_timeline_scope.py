"""
timeline scope 单元测试。

TimelineScope（时间线作用域）：
用于记录一次请求执行过程中的 timeline event（时间线事件）。

timeline event（时间线事件）：
表示系统运行过程中发生的一次关键动作，例如 node_start、tool_start、tool_success、error 等。

restore（恢复）：
把之前保存过的事件列表重新写回 TimelineScope。

clear（清空）：
清除当前 TimelineScope 中保存的所有事件。

monkeypatch（猴子补丁）：
pytest 提供的测试工具，可以在测试期间临时替换函数、属性或环境变量，测试结束后自动恢复。
"""

import pytest

from src.runtime.scopes.timeline_scope import TimelineScope

from src.runtime.timeline.timeline_event import TimelineEvent


def test_timeline_scope_can_be_created():
    """
    测试 TimelineScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    assert timeline_scope is not None
    assert isinstance(timeline_scope.events, list)
    assert len(timeline_scope.events) == 0


def test_timeline_scope_initial_events_should_be_empty():
    """
    测试 TimelineScope 初始事件列表是否为空。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    events = timeline_scope.get_events()

    assert events == []


def test_timeline_scope_add_event_should_append_created_event(
    monkeypatch,
):
    """
    测试 add_event 是否会创建事件并追加到 events 列表。

    这里使用 monkeypatch 临时替换 TimelineEvent.create，
    这样测试不依赖 TimelineEvent 内部字段结构，只关注 TimelineScope 的行为。

    参数：
        monkeypatch：
            pytest 内置 fixture，用于临时替换 TimelineEvent.create。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    created_calls = []
    fake_event = {
        "event_type": "node_start",
        "name": "tool_parse_node",
        "metadata": {
            "node": "tool_parse",
        },
    }

    def fake_create(
        event_type,
        name,
        metadata=None,
    ):
        """
        测试用假的 TimelineEvent.create 方法。

        参数：
            event_type：事件类型，字符串格式。
            name：事件名称，字符串格式。
            metadata：事件元数据，字典格式或 None。

        返回值：
            dict：测试用假的事件对象。
        """

        created_calls.append(
            {
                "event_type": event_type,
                "name": name,
                "metadata": metadata,
            }
        )

        return fake_event

    monkeypatch.setattr(
        TimelineEvent,
        "create",
        staticmethod(fake_create),
    )

    timeline_scope.add_event(
        event_type="node_start",
        name="tool_parse_node",
        metadata={
            "node": "tool_parse",
        },
    )

    events = timeline_scope.get_events()

    assert created_calls == [
        {
            "event_type": "node_start",
            "name": "tool_parse_node",
            "metadata": {
                "node": "tool_parse",
            },
        }
    ]

    assert events == [fake_event]


def test_timeline_scope_add_event_without_metadata_should_work(
    monkeypatch,
):
    """
    测试 add_event 在 metadata 为 None 时是否可以正常工作。

    参数：
        monkeypatch：
            pytest 内置 fixture，用于临时替换 TimelineEvent.create。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    fake_event = {
        "event_type": "graph_start",
        "name": "general_qa_agent",
        "metadata": None,
    }

    def fake_create(
        event_type,
        name,
        metadata=None,
    ):
        """
        测试用假的 TimelineEvent.create 方法。

        参数：
            event_type：事件类型。
            name：事件名称。
            metadata：事件元数据。

        返回值：
            dict：测试用假的事件对象。
        """

        return fake_event

    monkeypatch.setattr(
        TimelineEvent,
        "create",
        staticmethod(fake_create),
    )

    timeline_scope.add_event(
        event_type="graph_start",
        name="general_qa_agent",
    )

    assert timeline_scope.get_events() == [
        fake_event,
    ]


def test_timeline_scope_get_events_should_return_current_events(
    monkeypatch,
):
    """
    测试 get_events 是否可以返回当前保存的事件列表。

    参数：
        monkeypatch：
            pytest 内置 fixture，用于临时替换 TimelineEvent.create。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    first_event = {
        "event_type": "node_start",
        "name": "node_a",
    }

    second_event = {
        "event_type": "node_end",
        "name": "node_a",
    }

    fake_events = [
        first_event,
        second_event,
    ]

    def fake_create(
        event_type,
        name,
        metadata=None,
    ):
        """
        根据事件名称返回不同的测试事件。

        参数：
            event_type：事件类型。
            name：事件名称。
            metadata：事件元数据。

        返回值：
            dict：测试用事件对象。
        """

        if event_type == "node_start":
            return first_event

        return second_event

    monkeypatch.setattr(
        TimelineEvent,
        "create",
        staticmethod(fake_create),
    )

    timeline_scope.add_event(
        "node_start",
        "node_a",
    )

    timeline_scope.add_event(
        "node_end",
        "node_a",
    )

    assert timeline_scope.get_events() == fake_events


def test_timeline_scope_clear_should_remove_all_events(
    monkeypatch,
):
    """
    测试 clear 是否可以清空所有事件。

    参数：
        monkeypatch：
            pytest 内置 fixture，用于临时替换 TimelineEvent.create。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    def fake_create(
        event_type,
        name,
        metadata=None,
    ):
        """
        测试用假的 TimelineEvent.create 方法。

        参数：
            event_type：事件类型。
            name：事件名称。
            metadata：事件元数据。

        返回值：
            dict：测试用事件对象。
        """

        return {
            "event_type": event_type,
            "name": name,
            "metadata": metadata,
        }

    monkeypatch.setattr(
        TimelineEvent,
        "create",
        staticmethod(fake_create),
    )

    timeline_scope.add_event(
        "node_start",
        "node_a",
    )

    timeline_scope.add_event(
        "node_end",
        "node_a",
    )

    assert len(timeline_scope.get_events()) == 2

    timeline_scope.clear()

    assert timeline_scope.get_events() == []


def test_timeline_scope_restore_should_replace_events():
    """
    测试 restore 是否可以恢复事件列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    restored_events = [
        {
            "event_type": "node_start",
            "name": "node_a",
        },
        {
            "event_type": "node_end",
            "name": "node_a",
        },
    ]

    timeline_scope.restore(
        restored_events,
    )

    assert timeline_scope.get_events() == restored_events


@pytest.mark.asyncio
async def test_timeline_scope_startup_should_keep_existing_events():
    """
    测试 startup 是否不会清空已有事件。

    当前 TimelineScope.startup 是 pass，
    所以 startup 后 events 应该保持不变。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    existing_events = [
        {
            "event_type": "before_startup",
            "name": "existing_event",
        }
    ]

    timeline_scope.restore(
        existing_events,
    )

    await timeline_scope.startup()

    assert timeline_scope.get_events() == existing_events


@pytest.mark.asyncio
async def test_timeline_scope_shutdown_should_clear_events():
    """
    测试 shutdown 是否会清空事件列表。

    当前 TimelineScope.shutdown 内部调用 self.clear()，
    所以 shutdown 后 events 应该为空。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    timeline_scope = TimelineScope()

    existing_events = [
        {
            "event_type": "before_shutdown",
            "name": "existing_event",
        }
    ]

    timeline_scope.restore(
        existing_events,
    )

    await timeline_scope.shutdown()

    assert timeline_scope.get_events() == []