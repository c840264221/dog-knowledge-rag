"""
timeline event 单元测试。

TimelineEvent（时间线事件）：
用于描述系统运行过程中的一个关键事件，例如 node_start、tool_start、tool_success、error 等。

dataclass（数据类）：
Python 提供的简化类定义方式，适合用来保存结构化数据。

timestamp（时间戳）：
事件发生的时间，用于后续 trace（链路追踪）、timeline（时间线）和 debug report（调试报告）。

metadata（元数据）：
事件附加信息，通常是字典格式，例如 node 名称、tool 参数、耗时等。
"""

from datetime import datetime

from src.runtime.timeline.timeline_event import TimelineEvent


def test_timeline_event_can_be_created_directly():
    """
    测试 TimelineEvent 是否可以直接创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent(
        event_type="node_start",
        name="tool_parse_node",
        timestamp="2026-06-16T10:00:00",
        metadata={
            "node": "tool_parse",
        },
    )

    assert event.event_type == "node_start"
    assert event.name == "tool_parse_node"
    assert event.timestamp == "2026-06-16T10:00:00"
    assert event.metadata == {
        "node": "tool_parse",
    }


def test_timeline_event_create_should_return_timeline_event():
    """
    测试 TimelineEvent.create 是否返回 TimelineEvent 实例。

    create（创建方法）：
    用于统一创建 TimelineEvent，并自动生成 timestamp。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent.create(
        event_type="tool_start",
        name="date_tool",
        metadata={
            "tool": "date",
        },
    )

    assert isinstance(event, TimelineEvent)


def test_timeline_event_create_should_set_event_type_and_name():
    """
    测试 TimelineEvent.create 是否正确设置 event_type 和 name。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent.create(
        event_type="tool_success",
        name="weather_tool",
        metadata={
            "latency": 0.25,
        },
    )

    assert event.event_type == "tool_success"
    assert event.name == "weather_tool"


def test_timeline_event_create_should_keep_metadata():
    """
    测试 TimelineEvent.create 是否正确保存 metadata。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    metadata = {
        "tool": "weather",
        "latency": 0.42,
        "success": True,
    }

    event = TimelineEvent.create(
        event_type="tool_success",
        name="weather_tool",
        metadata=metadata,
    )

    assert event.metadata == metadata


def test_timeline_event_create_should_default_metadata_to_empty_dict():
    """
    测试 metadata 为 None 时是否默认转换成空字典。

    当前 TimelineEvent.create 中使用 metadata or {}，
    所以 metadata=None 时，最终 event.metadata 应该是 {}。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent.create(
        event_type="graph_start",
        name="general_qa_agent",
    )

    assert event.metadata == {}


def test_timeline_event_create_should_generate_timestamp():
    """
    测试 TimelineEvent.create 是否自动生成 timestamp。

    timestamp（时间戳）：
    当前实现使用 datetime.now().isoformat() 生成字符串格式时间。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent.create(
        event_type="node_start",
        name="answer_gen_node",
    )

    assert event.timestamp is not None
    assert isinstance(event.timestamp, str)
    assert len(event.timestamp) > 0


def test_timeline_event_timestamp_should_be_isoformat_string():
    """
    测试 timestamp 是否是 isoformat 可解析的时间字符串。

    isoformat（ISO 时间格式）：
    Python datetime.isoformat() 生成的标准时间字符串格式。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    event = TimelineEvent.create(
        event_type="node_end",
        name="answer_gen_node",
    )

    parsed_timestamp = datetime.fromisoformat(
        event.timestamp,
    )

    assert isinstance(
        parsed_timestamp,
        datetime,
    )


def test_timeline_event_create_should_use_current_datetime(
    monkeypatch,
):
    """
    测试 TimelineEvent.create 是否使用 datetime.now().isoformat() 生成 timestamp。

    monkeypatch（猴子补丁）：
    pytest 提供的临时替换工具，这里用于替换 timeline_event 模块中的 datetime。

    参数：
        monkeypatch：
            pytest 内置 fixture，用于临时替换对象。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    import src.runtime.timeline.timeline_event as timeline_event_module

    class FakeDateTime:
        """
        测试用假的 datetime 类。

        FakeDateTime（假时间类）：
        用于固定 datetime.now() 的返回结果，避免测试依赖真实当前时间。
        """

        @classmethod
        def now(cls):
            """
            返回固定时间。

            参数：
                无。

            返回值：
                datetime：固定的 datetime 对象。
            """

            return datetime(
                2026,
                6,
                16,
                12,
                00,
                0,
            )

    monkeypatch.setattr(
        timeline_event_module,
        "datetime",
        FakeDateTime,
    )

    event = TimelineEvent.create(
        event_type="fixed_time_event",
        name="fixed_time_node",
    )

    assert event.timestamp == "2026-06-16T12:00:00"