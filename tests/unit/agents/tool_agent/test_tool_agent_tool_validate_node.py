"""
ToolAgent tool_validate_node 测试。

功能：
    验证工具调用校验节点会根据工具目录过滤非法 tool_calls，
    并写入可观测的校验状态字段。
"""

from __future__ import annotations

from src.agents.tool_agent.nodes.tool_validate_node import (
    build_tool_agent_tool_validate_node,
)
from src.mcp.sqlite.tool_definitions import (
    SQLITE_SELECT_ROWS_TOOL_NAME,
)


class FakeStateScope:
    """
    测试用 StateScope。

    功能：
        记录当前节点名称。

    参数：
        无。

    返回值：
        FakeStateScope:
            测试用状态作用域。
    """

    def __init__(self) -> None:
        self.current_node: str | None = None

    def set_node(
        self,
        node_name: str,
    ) -> None:
        """
        设置当前节点名称。

        功能：
            模拟 RuntimeContext.state().set_node。

        参数：
            node_name:
                当前节点名称。

        返回值：
            None。
        """

        self.current_node = node_name


class FakeTimelineScope:
    """
    测试用 TimelineScope。

    功能：
        记录 timeline 事件。

    参数：
        无。

    返回值：
        FakeTimelineScope:
            测试用时间线作用域。
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def add_event(
        self,
        event_type: str,
        name: str,
        metadata: dict | None = None,
    ) -> None:
        """
        添加时间线事件。

        功能：
            模拟 RuntimeContext.timeline().add_event。

        参数：
            event_type:
                事件类型。

            name:
                事件名称。

            metadata:
                附加元数据。

        返回值：
            None。
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
    测试用 RuntimeContext。

    功能：
        提供 state 和 timeline 两个作用域。

    参数：
        无。

    返回值：
        FakeRuntimeContext:
            测试用运行时上下文。
    """

    def __init__(self) -> None:
        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self) -> FakeStateScope:
        """
        获取测试状态作用域。

        功能：
            返回 FakeStateScope。

        参数：
            无。

        返回值：
            FakeStateScope:
                测试用状态作用域。
        """

        return self.state_scope

    def timeline(self) -> FakeTimelineScope:
        """
        获取测试时间线作用域。

        功能：
            返回 FakeTimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope:
                测试用时间线作用域。
        """

        return self.timeline_scope


class FakeCheckpointManager:
    """
    测试用 CheckpointManager。

    功能：
        记录 save_checkpoint 调用次数。

    参数：
        无。

    返回值：
        FakeCheckpointManager:
            测试用检查点管理器。
    """

    def __init__(self) -> None:
        self.save_count = 0

    def save_checkpoint(self) -> None:
        """
        保存测试 checkpoint。

        功能：
            记录保存次数。

        参数：
            无。

        返回值：
            None。
        """

        self.save_count += 1


def build_test_tool_catalog() -> list[dict]:
    """
    构建测试工具目录。

    功能：
        创建 SQLite select_rows 工具目录条目。

    参数：
        无。

    返回值：
        list[dict]:
            测试工具目录。
    """

    return [
        {
            "name": SQLITE_SELECT_ROWS_TOOL_NAME,
            "description": "查看 SQLite 表前 N 行数据。",
            "require_confirm": False,
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                    },
                    "table_name": {
                        "type": "string",
                    },
                },
                "required": [
                    "database_name",
                    "table_name",
                ],
            },
            "source": "mcp",
        }
    ]


def test_tool_validate_node_should_keep_valid_tool_calls() -> None:
    """
    测试合法工具调用会被保留。

    功能：
        当 tool_calls 符合工具目录 input_schema 时，校验节点应返回 validation_ok=True。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_validate_node(
        runtime_context_getter=lambda: None,
    )

    update = node(
        {
            "tool_calls": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "args": {
                        "database_name": "memory",
                        "table_name": "dogs",
                    },
                }
            ],
            "tool_agent_tool_catalog": build_test_tool_catalog(),
        }
    )

    assert update["tool_call_validation_ok"] is True
    assert update["tool_call_validation_errors"] == []
    assert update["tool_calls"] == [
        {
            "name": SQLITE_SELECT_ROWS_TOOL_NAME,
            "args": {
                "database_name": "memory",
                "table_name": "dogs",
            },
        }
    ]


def test_tool_validate_node_should_filter_invalid_tool_calls() -> None:
    """
    测试非法工具调用会被过滤。

    功能：
        当 tool_calls 中包含未知工具时，校验节点应清空合法 tool_calls，
        并把错误写入 tool_call_validation_errors。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_validate_node(
        runtime_context_getter=lambda: None,
    )

    update = node(
        {
            "tool_calls": [
                {
                    "name": "unknown_tool",
                    "args": {},
                }
            ],
            "tool_agent_tool_catalog": build_test_tool_catalog(),
        }
    )

    assert update["tool_call_validation_ok"] is False
    assert update["tool_calls"] == []
    assert update["tool_call_validation_errors"][0]["code"] == "unknown_tool"
    assert update["tool_call_validation_invalid_calls"][0]["tool_call"][
        "name"
    ] == "unknown_tool"


def test_tool_validate_node_should_write_runtime_event_and_checkpoint() -> None:
    """
    测试校验节点写入运行时事件和 checkpoint。

    功能：
        确认节点执行时会记录当前 node，并在 checkpoint_manager 存在时保存 checkpoint。

    参数：
        无。

    返回值：
        None。
    """

    fake_ctx = FakeRuntimeContext()
    checkpoint_manager = FakeCheckpointManager()
    node = build_tool_agent_tool_validate_node(
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=lambda: fake_ctx,
    )

    update = node(
        {
            "tool_calls": [],
            "tool_agent_tool_catalog": build_test_tool_catalog(),
        }
    )

    assert update["tool_call_validation_ok"] is True
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_validate_node"
    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "tool_agent_tool_validate_node",
            "metadata": None,
        }
    ]
    assert checkpoint_manager.save_count == 1
