"""
ToolAgent registry adapter 测试。

功能：
    测试 ToolAgent 是否可以通过适配器读取底层工具注册表信息。

测试重点：
    1. 工具元数据按名称稳定排序。
    2. 工具目录输出普通 dict。
    3. 可以根据工具名读取单个工具元数据。
    4. 可以判断工具是否需要用户确认。
    5. 非法 registry 不会打断适配逻辑。
"""

from __future__ import annotations

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    build_tool_agent_tool_catalog,
    build_tool_agent_tool_catalog_state_update,
    dump_tool_metadata_for_agent,
    get_registered_tool_metadata,
    list_registered_tool_metadata,
    tool_requires_confirmation,
)
from src.graph.tools.schemas.tool_metadata import ToolMetadata


class FakeTool:
    """
    测试用工具对象。

    功能：
        模拟真实工具对象，只提供 metadata 字段。

    参数：
        metadata:
            工具元数据。

    返回值：
        FakeTool:
            测试用工具实例。
    """

    def __init__(
        self,
        metadata: ToolMetadata,
    ) -> None:
        self.metadata = metadata


class FakeRegistry:
    """
    测试用工具注册表。

    功能：
        模拟底层 ToolRegistry，只提供 tools 字段和 get_tool 方法。

    参数：
        tools:
            工具名到工具对象的映射。

    返回值：
        FakeRegistry:
            测试用注册表实例。
    """

    def __init__(
        self,
        tools: dict[str, FakeTool],
    ) -> None:
        self.tools = tools

    def get_tool(
        self,
        name: str,
    ) -> FakeTool | None:
        """
        根据工具名获取测试工具。

        功能：
            模拟真实 ToolRegistry.get_tool。

        参数：
            name:
                工具名称。

        返回值：
            FakeTool | None:
                找到时返回工具对象，找不到时返回 None。
        """

        return self.tools.get(
            name
        )


def build_fake_registry() -> FakeRegistry:
    """
    构建测试用工具注册表。

    功能：
        创建包含 weather 和 date 的假注册表，避免测试依赖真实工具实现。

    参数：
        无。

    返回值：
        FakeRegistry:
            测试用工具注册表。
    """

    return FakeRegistry(
        tools={
            "weather": FakeTool(
                ToolMetadata(
                    name="weather",
                    description="查询天气",
                    timeout=8,
                    retries=2,
                    require_confirm=True,
                )
            ),
            "date": FakeTool(
                ToolMetadata(
                    name="date",
                    description="查询日期",
                    timeout=3,
                    retries=1,
                    require_confirm=False,
                )
            ),
        }
    )


def test_list_registered_tool_metadata_should_return_sorted_metadata() -> None:
    """
    测试工具元数据按名称排序。

    功能：
        确认 registry_adapter 输出稳定顺序，方便调试和测试。

    参数：
        无。

    返回值：
        None。
    """

    metadata_items = list_registered_tool_metadata(
        tool_registry=build_fake_registry(),
    )

    assert [
        item.name
        for item in metadata_items
    ] == [
        "date",
        "weather",
    ]


def test_build_tool_agent_tool_catalog_should_return_plain_dicts() -> None:
    """
    测试工具目录输出普通 dict。

    功能：
        确认 ToolAgent 工具目录不会把 Pydantic 对象直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    catalog = build_tool_agent_tool_catalog(
        tool_registry=build_fake_registry(),
    )

    assert catalog == [
        {
            "name": "date",
            "description": "查询日期",
            "timeout": 3,
            "retries": 1,
            "require_confirm": False,
            "input_schema": {},
        },
        {
            "name": "weather",
            "description": "查询天气",
            "timeout": 8,
            "retries": 2,
            "require_confirm": True,
            "input_schema": {},
        },
    ]


def test_build_tool_agent_tool_catalog_state_update_should_use_standard_key() -> None:
    """
    测试工具目录 state update 使用标准 key。

    功能：
        确认后续节点可以直接返回该结果给 LangGraph 合并 state。

    参数：
        无。

    返回值：
        None。
    """

    update = build_tool_agent_tool_catalog_state_update(
        tool_registry=build_fake_registry(),
    )

    assert TOOL_AGENT_TOOL_CATALOG_STATE_KEY in update
    assert update[TOOL_AGENT_TOOL_CATALOG_STATE_KEY][0]["name"] == "date"


def test_get_registered_tool_metadata_should_return_single_metadata() -> None:
    """
    测试根据工具名读取单个元数据。

    功能：
        确认 ToolAgent 可以查询某个具体工具的配置。

    参数：
        无。

    返回值：
        None。
    """

    metadata = get_registered_tool_metadata(
        tool_name="weather",
        tool_registry=build_fake_registry(),
    )

    assert metadata is not None
    assert metadata.name == "weather"
    assert metadata.timeout == 8


def test_get_registered_tool_metadata_should_return_none_when_missing() -> None:
    """
    测试工具不存在时返回 None。

    功能：
        确认适配器不会因为工具缺失而抛出异常。

    参数：
        无。

    返回值：
        None。
    """

    metadata = get_registered_tool_metadata(
        tool_name="unknown",
        tool_registry=build_fake_registry(),
    )

    assert metadata is None


def test_tool_requires_confirmation_should_read_metadata_flag() -> None:
    """
    测试读取工具确认标记。

    功能：
        确认 require_confirm 可以被 ToolAgent 读取，用于后续确认节点。

    参数：
        无。

    返回值：
        None。
    """

    registry = build_fake_registry()

    assert tool_requires_confirmation(
        tool_name="weather",
        tool_registry=registry,
    ) is True
    assert tool_requires_confirmation(
        tool_name="date",
        tool_registry=registry,
    ) is False


def test_dump_tool_metadata_for_agent_should_return_plain_dict() -> None:
    """
    测试单个 ToolMetadata 转普通 dict。

    功能：
        确认 dump 函数输出可以安全写入 state。

    参数：
        无。

    返回值：
        None。
    """

    dumped = dump_tool_metadata_for_agent(
        ToolMetadata(
            name="date",
            description="查询日期",
        )
    )

    assert dumped == {
        "name": "date",
        "description": "查询日期",
        "timeout": 5,
        "retries": 3,
        "require_confirm": False,
        "input_schema": {},
    }


def test_list_registered_tool_metadata_should_return_empty_when_registry_invalid() -> None:
    """
    测试非法 registry 返回空列表。

    功能：
        确认适配器面对不符合预期的 registry 时保持保守，不打断主链路。

    参数：
        无。

    返回值：
        None。
    """

    class InvalidRegistry:
        """
        测试用非法注册表。

        功能：
            模拟 tools 字段不是 mapping 的情况。

        参数：
            无。

        返回值：
            InvalidRegistry:
                测试用非法注册表实例。
        """

        tools = "bad_tools"

    assert list_registered_tool_metadata(
        tool_registry=InvalidRegistry(),
    ) == []
