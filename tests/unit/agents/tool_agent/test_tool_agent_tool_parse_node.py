"""
ToolAgent tool_parse_node 测试。

功能：
    测试新版 ToolAgent 工具解析节点是否能通过注入 parser 生成工具调用 state。

测试重点：
    1. parser 返回 dict 时可以生成 tool_calls。
    2. parser 返回 ToolParseResult 时可以正常处理。
    3. 已有 tool_calls 时跳过重复解析。
    4. parser 抛异常时返回安全 fallback。
    5. 输出必须是普通 dict，不能把 Pydantic 对象写入 state。
"""

from __future__ import annotations

import pytest

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
)
from src.agents.tool_agent.nodes.tool_parse_node import (
    build_tool_catalog_prompt_text,
    build_tool_agent_tool_parse_node,
    call_tool_parser,
    parse_tool_parse_llm_output,
    normalize_tool_parse_result,
    recover_incomplete_sqlite_list_tables_call,
    render_input_schema_for_prompt,
    render_tool_catalog_item_for_prompt,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall, ToolParseResult
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
        记录节点事件。

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
        记录保存 checkpoint 的次数。

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
        模拟保存 checkpoint。

        功能：
            记录保存次数。

        参数：
            无。

        返回值：
            None。
        """

        self.save_count += 1


class FakeAinvokeParser:
    """
    测试用异步解析器。

    功能：
        模拟带 ainvoke 方法的 parser。

    参数：
        result:
            parser 返回值。

        error:
            parser 需要抛出的异常。

    返回值：
        FakeAinvokeParser:
            测试用解析器。
    """

    def __init__(
        self,
        result=None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.inputs: list[dict] = []

    async def ainvoke(
        self,
        parser_input: dict,
    ):
        """
        模拟异步 parser 调用。

        功能：
            记录输入，按配置返回结果或抛异常。

        参数：
            parser_input:
                parser 输入。

        返回值：
            object:
                配置的解析结果。
        """

        self.inputs.append(
            parser_input
        )

        if self.error is not None:
            raise self.error

        return self.result


class FakeLLMProvider:
    """
    测试用 LLM Provider。

    功能：
        模拟项目中的 llm_provider，提供 backup_llm 和 safe_ainvoke。

    参数：
        response_text:
            safe_ainvoke 返回的 LLM 文本。

        error:
            safe_ainvoke 需要抛出的异常。

    返回值：
        FakeLLMProvider:
            测试用 LLM Provider。
    """

    def __init__(
        self,
        response_text: str,
        error: Exception | None = None,
    ) -> None:
        self.backup_llm = object()
        self.response_text = response_text
        self.error = error
        self.calls: list[dict] = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response: str,
    ) -> str:
        """
        模拟安全调用 LLM。

        功能：
            记录调用参数，并返回预设文本。

        参数：
            llm:
                被调用的大语言模型。

            prompt:
                渲染后的 prompt。

            fallback_response:
                LLM 调用失败时的兜底文本。

        返回值：
            str:
                模拟 LLM 返回文本。
        """

        self.calls.append(
            {
                "llm": llm,
                "prompt": prompt,
                "fallback_response": fallback_response,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response_text


def build_test_node(
    parser,
    with_checkpoint: bool = True,
    with_runtime_context: bool = True,
):
    """
    构建测试用 ToolAgent 工具解析节点。

    功能：
        注入 fake parser、fake checkpoint_manager 和 fake runtime_context。

    参数：
        parser:
            测试用 parser。

        with_checkpoint:
            是否提供 checkpoint_manager。

        with_runtime_context:
            是否提供 runtime context。

    返回值：
        tuple:
            node, fake_ctx, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()
    checkpoint_manager = (
        FakeCheckpointManager()
        if with_checkpoint
        else None
    )

    def runtime_context_getter():
        """
        获取测试用 RuntimeContext。

        功能：
            根据 with_runtime_context 决定是否返回 fake_ctx。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None:
                测试用运行时上下文或 None。
        """

        if not with_runtime_context:
            return None

        return fake_ctx

    node = build_tool_agent_tool_parse_node(
        parser=parser,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        checkpoint_manager,
    )


def test_build_tool_catalog_prompt_text_should_use_default_when_catalog_missing() -> None:
    """
    测试缺少工具目录时使用默认工具说明。

    功能：
        当 state 中没有 tool_agent_tool_catalog 时，
        prompt 文本应回退到 date/weather 默认说明。

    参数：
        无。

    返回值：
        None。
    """

    prompt_text = build_tool_catalog_prompt_text(
        state={},
    )

    assert "date" in prompt_text
    assert "weather" in prompt_text


def test_build_tool_catalog_prompt_text_should_render_state_catalog() -> None:
    """
    测试根据 state 工具目录渲染 prompt 文本。

    功能：
        当 state 中存在 tool_agent_tool_catalog 时，
        prompt 文本应包含目录中的工具名称、描述和确认标记。

    参数：
        无。

    返回值：
        None。
    """

    prompt_text = build_tool_catalog_prompt_text(
        state={
            "tool_agent_tool_catalog": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "description": "查看 SQLite 表前 N 行数据。",
                    "require_confirm": False,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "database_name": {
                                "type": "string",
                                "description": "数据库白名单别名。",
                            },
                            "table_name": {
                                "type": "string",
                                "description": "要读取数据的表名。",
                            },
                        },
                        "required": [
                            "database_name",
                            "table_name",
                        ],
                    },
                    "source": "mcp",
                    "timeout": 5,
                    "retries": 0,
                }
            ],
        },
    )

    assert SQLITE_SELECT_ROWS_TOOL_NAME in prompt_text
    assert "查看 SQLite 表前 N 行数据。" in prompt_text
    assert "工具来源：" in prompt_text
    assert "- mcp" in prompt_text
    assert "- database_name: string，必填，数据库白名单别名。" in prompt_text
    assert "- table_name: string，必填，要读取数据的表名。" in prompt_text
    assert "是否需要确认：False" in prompt_text
    assert "timeout" not in prompt_text
    assert "retries" not in prompt_text


def test_render_tool_catalog_item_for_prompt_should_return_plain_text() -> None:
    """
    测试单个工具条目渲染为普通文本。

    功能：
        确认工具条目不会输出复杂对象，只输出 LLM 可读文本。

    参数：
        无。

    返回值：
        None。
    """

    prompt_item = render_tool_catalog_item_for_prompt(
        index=1,
        name="weather",
        description="查询天气",
        require_confirm=True,
    )

    assert prompt_item == (
        "1. weather\n"
        "功能：\n"
        "- 查询天气\n"
        "是否需要确认：True\n"
        "-----------------------------------"
    )


def test_render_input_schema_for_prompt_should_render_required_and_optional_fields() -> None:
    """
    测试 input_schema 渲染必填和可选字段。

    功能：
        确认 JSON Schema 风格参数结构会转换成 LLM 易读文本。

    参数：
        无。

    返回值：
        None。
    """

    rendered_schema = render_input_schema_for_prompt(
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少行。",
                },
            },
            "required": [
                "database_name",
            ],
        }
    )

    assert "- database_name: string，必填，数据库白名单别名。" in rendered_schema
    assert "- limit: integer，可选，最多返回多少行。" in rendered_schema


def test_render_input_schema_for_prompt_should_render_allowed_values() -> None:
    """
    测试 input_schema 渲染字段允许值。

    功能：
        当 database_name schema 中包含 enum 时，
        prompt 文本应明确告诉 LLM 合法数据库别名。

    参数：
        无。

    返回值：
        None。
    """

    rendered_schema = render_input_schema_for_prompt(
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                    "enum": [
                        "memory",
                        "rag",
                    ],
                },
            },
            "required": [
                "database_name",
            ],
        }
    )

    assert "合法值只能是：memory、rag。" in rendered_schema


def test_render_input_schema_should_show_explicit_user_input_contract() -> None:
    """
    测试参数结构会向 LLM 展示显式用户输入契约。

    功能：
        database_name 标记扩展契约后，Prompt 应明确禁止模型猜测参数值。

    参数：
        无。

    返回值：
        None。
    """

    rendered_schema = render_input_schema_for_prompt(
        input_schema={
            "type": "object",
            "properties": {
                "database_name": {
                    "type": "string",
                    "description": "数据库白名单别名。",
                    "x-requires-explicit-user-input": True,
                },
            },
            "required": ["database_name"],
        }
    )

    assert "必须由当前用户问题明确提供，禁止猜测" in rendered_schema


def test_recover_incomplete_sqlite_list_tables_call_should_build_empty_args() -> None:
    """
    测试数据库表清单问题的空调用可以恢复为待澄清调用。

    功能：
        模拟 LLM 因缺少 database_name 而返回 no_tool，确认恢复函数只补工具名，
        不猜测数据库参数，后续可由统一校验节点生成澄清请求。

    参数：
        无。

    返回值：
        None。
    """

    result = recover_incomplete_sqlite_list_tables_call(
        parse_result=ToolParseResult(
            need_tool=False,
            tool_calls=[],
            response="",
        ),
        question="帮我查一下数据库中都有什么表",
        state={
            "route_decision": {
                "requires_tool": True,
            },
            "tool_agent_tool_catalog": [
                {
                    "name": "sqlite_list_tables",
                }
            ],
        },
    )

    assert result.need_tool is True
    assert result.tool_calls[0].name == "sqlite_list_tables"
    assert result.tool_calls[0].args == {}


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_support_llm_provider() -> None:
    """
    测试节点支持注入 llm_provider。

    功能：
        不传 parser，只传 fake llm_provider，模拟旧 tool_parse_node 的 LLM JSON 输出。

    参数：
        无。

    返回值：
        None。
    """

    llm_provider = FakeLLMProvider(
        response_text="""
        {
          "need_tool": true,
          "tool_calls": [
            {
              "name": "weather",
              "args": {
                "city": "成都"
              }
            }
          ],
          "response": ""
        }
        """
    )
    fake_ctx = FakeRuntimeContext()
    checkpoint_manager = FakeCheckpointManager()

    def runtime_context_getter():
        """
        获取测试 RuntimeContext。

        功能：
            返回 fake_ctx。

        参数：
            无。

        返回值：
            FakeRuntimeContext:
                测试用运行时上下文。
        """

        return fake_ctx

    node = build_tool_agent_tool_parse_node(
        llm_provider=llm_provider,
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    update = await node(
        {
            "question": "今天成都天气怎么样？",
        }
    )

    assert update["need_tool"] is True
    assert update["tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "pending_confirmation"
    assert len(
        llm_provider.calls
    ) == 1
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_inject_tool_catalog_into_llm_prompt() -> None:
    """
    测试 LLM prompt 会注入 state 中的工具目录。

    功能：
        当 state 中包含 tool_agent_tool_catalog 时，
        LLM 工具解析 prompt 应包含 SQLite MCP 工具名称。

    参数：
        无。

    返回值：
        None。
    """

    llm_provider = FakeLLMProvider(
        response_text="""
        {
          "need_tool": true,
          "tool_calls": [
            {
              "name": "sqlite_select_rows",
              "args": {
                "database_name": "memory",
                "table_name": "dogs"
              }
            }
          ],
          "response": ""
        }
        """
    )
    node = build_tool_agent_tool_parse_node(
        llm_provider=llm_provider,
        runtime_context_getter=lambda: None,
    )

    update = await node(
        {
            "question": "查看 memory 数据库 dogs 表",
            "tool_agent_tool_catalog": [
                {
                    "name": SQLITE_SELECT_ROWS_TOOL_NAME,
                    "description": "查看 SQLite 表前 N 行数据。",
                    "require_confirm": False,
                }
            ],
        }
    )

    rendered_prompt = str(
        llm_provider.calls[0]["prompt"]
    )

    assert update["tool_calls"][0]["name"] == SQLITE_SELECT_ROWS_TOOL_NAME
    assert SQLITE_SELECT_ROWS_TOOL_NAME in rendered_prompt
    assert "查看 SQLite 表前 N 行数据。" in rendered_prompt


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_parse_dict_result() -> None:
    """
    测试 parser 返回 dict 时生成工具调用。

    功能：
        确认节点输出 need_tool、tool_calls、tool_results、tool_round 和 tool_agent_response。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
        }
    )
    (
        node,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天成都天气怎么样？",
            "tool_round": 2,
        }
    )

    assert update["need_tool"] is True
    assert update["tool_calls"] == [
        {
            "name": "weather",
            "args": {
                "city": "成都",
            },
        }
    ]
    assert update["tool_results"] == []
    assert update["tool_round"] == 3
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "pending_confirmation"
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_parse_node"
    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "tool_agent_tool_parse_node",
            "metadata": None,
        }
    ]
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_parse_tool_parse_result() -> None:
    """
    测试 parser 返回 ToolParseResult。

    功能：
        确认节点能处理底层工具 schema 中的 ToolParseResult。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result=ToolParseResult(
            need_tool=False,
            tool_calls=[],
            response="不需要工具。",
        )
    )
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "你好",
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update["tool_round"] == 1
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_skip_when_tool_calls_exist() -> None:
    """
    测试已有 tool_calls 时跳过解析。

    功能：
        避免重复解析覆盖上游已经生成的工具调用。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": False,
            "tool_calls": [],
        }
    )
    (
        node,
        fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天几号？",
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }
    )

    assert update == {}
    assert parser.inputs == []
    assert fake_ctx.state_scope.current_node == "tool_agent_tool_parse_node"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_fallback_when_parser_failed() -> None:
    """
    测试 parser 抛异常时返回安全 fallback。

    功能：
        确认解析失败不会打断 ToolAgent 链路。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        error=RuntimeError(
            "parser failed"
        )
    )
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node(
        {
            "question": "今天几号？",
        }
    )

    assert update["need_tool"] is False
    assert update["tool_calls"] == []
    assert update[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_tool_agent_tool_parse_node_should_fallback_when_question_missing() -> None:
    """
    测试缺少 question 时返回安全 fallback。

    功能：
        确认没有用户问题时不会调用 parser。

    参数：
        无。

    返回值：
        None。
    """

    parser = FakeAinvokeParser(
        result={
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }
    )
    (
        node,
        _fake_ctx,
        checkpoint_manager,
    ) = build_test_node(
        parser=parser,
    )

    update = await node({})

    assert update["need_tool"] is False
    assert parser.inputs == []
    assert checkpoint_manager is not None
    assert checkpoint_manager.save_count == 0


@pytest.mark.asyncio
async def test_call_tool_parser_should_support_callable_parser() -> None:
    """
    测试普通 callable parser。

    功能：
        确认 call_tool_parser 可以调用普通函数解析器。

    参数：
        无。

    返回值：
        None。
    """

    def parser(
        parser_input: dict,
    ) -> dict:
        """
        测试用普通解析函数。

        功能：
            返回 date 工具调用。

        参数：
            parser_input:
                工具解析输入。

        返回值：
            dict:
                工具解析结果。
        """

        assert parser_input["question"] == "今天几号？"
        return {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "date",
                    "args": {},
                }
            ],
        }

    result = await call_tool_parser(
        parser=parser,
        question="今天几号？",
        state={},
    )

    assert result["tool_calls"][0]["name"] == "date"


def test_normalize_tool_parse_result_should_skip_invalid_tool_calls() -> None:
    """
    测试归一化时跳过非法 tool_calls。

    功能：
        确认坏数据不会直接写入 state。

    参数：
        无。

    返回值：
        None。
    """

    result = normalize_tool_parse_result(
        {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                },
                {
                    "args": {},
                },
                "bad_call",
                ToolCall(
                    name="date",
                    args={},
                ),
            ],
        }
    )

    assert [
        tool_call.name
        for tool_call in result.tool_calls
    ] == [
        "weather",
        "date",
    ]


def test_parse_tool_parse_llm_output_should_extract_json_after_explanation() -> None:
    """
    测试 LLM 在 JSON 前输出解释文字时仍能解析工具调用。

    功能：
        复现真实冒烟日志中的“中文解释 + JSON”格式，
        确认解析器可以提取并校验后面的 ToolParseResult。

    参数：
        无。

    返回值：
        None。
    """

    result = parse_tool_parse_llm_output(
        raw_output=(
            "字段不是 required，但为了完整可以包含。\n"
            "下面给出结构化结果：\n"
            '{"need_tool": true, "tool_calls": ['
            '{"name": "sqlite_list_tables", '
            '"args": {"database_name": "memory"}}], '
            '"response": ""}'
        )
    )

    assert result.need_tool is True
    assert result.tool_calls[0].name == "sqlite_list_tables"
    assert result.tool_calls[0].args == {
        "database_name": "memory",
    }
