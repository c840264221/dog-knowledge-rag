"""
ToolAgent 工具解析节点。

功能：
    为新版 ToolAgent（工具智能体）提供独立的工具解析节点。

设计原则：
    1. 当前节点通过 parser 或 llm_provider 依赖注入完成解析。
    2. 当前节点不 import 旧位置 tool_parse_node，避免新 ToolAgent 回流旧图节点。
    3. 节点输出普通 dict，方便 LangGraph 合并 state，也避免 checkpoint 保存自定义对象。
    4. 节点文件放在 nodes 目录下，并以 _node.py 结尾。

专业名词：
    Node：图节点，LangGraph 中接收 state 并返回 state update 的执行单元。
    Parser：解析器，把用户问题解析成是否需要工具以及工具调用列表。
    Dependency Injection：依赖注入，把 parser 从外部传入，方便测试和后续替换实现。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall, ToolParseResult
from src.logger import logger
from src.runtime.context import runtime_ctx


ToolParseNode = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

tool_parse_output_parser = PydanticOutputParser(
    pydantic_object=ToolParseResult,
)


DEFAULT_TOOL_CATALOG_PROMPT_TEXT = """
1. date
功能：
- 获取今天日期
是否需要确认：False
-----------------------------------

2. weather
功能：
- 查询天气
是否需要确认：True
-----------------------------------
""".strip()


def build_tool_agent_tool_parse_node(
    parser: Any | None = None,
    llm_provider: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolParseNode:
    """
    构建 ToolAgent 工具解析节点。

    功能：
        创建一个可给 LangGraph 使用的 async node。
        节点从 state 中读取 question，调用注入的 parser 或 llm_provider，
        然后返回 need_tool、tool_calls、tool_results、tool_round 和 tool_agent_response。

    参数：
        parser:
            工具解析器。可以是 callable、带 ainvoke 方法的对象，或带 invoke 方法的对象。
            测试时可传入 fake parser。

        llm_provider:
            LLM Provider（大语言模型服务提供者）。
            当 parser=None 时，用它构建 LLM 工具解析器。

        checkpoint_manager:
            检查点管理器。解析成功时调用 save_checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。
            用于写入当前 node 和 timeline 事件。

    返回值：
        ToolParseNode:
            async 节点函数，接收 state，返回可合并进 state 的 dict。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    resolved_parser = parser or build_llm_tool_parser(
        llm_provider=llm_provider,
    )

    async def tool_agent_tool_parse_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        解析当前用户问题是否需要工具。

        功能：
            1. 写入当前运行时 node 信息。
            2. 如果 state 已有 tool_calls，则返回空 dict，避免重复解析。
            3. 读取 question 并调用 parser。
            4. 将 parser 输出归一化为 ToolParseResult。
            5. 输出普通 dict 格式的 state update。
            6. 解析异常时返回安全 fallback。

        参数：
            state:
                当前 LangGraph state，主要读取 question、tool_calls、tool_round。

        返回值：
            dict[str, Any]:
                需要合并进 LangGraph state 的字段。
        """

        write_tool_parse_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        log_tool_agent_state(
            node_name="tool_parse",
            event="tool_parse_start",
            state=state,
        )

        if state.get(
            "tool_calls"
        ):
            log_tool_agent_state(
                node_name="tool_parse",
                event="tool_parse_skip_existing_tool_calls",
                state=state,
                extra={
                    "reason": "state 已经存在 tool_calls，跳过重复解析。",
                },
            )
            return {}

        question = str(
            state.get(
                "question",
                "",
            )
            or ""
        ).strip()

        if not question:
            logger.warning(
                "tool_agent_tool_parse_node 缺少 question，已兜底为 need_tool=False"
            )

            update = build_tool_parse_fallback_update(
                state=state,
            )
            log_tool_agent_state(
                node_name="tool_parse",
                event="tool_parse_fallback_missing_question",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "reason": "缺少 question。",
                },
            )
            return update

        try:
            raw_result = await call_tool_parser(
                parser=resolved_parser,
                question=question,
                state=state,
            )
            parse_result = normalize_tool_parse_result(
                raw_result=raw_result,
            )
            original_tool_call_count = len(
                parse_result.tool_calls
            )

            # 缺少显式参数时仍需保留不完整调用，交给统一校验节点生成澄清请求。
            parse_result = recover_incomplete_sqlite_list_tables_call(
                parse_result=parse_result,
                question=question,
                state=state,
            )
            incomplete_call_recovered = (
                original_tool_call_count == 0
                and bool(parse_result.tool_calls)
            )
        except Exception as exc:
            logger.exception(
                f"ToolAgent 工具解析失败: {exc}"
            )

            update = build_tool_parse_fallback_update(
                state=state,
            )
            log_tool_agent_state(
                node_name="tool_parse",
                event="tool_parse_fallback_parser_error",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "error": str(
                        exc
                    ),
                },
            )
            return update

        update = dump_tool_parse_result_for_state(
            parse_result=parse_result,
            state=state,
        )

        log_tool_agent_state(
            node_name="tool_parse",
            event="tool_parse_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "parsed_need_tool": parse_result.need_tool,
                "parsed_tool_call_count": len(
                    parse_result.tool_calls
                ),
                "incomplete_call_recovered": incomplete_call_recovered,
            },
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return update

    return tool_agent_tool_parse_node


def build_llm_tool_parser(
    llm_provider: Any | None,
) -> Any:
    """
    构建 LLM 工具解析器。

    功能：
        参考旧 tool_parse_node 的实现方式，使用 backup_llm、safe_ainvoke、
        ChatPromptTemplate 和 PydanticOutputParser 解析用户问题。

    参数：
        llm_provider:
            LLM Provider（大语言模型服务提供者）。
            必须提供 backup_llm 字段和 safe_ainvoke 方法。

    返回值：
        Any:
            可被 call_tool_parser 调用的 LangChain Runnable 解析链。
    """

    if llm_provider is None:
        raise ValueError(
            "build_tool_agent_tool_parse_node 需要 parser 或 llm_provider。"
        )

    backup_llm = llm_provider.backup_llm

    tool_parse_prompt = """
    你是一个工具调用分析助手。

    你可以使用以下工具：

    {tool_catalog_text}

    规则：

    1. 如果问题需要工具：
    返回 need_tool=true

    2. 如果不需要工具：
    返回 need_tool=false

    3. 必须严格输出 JSON

    4. 不允许输出 Markdown

    5. 不允许输出解释

    6. args 必须是 JSON 对象

    7. tool_calls 必须是数组

    8. 如果用户问题匹配某个工具能力，则 tool_calls 中必须包含该工具名称

    9. 如果用户问题不需要任何工具，则 need_tool=false 且 tool_calls=[]

    10. 参数结构中标记 x-requires-explicit-user-input=true 的字段，
    只有当前用户问题明确给出参数值时才能写入 args；否则必须省略，禁止从历史、候选值或默认值猜测

    11. 缺少参数不等于不需要工具。如果问题明确匹配某个工具，但缺少必填参数，
    仍然必须返回 need_tool=true，并把匹配的工具放入 tool_calls；缺少的参数不要写入 args

    示例：用户要求查询数据库中的表，但没有说明数据库别名时，应输出：
    {{"need_tool": true, "tool_calls": [{{"name": "sqlite_list_tables", "args": {{}}}}], "response": ""}}
    -----------------------------------

    用户问题：
    {question}
    """

    prompt = ChatPromptTemplate.from_template(
        tool_parse_prompt
        + "\n\n"
        + "{format_instructions}"
    )

    async def safe_llm_ainvoke(
        prompt_value: Any,
    ) -> str:
        """
        安全调用 LLM。

        功能：
            调用 llm_provider.safe_ainvoke，并使用 backup_llm 执行工具解析。

        参数：
            prompt_value:
                ChatPromptTemplate 渲染后的 prompt 输入。

        返回值：
            str:
                LLM 返回的文本结果。
        """

        return await llm_provider.safe_ainvoke(
            llm=backup_llm,
            prompt=prompt_value,
            fallback_response="调用LLM失败",
        )

    safe_llm = RunnableLambda(
        safe_llm_ainvoke
    )

    # 先保留 LLM 原始文本，再由解析函数兼容“解释文字 + JSON”的输出。
    chain = prompt | safe_llm

    async def llm_tool_parser(
        parser_input: Mapping[str, Any],
    ) -> ToolParseResult:
        """
        执行 LLM 工具解析链。

        功能：
            从统一 parser_input 中取出 question，
            补充 PydanticOutputParser 需要的格式说明，
            然后调用 LangChain chain 得到 ToolParseResult。

        参数：
            parser_input:
                call_tool_parser 传入的统一解析输入。

        返回值：
            ToolParseResult:
                LLM 解析后的工具解析结果。
        """

        raw_output = await chain.ainvoke(
            {
                "question": parser_input.get(
                    "question",
                    "",
                ),
                "tool_catalog_text": build_tool_catalog_prompt_text(
                    state=parser_input.get(
                        "state",
                        {},
                    ),
                ),
                "format_instructions": (
                    tool_parse_output_parser.get_format_instructions()
                ),
            }
        )

        return parse_tool_parse_llm_output(
            raw_output=raw_output,
        )

    return llm_tool_parser


def parse_tool_parse_llm_output(
    raw_output: Any,
) -> ToolParseResult:
    """
    将 LLM 原始输出解析成 ToolParseResult。

    功能：
        第一阶段使用 PydanticOutputParser 解析标准纯 JSON；
        第一阶段失败后，扫描文本中的 JSON 对象，并逐个执行 Pydantic 校验。
        这样可以兼容 LLM 在 JSON 前后附加少量解释文字的情况。

    参数：
        raw_output:
            LLM 返回的字符串或带 content 属性的消息对象。

    返回值：
        ToolParseResult:
            通过 Pydantic 数据校验的工具解析结果。

    异常：
        Exception:
            文本中不存在符合 ToolParseResult 的 JSON 对象时，重新抛出原始解析异常。
    """

    output_text = extract_llm_output_text(
        raw_output=raw_output,
    )

    try:
        return tool_parse_output_parser.parse(
            output_text
        )
    except Exception as original_error:
        decoder = json.JSONDecoder()

        for index, character in enumerate(
            output_text
        ):
            if character != "{":
                continue

            try:
                candidate, _ = decoder.raw_decode(
                    output_text[index:]
                )
            except json.JSONDecodeError:
                continue

            if not isinstance(
                candidate,
                Mapping,
            ):
                continue

            try:
                return ToolParseResult.model_validate(
                    candidate
                )
            except ValidationError:
                continue

        raise original_error


def extract_llm_output_text(
    raw_output: Any,
) -> str:
    """
    从 LLM 返回值中提取文本。

    功能：
        兼容普通字符串和带 content 属性的 LangChain 消息对象。

    参数：
        raw_output:
            LLM 原始返回值。

    返回值：
        str:
            用于 JSON 解析的文本。
    """

    if isinstance(
        raw_output,
        str,
    ):
        return raw_output.strip()

    if hasattr(
        raw_output,
        "content",
    ):
        return str(
            raw_output.content
            or ""
        ).strip()

    return str(
        raw_output
        or ""
    ).strip()


def build_tool_catalog_prompt_text(
    state: Mapping[str, Any],
) -> str:
    """
    构建工具目录 prompt 文本。

    功能：
        从 state["tool_agent_tool_catalog"] 读取工具目录，
        并渲染成 LLM 可以阅读的工具列表文本。
        如果 state 中没有工具目录，则返回默认 date/weather 工具说明。

    参数：
        state:
            当前 LangGraph state，可能包含 tool_agent_tool_catalog。

    返回值：
        str:
            可注入工具解析 prompt 的工具目录文本。
    """

    raw_catalog = state.get(
        "tool_agent_tool_catalog",
        [],
    )

    if not isinstance(
        raw_catalog,
        list,
    ) or not raw_catalog:
        return DEFAULT_TOOL_CATALOG_PROMPT_TEXT

    rendered_tools: list[str] = []

    for index, raw_tool in enumerate(
        raw_catalog,
        start=1,
    ):
        if not isinstance(
            raw_tool,
            Mapping,
        ):
            continue

        tool_name = str(
            raw_tool.get(
                "name",
                "",
            )
            or ""
        ).strip()

        if not tool_name:
            continue

        description = str(
            raw_tool.get(
                "description",
                "",
            )
            or ""
        ).strip()
        require_confirm = bool(
            raw_tool.get(
                "require_confirm",
                False,
            )
        )
        input_schema = raw_tool.get(
            "input_schema",
            {},
        )
        source = str(
            raw_tool.get(
                "source",
                "",
            )
            or ""
        ).strip()

        rendered_tools.append(
            render_tool_catalog_item_for_prompt(
                index=index,
                name=tool_name,
                description=description,
                require_confirm=require_confirm,
                input_schema=input_schema
                if isinstance(
                    input_schema,
                    Mapping,
                )
                else {},
                source=source,
            )
        )

    if not rendered_tools:
        return DEFAULT_TOOL_CATALOG_PROMPT_TEXT

    return "\n\n".join(
        rendered_tools
    )


def render_tool_catalog_item_for_prompt(
    index: int,
    name: str,
    description: str,
    require_confirm: bool,
    input_schema: Mapping[str, Any] | None = None,
    source: str = "",
) -> str:
    """
    渲染单个工具目录条目。

    功能：
        将单个工具的名称、描述、确认标记和参数结构转换成 prompt 文本。

    参数：
        index:
            工具序号。

        name:
            工具名称。

        description:
            工具描述。

        require_confirm:
            是否需要用户确认。

        input_schema:
            工具输入参数结构。为空时不输出参数说明。

        source:
            工具来源，例如 local 或 mcp。为空时不输出来源说明。

    返回值：
        str:
            单个工具的 prompt 文本。
    """

    safe_description = description or "无描述"
    rendered_parts = [
        f"{index}. {name}",
        "功能：",
        f"- {safe_description}",
    ]

    if source:
        rendered_parts.extend(
            [
                "工具来源：",
                f"- {source}",
            ]
        )

    input_schema_text = render_input_schema_for_prompt(
        input_schema=input_schema or {},
    )

    if input_schema_text:
        rendered_parts.extend(
            [
                "参数结构：",
                input_schema_text,
            ]
        )

    rendered_parts.extend(
        [
            f"是否需要确认：{require_confirm}",
            "-----------------------------------",
        ]
    )

    return "\n".join(
        rendered_parts
    )


def render_input_schema_for_prompt(
    input_schema: Mapping[str, Any],
) -> str:
    """
    渲染工具输入参数结构。

    功能：
        从 JSON Schema 风格的 input_schema 中读取 properties 和 required，
        转换成 LLM 更容易阅读的普通文本。

    参数：
        input_schema:
            工具输入参数结构，通常来自 MCP 工具定义。

    返回值：
        str:
            参数结构文本。没有可用字段时返回空字符串。
    """

    raw_properties = input_schema.get(
        "properties",
        {},
    )

    if not isinstance(
        raw_properties,
        Mapping,
    ) or not raw_properties:
        return ""

    raw_required = input_schema.get(
        "required",
        [],
    )
    required_fields = {
        str(
            item
        )
        for item in raw_required
        if isinstance(
            item,
            str,
        )
    } if isinstance(
        raw_required,
        list,
    ) else set()

    rendered_fields: list[str] = []

    for field_name, raw_field_schema in raw_properties.items():
        if not isinstance(
            raw_field_schema,
            Mapping,
        ):
            continue

        rendered_fields.append(
            render_input_schema_field_for_prompt(
                field_name=str(
                    field_name
                ),
                field_schema=raw_field_schema,
                required=str(
                    field_name
                )
                in required_fields,
            )
        )

    if not rendered_fields:
        return ""

    return "\n".join(
        rendered_fields
    )


def render_input_schema_field_for_prompt(
    field_name: str,
    field_schema: Mapping[str, Any],
    required: bool,
) -> str:
    """
    渲染单个工具参数字段。

    功能：
        将 JSON Schema 中的单个字段描述转换成一行 prompt 文本。

    参数：
        field_name:
            参数字段名。

        field_schema:
            参数字段的 schema 描述。

        required:
            该参数是否必填。

    返回值：
        str:
            单行参数说明文本。
    """

    field_type = str(
        field_schema.get(
            "type",
            "unknown",
        )
        or "unknown"
    )
    description = str(
        field_schema.get(
            "description",
            "",
        )
        or ""
    ).strip()
    required_text = (
        "必填"
        if required
        else "可选"
    )
    explicit_input_text = (
        "，必须由当前用户问题明确提供，禁止猜测"
        if field_schema.get(
            "x-requires-explicit-user-input",
            False,
        )
        else ""
    )

    if description:
        allowed_values_text = render_schema_allowed_values_for_prompt(
            field_schema=field_schema,
        )
        return (
            f"- {field_name}: {field_type}，{required_text}，{description}"
            f"{explicit_input_text}{allowed_values_text}"
        )

    allowed_values_text = render_schema_allowed_values_for_prompt(
        field_schema=field_schema,
    )

    return (
        f"- {field_name}: {field_type}，{required_text}"
        f"{explicit_input_text}{allowed_values_text}"
    )


def render_schema_allowed_values_for_prompt(
    field_schema: Mapping[str, Any],
) -> str:
    """
    渲染字段允许值说明。

    功能：
        从字段 schema 中读取 enum 或 allowed_values，
        转换成 LLM 可读的“合法值只能是...”文本。

    参数：
        field_schema:
            单个字段的 JSON Schema 描述。

    返回值：
        str:
            允许值说明；没有允许值时返回空字符串。
    """

    raw_allowed_values = field_schema.get(
        "enum",
        field_schema.get(
            "allowed_values",
            [],
        ),
    )

    if not isinstance(
        raw_allowed_values,
        list,
    ) or not raw_allowed_values:
        return ""

    allowed_values = [
        str(
            value
        )
        for value in raw_allowed_values
        if str(
            value
        ).strip()
    ]

    if not allowed_values:
        return ""

    return " 合法值只能是：" + "、".join(
        allowed_values
    ) + "。"


def write_tool_parse_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具解析节点运行时事件。

    功能：
        如果存在 RuntimeContext，则记录当前 node 和 timeline 事件。
        如果不存在，则静默跳过，保证单元测试和脚本环境也能运行。

    参数：
        runtime_context:
            当前请求的 RuntimeContext，可能为 None。

    返回值：
        None。
    """

    if runtime_context is None:
        return

    runtime_context.state().set_node(
        "tool_agent_tool_parse_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_parse_node",
    )


async def call_tool_parser(
    parser: Any,
    question: str,
    state: Mapping[str, Any],
) -> Any:
    """
    调用工具解析器。

    功能：
        兼容三种 parser：
        1. 带 ainvoke 方法的异步解析器。
        2. 带 invoke 方法的同步解析器。
        3. 普通 callable 解析函数。

    参数：
        parser:
            工具解析器对象或函数。

        question:
            用户问题文本。

        state:
            当前 LangGraph state。

    返回值：
        Any:
            parser 的原始返回结果，后续由 normalize_tool_parse_result 归一化。
    """

    parser_input = {
        "question": question,
        "state": dict(
            state
        ),
    }

    if hasattr(
        parser,
        "ainvoke",
    ):
        result = parser.ainvoke(
            parser_input
        )
        return await maybe_await(
            result
        )

    if hasattr(
        parser,
        "invoke",
    ):
        result = parser.invoke(
            parser_input
        )
        return await maybe_await(
            result
        )

    if callable(
        parser
    ):
        result = parser(
            parser_input
        )
        return await maybe_await(
            result
        )

    raise TypeError(
        "parser 必须是 callable，或提供 ainvoke/invoke 方法。"
    )


async def maybe_await(
    value: Any,
) -> Any:
    """
    按需等待 awaitable 对象。

    功能：
        如果 value 是 Awaitable（可等待对象），则 await 后返回结果；
        如果不是，则直接返回原值。

    参数：
        value:
            任意值，可能是 coroutine、Future 或普通对象。

    返回值：
        Any:
            await 后或原始的值。
    """

    if inspect.isawaitable(
        value
    ):
        return await value

    return value


def normalize_tool_parse_result(
    raw_result: Any,
) -> ToolParseResult:
    """
    归一化工具解析结果。

    功能：
        将 ToolParseResult 或 dict 格式的 parser 输出统一转换成 ToolParseResult。

    参数：
        raw_result:
            parser 原始输出，可以是 ToolParseResult 或 Mapping。

    返回值：
        ToolParseResult:
            标准工具解析结果。
    """

    if isinstance(
        raw_result,
        ToolParseResult,
    ):
        return raw_result

    if not isinstance(
        raw_result,
        Mapping,
    ):
        raise TypeError(
            "parser 返回值必须是 ToolParseResult 或 dict。"
        )

    tool_calls = [
        tool_call
        for tool_call in (
            parse_tool_call_item(
                item
            )
            for item in raw_result.get(
                "tool_calls",
                [],
            )
        )
        if tool_call is not None
    ]

    return ToolParseResult(
        need_tool=bool(
            raw_result.get(
                "need_tool",
                bool(
                    tool_calls
                ),
            )
        ),
        tool_calls=tool_calls,
        response=str(
            raw_result.get(
                "response",
                "",
            )
            or ""
        ),
    )


def recover_incomplete_sqlite_list_tables_call(
    parse_result: ToolParseResult,
    question: str,
    state: Mapping[str, Any],
) -> ToolParseResult:
    """
    恢复被 LLM 静默丢弃的 SQLite 表清单不完整调用。

    功能：
        当根路由已经明确要求进入工具链路、用户问题明显是在查询数据库表清单、
        工具目录中也确实存在 sqlite_list_tables，但 LLM 返回空 tool_calls 时，
        构造 args={} 的不完整调用，交给后续参数校验节点生成澄清请求。

    参数：
        parse_result:
            LLM 归一化后的工具解析结果。
        question:
            当前轮用户原始问题。
        state:
            当前 ToolAgent state，用于读取根路由决策和工具目录。

    返回值：
        ToolParseResult:
            不满足恢复条件时返回原结果；满足条件时返回包含空参数调用的新结果。
    """

    if parse_result.tool_calls:
        return parse_result

    route_decision = state.get(
        "route_decision",
        {},
    )
    if not isinstance(
        route_decision,
        Mapping,
    ) or not bool(
        route_decision.get(
            "requires_tool",
            False,
        )
    ):
        return parse_result

    normalized_question = str(
        question
        or ""
    ).strip().lower()
    if not (
        "数据库" in normalized_question
        and "表" in normalized_question
    ):
        return parse_result

    tool_catalog = state.get(
        "tool_agent_tool_catalog",
        [],
    )
    has_list_tables_tool = any(
        isinstance(item, Mapping)
        and str(
            item.get(
                "name",
                "",
            )
            or ""
        ) == "sqlite_list_tables"
        for item in tool_catalog
    ) if isinstance(
        tool_catalog,
        list,
    ) else False

    if not has_list_tables_tool:
        return parse_result

    return ToolParseResult(
        need_tool=True,
        tool_calls=[
            ToolCall(
                name="sqlite_list_tables",
                args={},
            )
        ],
        response=parse_result.response,
    )


def parse_tool_call_item(
    item: Any,
) -> ToolCall | None:
    """
    解析单个工具调用条目。

    功能：
        将 ToolCall 或 dict 转换成 ToolCall。
        如果缺少 name 或格式非法，则返回 None。

    参数：
        item:
            单个工具调用条目。

    返回值：
        ToolCall | None:
            解析成功返回 ToolCall，失败返回 None。
    """

    if isinstance(
        item,
        ToolCall,
    ):
        return item

    if not isinstance(
        item,
        Mapping,
    ):
        return None

    name = item.get(
        "name",
        "",
    )

    if not name:
        return None

    try:
        return ToolCall(
            name=str(
                name
            ),
            args=dict(
                item.get(
                    "args",
                    {},
                )
                or {}
            ),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def dump_tool_parse_result_for_state(
    parse_result: ToolParseResult,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将工具解析结果转换成 state update。

    功能：
        输出旧工具链路兼容字段 need_tool、tool_calls、tool_results、tool_round，
        并额外输出 tool_agent_response，方便后续 ToolAgent 响应契约收敛。

    参数：
        parse_result:
            标准工具解析结果。

        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的普通字典。
    """

    update: dict[str, Any] = {
        "need_tool": parse_result.need_tool,
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in parse_result.tool_calls
        ],
        "tool_results": [],
        "tool_round": int(
            state.get(
                "tool_round",
                0,
            )
            or 0
        )
        + 1,
    }

    merged_state = {
        **dict(
            state
        ),
        **update,
    }
    response = build_tool_agent_response_from_state(
        state=merged_state,
    )
    update[TOOL_AGENT_RESPONSE_STATE_KEY] = response.model_dump()

    return update


def build_tool_parse_fallback_update(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构建工具解析失败时的兜底 state update。

    功能：
        当缺少 question、parser 抛错或 parser 输出非法时，返回安全默认值。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            表示不需要工具的 state update。
    """

    return dump_tool_parse_result_for_state(
        parse_result=ToolParseResult(
            need_tool=False,
            tool_calls=[],
            response="",
        ),
        state=state,
    )
