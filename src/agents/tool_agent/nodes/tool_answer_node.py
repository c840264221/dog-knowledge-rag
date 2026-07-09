"""
ToolAgent 工具答案节点。

功能：
    将工具执行结果转换成用户可读的 final_answer。

设计原则：
    1. 当前 MVP 使用规则格式化，不调用 LLM。
    2. 不修改 tool_results 原始结构，只新增 final_answer。
    3. 输出普通 dict，避免 checkpoint 保存自定义对象。
    4. 节点文件放在 nodes 目录下，并以 _node.py 结尾。

专业名词：
    Formatter：格式化器，把结构化数据转换成可读文本。
    Final Answer：最终回答，展示给用户的文本。
    Tool Result：工具结果，工具执行后的结构化返回值。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    normalize_tool_results,
)
from src.agents.tool_agent.debug.state_logging import (
    log_tool_agent_state,
)
from src.graph.tools.schemas.tool_result_schema import ToolResult
from src.runtime.context import runtime_ctx


ToolAnswerNode = Callable[
    [Mapping[str, Any]],
    dict[str, Any],
]


def build_tool_agent_tool_answer_node(
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolAnswerNode:
    """
    构建 ToolAgent 工具答案节点。

    功能：
        创建一个同步节点。
        节点读取 state.tool_results，生成用户可读的 final_answer。

    参数：
        checkpoint_manager:
            检查点管理器。生成 final_answer 后按需保存 checkpoint。

        runtime_context_getter:
            RuntimeContext 获取函数。默认使用 runtime_ctx.get。

    返回值：
        ToolAnswerNode:
            同步节点函数，接收 state，返回可合并进 state 的 dict。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    def tool_agent_tool_answer_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        生成工具结果回答。

        功能：
            1. 写入当前运行时 node 信息。
            2. 读取并归一化 tool_results。
            3. 根据工具名和执行成功状态生成 final_answer。
            4. 刷新 tool_agent_response。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                包含 final_answer 和 tool_agent_response 的 state update。
        """

        write_tool_answer_runtime_event(
            runtime_context=runtime_context_getter(),
        )

        log_tool_agent_state(
            node_name="tool_answer",
            event="tool_answer_start",
            state=state,
        )

        tool_results = normalize_tool_results(
            state.get(
                "tool_results",
                [],
            )
        )

        if not tool_results:
            update = build_tool_answer_update(
                state=state,
                final_answer=str(
                    state.get(
                        "final_answer",
                        "",
                    )
                    or ""
                ),
                answer_source="empty_tool_results",
            )
            log_tool_agent_state(
                node_name="tool_answer",
                event="tool_answer_empty_results",
                state={
                    **dict(
                        state
                    ),
                    **update,
                },
                extra={
                    "tool_result_count": 0,
                },
            )
            return update

        final_answer = format_tool_results_answer(
            tool_results=tool_results,
        )
        update = build_tool_answer_update(
            state=state,
            final_answer=final_answer,
            answer_source="tool_results_formatter",
        )

        log_tool_agent_state(
            node_name="tool_answer",
            event="tool_answer_success",
            state={
                **dict(
                    state
                ),
                **update,
            },
            extra={
                "tool_result_count": len(
                    tool_results
                ),
                "final_answer_preview": final_answer[:120],
            },
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return update

    return tool_agent_tool_answer_node


def write_tool_answer_runtime_event(
    runtime_context: Any,
) -> None:
    """
    写入工具答案节点运行时事件。

    功能：
        如果存在 RuntimeContext，则记录当前 node 和 timeline 事件。

    参数：
        runtime_context:
            当前请求的 RuntimeContext，可能为 None。

    返回值：
        None。
    """

    if runtime_context is None:
        return

    runtime_context.state().set_node(
        "tool_agent_tool_answer_node"
    )
    runtime_context.timeline().add_event(
        event_type="node",
        name="tool_agent_tool_answer_node",
    )


def format_tool_results_answer(
    tool_results: list[ToolResult],
) -> str:
    """
    格式化工具结果回答。

    功能：
        将一个或多个 ToolResult 转换成用户可读文本。

    参数：
        tool_results:
            归一化后的工具结果列表。

    返回值：
        str:
            用户可读回答文本。
    """

    formatted_answers = [
        format_single_tool_result(
            tool_result=tool_result,
        )
        for tool_result in tool_results
    ]

    return "\n".join(
        answer
        for answer in formatted_answers
        if answer
    )


def format_single_tool_result(
    tool_result: ToolResult,
) -> str:
    """
    格式化单个工具结果。

    功能：
        根据工具执行成功状态和工具名称，生成对应的回答文本。

    参数：
        tool_result:
            单个工具执行结果。

    返回值：
        str:
            单个工具的用户可读回答。
    """

    if not tool_result.success:
        return format_failed_tool_result(
            tool_result=tool_result,
        )

    if tool_result.tool_name == "weather":
        return format_weather_tool_result(
            tool_result=tool_result,
        )

    if tool_result.tool_name == "date":
        return format_date_tool_result(
            tool_result=tool_result,
        )

    return format_generic_tool_result(
        tool_result=tool_result,
    )


def format_failed_tool_result(
    tool_result: ToolResult,
) -> str:
    """
    格式化失败工具结果。

    功能：
        将工具失败信息转换成用户可读文本。

    参数：
        tool_result:
            失败的工具结果。

    返回值：
        str:
            失败提示文本。
    """

    error = tool_result.error or "未知错误"

    return f"工具调用失败：{error}"


def format_weather_tool_result(
    tool_result: ToolResult,
) -> str:
    """
    格式化天气工具结果。

    功能：
        优先从 dict content 中读取城市、天气、温度、风速。
        如果 content 是字符串，则直接拼接成天气回答。

    参数：
        tool_result:
            weather 工具执行结果。

    返回值：
        str:
            天气回答文本。
    """

    content = tool_result.content

    if isinstance(
        content,
        Mapping,
    ):
        city = content.get(
            "city",
            "",
        ) or content.get(
            "city_name",
            "",
        )
        weather = content.get(
            "weather",
            "",
        ) or content.get(
            "condition",
            "",
        )
        temperature = content.get(
            "temperature",
            "",
        ) or content.get(
            "temp",
            "",
        )
        wind_speed = content.get(
            "wind_speed",
            "",
        ) or content.get(
            "windspeed",
            "",
        )

        parts = [
            f"{city}天气"
            if city
            else "天气",
            str(
                weather
            )
            if weather
            else "",
            f"温度约 {temperature}"
            if temperature
            else "",
            f"风速约 {wind_speed}"
            if wind_speed
            else "",
        ]

        return "，".join(
            part
            for part in parts
            if part
        ) + "。"

    content_text = stringify_tool_content(
        content=content,
    )

    if content_text:
        return f"天气查询结果：{content_text}"

    return "天气查询完成，但工具没有返回具体天气内容。"


def format_date_tool_result(
    tool_result: ToolResult,
) -> str:
    """
    格式化日期工具结果。

    功能：
        将 date 工具返回内容转换成用户可读日期回答。

    参数：
        tool_result:
            date 工具执行结果。

    返回值：
        str:
            日期回答文本。
    """

    content_text = stringify_tool_content(
        content=tool_result.content,
    )

    if content_text:
        return f"今天的日期是 {content_text}。"

    return "日期查询完成，但工具没有返回具体日期内容。"


def format_generic_tool_result(
    tool_result: ToolResult,
) -> str:
    """
    格式化通用工具结果。

    功能：
        当工具没有专门格式化规则时，使用通用文本输出。

    参数：
        tool_result:
            工具执行结果。

    返回值：
        str:
            通用工具回答文本。
    """

    content_text = stringify_tool_content(
        content=tool_result.content,
    )

    if content_text:
        return f"{tool_result.tool_name} 工具返回：{content_text}"

    return f"{tool_result.tool_name} 工具执行完成。"


def stringify_tool_content(
    content: Any,
) -> str:
    """
    将工具内容转换成文本。

    功能：
        兼容字符串、dict、list、None 等不同工具返回格式。

    参数：
        content:
            工具返回内容。

    返回值：
        str:
            转换后的文本。
    """

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content

    return str(
        content
    )


def build_tool_answer_update(
    state: Mapping[str, Any],
    final_answer: str,
    answer_source: str,
) -> dict[str, Any]:
    """
    构建工具答案 state update。

    功能：
        写入 final_answer，并刷新 tool_agent_response。

    参数：
        state:
            当前 LangGraph state。

        final_answer:
            工具结果格式化后的最终回答。

        answer_source:
            回答来源，写入 metadata 方便调试。

    返回值：
        dict[str, Any]:
            可写回 LangGraph state 的回答结果。
    """

    update = {
        "final_answer": final_answer,
        "tool_agent_answer_source": answer_source,
    }
    merged_state = {
        **dict(
            state
        ),
        **update,
    }
    response = build_tool_agent_response_from_state(
        state=merged_state,
    ).model_dump()
    metadata = dict(
        response.get(
            "metadata",
            {},
        )
        or {}
    )
    metadata["answer_source"] = answer_source
    response["metadata"] = metadata
    update[TOOL_AGENT_RESPONSE_STATE_KEY] = response

    return update
