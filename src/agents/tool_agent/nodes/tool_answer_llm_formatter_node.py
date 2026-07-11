"""
ToolAgent LLM 工具答案格式化节点。

功能：
    在规则答案生成后，尝试使用 LLM 把成功工具结果转换成自然语言；
    LLM 不可用或失败时保留已有规则答案。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.tool_agent.adapters.state_adapter import normalize_tool_results
from src.agents.tool_agent.adapters.tool_answer_formatter_adapter import (
    format_tool_results_with_llm,
)
from src.agents.tool_agent.debug.state_logging import log_tool_agent_state
from src.agents.tool_agent.nodes.tool_answer_node import build_tool_answer_update
from src.logger import logger
from src.runtime.context import runtime_ctx


ToolAnswerLlmFormatterNode = Callable[
    [Mapping[str, Any]],
    Awaitable[dict[str, Any]],
]


def build_tool_agent_tool_answer_llm_formatter_node(
    llm_provider: Any | None = None,
    checkpoint_manager: Any | None = None,
    runtime_context_getter: Callable[[], Any] | None = None,
) -> ToolAnswerLlmFormatterNode:
    """
    构建 ToolAgent LLM 答案格式化节点。

    参数：
        llm_provider:
            LLM 服务提供者；为空时节点直接跳过。
        checkpoint_manager:
            自定义 Runtime Checkpoint 管理器。
        runtime_context_getter:
            RuntimeContext 获取函数。

    返回值：
        ToolAnswerLlmFormatterNode:
            接收 state 并异步返回普通字典 update 的节点函数。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def tool_agent_tool_answer_llm_formatter_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        尝试生成 LLM 自然语言工具答案。

        参数：
            state:
                当前 LangGraph state。

        返回值：
            dict[str, Any]:
                成功时返回新的 final_answer；跳过或失败时返回空字典。
        """

        runtime_context = runtime_context_getter()
        if runtime_context is not None:
            runtime_context.state().set_node(
                "tool_agent_tool_answer_llm_formatter_node"
            )
            runtime_context.timeline().add_event(
                event_type="node",
                name="tool_agent_tool_answer_llm_formatter_node",
            )

        tool_results = normalize_tool_results(
            state.get(
                "tool_results",
                [],
            )
        )
        successful_results = [
            result.model_dump()
            for result in tool_results
            if result.success
        ]

        if llm_provider is None or not successful_results:
            log_tool_agent_state(
                node_name="tool_answer_llm_formatter",
                event="tool_answer_llm_formatter_skipped",
                state=state,
                extra={
                    "has_llm_provider": llm_provider is not None,
                    "successful_result_count": len(successful_results),
                },
            )
            return {}

        try:
            final_answer = await format_tool_results_with_llm(
                question=str(
                    state.get(
                        "question",
                        "",
                    )
                    or ""
                ),
                tool_results=successful_results,
                llm_provider=llm_provider,
            )
        except Exception as exc:
            logger.warning(
                f"ToolAgent LLM 答案格式化失败，保留规则答案: {exc}"
            )
            log_tool_agent_state(
                node_name="tool_answer_llm_formatter",
                event="tool_answer_llm_formatter_fallback",
                state=state,
                extra={
                    "error": str(exc),
                },
            )
            return {}

        update = build_tool_answer_update(
            state=state,
            final_answer=final_answer,
            answer_source="llm_tool_result_formatter",
        )
        update["tool_agent_llm_answer_used"] = True

        log_tool_agent_state(
            node_name="tool_answer_llm_formatter",
            event="tool_answer_llm_formatter_success",
            state={
                **dict(state),
                **update,
            },
            extra={
                "final_answer_preview": final_answer[:120],
            },
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return update

    return tool_agent_tool_answer_llm_formatter_node
