from typing import Any

from src.graph.states.state import DogState
from src.graph.tools.runtime.tool_executor import ToolExecutor
from src.logger import logger
from src.runtime.context import runtime_ctx


def build_execute_tool_node(
    tool_executor=None,
    checkpoint_manager=None,
    runtime_context_getter=None,
):
    """
    构建 execute_tool_node 节点。

    功能：
        创建一个真正给 LangGraph 使用的工具执行节点。
        外层函数负责接收 tool_executor、checkpoint_manager、runtime_context_getter 等依赖。
        内层 execute_tool_node 保持 LangGraph 需要的 state -> dict 调用格式。

    参数：
        tool_executor：
            ToolExecutor（工具执行器）。
            用于根据工具名称和参数执行真实工具。
            如果不传，则默认创建 ToolExecutor。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于工具执行完成后保存 checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        callable：
            返回一个 async node 函数。
            该函数接收 DogState，返回 dict，供 LangGraph 合并 state。

    专业名词：
        Dependency Injection，DI（依赖注入）：
            不在节点内部直接 import container，而是从外部传入依赖。

        ToolExecutor（工具执行器）：
            负责真正执行工具调用的运行时组件。

        Tool Results（工具结果）：
            工具执行后的结果集合。这里统一使用 list 结构，支持多个工具结果。
    """

    if tool_executor is None:
        tool_executor = ToolExecutor()

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def execute_tool_node(state: DogState) -> dict:
        """
        执行当前待调用工具。

        功能：
            1. 写入当前 node 状态
            2. 记录 node timeline 事件
            3. 从 state.tool_calls 中取出第一个工具调用
            4. 执行该工具
            5. 将工具结果追加到 tool_results 列表
            6. 移除已经执行过的工具调用
            7. 如果还有剩余工具，则 need_tool=True
            8. 如果没有剩余工具，则 need_tool=False
            9. 工具执行完成后保存 checkpoint

        参数：
            state：
                DogState，LangGraph 当前状态。
                主要读取 tool_calls、tool_results、tool_round 字段。

        返回值：
            dict：
                返回需要合并进 LangGraph state 的字段。
                包括 tool_results、tool_calls、need_tool、tool_round。
        """

        ctx = runtime_context_getter()

        if ctx is not None:
            ctx.state().set_node(
                "execute_tool_node"
            )

            ctx.timeline().add_event(
                event_type="node",
                name="execute_tool_node"
            )

        logger.info(
            "进入执行工具节点......"
        )

        tool_calls = state.get(
            "tool_calls",
            []
        )

        current_results = _normalize_tool_results(
            state.get(
                "tool_results",
                []
            )
        )

        if not tool_calls:
            return {
                "need_tool": False,
                "tool_results": current_results
            }

        tool_call = tool_calls[0]

        name = tool_call.get(
            "name"
        )

        args = tool_call.get(
            "args",
            {}
        )

        logger.debug(
            f"tool_name: {name}...tool_args: {args}"
        )

        if ctx is not None:
            ctx.state().set_tool(
                name
            )

            ctx.timeline().add_event(
                event_type="tool",
                name=name,
                metadata={
                    "args": args
                }
            )

        result = await tool_executor.execute(
            name,
            args
        )

        dumped_result = _dump_tool_result(
            result
        )

        new_results = current_results + [
            dumped_result
        ]

        remaining_calls = tool_calls[1:]

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return {
            "tool_results": new_results,
            "tool_calls": remaining_calls,
            "need_tool": bool(
                remaining_calls
            ),
            "tool_round": (
                state.get(
                    "tool_round",
                    0
                )
                + 1
            ),
        }

    return execute_tool_node


def _normalize_tool_results(
    tool_results: Any,
) -> list:
    """
    归一化工具结果列表。

    功能：
        将历史遗留的字符串 tool_results 转换成 list。
        将 None 转换成空 list。
        如果本身就是 list，则直接返回。
        这样可以保证后续多工具结果追加时不会出错。

    参数：
        tool_results：
            当前 state 中已有的工具结果。
            可能是 list、str、None 或其他类型。

    返回值：
        list：
            归一化后的工具结果列表。

    专业名词：
        Normalize（归一化）：
            把不同格式的数据统一转换成一种稳定格式。
    """

    if tool_results is None:
        return []

    if isinstance(
        tool_results,
        list
    ):
        return tool_results

    if isinstance(
        tool_results,
        str
    ):
        return [
            tool_results
        ]

    return [
        tool_results
    ]


def _dump_tool_result(
    result: Any,
) -> Any:
    """
    转换工具执行结果。

    功能：
        如果工具结果是 Pydantic Model，则使用 model_dump 转换成 dict。
        如果工具结果本身是 dict、str 或其他普通对象，则直接返回。

    参数：
        result：
            tool_executor.execute 返回的工具执行结果。

    返回值：
        Any：
            可放入 tool_results 列表中的工具结果。

    专业名词：
        Pydantic Model（Pydantic 模型）：
            Pydantic 提供的数据结构对象，通常支持 model_dump 方法转换成 dict。
    """

    if hasattr(
        result,
        "model_dump"
    ):
        return result.model_dump()

    return result