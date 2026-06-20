from langgraph.types import interrupt

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx


def build_ask_confirm_tool_node(
    checkpoint_manager=None,
    interrupt_func=None,
    runtime_context_getter=None,
):
    """
    构建 ask_confirm_tool_node 节点。

    功能：
        创建一个真正给 LangGraph 使用的工具确认节点。
        外层函数负责接收 checkpoint_manager、interrupt_func、runtime_context_getter 等依赖。
        内层 ask_confirm_tool_node 保持 LangGraph 需要的 state -> dict 调用格式。

    参数：
        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于在中断前、用户确认后、用户拒绝后保存 checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        interrupt_func：
            Interrupt Function（中断函数）。
            默认使用 langgraph.types.interrupt。
            测试时可以传入 fake interrupt 函数，避免真的中断 Graph。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        callable：
            返回一个同步 node 函数。
            该函数接收 DogState，返回 dict，供 LangGraph 合并 state。

    专业名词：
        Dependency Injection，DI（依赖注入）：
            不在节点内部直接 import container，而是从外部传入依赖。

        Interrupt（中断）：
            LangGraph 中用于暂停当前 Graph 执行，并等待用户输入的机制。

        Tool Results（工具结果）：
            工具执行后的结果集合。这里统一使用 list 结构，
            为后续多个工具调用结果做准备。
    """

    if interrupt_func is None:
        interrupt_func = interrupt

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    def ask_confirm_tool_node(state: DogState) -> dict:
        """
        在执行工具之前询问用户是否确认调用工具。

        功能：
            1. 写入当前 node 状态
            2. 记录 timeline 事件
            3. 检查 state 中是否存在 tool_calls
            4. 如果没有 tool_calls，则返回 need_tool=False
            5. 如果有 tool_calls，则生成确认提示
            6. 使用 interrupt_func 中断并等待用户输入
            7. 用户输入 y 时返回空 dict，表示继续执行工具
            8. 用户输入非 y 时取消工具调用
            9. tool_results 统一使用 List[str]，不再返回单个字符串

        参数：
            state：
                DogState，LangGraph 当前状态。
                主要读取 tool_calls 字段。

        返回值：
            dict：
                返回需要合并进 LangGraph state 的字段。
                用户确认时返回 {}。
                用户拒绝时返回 need_tool=False、tool_results、tool_calls=[]。
        """

        ctx = runtime_context_getter()

        if ctx is not None:
            ctx.state().set_node(
                "ask_confirm_tool_node"
            )

            ctx.timeline().add_event(
                event_type="node",
                name="ask_confirm_tool_node"
            )

        tool_calls = state.get(
            "tool_calls",
            []
        )

        if not tool_calls:
            return {
                "need_tool": False,
                "tool_results": [
                    "没有工具需要确认。"
                ]
            }

        prompt = _build_confirm_prompt(
            tool_calls
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        user_input = interrupt_func(
            prompt
        )

        if user_input.strip().lower() == "y":
            logger.debug(
                f"DEBUG: state = {state}------用户确认执行工具"
            )

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint()

            return {}

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return {
            "need_tool": False,
            "tool_results": [
                "用户取消了工具调用。"
            ],
            "tool_calls": []
        }

    return ask_confirm_tool_node


def _build_confirm_prompt(
    tool_calls: list[dict],
) -> str:
    """
    构建工具调用确认提示。

    功能：
        将 tool_calls 中的所有待执行工具格式化成用户可读的确认文本。
        不再只展示第一个工具，而是展示全部工具调用。

    参数：
        tool_calls：
            工具调用列表。
            每一项通常包含 name 和 args。

    返回值：
        str：
            用户确认提示文本。

    专业名词：
        Prompt（提示词）：
            展示给用户或模型看的输入文本。

        Tool Call（工具调用）：
            表示需要执行的工具名称和参数。
    """

    lines = [
        "即将执行以下工具："
    ]

    for index, tool_call in enumerate(
        tool_calls,
        start=1,
    ):
        tool_name = tool_call.get(
            "name",
            "unknown_tool"
        )

        tool_args = tool_call.get(
            "args",
            {}
        )

        lines.append(
            f"{index}. 工具：【{tool_name}】，参数：【{tool_args}】"
        )

    lines.append(
        "是否继续？(y/n)"
    )

    return "\n".join(
        lines
    )

