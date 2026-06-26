from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class BaseAgentState(TypedDict, total=False):
    """
    基础 Agent 状态。

    功能：
        定义所有 Graph（图）和 Agent Subgraph（智能体子图）共享的基础状态字段。
        这样可以保证主图和子图在 LangGraph 中使用相同的 channel（通道）定义，
        避免 messages 字段类型不一致导致运行时报错。

    参数：
        无。TypedDict 是类型结构定义，不需要初始化参数。

    字段：
        messages:
            消息列表，用于保存用户输入、AI 回复、工具消息等。
            使用 add_messages 合并策略，表示新消息会追加到旧消息后面。

        user_id:
            用户 ID，用于长期记忆 Memory（记忆）、权限隔离和个性化。

        session_id:
            会话 ID，用于区分不同对话会话。

        trace_id:
            追踪 ID，用于 Trace（链路追踪）、Timeline（时间线）和日志系统。

    返回值：
        无直接返回值。该类用于给 LangGraph State 提供类型约束。
    """

    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    trace_id: str