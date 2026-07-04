from typing import (
    TypedDict,
    List,
    Optional,
    Dict,
    Any,
    Annotated,
)

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.rag.schemas import RagQuery, RagContext


class DogState(TypedDict, total=False):
    """
    Dog Agent Framework 主状态。

    功能：
        定义 Dog Agent Framework 中 Main Graph（主图）、Agent Subgraph（智能体子图）、
        RAG（检索增强生成）、Tool（工具调用）、Memory（记忆）等模块共享的状态字段。

        v1.5 阶段采用“兼容迁移”策略：
        1. 保留旧字段，避免现有节点立刻报错。
        2. 新增结构化字段，支持 Graph Redesign。
        3. 后续逐步把旧字段迁移到新对象中。

    参数：
        无。TypedDict（类型字典）只用于描述 state 的字段结构，不需要初始化参数。

    返回值：
        无直接返回值。该类用于给 LangGraph 的 StateGraph 提供状态结构约束。
    """

    # =========================
    # 1. 用户输入 / 对话历史
    # =========================

    question: str
    messages: Annotated[List[BaseMessage], add_messages]

    # =========================
    # 2. 运行时上下文 Runtime Context
    # =========================

    user_id: str
    session_id: str
    trace_id: str

    # =========================
    # 3. 旧版 RAG 查询解析字段
    # 后续逐步迁移到 rag_query
    # =========================

    intent: str
    strategy: str
    filters: dict
    tags: List[str]
    features: List[str]
    dog_name: Optional[str]
    top_k: int

    # =========================
    # 4. 新版 RAG 结构化字段
    # v1.4 / v1.5 推荐使用
    # =========================

    rag_query: RagQuery
    rag_context: RagContext

    retrieval_quality: dict[str, Any]

    retrieval_failure_type: str

    retrieval_evaluated: bool

    # =========================
    # 5. 旧版 RAG 检索结果字段
    # 后续逐步迁移到 rag_context
    # =========================

    docs: List[Document]
    retrieval_ok: bool
    retry_count: int

    # =========================
    # 6. 答案输出字段
    # answer 用于旧链路兼容
    # final_answer 用于 v1.5 统一最终输出
    # =========================

    answer: str
    final_answer: str

    answer_strategy: Dict[str, Any]

    # =========================
    # 6.1 DogKnowledgeAgent v1.7.3 标准答案契约字段
    # =========================

    dog_knowledge_answer: Any
    dog_knowledge_answer_public: Dict[str, Any]

    # =========================
    # 7. 人机交互 / 澄清字段
    # =========================

    user_feedback: Optional[str]
    has_asked_user: bool
    pending_prompt: str
    waiting_user_input: bool

    # =========================
    # 8. 工具调用字段
    # =========================

    tool_calls: List[Dict[str, Any]]
    tool_results: List[str]
    need_tool: bool
    tool_round: int
    tool_confirmed: str
    tool_executed: bool

    # =========================
    # 9. 旧版 Supervisor 路由字段
    # 后续逐步迁移到 route_decision
    # =========================

    next_worker: str
    next_agent: str
    current_agent: str

    # =========================
    # 10. 新版结构化路由字段  因为graph中使用了sqlite checkpoint 所以这里将对象换为字典方便存储
    # =========================

    route_decision: Dict[str, Any]
    root_observability: Dict[str, Any]
    root_debug_report: Dict[str, Any]

    # =========================
    # 11. DogKnowledgeAgent pipeline skeleton(狗狗知识智能体的管线骨架)相关字段
    # =========================

    dog_knowledge_pipeline_status: str

    dog_knowledge_pipeline_version: str

    dog_knowledge_pipeline_question: str

    dog_knowledge_pipeline_steps: list[dict[str, Any]]

    dog_knowledge_pipeline_trace: list[dict[str, Any]]

    dog_knowledge_debug_report: dict[str, Any]

    # =========================
    # 12. Memory 记忆上下文
    # =========================

    memory_context: str

    # =========================
    # 12. 错误字段
    # =========================

    error: str
