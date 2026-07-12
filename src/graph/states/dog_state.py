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

    # 用户当前这一轮输入的问题，是路由、检索、工具解析和答案生成的主要输入。
    question: str

    # 当前对话累计的消息历史；add_messages 会把节点返回的新消息追加到旧消息中。
    messages: Annotated[List[BaseMessage], add_messages]

    # =========================
    # 2. 运行时上下文 Runtime Context
    # =========================

    # 用户唯一编号，用于隔离不同用户的长期记忆和业务数据。
    user_id: str

    # 会话编号，用于标识同一个对话窗口或一次连续会话。
    session_id: str

    # 单次请求追踪编号，用于串联本轮执行产生的日志、事件和调试信息。
    trace_id: str

    # =========================
    # 3. 旧版 RAG 查询解析字段
    # 后续逐步迁移到 rag_query
    # =========================

    # 旧版问题意图，例如精确查询、推荐或普通问答；新版优先读取 rag_query。
    intent: str

    # 旧版检索策略，描述本轮应采用的检索方式；新版逐步迁移到 rag_query。
    strategy: str

    # 旧版 RAG metadata 过滤条件，用于限定犬种、体型等检索范围。
    filters: dict

    # 旧版问题标签列表，用于辅助规则过滤和问题分类。
    tags: List[str]

    # 旧版犬种特征列表，例如体型、性格或护理特点。
    features: List[str]

    # 从问题中识别出的犬种名称；没有明确犬种时为 None。
    dog_name: Optional[str]

    # 希望从检索器中取得的候选结果数量。
    top_k: int

    # =========================
    # 4. 新版 RAG 结构化字段
    # v1.4 / v1.5 推荐使用
    # =========================

    # 新版标准 RAG 查询对象，集中保存问题、过滤条件和召回数量等检索输入。
    rag_query: RagQuery

    # 新版标准 RAG 上下文，集中保存召回片段、上下文文本和召回状态。
    rag_context: RagContext

    # 检索质量评估明细，例如质量等级、原因和诊断指标。
    retrieval_quality: dict[str, Any]

    # 检索失败类型，例如结果为空、质量不足或检索异常。
    retrieval_failure_type: str

    # 是否已经执行过检索质量评估，防止节点重复评估。
    retrieval_evaluated: bool

    # =========================
    # 5. 旧版 RAG 检索结果字段
    # 后续逐步迁移到 rag_context
    # =========================

    # 旧版检索文档列表，每个 Document 包含正文和 metadata；新版优先使用 rag_context。
    docs: List[Document]

    # 旧版检索是否可继续标记；True 表示结果满足后续生成条件。
    retrieval_ok: bool

    # 当前链路累计重试次数，用于限制检索或生成失败后的循环次数。
    retry_count: int

    # =========================
    # 6. 答案输出字段
    # answer 用于旧链路兼容
    # final_answer 用于 v1.5 统一最终输出
    # =========================

    # 旧版节点生成的答案文本，为兼容旧链路暂时保留。
    answer: str

    # 主图对外返回的统一最终答案文本。
    final_answer: str

    # 答案生成策略，例如任务类型、回答结构和应使用的数据来源。
    answer_strategy: Dict[str, Any]

    # =========================
    # 6.1 DogKnowledgeAgent v1.7.3 标准答案契约字段
    # =========================

    # DogKnowledgeAgent 内部标准答案对象；运行时可能是 Schema 实例或兼容数据。
    dog_knowledge_answer: Any

    # 可安全写入 checkpoint 并对外暴露的字典版狗狗知识答案。
    dog_knowledge_answer_public: Dict[str, Any]

    # =========================
    # 6.2 DogKnowledgeAgent v1.7.4 Layer Contract 中间产物字段
    # 这里使用 dict 保存，方便 LangGraph checkpoint 序列化。
    # =========================

    # 问题解析层的标准中间产物，记录问题类型、目标字段和过滤条件等结果。
    dog_query_result: Dict[str, Any]

    # 检索层的标准中间产物，记录召回是否成功、上下文和检索诊断信息。
    dog_retrieval_result: Dict[str, Any]

    # 推荐层的标准中间产物，记录候选犬种及推荐理由。
    dog_recommendation_result: Dict[str, Any]

    # 答案生成层的标准中间产物，记录生成文本及其来源信息。
    dog_generation_result: Dict[str, Any]

    # 降级层的标准中间产物，记录降级原因和兜底答案。
    dog_fallback_result: Dict[str, Any]

    # 汇总各分层产物后的 DogKnowledgeAgent pipeline 最终结果。
    dog_knowledge_pipeline_result: Dict[str, Any]

    # =========================
    # 7. 人机交互 / 澄清字段
    # =========================

    # 用户针对确认或澄清提示返回的反馈内容。
    user_feedback: Optional[str]

    # 是否已经询问过用户，用于避免同一问题被无限重复询问。
    has_asked_user: bool

    # 等待用户处理的提示文本，例如工具确认问题或参数澄清问题。
    pending_prompt: str

    # 当前图是否正在等待用户输入；True 表示本轮不能继续自动执行。
    waiting_user_input: bool

    # =========================
    # 8. 工具调用字段
    # =========================

    # LLM 解析出的旧版工具调用列表，单项通常包含工具 name 和调用 args。
    tool_calls: List[Dict[str, Any]]

    # 工具执行结果列表，顺序通常与实际执行的 tool_calls 对应。
    tool_results: List[Any]

    # 本轮问题是否需要调用工具；False 时 ToolAgent 可以直接生成无工具响应。
    need_tool: bool

    # 当前请求已进入工具处理的轮数，用于防止工具解析与执行无限循环。
    tool_round: int

    # 旧版工具确认结果，例如用户同意、拒绝或尚未确认。
    tool_confirmed: str

    # 旧版工具是否已经执行完成的标记，用于兼容原工具链路。
    tool_executed: bool

    # =========================
    # 8.1 ToolAgent v1.8 新版工具智能体字段
    # 这些字段用于新版 ToolAgent 子图在 LangGraph state 中保留确认、执行和响应契约。
    # =========================

    # 本轮计划中是否至少存在一个必须经过人工确认的工具调用。
    tool_confirmation_required: bool

    # 工具确认模式，描述按单个调用确认、批量确认或无需确认。
    tool_confirmation_mode: str

    # 展示给前端和用户的工具确认提示文本。
    tool_confirmation_prompt: str

    # ToolAgent 可用工具目录；每项描述工具名称、用途、参数契约和确认要求。
    tool_agent_tool_catalog: List[Dict[str, Any]]

    # SQLite MCP 允许访问的数据库别名与实际目标映射，例如 memory 和 rag。
    tool_agent_allowed_databases: Dict[str, str]

    # 工具调用的权限判断结果，记录状态、待确认调用编号、提示和原因。
    tool_agent_permission: Dict[str, Any]

    # ToolAgent 标准响应契约，汇总意图、计划调用、权限、执行记录和最终答案。
    tool_agent_response: Dict[str, Any]

    # 工具运行时产生的标准执行记录列表，包含调用参数、状态、结果和错误信息。
    tool_agent_runtime_execution_records: List[Dict[str, Any]]

    # 执行节点是否主动跳过工具执行，例如权限待确认、拒绝或参数校验失败。
    tool_agent_execute_skipped: bool

    # 工具执行被跳过的具体原因，供路由、最终回答和调试日志使用。
    tool_agent_execute_skip_reason: str

    # 最终工具答案的数据来源，例如 LLM 格式化、规则格式化或错误降级。
    tool_agent_answer_source: str

    # 最终工具答案是否成功使用 LLM 进行自然语言整理。
    tool_agent_llm_answer_used: bool

    # 所有工具调用是否通过工具存在性、参数类型和必填字段等契约校验。
    tool_call_validation_ok: bool

    # 本轮是否跳过了工具调用校验，例如根本没有需要执行的工具调用。
    tool_call_validation_skipped: bool

    # 工具调用校验发现的结构化错误列表，用于反馈具体工具和参数问题。
    tool_call_validation_errors: List[Dict[str, Any]]

    # 未通过校验的原始工具调用列表，保留错误输入以便审计和排查。
    tool_call_validation_invalid_calls: List[Dict[str, Any]]

    # 参数缺失时生成的澄清请求，记录缺失字段、候选值和用户提示。
    tool_agent_clarification_request: Optional[Dict[str, Any]]

    # 等待用户补参的原始工具调用；下一轮会在此调用上继续填充参数。
    tool_agent_pending_tool_call: Optional[Dict[str, Any]]

    # 产生待补参工具调用时的原始用户问题，用于恢复完整任务语义。
    tool_agent_pending_original_question: str

    # 待补参任务的创建时间，用于调试以及后续实现澄清任务过期策略。
    tool_agent_pending_created_at: str

    # 当前用户输入是否已被识别为可用于恢复待补参工具调用的内容。
    tool_agent_clarification_resume_ready: bool

    # 澄清输入解析结果，记录继续补参、仍需澄清或切换新任务等判断。
    tool_agent_clarification_resolution: Dict[str, Any]

    # =========================
    # 9. 旧版 Supervisor 路由字段
    # 后续逐步迁移到 route_decision
    # =========================

    # 旧版多智能体调度中下一位要执行的 worker 名称。
    next_worker: str

    # 旧版路由判断出的下一 Agent 名称。
    next_agent: str

    # 当前正在执行或最近执行的 Agent 名称。
    current_agent: str

    # =========================
    # 10. 新版结构化路由字段  因为graph中使用了sqlite checkpoint 所以这里将对象换为字典方便存储
    # =========================

    # RootAgent 标准路由决策的字典形式，包含 route、query_type、confidence 和 reason。
    route_decision: Dict[str, Any]

    # RootAgent 路由过程的可观测数据，例如时间线记录和最终目标 Agent。
    root_observability: Dict[str, Any]

    # RootAgent 调试报告，用于解释为什么问题被路由到某个 Agent。
    root_debug_report: Dict[str, Any]

    # =========================
    # 11. DogKnowledgeAgent pipeline skeleton(狗狗知识智能体的管线骨架)相关字段
    # =========================

    # DogKnowledgeAgent pipeline 当前状态，例如准备、运行中、完成或失败。
    dog_knowledge_pipeline_status: str

    # 管线骨架元数据使用的版本号，用于区分不同阶段的结构契约。
    dog_knowledge_pipeline_version: str

    # 创建管线骨架时记录的用户问题快照。
    dog_knowledge_pipeline_question: str

    # 静态步骤说明列表，描述每一步名称、期望输入和期望输出，不表示实时执行记录。
    dog_knowledge_pipeline_steps: list[dict[str, Any]]

    # 管线实际运行轨迹列表，用于记录执行期间发生的步骤和状态变化。
    dog_knowledge_pipeline_trace: list[dict[str, Any]]

    # DogKnowledgeAgent 调试报告，汇总查询、检索、生成和答案契约信息。
    dog_knowledge_debug_report: dict[str, Any]

    # =========================
    # 12. Memory 记忆上下文
    # =========================

    # 根据当前问题召回并格式化后的长期记忆文本，会注入后续答案生成提示词。
    memory_context: str

    # 记忆召回的结构化可观测结果，包含召回状态、候选数量、语义门槛、采用数量和记忆 ID。
    memory_recall_result: Dict[str, Any]

    # 当前输入是否已经成功创建、强化或重新激活长期记忆。
    memory_saved: bool

    # LLM 对当前输入的记忆抽取结果，包含 should_save、类型、内容、可信度和原因。
    memory_extract_result: Dict[str, Any]

    # MemoryManager 实际保存结果，包含 action、memory_id、强度和失效数量等字段；未保存时为 None。
    memory_save_result: Optional[Dict[str, Any]]

    # =========================
    # 13. 错误字段
    # =========================

    # 节点或主图执行失败时记录的错误信息，供降级回答和调试使用。
    error: str
