from src.graph.tools.runtime.graph_stream_runtime import safe_stream_graph

from src.runtime.container.init import (
    container
)

from typing import Any

from src.logger import logger
from src.user.user_manager import get_user_id

from collections.abc import Mapping

from src.settings import settings
from src.rag.debug.retriever_debug_report import (
    cleanup_old_rag_debug_reports,
    save_rag_debug_report,
)


def create_initial_state(
        question: str,
        trace_id: str,
        session_id: str | None = None,
) -> dict[str, Any]:
    """
    创建每次 Graph 运行的初始状态。

    功能：
        为每一次用户问题创建一个干净的 DogState 初始字典。

        v1.5 当前设计：
        1. 每个新问题都要重置临时运行状态。
        2. 保留 user_id、trace_id、session_id 等运行身份字段。
        3. 初始化新版 RAG 字段：
           rag_query、rag_context、retrieval_quality、retrieval_failure_type。
        4. 初始化新版路由字段：
           route_decision、next_agent、next_worker、current_agent。
        5. 初始化工具调用字段：
           tool_calls、tool_results、need_tool、tool_round、tool_confirmed、tool_executed。
        6. 初始化生成结果字段：
           answer、final_answer。
        7. 兼容旧字段：
           filters、tags、features、dog_name、docs、retry_count。

    重要说明：
        这里 messages 暂时保持为空列表。
        因为当前 semantic_router_node 内部会把 HumanMessage(question) 追加进去。
        如果这里也放 HumanMessage，就会导致消息重复。

    技术名词：
        State：
            状态。LangGraph 节点之间共享和传递的数据。

        Initial State：
            初始状态。每次图运行开始时创建的状态。

        Transient State：
            临时状态。只属于当前一次问题处理流程的数据，
            例如 docs、rag_context、answer、route_decision。

        RAG：
            Retrieval-Augmented Generation，检索增强生成。

        Trace ID：
            链路追踪 ID。用于定位一次完整请求的日志。

        Session ID：
            会话 ID。用于表示一次连续对话会话。

    参数：
        question:
            用户本次输入的问题。

        trace_id:
            当前请求的链路追踪 ID。

        session_id:
            当前会话 ID。
            如果外部没有传入，则默认使用 trace_id 作为 session_id。

    返回值：
        dict[str, Any]:
            DogState 初始状态字典。
    """

    logger.info(
        "初始化 state..."
    )

    clean_question = str(
        question
        or ""
    ).strip()

    if not clean_question:
        raise ValueError(
            "create_initial_state 失败：question 不能为空"
        )

    user_id = get_user_id()

    resolved_session_id = (
        session_id
        or trace_id
    )

    return {
        # ========= 用户输入 =========
        "question": clean_question,
        "messages": [],

        # ========= 身份与追踪 =========
        "user_id": user_id,
        "session_id": resolved_session_id,
        "trace_id": trace_id,

        # ========= 路由字段 =========
        "intent": "",
        "strategy": None,
        "next_agent": "",
        "current_agent": "",
        "next_worker": "",
        "route_decision": {},

        # ========= 旧版兼容检索字段 =========
        "filters": {},
        "tags": [],
        "features": [],
        "dog_name": None,
        "top_k": 5,
        "docs": [],

        # ========= 新版 RAG 字段 =========
        "rag_query": None,
        "rag_context": None,

        # ========= 召回评估字段 =========
        "retrieval_ok": False,
        "retrieval_evaluated": False,
        "retrieval_quality": None,
        "retrieval_failure_type": None,
        "retrieval_retry_strategy": None,
        "retry_count": 0,

        # ========= 生成结果字段 =========
        "answer": "",
        "final_answer": "",
        "answer_strategy": {},

        # ========= 用户反馈 / 追问字段 =========
        "user_feedback": "",
        "has_asked_user": False,
        "pending_prompt": "",
        "waiting_user_input": False,

        # ========= 工具调用字段 =========
        "tool_calls": [],
        "tool_results": [],
        "need_tool": False,
        "tool_round": 0,
        "tool_confirmed": "",
        "tool_executed": False,

        # ========= 记忆字段 =========
        "memory_context": "",

        # ========= 错误字段 =========
        "error": "",
    }


def should_write_rag_debug_report(
        state: Mapping[str, Any],
) -> bool:
    """
    判断是否应该写入 RAG Debug Report。

    功能：
        根据 settings.observability 的配置，
        以及当前 state 是否包含 RAG 相关字段，
        判断是否需要保存 RAG Debug Report。

    参数：
        state:
            当前 LangGraph 最终状态。

    返回值：
        bool:
            True 表示需要保存报告。
            False 表示不保存报告。

    专业名词：
        RAG Debug Report：
            RAG 调试报告，用于记录一次 RAG 链路的查询、检索、评估、精排、生成结果。
    """

    if not settings.observability.ENABLE_RAG_DEBUG_REPORT:
        return False

    if not settings.observability.RAG_DEBUG_REPORT_TO_FILE:
        return False

    if not isinstance(
            state,
            Mapping,
    ):
        return False

    has_rag_data = any(
        [
            state.get(
                "rag_query"
            ),
            state.get(
                "rag_context"
            ),
            state.get(
                "retrieval_quality"
            ),
        ]
    )

    return bool(
        has_rag_data
    )


def write_rag_debug_report_if_enabled(
        state: Mapping[str, Any],
        trace_id: str | None = None,
) -> None:
    """
    按配置写入 RAG Debug Report。

    功能：
        如果 settings 中开启了 RAG Debug Report 文件输出，
        则将当前 state 写入 Markdown 报告文件。

        当前支持：
        1. 按 trace_id 命名报告文件。
        2. 按日期分目录保存。
        3. 写入后打印文件路径。
        4. 可选清理过期报告目录。

    参数：
        state:
            当前 LangGraph 最终状态。

        trace_id:
            当前请求 trace_id。

    返回值：
        None。
    """

    if not should_write_rag_debug_report(
            state=state,
    ):
        return

    try:
        report_path = save_rag_debug_report(
            state=state,
            report_dir=settings.path.RAG_DEBUG_REPORT_DIR,
            trace_id=trace_id,
            max_context_chars=settings.observability.RAG_DEBUG_CONTEXT_MAX_CHARS,
            max_answer_chars=settings.observability.RAG_DEBUG_ANSWER_MAX_CHARS,
            use_date_dir=settings.observability.RAG_DEBUG_REPORT_USE_DATE_DIR,
        )

        logger.info(
            f"RAG Debug Report saved: {report_path.resolve()} "
            f"exists={report_path.exists()}"
        )

        if settings.observability.RAG_DEBUG_REPORT_CLEANUP_ON_WRITE:
            removed_count = cleanup_old_rag_debug_reports(
                report_dir=settings.path.RAG_DEBUG_REPORT_DIR,
                retention_days=settings.observability.RAG_DEBUG_REPORT_RETENTION_DAYS,
            )

            if removed_count > 0:
                logger.info(
                    f"RAG Debug Report 清理完成，removed_dirs={removed_count}"
                )

    except Exception as e:
        logger.warning(
            f"RAG Debug Report 保存失败: {e}"
        )


def get_final_state_values(
        current_state,
) -> dict[str, Any]:
    """
    从 LangGraph current_state 中提取最终 state。

    功能：
        app.aget_state(config) 返回的对象中，
        values 通常保存当前图状态。
        这里统一转换为 dict，方便后续写报告。

    参数：
        current_state:
            LangGraph 当前状态对象。

    返回值：
        dict[str, Any]:
            当前图状态字典。
    """

    values = getattr(
        current_state,
        "values",
        {},
    )

    if isinstance(
            values,
            Mapping,
    ):
        return dict(
            values
        )

    return {}


async def run_main_graph_with_stream(question: str, thread_id:str="default_user", trace_id=None) -> str:
    # 启用runtime上下文管理  废弃trace_ctx
    from src.runtime.context import runtime_ctx

    runtime_context = runtime_ctx.get()


    graph_runtime = container.get(
        "graph_runtime"
    )

    app = graph_runtime.graph

    if trace_id:
        # trace_ctx.set_trace_id(trace_id)
        runtime_context.trace_id = trace_id

    if isinstance(question, list) and len(question) > 0 and "text" in question[0]:
        question = question[0]["text"]

    # 初始化state
    state = create_initial_state(question, trace_id)

    # 设置 user_id 和 session_id 到 contextvar（确保即使没有经过 Gradio 入口也能记录）
    # trace_ctx.set_user_id(state.get("user_id", "unknown"))
    # trace_ctx.set_session_id(thread_id)


    runtime_context.user_id = state.get("user_id", "unknown")
    runtime_context.set_session_id = thread_id

    logger.info(f"收到用户 [{state["user_id"]}] 问题: {question}")

    # 用user_id作为线程的id 每次对话都可以根据线程id来获取之前对话历史  达到记忆目的
    config = {"configurable": {"thread_id": thread_id},
              "run_name": f"query_{question[:20]}",  # LangSmith 显示的名称
              "tags": ["dog_agent", "memory_test"],
              }

    from langgraph.types import Command


    # 根据question判断是恢复断点还是新的问题
    if question.startswith("RESUME:"):
        user_input = question[7:].strip()

        async def get_command_events():
            events = []
            async for chunk in app.astream(Command(resume=user_input), config, stream_mode="values"):
                events.append(chunk)
            return events

        # events = list(app_2.astream(Command(resume=user_input), config, stream_mode="values"))

        events = await get_command_events()


        # 遍历事件  获取最后的answer
        for ev in events:
            if "answer" in ev and ev["answer"]:
                return ev["answer"]


        # 恢复后又中断了  虽然不太可能  但为了保险起见
        current = await app.aget_state(
            config
        )

        if current.next:
            prompt = extract_interrupt_prompt(
                current
            )

            return f"__INTERRUPT__:{prompt}"

        final_state = get_final_state_values(
            current_state=current
        )

        write_rag_debug_report_if_enabled(
            state=final_state,
            trace_id=trace_id,
        )

        answer = (
                final_state.get(
                    "answer"
                )
                or final_state.get(
            "final_answer"
        )
                or "无答案"
        )

        return str(
            answer
        )
    else:

        # 使用uuid作为线程id 每次对话创建一个新的id
        # config = {"configurable": {"thread_id": uuid.uuid4().hex}}

        # 第一次调用，可能因中断而提前结束迭代
        #     events = list(app_2.stream(state, config, stream_mode="values"))
        #     改成safe_stream_graph方式调用 将运行和业务逻辑解耦 并添加上错误处理
        # events = list(
        #     safe_stream_graph(
        #         graph=app_2,
        #         state=state,
        #         config=config,
        #         # stream_mode="values"
        #         stream_mode="updates"
        #     )
        # )


        # safe_stream_graph改为异步之后 调用方式变为async for来收集
        async def collect_stream():
            events = []
            async for chunk in safe_stream_graph(graph=app, state=state, config=config, stream_mode="updates"):
                events.append(chunk)
            return events

        events = await collect_stream()

        logger.debug(
            f"图执行产生 {len(events)} 个状态快照"
        )

        current = await app.aget_state(
            config
        )

        if current.next:
            prompt = extract_interrupt_prompt(
                current
            )

            return f"__INTERRUPT__:{prompt}"

        final_state = get_final_state_values(
            current_state=current
        )

        write_rag_debug_report_if_enabled(
            state=final_state,
            trace_id=trace_id,
        )

        answer = (
                final_state.get(
                    "answer"
                )
                or final_state.get(
            "final_answer"
        )
                or "无答案"
        )

        logger.info(
            f"返回答案长度: {len(str(answer))} 字符"
        )

        return str(
            answer
        )


        # 如果没有 answer，说明图在中断点暂停了，需要循环处理
        # while True:
        #     # 获取当前状态，检查中断信息
        #     current_state = app_2.get_state(config)
        #     if not current_state.next:  # 图已结束
        #         return current_state.values.get("answer", "无答案")
        #
        #     # 提取中断附带的消息（由 ask_user_node 中的 interrupt() 传入）
        #     # 在 LangGraph 中，中断信息存储在 current_state.tasks[0].interrupts 或 __interrupt__ 字段中
        #     # 不同版本略有差异，常用方式：
        #     if hasattr(current_state, 'tasks') and current_state.tasks:
        #         interrupts = current_state.tasks[0].interrupts
        #         if interrupts:
        #             # 取第一个中断的消息
        #             prompt_message = interrupts[0].value  # 这就是 interrupt(question) 中的 question
        #             logger.debug(f"第一个中断的信息: {prompt_message}")
        #         else:
        #             prompt_message = "请做出选择："
        #     else:
        #         # 备用：从状态中的某个字段获取（但这需要你在 ask_user_node 中额外保存）
        #         prompt_message = "请做出选择："
        #
        #     # 显示从节点中提取的提示，而不是硬编码
        #     logger.debug(f"节点中提取的提示信息: {prompt_message}")
        #     user_input = input("您的输入：").strip()
        #
        #     # 恢复执行
        #     for event in app_2.stream(Command(resume=user_input), config, stream_mode="values"):
        #         if "answer" in event and event["answer"]:
        #             return event["answer"]


def extract_interrupt_prompt(current_state):
    if hasattr(current_state, 'tasks') and current_state.tasks:
        interrupts = current_state.tasks[0].interrupts
        if interrupts:
            return interrupts[0].value
    return "请做出选择（1/2/3）："