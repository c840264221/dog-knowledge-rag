from src.graph.tools.runtime.graph_stream_runtime import safe_stream_graph

from src.runtime.container.init import (
    container
)
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
    GraphInterruptType,
)
from src.runtime.resume.legacy_protocol import (
    encode_legacy_interrupt_result,
    parse_legacy_resume_message,
)
from src.runtime.services.checkpoint_config import build_graph_checkpoint_config

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
        "tool_agent_llm_answer_used": False,

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


async def run_main_graph_with_result(
        question: str,
        thread_id: str = "default_user",
        trace_id: str | None = None,
        resume_value: str | None = None,
        graph_app: Any | None = None,
        runtime_context: Any | None = None,
        stream_runner: Any | None = None,
) -> GraphFinalResult | GraphInterruptResult:
    """
    运行主图并返回结构化结果。

    功能：
        执行 Dog Agent Framework 主图，并用 GraphFinalResult / GraphInterruptResult
        表达运行结果，避免在运行入口内部继续依赖字符串前缀表示状态。

    参数：
        question:
            用户输入的问题，或者旧 UI 传入的恢复消息。
        thread_id:
            LangGraph thread_id，用于定位同一条可恢复图执行线程。
        trace_id:
            当前请求链路追踪 ID。
        resume_value:
            显式恢复值。UI / API 已经知道当前是在恢复中断时，可以直接传入该值，
            避免继续拼接旧版 RESUME 字符串。
        graph_app:
            可选的 LangGraph app。测试时可以传入 mock app，真实运行时默认从 container 获取。
        runtime_context:
            可选 RuntimeContext。测试时可以传入 mock context，真实运行时默认从 runtime_ctx 获取。
        stream_runner:
            可选流式执行函数。测试时可以传入 mock async generator，真实运行时默认使用 safe_stream_graph。

    返回值：
        GraphFinalResult | GraphInterruptResult:
            GraphFinalResult 表示图正常完成；
            GraphInterruptResult 表示图被 interrupt 暂停，需要用户继续输入。
    """

    app = graph_app or _get_graph_app_from_container()
    resolved_runtime_context = runtime_context or _get_runtime_context()
    resolved_stream_runner = stream_runner or safe_stream_graph
    normalized_question = normalize_graph_question(question)

    if trace_id:
        resolved_runtime_context.trace_id = trace_id

    state = create_initial_state(
        normalized_question,
        trace_id,
    )

    resolved_runtime_context.user_id = state.get(
        "user_id",
        "unknown",
    )
    resolved_runtime_context.session_id = thread_id

    logger.info(
        f"收到用户 [{state['user_id']}] 问题: {normalized_question}"
    )

    config = build_graph_checkpoint_config(
        thread_id=thread_id,
        run_name=f"query_{normalized_question[:20]}",
        tags=[
            "dog_agent",
            "memory_test",
        ],
        metadata={
            "trace_id": trace_id,
        },
    )
    resume_checkpoint_ns = config["configurable"].pop(
        "checkpoint_ns"
    )

    # 新一轮问题启动前，只恢复参数澄清所需字段，避免完整旧 state 污染本轮输入。
    state = await restore_pending_tool_clarification_state(
        app=app,
        config=config,
        state=state,
    )

    resume_request = parse_legacy_resume_message(
        message=normalized_question,
        thread_id=thread_id,
        checkpoint_ns=resume_checkpoint_ns,
        trace_id=trace_id,
    )
    resolved_resume_value = (
        resume_value
        if resume_value is not None
        else (
            resume_request.resume_value
            if resume_request is not None
            else None
        )
    )

    if resolved_resume_value is not None:
        return await _resume_main_graph_with_result(
            app=app,
            config=config,
            resume_value=str(resolved_resume_value),
            thread_id=thread_id,
            checkpoint_ns=resume_checkpoint_ns,
            trace_id=trace_id,
        )

    return await _start_main_graph_with_result(
        app=app,
        state=state,
        config=config,
        stream_runner=resolved_stream_runner,
        thread_id=thread_id,
        checkpoint_ns=resume_checkpoint_ns,
        trace_id=trace_id,
    )


async def restore_pending_tool_clarification_state(
        app: Any,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从当前线程检查点恢复待处理的工具参数澄清字段。

    功能：
        在新一轮主图执行前读取相同 thread_id 的最新 Checkpoint，
        只恢复澄清请求、待补全调用及其辅助字段，不恢复旧答案和旧工具结果。

    参数：
        app:
            已编译的 LangGraph 主图对象，需要提供 aget_state 方法。
        config:
            当前 LangGraph 执行配置，包含 thread_id。
        state:
            本轮新创建的干净初始状态。

    返回值：
        dict[str, Any]:
            合并待澄清字段后的本轮初始状态；没有待澄清状态时返回原状态副本。
    """

    restored_state = dict(
        state
    )
    try:
        current_state = await app.aget_state(
            config
        )
        checkpoint_values = get_final_state_values(
            current_state=current_state,
        )
    except Exception as exc:
        logger.debug(
            f"读取参数澄清 Checkpoint 失败，按新问题继续: {exc}"
        )
        return restored_state

    clarification_request = checkpoint_values.get(
        "tool_agent_clarification_request"
    )
    pending_tool_call = checkpoint_values.get(
        "tool_agent_pending_tool_call"
    )
    if not isinstance(
            clarification_request,
            Mapping,
    ) or not isinstance(
            pending_tool_call,
            Mapping,
    ):
        return restored_state

    clarification_keys = (
        "tool_agent_clarification_request",
        "tool_agent_pending_tool_call",
        "tool_agent_pending_original_question",
        "tool_agent_pending_created_at",
    )
    for key in clarification_keys:
        if key in checkpoint_values:
            restored_state[key] = checkpoint_values[key]

    logger.info(
        "已从当前 thread_id 的 Checkpoint 恢复待处理工具参数澄清状态。"
    )
    return restored_state


async def run_main_graph_with_stream(
        question: str,
        thread_id: str = "default_user",
        trace_id: str | None = None,
) -> str:
    """
    运行主图并返回旧 UI 兼容字符串。

    功能：
        兼容 Gradio UI 当前调用方式。内部调用结构化的 run_main_graph_with_result，
        再把 GraphInterruptResult 编码为旧版中断字符串，把 GraphFinalResult 转成答案文本。

    参数：
        question:
            用户输入问题或旧版恢复消息。
        thread_id:
            LangGraph thread_id。
        trace_id:
            当前请求链路追踪 ID。

    返回值：
        str:
            普通完成时返回答案文本；
            中断时返回旧 UI 可识别的中断字符串。
    """

    result = await run_main_graph_with_result(
        question=question,
        thread_id=thread_id,
        trace_id=trace_id,
    )

    if isinstance(
            result,
            GraphInterruptResult,
    ):
        return encode_legacy_interrupt_result(
            result
        )

    return result.answer


def normalize_graph_question(
        question: Any,
) -> str:
    """
    归一化 Graph 输入问题。

    功能：
        兼容 Gradio 多模态输入中 question 可能是 [{"text": "..."}] 的情况，
        并统一转换为字符串。

    参数：
        question:
            原始输入，可能是 str 或包含 text 字段的 list。

    返回值：
        str:
            归一化后的问题文本。
    """

    if (
            isinstance(question, list)
            and len(question) > 0
            and isinstance(question[0], Mapping)
            and "text" in question[0]
    ):
        return str(
            question[0]["text"]
        )

    return str(
        question
    )


def _get_runtime_context() -> Any:
    """
    获取当前 RuntimeContext。

    功能：
        从 runtime_ctx contextvar 中读取当前运行时上下文。
        单独抽出函数后，测试可以通过 run_main_graph_with_result 的 runtime_context 参数绕过真实上下文。

    参数：
        无。

    返回值：
        Any:
            当前 RuntimeContext。
    """

    from src.runtime.context import runtime_ctx

    return runtime_ctx.get()


def _get_graph_app_from_container() -> Any:
    """
    从 container 中获取主图 app。

    功能：
        读取 graph_runtime 服务，并返回已经 compile 的 LangGraph app。

    参数：
        无。

    返回值：
        Any:
            LangGraph compiled graph app。
    """

    graph_runtime = container.get(
        "graph_runtime"
    )

    return graph_runtime.graph


async def _resume_main_graph_with_result(
        app: Any,
        config: dict[str, Any],
        resume_value: str,
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None = None,
) -> GraphFinalResult | GraphInterruptResult:
    """
    恢复已中断的主图并返回结构化结果。

    功能：
        使用 LangGraph Command(resume=...) 恢复图执行。
        如果恢复后正常完成，返回 GraphFinalResult；
        如果恢复后再次中断，返回 GraphInterruptResult。

    参数：
        app:
            LangGraph compiled graph app。
        config:
            LangGraph 执行配置。
        resume_value:
            用户确认或补充输入的恢复值。
        thread_id:
            LangGraph thread_id。
        checkpoint_ns:
            恢复契约中记录的 checkpoint namespace。
        trace_id:
            当前请求链路追踪 ID。

    返回值：
        GraphFinalResult | GraphInterruptResult:
            结构化图运行结果。
    """

    from langgraph.types import Command

    async for event in app.astream(
            Command(resume=resume_value),
            config,
            stream_mode="values",
    ):
        answer = extract_answer_from_state(event)

        if answer:
            return build_graph_final_result(
                answer=answer,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                trace_id=trace_id,
                metadata={
                    "source": "resume_stream_event",
                },
            )

    current = await app.aget_state(
        config
    )

    return build_graph_result_from_current_state(
        current_state=current,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
    )


async def _start_main_graph_with_result(
        app: Any,
        state: dict[str, Any],
        config: dict[str, Any],
        stream_runner: Any,
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None = None,
) -> GraphFinalResult | GraphInterruptResult:
    """
    启动一次新的主图运行并返回结构化结果。

    功能：
        调用 safe_stream_graph 执行主图，然后读取 current state 判断是否完成或中断。

    参数：
        app:
            LangGraph compiled graph app。
        state:
            Graph 初始 state。
        config:
            LangGraph 执行配置。
        stream_runner:
            图流式执行函数，真实运行使用 safe_stream_graph，测试可传 mock。
        thread_id:
            LangGraph thread_id。
        checkpoint_ns:
            恢复契约中记录的 checkpoint namespace。
        trace_id:
            当前请求链路追踪 ID。

    返回值：
        GraphFinalResult | GraphInterruptResult:
            结构化图运行结果。
    """

    events = []

    async for chunk in stream_runner(
            graph=app,
            state=state,
            config=config,
            stream_mode="updates",
    ):
        events.append(chunk)

    logger.debug(
        f"图执行产生 {len(events)} 个状态快照"
    )

    current = await app.aget_state(
        config
    )

    return build_graph_result_from_current_state(
        current_state=current,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
    )


def build_graph_result_from_current_state(
        current_state: Any,
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None = None,
) -> GraphFinalResult | GraphInterruptResult:
    """
    根据 LangGraph current state 构建结构化运行结果。

    功能：
        如果 current_state.next 存在，说明图停在 interrupt 上，返回 GraphInterruptResult；
        否则提取最终 state 中的答案并返回 GraphFinalResult。

    参数：
        current_state:
            LangGraph 当前状态对象。
        thread_id:
            LangGraph thread_id。
        checkpoint_ns:
            恢复契约中记录的 checkpoint namespace。
        trace_id:
            当前请求链路追踪 ID。

    返回值：
        GraphFinalResult | GraphInterruptResult:
            结构化图运行结果。
    """

    if getattr(
            current_state,
            "next",
            None,
    ):
        prompt = extract_interrupt_prompt(
            current_state
        )
        current_values = get_final_state_values(
            current_state=current_state
        )
        interrupt_metadata = build_interrupt_metadata_from_state(
            state=current_values,
        )

        return GraphInterruptResult(
            prompt=prompt,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            trace_id=trace_id,
            interrupt_type=resolve_interrupt_type_from_state(
                state=current_values,
            ),
            metadata=interrupt_metadata,
        )

    final_state = get_final_state_values(
        current_state=current_state
    )

    write_rag_debug_report_if_enabled(
        state=final_state,
        trace_id=trace_id,
    )

    answer = (
            extract_answer_from_state(final_state)
            or "无答案"
    )

    logger.info(
        f"返回答案长度: {len(str(answer))} 字符"
    )

    return build_graph_final_result(
        answer=answer,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        metadata={
            "source": "current_state",
        },
    )


def build_graph_final_result(
        answer: Any,
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
) -> GraphFinalResult:
    """
    构建 GraphFinalResult。

    功能：
        统一把答案转换成字符串，并补齐 thread_id、checkpoint_ns、trace_id 和 metadata。

    参数：
        answer:
            原始答案内容。
        thread_id:
            LangGraph thread_id。
        checkpoint_ns:
            恢复契约中记录的 checkpoint namespace。
        trace_id:
            当前请求链路追踪 ID。
        metadata:
            可选扩展元数据。

    返回值：
        GraphFinalResult:
            结构化最终完成结果。
    """

    return GraphFinalResult(
        answer=str(answer),
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        metadata=dict(metadata or {}),
    )


def extract_answer_from_state(
        state: Any,
) -> str | None:
    """
    从 state 或事件中提取答案文本。

    功能：
        优先读取 answer 字段，其次读取 final_answer 字段。
        兼容 dict / Mapping 格式的事件和最终 state。

    参数：
        state:
            Graph state 或 stream event。

    返回值：
        str | None:
            提取到的答案文本；没有可用答案时返回 None。
    """

    if not isinstance(
            state,
            Mapping,
    ):
        return None

    answer = (
            state.get("answer")
            or state.get("final_answer")
    )

    if answer:
        return str(
            answer
        )

    return None


def extract_interrupt_prompt(current_state):
    if hasattr(current_state, 'tasks') and current_state.tasks:
        interrupts = current_state.tasks[0].interrupts
        if interrupts:
            return interrupts[0].value
    return "请做出选择（1/2/3）："


def resolve_interrupt_type_from_state(
        state: Mapping[str, Any],
) -> GraphInterruptType:
    """
    根据当前 state 判断中断类型。

    功能：
        如果 state 中存在工具确认字段，则标记为工具确认中断。
        其他情况兜底为 unknown。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        GraphInterruptType:
            结构化中断类型。
    """

    if not isinstance(
            state,
            Mapping,
    ):
        return GraphInterruptType.UNKNOWN

    if (
            state.get("tool_confirmation_required")
            or state.get("tool_confirmation_prompt")
            or state.get("tool_agent_permission")
    ):
        return GraphInterruptType.TOOL_CONFIRMATION

    return GraphInterruptType.UNKNOWN


def build_interrupt_metadata_from_state(
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从当前 state 构建中断元数据。

    功能：
        提取 UI 恢复和日志排查最需要的字段，
        避免 UI 继续写死 general_agent。

    参数：
        state:
            当前 LangGraph state。

    返回值：
        dict[str, Any]:
            中断元数据。
    """

    if not isinstance(
            state,
            Mapping,
    ):
        return {}

    route_decision = state.get(
        "route_decision",
        {},
    )
    route = (
        route_decision.get("route")
        if isinstance(route_decision, Mapping)
        else ""
    )

    current_agent = (
            state.get("next_agent")
            or route
            or state.get("current_agent")
            or ""
    )

    return {
        "current_agent": current_agent,
        "current_node": state.get(
            "current_node",
            "",
        ),
        "route": route,
        "tool_calls": state.get(
            "tool_calls",
            [],
        ),
        "tool_confirmed": state.get(
            "tool_confirmed",
            "",
        ),
        "tool_confirmation_required": state.get(
            "tool_confirmation_required",
            False,
        ),
        "tool_agent_permission": state.get(
            "tool_agent_permission",
            {},
        ),
    }
