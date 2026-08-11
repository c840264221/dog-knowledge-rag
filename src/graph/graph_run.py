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
from src.agents.collaboration.contracts import MultiAgentTaskResult

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
        "raw_user_input": clean_question,
        "question": clean_question,
        "memory_source_text": clean_question,
        "memory_retrieval_text": clean_question,
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
        "task_relation_decision": {},
        "task_relation_pending_kind": "",
        "task_relation_requires_confirmation": False,
        "task_relation_guard_processed": False,

        # ========= 多 Agent 跨轮恢复字段 =========
        "multi_agent_task_result": {},
        "multi_agent_resume_action": "none",
        "multi_agent_resume_inputs": {},
        "multi_agent_step_resume_decisions": {},
        "multi_agent_resume_ready": False,
        "multi_agent_clarification_extraction": {},
        "multi_agent_pending_prompt": "",

        # ========= Skill 技能准备与跨轮恢复字段 =========
        "skill_runtime_result": {},
        "skill_selected_id": "",
        "skill_inputs": {},
        "skill_status": "no_skill",
        "skill_pending_prompt": "",
        "skill_context": "",
        "skill_original_question": "",
        "skill_target_agent": "",
        "skill_execution_mode": "standard",
        "skill_ignored_input_ids": [],
        "skill_degradation_reason": "",
        "skill_degradation_user_input": "",
        "retrieval_question": "",

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
        "memory_recall_result": {},
        "memory_saved": False,
        "memory_extract_result": {},
        "memory_retention_result": {},
        "memory_save_result": None,
        "pet_profile_extraction_result": {},
        "pet_profile_save_result": {},
        "active_pet_key": "",
        "active_pet_name": "",
        "pet_profile_recall_result": {},
        "skill_profile_recall_result": {},
        "pet_profile_suggested_attributes": [],
        "skill_required_pet_profile_attributes": [],
        "skill_profile_access_decision": {},
        "answer_profile_access_decision": {},
        "dog_query_understanding_result": {},

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

    # 新一轮问题启动前，只恢复当前宠物和各业务等待状态的白名单字段。
    state = await restore_active_pet_state(
        app=app,
        config=config,
        state=state,
    )
    state = await restore_pending_tool_clarification_state(
        app=app,
        config=config,
        state=state,
    )
    state = await restore_pending_multi_agent_state(
        app=app,
        config=config,
        state=state,
    )
    state = await restore_pending_skill_state(
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

    if (
            resolved_resume_value is not None
            and (
                _has_pending_multi_agent_resume_state(state)
                or _has_pending_skill_resume_state(state)
            )
    ):
        return await _start_main_graph_with_result(
            app=app,
            state=state,
            config=config,
            stream_runner=resolved_stream_runner,
            thread_id=thread_id,
            checkpoint_ns=resume_checkpoint_ns,
            trace_id=trace_id,
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


async def restore_active_pet_state(
        app: Any,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从当前线程检查点恢复上一轮明确选中的宠物身份。

    功能：
        读取相同 thread_id（检查点线程标识）的最新 Checkpoint（检查点），
        确认检查点与本轮属于同一用户后，只恢复 active_pet_key（当前宠物
        稳定标识）和 active_pet_name（当前宠物名称）。旧答案、路由、工具
        结果和检索结果都不会恢复。

    参数含义：
        app：
            已编译的 LangGraph 主图对象，需要提供 aget_state 方法。
        config：
            当前图执行配置，包含用于定位检查点的 thread_id。
        state：
            本轮根据新用户输入创建的干净初始状态。

    返回值含义：
        dict[str, Any]：
            合并当前宠物身份白名单后的新状态；检查点无效时返回原状态副本。
    """

    restored_state = dict(state)
    try:
        current_state = await app.aget_state(config)
        checkpoint_values = get_final_state_values(
            current_state=current_state,
        )
    except Exception as exc:
        logger.debug(
            f"读取当前宠物 Checkpoint 失败，按无选中宠物继续: {exc}"
        )
        return restored_state

    # thread_id 只能定位会话，仍需校验 user_id，避免错误会话标识造成数据串用。
    current_user_id = str(state.get("user_id") or "").strip()
    checkpoint_user_id = str(
        checkpoint_values.get("user_id") or ""
    ).strip()
    if (
        not current_user_id
        or checkpoint_user_id != current_user_id
    ):
        return restored_state

    active_pet_key = str(
        checkpoint_values.get("active_pet_key") or ""
    ).strip()
    active_pet_name = str(
        checkpoint_values.get("active_pet_name") or ""
    ).strip()
    if (
        not active_pet_key
        or len(active_pet_key) > 200
        or len(active_pet_name) > 100
    ):
        return restored_state

    restored_state["active_pet_key"] = active_pet_key
    restored_state["active_pet_name"] = active_pet_name
    logger.info(
        "已从当前 thread_id 的 Checkpoint 恢复当前宠物身份: "
        f"pet_key={active_pet_key}"
    )
    return restored_state


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


async def restore_pending_multi_agent_state(
        app: Any,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从当前线程检查点恢复暂停中的多 Agent 任务。

    功能：
        读取相同 thread_id 的最新 Checkpoint，只恢复状态为 awaiting_input
        的多 Agent 任务结果和待展示提示，不把旧答案或旧路由带入新一轮。

    参数：
        app:
            已编译的 LangGraph 主图对象，需要提供 aget_state 方法。
        config:
            当前 LangGraph 执行配置，包含 thread_id。
        state:
            本轮新创建的干净初始状态。

    返回值：
        dict[str, Any]:
            合并暂停任务白名单字段后的新状态；没有合法暂停任务时返回原状态副本。
    """

    restored_state = dict(state)
    try:
        current_state = await app.aget_state(config)
        checkpoint_values = get_final_state_values(
            current_state=current_state,
        )
    except Exception as exc:
        logger.debug(
            f"读取多 Agent Checkpoint 失败，按新问题继续: {exc}"
        )
        return restored_state

    raw_task_result = checkpoint_values.get(
        "multi_agent_task_result"
    )
    if not isinstance(raw_task_result, Mapping):
        return restored_state
    try:
        task_result = MultiAgentTaskResult.model_validate(raw_task_result)
    except (TypeError, ValueError):
        return restored_state
    if task_result.status != "awaiting_input":
        return restored_state

    restored_state["multi_agent_task_result"] = task_result.model_dump(
        mode="python"
    )
    raw_resume_inputs = checkpoint_values.get("multi_agent_resume_inputs")
    if isinstance(raw_resume_inputs, Mapping):
        restored_state["multi_agent_resume_inputs"] = dict(
            raw_resume_inputs
        )
    raw_resume_decisions = checkpoint_values.get(
        "multi_agent_step_resume_decisions"
    )
    if isinstance(raw_resume_decisions, Mapping):
        restored_state["multi_agent_step_resume_decisions"] = dict(
            raw_resume_decisions
        )
    raw_extraction = checkpoint_values.get(
        "multi_agent_clarification_extraction"
    )
    if isinstance(raw_extraction, Mapping):
        restored_state["multi_agent_clarification_extraction"] = dict(
            raw_extraction
        )
    restored_state["multi_agent_pending_prompt"] = str(
        checkpoint_values.get("multi_agent_pending_prompt")
        or task_result.plan.clarification_prompt
        or ""
    )
    logger.info(
        "已从当前 thread_id 的 Checkpoint 恢复暂停中的多 Agent 任务。"
    )
    return restored_state


async def restore_pending_skill_state(
        app: Any,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从当前线程检查点恢复等待输入的顶层 Skill 状态。

    功能：
        读取相同 thread_id 的最新 Checkpoint，只恢复 Skill 继续准备所需的
        技能编号、已提取输入、原始问题和目标 Agent，不恢复旧答案或旧检索结果。

    参数含义：
        app:
            已编译的 LangGraph 主图对象，需要提供 aget_state 方法。
        config:
            当前图执行配置，包含用于定位检查点的 thread_id。
        state:
            本轮根据用户补充回答创建的干净初始状态。

    返回值含义：
        dict[str, Any]:
            合并 Skill 白名单字段后的状态；没有合法等待状态时返回原状态副本。
    """

    restored_state = dict(state)
    try:
        current_state = await app.aget_state(config)
        checkpoint_values = get_final_state_values(
            current_state=current_state,
        )
    except Exception as exc:
        logger.debug(
            f"读取 Skill Checkpoint 失败，按新问题继续: {exc}"
        )
        return restored_state

    if str(checkpoint_values.get("skill_status") or "").strip() != (
        "awaiting_input"
    ):
        return restored_state

    selected_skill_id = str(
        checkpoint_values.get("skill_selected_id") or ""
    ).strip()
    original_question = str(
        checkpoint_values.get("skill_original_question") or ""
    ).strip()
    target_agent = str(
        checkpoint_values.get("skill_target_agent") or ""
    ).strip()
    raw_inputs = checkpoint_values.get("skill_inputs")
    if (
        not selected_skill_id
        or not original_question
        or target_agent not in {
            "dog_knowledge_agent",
            "general_agent",
        }
        or not isinstance(raw_inputs, Mapping)
    ):
        return restored_state

    # 这里只恢复 Skill 继续工作必需的数据，避免旧答案和旧业务结果污染本轮。
    restored_state.update(
        {
            "skill_runtime_result": dict(
                checkpoint_values.get("skill_runtime_result") or {}
            ),
            "skill_selected_id": selected_skill_id,
            "skill_inputs": dict(raw_inputs),
            "skill_status": "awaiting_input",
            "skill_pending_prompt": str(
                checkpoint_values.get("skill_pending_prompt") or ""
            ),
            "skill_context": "",
            "skill_original_question": original_question,
            "skill_target_agent": target_agent,
            "skill_execution_mode": str(
                checkpoint_values.get("skill_execution_mode")
                or "standard"
            ),
            "skill_ignored_input_ids": list(
                checkpoint_values.get("skill_ignored_input_ids") or []
            ),
            "skill_degradation_reason": str(
                checkpoint_values.get("skill_degradation_reason") or ""
            ),
            "skill_degradation_user_input": str(
                checkpoint_values.get("skill_degradation_user_input") or ""
            ),
        }
    )
    logger.info(
        "已从当前 thread_id 的 Checkpoint 恢复等待输入的顶层 Skill。"
    )
    return restored_state


def _has_pending_multi_agent_resume_state(
        state: Mapping[str, Any],
) -> bool:
    """
    判断本轮是否应通过新主图输入恢复多 Agent 逻辑等待。

    功能：
        多 Agent awaiting_input 会写入 Checkpoint 后走到主图 END，不存在
        LangGraph 原生 interrupt。只要恢复出的标准任务结果仍在等待，就让
        用户补充内容重新经过语义路由和多 Agent 恢复适配器，而不是调用
        只适用于原生中断的 Command(resume=...)。

    参数：
        state:
            已合并暂停任务白名单字段的本轮主图初始 State。

    返回值：
        bool:
            存在合法 awaiting_input 多 Agent 结果时返回 True。
    """

    raw_task_result = state.get("multi_agent_task_result")
    return (
        isinstance(raw_task_result, Mapping)
        and str(raw_task_result.get("status") or "").strip()
        == "awaiting_input"
    )


def _has_pending_skill_resume_state(
        state: Mapping[str, Any],
) -> bool:
    """
    判断本轮是否应通过新主图输入恢复顶层 Skill。

    功能：
        Skill 的 awaiting_input 是写入 State 后走到 END 的逻辑等待，不是
        LangGraph 原生 interrupt；因此需要把用户补充回答作为新问题重新入图。

    参数含义：
        state:
            已合并检查点 Skill 白名单字段的本轮初始状态。

    返回值含义：
        bool:
            存在合法的等待 Skill、原始问题和目标 Agent 时返回 True。
    """

    return (
        str(state.get("skill_status") or "").strip() == "awaiting_input"
        and bool(str(state.get("skill_selected_id") or "").strip())
        and bool(str(state.get("skill_original_question") or "").strip())
        and str(state.get("skill_target_agent") or "").strip()
        in {
            "dog_knowledge_agent",
            "general_agent",
        }
    )


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
        logical_interrupt = build_multi_agent_interrupt_result_from_state(
            state=event,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            trace_id=trace_id,
            source="resume_stream_event",
        )
        if logical_interrupt is not None:
            return logical_interrupt

        answer = extract_answer_from_state(event)

        if answer:
            metadata = build_graph_business_summary(event)
            metadata["source"] = "resume_stream_event"
            return build_graph_final_result(
                answer=answer,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                trace_id=trace_id,
                metadata=metadata,
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

    task_relation_interrupt = build_task_relation_interrupt_result_from_state(
        state=final_state,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        source="current_state",
    )
    if task_relation_interrupt is not None:
        return task_relation_interrupt

    logical_interrupt = build_multi_agent_interrupt_result_from_state(
        state=final_state,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        source="current_state",
    )
    if logical_interrupt is not None:
        return logical_interrupt

    skill_interrupt = build_skill_interrupt_result_from_state(
        state=final_state,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        source="current_state",
    )
    if skill_interrupt is not None:
        return skill_interrupt

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
        metadata=build_graph_business_summary(final_state),
    )


def build_task_relation_interrupt_result_from_state(
        *,
        state: Mapping[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None,
        source: str,
) -> GraphInterruptResult | None:
    """
    把无法区分新旧任务的状态转换成统一主图中断结果。

    功能：
        当通用任务关系判断返回 ambiguous 时，不让 Tool、Skill 或多智能体
        猜测用户意图，而是把确认提示转换为 API 和 UI 已支持的
        GraphInterruptResult。

    参数含义：
        state:
            当前主图最终状态。
        thread_id:
            后续继续同一会话所需的主图线程编号。
        checkpoint_ns:
            检查点命名空间。
        trace_id:
            当前请求的链路追踪编号。
        source:
            当前状态来自最终快照还是流式恢复事件。

    返回值含义：
        GraphInterruptResult | None:
            需要用户明确新旧任务关系时返回中断结果，否则返回 None。
    """

    if (
        not isinstance(state, Mapping)
        or not bool(state.get("task_relation_requires_confirmation"))
    ):
        return None

    raw_decision = state.get("task_relation_decision")
    if (
        not isinstance(raw_decision, Mapping)
        or str(raw_decision.get("relation") or "").strip()
        != "ambiguous"
    ):
        return None

    prompt = str(
        state.get("pending_prompt")
        or state.get("final_answer")
        or "请明确说明是继续上一条任务，还是开始一个新问题。"
    ).strip()
    metadata = build_interrupt_metadata_from_state(state)
    metadata.update(
        {
            "source": source,
            "logical_interrupt": True,
            "business_status": "awaiting_input",
            "state_waiting_user_input": bool(
                state.get("waiting_user_input")
            ),
            "task_relation": dict(raw_decision),
            "pending_task_kind": str(
                state.get("task_relation_pending_kind") or ""
            ),
        }
    )
    return GraphInterruptResult(
        prompt=prompt,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        interrupt_type=GraphInterruptType.USER_CLARIFICATION,
        metadata=metadata,
    )


def build_multi_agent_interrupt_result_from_state(
        *,
        state: Mapping[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None,
        source: str,
) -> GraphInterruptResult | None:
    """
    把多 Agent 写入 State 的逻辑等待状态转换成主图中断结果。

    功能：
        多 Agent 节点当前会把 awaiting_input 写入 DogState 后走到 END，
        不会产生 LangGraph 原生 interrupt。本函数识别这类逻辑等待，并
        将它统一转换成 API 和 UI 已支持的 GraphInterruptResult。
        多 Agent 标准结果是业务状态的权威来源；DogState 中的通用等待
        标记只用于记录两层状态是否一致，不参与阻断判断。

    参数：
        state:
            当前主图 State 字段值。
        thread_id:
            后续恢复同一主图线程使用的编号。
        checkpoint_ns:
            后续定位检查点使用的命名空间。
        trace_id:
            当前请求链路追踪编号。
        source:
            当前 State 来自最终快照还是恢复执行事件。

    返回值：
        GraphInterruptResult | None:
            多 Agent 正在等待输入时返回中断结果，否则返回 None。
    """

    if not isinstance(state, Mapping):
        return None

    raw_task_result = state.get("multi_agent_task_result")
    if (
            not isinstance(raw_task_result, Mapping)
            or str(raw_task_result.get("status") or "").strip()
            != "awaiting_input"
    ):
        return None

    state_waiting_user_input = bool(state.get("waiting_user_input"))
    prompt = str(
        state.get("multi_agent_pending_prompt")
        or state.get("pending_prompt")
        or raw_task_result.get("final_answer")
        or "多 Agent 任务正在等待用户补充信息。"
    ).strip()
    metadata = build_interrupt_metadata_from_state(state)
    metadata.update(
        {
            "source": source,
            "logical_interrupt": True,
            "business_status": "awaiting_input",
            "state_waiting_user_input": state_waiting_user_input,
            "waiting_state_consistent": state_waiting_user_input,
            "multi_agent_task_id": str(
                raw_task_result.get("collaboration_id") or ""
            ),
        }
    )
    return GraphInterruptResult(
        prompt=prompt,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        interrupt_type=GraphInterruptType.USER_CLARIFICATION,
        metadata=metadata,
    )


def build_skill_interrupt_result_from_state(
        *,
        state: Mapping[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        trace_id: str | None,
        source: str,
) -> GraphInterruptResult | None:
    """
    把顶层 Skill 写入 State 的逻辑等待转换成主图中断结果。

    功能：
        Skill 缺少必需输入时会保存 awaiting_input 并走到主图 END。本函数
        将该业务等待统一转换为 API 和 UI 已支持的 GraphInterruptResult。

    参数含义：
        state:
            当前主图最终状态。
        thread_id:
            后续读取同一检查点使用的主图线程编号。
        checkpoint_ns:
            检查点命名空间。
        trace_id:
            当前请求的链路追踪编号。
        source:
            当前状态结果的来源说明。

    返回值含义：
        GraphInterruptResult | None:
            Skill 正在等待输入时返回统一中断结果，否则返回 None。
    """

    if (
        not isinstance(state, Mapping)
        or str(state.get("skill_status") or "").strip()
        != "awaiting_input"
    ):
        return None

    prompt = str(
        state.get("skill_pending_prompt")
        or state.get("pending_prompt")
        or "当前 Skill 正在等待用户补充信息。"
    ).strip()
    metadata = build_interrupt_metadata_from_state(state)
    metadata.update(
        {
            "source": source,
            "logical_interrupt": True,
            "business_status": "awaiting_input",
            "state_waiting_user_input": bool(
                state.get("waiting_user_input")
            ),
            "skill_selected_id": str(
                state.get("skill_selected_id") or ""
            ),
            "skill_target_agent": str(
                state.get("skill_target_agent") or ""
            ),
        }
    )
    return GraphInterruptResult(
        prompt=prompt,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
        interrupt_type=GraphInterruptType.USER_CLARIFICATION,
        metadata=metadata,
    )


def build_graph_business_summary(
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从主图最终 State 提取可安全向 API 透传的业务结果摘要。

    功能：
        区分“主图已经执行结束”和“用户业务任务是否成功”。普通 Agent
        默认返回 completed；多 Agent 结果会透传 completed、partial、
        failed 或 cancelled，并为失败、取消构建精简结构化原因。

    参数：
        state:
            主图执行结束后的最终 State。

    返回值：
        dict[str, Any]:
            包含 source、business_status 和可选 business_error 的元数据。
    """

    summary: dict[str, Any] = {
        "source": "current_state",
        "business_status": "completed",
        "business_error": None,
    }
    if not isinstance(state, Mapping):
        return summary

    raw_task_result = state.get("multi_agent_task_result")
    if not isinstance(raw_task_result, Mapping) or not raw_task_result:
        return summary

    raw_status = str(raw_task_result.get("status") or "").strip()
    if raw_status not in {
        "completed",
        "partial",
        "failed",
        "cancelled",
    }:
        return summary

    summary["business_status"] = raw_status
    if raw_status == "failed":
        summary["business_error"] = _build_multi_agent_failure_summary(
            raw_task_result
        )
    elif raw_status == "cancelled":
        summary["business_error"] = {
            "code": "MULTI_AGENT_TASK_CANCELLED",
            "message": str(
                raw_task_result.get("final_answer")
                or "多 Agent 任务已取消。"
            ),
            "details": {},
        }
    return summary


def _build_multi_agent_failure_summary(
        task_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构建多 Agent 失败结果的公开结构化摘要。

    功能：
        扫描步骤结果中的 timed_out 元数据；存在超时步骤时返回专用错误码
        和步骤级超时详情，否则返回通用多 Agent 任务失败错误。

    参数：
        task_result:
            DogState 中保存的可序列化 MultiAgentTaskResult。

    返回值：
        dict[str, Any]:
            包含 code、message 和 details 的业务错误摘要。
    """

    timed_out_steps: list[dict[str, Any]] = []
    raw_results = task_result.get("task_results", [])
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping):
                continue
            metadata = raw_result.get("metadata", {})
            if (
                    not isinstance(metadata, Mapping)
                    or not bool(metadata.get("timed_out"))
            ):
                continue
            timed_out_steps.append(
                {
                    "step_id": str(raw_result.get("step_id") or ""),
                    "timeout_seconds": metadata.get("timeout_seconds"),
                    "attempt_count": int(
                        metadata.get("scheduler_attempt_count", 0) or 0
                    ),
                }
            )

    message = str(
        task_result.get("error_message")
        or "多 Agent 任务执行失败。"
    )
    if timed_out_steps:
        return {
            "code": "MULTI_AGENT_STEP_TIMEOUT",
            "message": message,
            "details": {
                "timed_out_steps": timed_out_steps,
            },
        }
    return {
        "code": "MULTI_AGENT_TASK_FAILED",
        "message": message,
        "details": {},
    }


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
