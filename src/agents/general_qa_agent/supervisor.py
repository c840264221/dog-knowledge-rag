import json

from langchain_core.messages import AIMessage

from src.agents.general_qa_agent.prompts import (
    GENERAL_QA_SUPERVISOR_PROMPT
)
from src.agents.general_qa_agent.valid_workers import (
    VALID_WORKERS,
    TERMINAL_SIGNALS,
)
from src.logger import logger
from src.runtime.context import runtime_ctx


def build_general_qa_supervisor_node(
    llm_provider,
    checkpoint_manager=None,
    runtime_context_getter=None,
):
    """
    构建 general_qa_supervisor_node 节点。

    功能：
        创建一个真正给 LangGraph 使用的 supervisor 节点。
        外层函数负责接收 llm_provider、checkpoint_manager、runtime_context_getter 等依赖。
        内层 general_qa_supervisor_node 保持 LangGraph 需要的 state -> dict 调用格式。

    参数：
        llm_provider：
            LLM Provider（大语言模型提供者）。
            用于调用 main_llm，让模型决定下一个 worker。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于 supervisor 决策完成后保存 checkpoint。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        callable：
            返回一个 async node 函数。
            该函数接收 state，返回 dict，供 LangGraph 合并 state。

    专业名词：
        Supervisor（监督者节点）：
            负责根据当前 state 决定下一步应该交给哪个 worker 执行。

        Worker（工作节点）：
            Graph 中执行具体任务的节点，例如 tool_parse、answer_gen。

        Dependency Injection，DI（依赖注入）：
            不在 node 内部直接 import container，而是从外部传入依赖。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    valid_workers = VALID_WORKERS + TERMINAL_SIGNALS
    valid_worker_names = {
        worker.lower()
        for worker in valid_workers
    }

    async def general_qa_supervisor_node(
        state,
    ) -> dict:
        """
        根据当前 state 决定 general_qa_agent 下一步执行哪个 worker。

        功能：
            1. 写入当前 node 状态
            2. 记录 timeline 事件
            3. 从 state 中提取关键摘要
            4. 将摘要注入 supervisor prompt
            5. 调用 LLM 生成下一步 worker 决策
            6. 校验决策是否合法
            7. 非法决策兜底到 answer_gen
            8. 保存 checkpoint
            9. 返回 next_worker 和 supervisor 决策消息

        参数：
            state：
                LangGraph 当前状态。
                主要读取 question、need_tool、tool_calls、tool_results、answer、tool_confirmed 等字段。

        返回值：
            dict：
                返回需要合并进 state 的字段。
                包括 next_worker 和 messages。
        """

        node_name = "general_qa_supervisor_node"

        ctx = runtime_context_getter()

        if ctx is not None:
            ctx.state().set_node(
                node_name
            )

            ctx.timeline().add_event(
                event_type="node",
                name=node_name
            )

        logger.info(
            "进入 general qa supervisor"
        )

        summary = build_state_summary(
            state
        )

        # 最终回答已经存在时使用确定性规则结束，不再让 LLM 决定是否重复生成。
        if summary.get("has_answer"):
            logger.info(
                "Supervisor确定性决策: finish"
            )

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint()

            return {
                "next_worker": "finish",
                "messages": [
                    AIMessage(
                        content="Supervisor决策: finish"
                    )
                ]
            }

        forced_worker = decide_forced_tool_parse_worker(
            summary=summary,
        )

        if forced_worker is not None:
            logger.info(
                f"Supervisor确定性决策: {forced_worker}"
            )

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint()

            return {
                "next_worker": forced_worker,
                "messages": [
                    AIMessage(
                        content=f"Supervisor决策: {forced_worker}"
                    )
                ]
            }

        response = await llm_provider.safe_ainvoke(
            llm=llm_provider.main_llm,
            prompt=GENERAL_QA_SUPERVISOR_PROMPT.format_messages(
                state_summary=json.dumps(
                    summary,
                    ensure_ascii=False
                )
            ),
            fallback_response="所有模型均不可用！"
        )

        decision = str(
            getattr(
                response,
                "content",
                response,
            )
        ).strip().lower()

        logger.debug(
            f"valid_workers为{valid_workers}"
        )

        if decision not in valid_worker_names:
            logger.warning(
                f"非法worker: {decision}"
            )

            decision = "answer_gen"

        logger.info(
            f"Supervisor决策: {decision}"
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_checkpoint()

        return {
            "next_worker": decision,
            "messages": [
                AIMessage(
                    content=f"Supervisor决策: {decision}"
                )
            ]
        }

    return general_qa_supervisor_node


def build_state_summary(
    state,
) -> dict:
    """
    构建 supervisor 使用的 state 摘要。

    功能：
        从完整 state 中提取 supervisor 决策需要的关键字段。
        避免把整个 state 直接塞给 LLM，降低 prompt 噪音。

    参数：
        state：
            LangGraph 当前状态。

    返回值：
        dict：
            用于 supervisor prompt 的状态摘要。

    专业名词：
        State Summary（状态摘要）：
            从完整运行状态中提取出的简化信息。
    """

    return {
        "question": state.get(
            "question"
        ),
        "need_tool": state.get(
            "need_tool"
        ),
        "tool_calls": state.get(
            "tool_calls"
        ),
        "tool_results": state.get(
            "tool_results"
        ),
        "has_answer": bool(
            state.get(
                "answer"
            )
        ),
        "tool_confirmed": state.get(
            "tool_confirmed"
        ),
        "tool_round": state.get(
            "tool_round",
            0,
        ),
        "next_agent": state.get(
            "next_agent"
        ),
        "current_agent": state.get(
            "current_agent"
        ),
        "route_decision": state.get(
            "route_decision"
        ),
    }


def decide_forced_tool_parse_worker(
    summary: dict,
) -> str | None:
    """
    判断是否需要确定性地进入 tool_parse。

    功能：
        当 RootAgent 已经把问题识别为 tool_agent / tool_request 时，
        general_qa_agent 当前只是临时承接工具链路。
        此时第一次进入 general supervisor 应优先进入 tool_parse，
        避免再次依赖 LLM 判断导致天气等工具问题被直接送到 answer_gen。

    参数：
        summary：
            build_state_summary 生成的状态摘要。

    返回值：
        str | None：
            返回 "tool_parse" 表示强制进入工具解析；
            返回 None 表示继续走原有 LLM supervisor 决策。
    """

    if not is_root_tool_request(
        summary
    ):
        return None

    if summary.get(
        "has_answer"
    ):
        return None

    if summary.get(
        "tool_calls"
    ):
        return None

    if summary.get(
        "tool_results"
    ):
        return None

    if summary.get(
        "tool_confirmed"
    ):
        return None

    tool_round = summary.get(
        "tool_round",
        0,
    ) or 0

    try:
        normalized_tool_round = int(
            tool_round
        )
    except (
        TypeError,
        ValueError,
    ):
        normalized_tool_round = 0

    if normalized_tool_round > 0:
        return None

    return "tool_parse"


def is_root_tool_request(
    summary: dict,
) -> bool:
    """
    判断 RootAgent 是否已经将当前问题路由为工具请求。

    功能：
        兼容 RootAgent 写入 state 的多个字段：
        1. next_agent == tool_agent
        2. route_decision.route == tool_agent
        3. route_decision.query_type == tool_request
        4. route_decision.requires_tool == True

    参数：
        summary：
            build_state_summary 生成的状态摘要。

    返回值：
        bool：
            True 表示当前问题是 RootAgent 判定的工具请求；
            False 表示不是。
    """

    if summary.get(
        "next_agent"
    ) == "tool_agent":
        return True

    route_decision = summary.get(
        "route_decision"
    )

    if not isinstance(
        route_decision,
        dict,
    ):
        return False

    if route_decision.get(
        "route"
    ) == "tool_agent":
        return True

    if route_decision.get(
        "query_type"
    ) == "tool_request":
        return True

    return bool(
        route_decision.get(
            "requires_tool"
        )
    )
