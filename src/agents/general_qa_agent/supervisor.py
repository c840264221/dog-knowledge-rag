import json

from langchain_core.messages import AIMessage

from src.agents.general_qa_agent.prompts import (
    GENERAL_QA_SUPERVISOR_PROMPT
)

from src.agents.general_qa_agent.valid_workers import VALID_WORKERS,TERMINAL_SIGNALS

from src.logger import logger

from src.runtime.context import runtime_ctx


valid_workers = VALID_WORKERS + TERMINAL_SIGNALS

async def general_qa_supervisor_node(state):

    runtime_ctx.get().state().set_node(
        "general_qa_supervisor_node"
    )

    # 记录时间线
    runtime_ctx.get().timeline().add_event(

        event_type="node",

        name="general_qa_supervisor_node"
    )

    def get_llm_provider():
        from src.runtime.container.init import container
        return container.get("llm")

    llm_provider = get_llm_provider()

    main_llm = llm_provider.main_llm

    logger.info(
        "进入 general qa supervisor"
    )

    summary = {

        "question":
            state.get("question"),

        "need_tool":
            state.get("need_tool"),

        "tool_calls":
            state.get("tool_calls"),

        "tool_results":
            state.get("tool_results"),

        "has_answer":
            bool(state.get("answer")),

        "tool_confirmed":
            state.get("tool_confirmed")
    }

    response = await llm_provider.safe_ainvoke(
            main_llm,

            GENERAL_QA_SUPERVISOR_PROMPT.format_messages(

                state_summary=json.dumps(
                    summary,
                    ensure_ascii=False
                )
        ),
            fallback_response="所有模型均不可用！"
    )

    decision = response.content.strip().lower()

    logger.debug(f"valid_workers为{valid_workers}")

    if decision not in [s.lower() for s in valid_workers]:

        logger.warning(
            f"非法worker: {decision}"
        )

        decision = "answer_gen"

    logger.info(
        f"Supervisor决策: {decision}"
    )

    from src.runtime.container.init import container

    container.get("checkpoint").manager.save_checkpoint()

    return {

        "next_worker": decision,

        "messages": [

            AIMessage(
                content=f"Supervisor决策: {decision}"
            )
        ]
    }