import json

from langchain_core.messages import AIMessage

from src.agents.exact_search_agent.prompts import (
    EXACT_AGENT_SUPERVISOR_PROMPT
)

# from src.models.llm import safe_llm_ainvoke, get_instance_llm

from src.logger import logger

from src.agents.exact_search_agent.valid_workers import VALID_WORKERS, TERMINAL_SIGNALS

from src.runtime.context import runtime_ctx

async def exact_search_supervisor_node(state):

    runtime_ctx.get().state().set_node(
        "exact_search_supervisor_node"
    )

    # 记录时间线
    runtime_ctx.get().timeline().add_event(

        event_type="node",

        name="exact_search_supervisor_node"
    )


    from src.runtime.container.init import container

    def get_llm_provider():
        return container.get("llm")

    llm_provider = get_llm_provider()

    backup_llm = llm_provider.backup_llm

    logger.info(
        "进入 exact_search supervisor"
    )

    summary = {

        "question":
            state.get("question"),

        "filters":
            state.get("filters"),

        "docs_count":
            len(state.get("docs", [])),

        "retrieval_ok":
            state.get("retrieval_ok"),

        "has_answer":
            bool(state.get("answer"))
    }

    response = await llm_provider.safe_ainvoke(
        backup_llm,
        EXACT_AGENT_SUPERVISOR_PROMPT.format_messages(
            state_summary=json.dumps(
                summary,
                ensure_ascii=False
            )
        ),
        fallback_response="模型均不可用"
    )

    logger.debug(f"解析完成，结果response为：{response}")
    decision = response.content.strip().lower()

    valid_workers = VALID_WORKERS + TERMINAL_SIGNALS

    if decision not in valid_workers:

        logger.warning(
            f"非法worker: {decision}"
        )

        decision = "generate"

    logger.info(
        f"Supervisor决策: {decision}"
    )

    container.get("checkpoint").manager.save_checkpoint()

    return {

        "next_worker": decision,

        "messages": [

            AIMessage(
                content=f"Supervisor决策: {decision}"
            )
        ]
    }