import json

from langchain_core.messages import AIMessage

from src.agents.exact_search_agent.prompts import (
    EXACT_AGENT_SUPERVISOR_PROMPT
)

from src.models.llm import get_backup_llm, safe_llm_invoke, get_instance_llm

from src.logger import logger

from src.agents.exact_search_agent.valid_workers import VALID_WORKERS, TERMINAL_SIGNALS


llm = get_backup_llm()
# llm = get_instance_llm()

def exact_search_supervisor_node(state):

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

    response = safe_llm_invoke(
        llm,
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

    return {

        "next_worker": decision,

        "messages": [

            AIMessage(
                content=f"Supervisor决策: {decision}"
            )
        ]
    }