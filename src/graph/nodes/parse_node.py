from langchain_core.messages import HumanMessage

from src.parser.query_parser import parse_query_with_llm
from src.retrieval.alias_loader import get_alias_dict
from src.common.decorators.validation_input import validate_question
from src.common.decorators.state_validation import validate_state
from src.common.decorators.validate_llm_output import validate_llm_output, validate_query_parse_result, default_parse_result
from src.common.decorators.safe_node import safe_node
from src.logger import logger
from src.parser.schema import Intent,QueryParseResult


alias_dict = get_alias_dict()


@safe_node(fallback=lambda state,e: {"intent":Intent.ASK_INFO.value})
@validate_question
@validate_state(["question"])
@validate_llm_output(
    validator=validate_query_parse_result,
    fallback_factory=default_parse_result
)
def parse_node(state):
    logger.info(f"进入解析用户问题节点: {state['question']}")

    messages = state.get("messages",[])
    messages.append(HumanMessage(content=state["question"]))
    result = QueryParseResult(
            intent=Intent.GENERAL.value,
            filters={},
            tags=["general"],
            features=["general"],
            dog_name=None
        ).model_dump()
    try:
        result = parse_query_with_llm(state["question"]).model_dump()
    except Exception as e:
        logger.exception(f"解析用户问题节点出错：{e}")

    dog_name = result.get("dog_name", None)
    logger.debug(f"提取到的狗狗名称: {dog_name}")
    if dog_name:
        for dog, aliases in alias_dict.items():
            if dog_name.lower() in aliases:
                dog_name = dog
                break
    logger.info(f"解析用户问题节点执行完毕: result={result}")
    return {
        "intent": result["intent"],
        "filters": result["filters"],
        "tags": result["tags"],
        "features": result["features"],
        "dog_name": dog_name,
        "messages": messages,
    }