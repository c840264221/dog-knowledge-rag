from langchain_core.messages import HumanMessage

from src.parser.query_parser import parse_query_with_llm
from src.retrieval.alias_loader import get_alias_dict
from src.logger import logger


alias_dict = get_alias_dict()

def parse_node(state):
    logger.info(f"进入解析用户问题节点: {state['question']}")

    messages = state.get("messages",[])
    messages.append(HumanMessage(content=state["question"]))
    try:
        result = parse_query_with_llm(state["question"])
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