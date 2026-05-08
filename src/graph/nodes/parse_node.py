from src.parser.query_parser import parse_query_with_llm
from src.retrieval.alias_loader import get_alias_dict

alias_dict = get_alias_dict()

def parse_node(state):
    print("parse_node开始......")
    print("当前state为:", state)
    result = parse_query_with_llm(state["question"])
    dog_name = result.get("dog_name", None)
    if dog_name:
        for dog, aliases in alias_dict.items():
            if dog_name.lower() in aliases:
                dog_name = dog
                break
    print("parse_node完成，输出结果为：", result)
    print("当前state为:", state)

    return {
        "intent": result["intent"],
        "filters": result["filters"],
        "tags": result["tags"],
        "features": result["features"],
        "dog_name": dog_name,
    }