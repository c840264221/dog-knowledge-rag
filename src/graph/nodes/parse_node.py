from src.parser.query_parser import parse_query_with_llm


def parse_node(state):
    print("parse_node开始......")
    print("当前state为:", state)
    result = parse_query_with_llm(state["question"])
    print("parse_node完成，输出结果为：", result)
    print("当前state为:", state)

    return {
        "intent": result["intent"],
        "filters": result["filters"],
        "tags": result["tags"],
        "features": result["features"],
        "dog_name": result["dog_name"]
    }