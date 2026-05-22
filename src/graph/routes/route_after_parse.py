from src.parser.schema import Intent

def route_after_parse(state):
    print("进入route_after_parse分流......")
    intent = state.get("intent")
    dog_name = state.get("dog_name")

    if intent == Intent.RECOMMEND.value:
        print("进入recommend分流......")
        return "recommend"

    if intent == Intent.ASK_INFO.value:
        if dog_name:
            print("进入qa_with_name分流......")
            return "qa_with_name"
        else:
            print("进入qa_general分流......")
            return "qa_general"

    return "qa_general"