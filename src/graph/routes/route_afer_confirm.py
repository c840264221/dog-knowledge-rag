from src.graph.states.state import DogState

def route_after_confirm(state: DogState) -> str:
    if state.get("need_tool", False) and state.get("tool_calls"):
        return "call_tool"
    else:
        return "no_call_tool"